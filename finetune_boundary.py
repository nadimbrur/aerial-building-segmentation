"""Short boundary-aware fine-tune of the trained SegFormer-B3 (city-disjoint split, test city untouched).

Loss   : existing 0.5*weighted-BCE + 0.5*Dice  +  edge_weight * BCE over a +/-3 px zone around every
         ground-truth contour (pushes the network to place edges precisely).
Select : the checkpoint with the best validation-city (Vienna) score, where
         score = 0.5 * (IoU + Boundary F1), maximised over a few thresholds.
         The test city is NOT loaded here at all.

Run:
    python finetune_boundary.py --init-checkpoint outputs_city_split_segformer/best_val_iou.pt
Resume after a power cut / crash: run the SAME command again. It automatically continues from
<out-dir>/last_checkpoint.pt (saved atomically after every epoch). Add --no-resume to start over.
Options worth trying: --edge-weight 1.0, --epochs 15, --lr 2e-5
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_base as pipeline
import train_segformer as seg
from boundary_utils import (BoundaryAwareLoss, IndexedPatches, confusion_and_boundary, gt_terms, summarize,
                            valid_mask)


def atomic_save(obj, path):
    """Write to a temp file, flush to disk, then rename: a power cut leaves the old file intact."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        torch.save(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@torch.no_grad()
def validate(model, loader, dataset, thresholds, args, device):
    model.eval()
    stats = np.zeros((len(thresholds), 10), dtype=np.float64)
    for images, masks, indices in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        valid = valid_mask(dataset.patch_index, indices, args.patch_size, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)["out"]
        probs = torch.sigmoid(logits.float())
        gt = gt_terms(masks, args.boundary_dilation, args.boundary_tolerance)
        for t, threshold in enumerate(thresholds):
            rows = confusion_and_boundary((probs >= threshold).float(), masks, valid,
                                          args.boundary_dilation, args.boundary_tolerance, gt)
            stats[t] += rows.sum(0).cpu().numpy()
    return [summarize(row) for row in stats]


def main():
    parser = argparse.ArgumentParser(description="Boundary-aware fine-tuning")
    parser.add_argument("--data-root", type=Path, default=Path("AerialImageDataset"))
    parser.add_argument("--init-checkpoint", type=Path, default=Path("outputs_city_split_segformer/best_val_iou.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_city_split_segformer_boundary"))
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pos-weight", type=float, default=5.92, help="Same BCE positive weight as the original training run")
    parser.add_argument("--edge-weight", type=float, default=0.5)
    parser.add_argument("--edge-radius", type=int, default=3, help="Half-width (px) of the edge zone used in the loss")
    parser.add_argument("--boundary-dilation", type=int, default=3)
    parser.add_argument("--boundary-tolerance", type=int, default=3)
    parser.add_argument("--select-thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.85])
    parser.add_argument("--no-resume", action="store_true", help="Ignore last_checkpoint.pt and start a fresh fine-tune")
    args = parser.parse_args()

    pipeline.set_seed(pipeline.SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = seg.build_model(pretrained=False).to(device)  # architecture only; weights come from the checkpoint
    checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"Initialised from {args.init_checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    train_ids = pipeline.tile_ids_for_cities(args.data_root, pipeline.TRAIN_CITIES)
    val_ids = pipeline.tile_ids_for_cities(args.data_root, (pipeline.VAL_CITY,))
    assert not set(train_ids) & set(val_ids)
    train_ds = pipeline.InriaPatches(pipeline.build_patch_index(train_ids, args.data_root, args.patch_size), args.data_root, args.patch_size, augment=True)
    val_ds = pipeline.InriaPatches(pipeline.build_patch_index(val_ids, args.data_root, args.patch_size), args.data_root, args.patch_size)
    print(f"Train patches: {len(train_ds)} | Val patches: {len(val_ds)} | test city not loaded")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,
                              pin_memory=pin, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(IndexedPatches(val_ds), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    loss_fn = BoundaryAwareLoss(pipeline.DiceWeightedBCELoss(args.pos_weight).to(device), args.edge_weight, args.edge_radius)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=pin)

    last_path = args.out_dir / "last_checkpoint.pt"
    best_path = args.out_dir / "best_boundary.pt"
    start_epoch = 1

    if last_path.exists() and not args.no_resume:
        # Resume after an interruption (e.g. power cut): restores weights, optimiser, scheduler, AMP scaler,
        # best score and history, and continues from the next epoch. Keep the same --epochs value.
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best_score, history, start_epoch = state["best_score"], state["history"], state["epoch"] + 1
        print(f"Resuming from {last_path}: completed epoch {state['epoch']}, best val score so far={best_score:.4f}")
    else:
        # Reference: the starting checkpoint on the validation city (so improvement is measurable).
        base = validate(model, val_loader, val_ds, args.select_thresholds, args, device)
        base_scores = [0.5 * (m["iou"] + m["boundary_f1"]) for m in base]
        best_score = max(base_scores)
        print(f"Start (before fine-tuning): best val score={best_score:.4f} at thr={args.select_thresholds[int(np.argmax(base_scores))]}")
        history = []

    if start_epoch > args.epochs:
        print(f"Nothing to do: {start_epoch - 1} epochs already completed (--epochs {args.epochs}).")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        torch.manual_seed(pipeline.SEED + epoch)  # different shuffle each epoch, identical after a resume
        model.train()
        running, batches = 0.0, 0
        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=pin):
                logits = model(images)["out"]
            loss = loss_fn(logits.float(), masks)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running, batches = running + loss.item(), batches + 1
            print(f"\r  [train] epoch {epoch} batch {batches}/{len(train_loader)} loss={loss.item():.4f}", end="", flush=True)
        print()
        scheduler.step()

        metrics = validate(model, val_loader, val_ds, args.select_thresholds, args, device)
        scores = [0.5 * (m["iou"] + m["boundary_f1"]) for m in metrics]
        best_t = int(np.argmax(scores))
        m = metrics[best_t]
        history.append({"epoch": epoch, "train_loss": running / batches, "val_score": scores[best_t], "threshold": args.select_thresholds[best_t],
                        "val_iou": m["iou"], "val_boundary_f1": m["boundary_f1"], "val_boundary_iou": m["boundary_iou"]})
        print(f"Epoch {epoch}/{args.epochs}: train loss={running / batches:.4f} | val thr={args.select_thresholds[best_t]:.2f} "
              f"IoU={m['iou']:.4f} BIoU={m['boundary_iou']:.4f} BF1={m['boundary_f1']:.4f} score={scores[best_t]:.4f}")

        if scores[best_t] > best_score:
            best_score = scores[best_t]
            atomic_save({"model": model.state_dict(), "epoch": epoch, "threshold": args.select_thresholds[best_t], "val_metrics": m,
                         "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, best_path)
            print(f"  Saved new best (val score={best_score:.4f})")
        (args.out_dir / "finetune_history.json").write_text(json.dumps(history, indent=2))
        atomic_save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "best_score": best_score, "history": history}, last_path)
        print(f"  Saved resume checkpoint ({last_path.name}, epoch {epoch})")

    if not best_path.exists():
        print("\nNo epoch beat the starting checkpoint on validation; try --edge-weight 1.0 or a different --lr.")
    else:
        print(f"\nDone. Best checkpoint: {args.out_dir / 'best_boundary.pt'}")


if __name__ == "__main__":
    main()