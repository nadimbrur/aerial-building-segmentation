
import argparse
import csv
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Patch
from PIL import Image
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

import train_base as pipeline
import train_segformer as seg

TILE_SIZE = 5000  # Inria tiles are 5000 x 5000 px
SPLITS = ("train", "val", "test")
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
# Columns of the per-tile statistics array
COLS = ["tp", "fp", "fn", "tn", "b_inter", "b_union", "bp_hit", "bp_sum", "br_hit", "br_sum"]
METRIC_COLS = ["iou", "dice", "precision", "recall", "pixel_accuracy", "boundary_iou", "boundary_f1"]


def city_of(tile_id):
    return re.sub(r"\d+$", "", tile_id)


# --------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------
class IndexedPatches(torch.utils.data.Dataset):
    """Wraps InriaPatches so each item also returns its index (to recover tile id / position)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, mask = self.base[index]
        return image, mask, index


def valid_mask(patch_index, indices, patch_size, device):
    """1 for pixels that lie inside the real 5000 x 5000 tile, 0 for reflect/zero padding."""
    valid = torch.zeros(len(indices), 1, patch_size, patch_size, device=device)
    for k, i in enumerate(indices.tolist()):
        _, x, y = patch_index[i]
        valid[k, :, : max(0, min(patch_size, TILE_SIZE - y)), : max(0, min(patch_size, TILE_SIZE - x))] = 1.0
    return valid


def check_tile_sizes(data_root, tile_ids):
    image_dir = data_root / "train" / "images"
    for tile_id in tile_ids:
        with Image.open(image_dir / f"{tile_id}.tif") as image:
            assert image.size == (TILE_SIZE, TILE_SIZE), f"{tile_id} has size {image.size}, expected 5000x5000"


# --------------------------------------------------------------------------------------
# Leakage audit
# --------------------------------------------------------------------------------------
def audit_split(ids_by_split, datasets, patch_size, out_path):
    print("\n=== Spatial-leakage audit ===")
    for a, b in combinations(SPLITS, 2):
        overlap = set(ids_by_split[a]) & set(ids_by_split[b])
        assert not overlap, f"Tile overlap between {a} and {b}: {sorted(overlap)}"
    print("  [OK] tile IDs are pairwise disjoint across train / val / test")

    cities = {name: sorted({city_of(t) for t in ids}) for name, ids in ids_by_split.items()}
    for a, b in combinations(SPLITS, 2):
        overlap = set(cities[a]) & set(cities[b])
        assert not overlap, f"City overlap between {a} and {b}: {sorted(overlap)}"
    print(f"  [OK] cities are pairwise disjoint: {cities}")

    for name, dataset in datasets.items():
        patch_index = dataset.patch_index
        assert {t for t, _, _ in patch_index} == set(ids_by_split[name]), f"{name}: patches reference foreign tiles"
        assert len(set(patch_index)) == len(patch_index), f"{name}: duplicate patches"
        assert all(x % patch_size == 0 and y % patch_size == 0 for _, x, y in patch_index), f"{name}: patch grid is not aligned"
    print(f"  [OK] every patch belongs to exactly one split; patches sit on a non-overlapping {patch_size}px grid")

    manifest = {
        "patch_size": patch_size,
        "splits": {
            name: {"cities": cities[name], "n_tiles": len(ids_by_split[name]), "n_patches": len(datasets[name]), "tiles": ids_by_split[name]}
            for name in SPLITS
        },
    }
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Saved split manifest to {out_path}")


# --------------------------------------------------------------------------------------
# Boundary operators (square structuring element; padding never creates a border contour)
# --------------------------------------------------------------------------------------
def erode(mask, radius):
    return -F.max_pool2d(-mask, kernel_size=2 * radius + 1, stride=1, padding=radius)


def dilate(mask, radius):
    return F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=radius)


def boundary_band(mask, radius):
    """Mask pixels within `radius` px of the mask edge (Boundary IoU region)."""
    return mask * (1.0 - erode(mask, radius))


# --------------------------------------------------------------------------------------
# Core evaluation pass
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_split(model, dataset, args, device, loss_fn, thresholds, with_boundary, sample_pixels=0, record_thr=None):
    loader = torch.utils.data.DataLoader(
        IndexedPatches(dataset), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    n_thr = len(thresholds)
    tile_stats = defaultdict(lambda: np.zeros((n_thr, len(COLS)), dtype=np.float64))
    total_loss, n_seen = 0.0, 0
    budget = max(1, sample_pixels // max(len(loader), 1))
    probs_sample, labels_sample, records = [], [], []
    model.eval()

    for images, masks, indices in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid = valid_mask(dataset.patch_index, indices, args.patch_size, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)["out"]
        logits = logits.float()
        total_loss += loss_fn(logits, masks).item() * images.size(0)
        n_seen += images.size(0)
        probs = torch.sigmoid(logits)
        tiles = [dataset.patch_index[i][0] for i in indices.tolist()]

        if sample_pixels:
            inside = valid > 0
            flat_p, flat_l = probs[inside], masks[inside]
            keep = torch.randperm(flat_p.numel(), device=device)[: min(flat_p.numel(), budget)]
            probs_sample.append(flat_p[keep].cpu().numpy())
            labels_sample.append(flat_l[keep].cpu().numpy())

        if with_boundary:  # ground-truth boundary terms do not depend on the threshold
            gt_band = boundary_band(masks, args.boundary_dilation)
            gt_contour = boundary_band(masks, 1)
        gt_count = masks.sum((1, 2, 3))

        for t, thr in enumerate(thresholds):
            preds = (probs >= thr).float() * valid  # padding never counts as prediction
            tp = (preds * masks).sum((1, 2, 3))
            fp = (preds * (1 - masks)).sum((1, 2, 3))
            fn = ((1 - preds) * masks).sum((1, 2, 3))
            tn = (valid * (1 - preds) * (1 - masks)).sum((1, 2, 3))
            columns = [tp, fp, fn, tn]
            if with_boundary:
                pred_band = boundary_band(preds, args.boundary_dilation)
                inter = (gt_band * pred_band).sum((1, 2, 3))
                union = (gt_band + pred_band - gt_band * pred_band).sum((1, 2, 3))
                pred_contour = boundary_band(preds, 1)
                columns += [
                    inter, union,
                    (pred_contour * dilate(gt_contour, args.boundary_tolerance)).sum((1, 2, 3)), pred_contour.sum((1, 2, 3)),
                    (gt_contour * dilate(pred_contour, args.boundary_tolerance)).sum((1, 2, 3)), gt_contour.sum((1, 2, 3)),
                ]
            else:
                columns += [torch.zeros_like(tp)] * 6
            rows = torch.stack(columns, dim=1).cpu().numpy()
            for b, tile in enumerate(tiles):
                tile_stats[tile][t] += rows[b]

            if record_thr is not None and t == record_thr:
                valid_frac = valid.mean((1, 2, 3)).cpu().numpy()
                for b, index in enumerate(indices.tolist()):
                    tpb, fpb, fnb = (float(v) for v in (tp[b], fp[b], fn[b]))
                    records.append({
                        "idx": index, "tile": tiles[b], "x": dataset.patch_index[index][1], "y": dataset.patch_index[index][2],
                        "iou": tpb / (tpb + fpb + fnb + 1e-7), "tp": tpb, "fp": fpb, "fn": fnb,
                        "gt_frac": float(gt_count[b]) / (args.patch_size ** 2), "valid_frac": float(valid_frac[b]),
                    })

    samples = None
    if sample_pixels:
        samples = (np.concatenate(probs_sample), np.concatenate(labels_sample))
    return {"loss": total_loss / n_seen, "tile_stats": dict(tile_stats), "samples": samples, "records": records}


# --------------------------------------------------------------------------------------
# Metrics / aggregation
# --------------------------------------------------------------------------------------
def aggregate(tile_stats, predicate=lambda tile: True):
    return np.sum([v for tile, v in tile_stats.items() if predicate(tile)], axis=0)  # (n_thr, n_cols)


def summarize(row, eps=1e-7):
    tp, fp, fn, tn, b_inter, b_union, bp_hit, bp_sum, br_hit, br_sum = row
    metrics = pipeline.metrics_from_counts(tp, fp, fn, tn)
    b_precision, b_recall = bp_hit / (bp_sum + eps), br_hit / (br_sum + eps)
    metrics["boundary_iou"] = b_inter / (b_union + eps)
    metrics["boundary_f1"] = 2 * b_precision * b_recall / (b_precision + b_recall + eps)
    metrics["boundary_precision"], metrics["boundary_recall"] = b_precision, b_recall
    metrics["fg_fraction"] = (tp + fn) / (tp + fp + fn + tn + eps)
    return metrics


def print_table(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'Split / scope':<22}" + "".join(f"{c:>15}" for c in METRIC_COLS))
    for label, metrics in rows:
        print(f"{label:<22}" + "".join(f"{metrics[c]:>15.4f}" for c in METRIC_COLS))


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_threshold_sweep(thresholds, val_stats, best_thr, out_path):
    metrics = [summarize(row) for row in val_stats]
    figure, axis = plt.subplots(figsize=(6, 4))
    for key in ("iou", "precision", "recall"):
        axis.plot(thresholds, [m[key] for m in metrics], marker="o", markersize=3, label=key)
    axis.axvline(best_thr, color="k", linestyle="--", linewidth=1, label=f"best thr={best_thr:.2f}")
    axis.set_xlabel("Probability threshold")
    axis.set_ylabel("Score")
    axis.set_title("Threshold sweep on validation city (Vienna)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def plot_confusion_matrices(counts_by_split, threshold, out_path):
    figure, axes = plt.subplots(1, len(counts_by_split), figsize=(4.5 * len(counts_by_split), 4))
    for axis, (split, (tp, fp, fn, tn)) in zip(np.atleast_1d(axes), counts_by_split.items()):
        matrix = np.array([[tn, fp], [fn, tp]])
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        axis.set_xticks([0, 1], ["Background", "Building"])
        axis.set_yticks([0, 1], ["Background", "Building"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Ground truth")
        axis.set_title(f"{split} (thr={threshold:.2f})")
        for r in range(2):
            for c in range(2):
                axis.text(c, r, f"{int(matrix[r, c]):,}\n({normalized[r, c]:.1%})", ha="center", va="center",
                          color="white" if normalized[r, c] > 0.5 else "black", fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def plot_roc_pr(samples_by_split, out_dir):
    roc_fig, roc_axis = plt.subplots(figsize=(5, 5))
    pr_fig, pr_axis = plt.subplots(figsize=(5, 5))
    summary = {}
    for split in SPLITS:
        probs, labels = samples_by_split[split]
        if labels.min() == labels.max():
            print(f"  Skipping ROC/PR for {split}: only one class in the pixel sample.")
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        precision, recall, _ = precision_recall_curve(labels, probs)
        summary[split] = {"roc_auc": auc(fpr, tpr), "pr_auc": average_precision_score(labels, probs)}
        roc_axis.plot(fpr, tpr, label=f"{split} (AUC={summary[split]['roc_auc']:.3f})")
        pr_axis.plot(recall, precision, label=f"{split} (AP={summary[split]['pr_auc']:.3f})")
    roc_axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    roc_axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (pixel-level, subsampled)")
    roc_axis.legend(loc="lower right", fontsize=8)
    pr_axis.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (pixel-level, subsampled)")
    pr_axis.legend(loc="lower left", fontsize=8)
    for figure, name in ((roc_fig, "roc_curve.png"), (pr_fig, "pr_curve.png")):
        figure.tight_layout()
        figure.savefig(out_dir / name, dpi=150)
        plt.close(figure)
    return summary


def plot_learning_curves(out_dir, best_epoch, out_path):
    last_path = out_dir / "last_checkpoint.pt"
    if not last_path.exists():
        print("  No last_checkpoint.pt found; skipping learning curves.")
        return
    history = torch.load(last_path, map_location="cpu", weights_only=False).get("history")
    if not history:
        print("  Checkpoint has no history; skipping learning curves.")
        return
    epochs = [h["epoch"] for h in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, keys, ylabel in ((axes[0], ("train_loss", "val_loss"), "Loss"), (axes[1], ("train_iou", "val_iou"), "IoU")):
        for key in keys:
            axis.plot(epochs, [h[key] for h in history], marker="o", markersize=3, label=key)
        if isinstance(best_epoch, int):
            axis.axvline(best_epoch, color="k", linestyle="--", linewidth=1, label=f"selected epoch {best_epoch}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"  Saved learning curves to {out_path}")


# --------------------------------------------------------------------------------------
# Qualitative analysis
# --------------------------------------------------------------------------------------
def pick(records, key, k, descending):
    """Top-k records by `key`, at most one patch per tile so examples are diverse."""
    chosen, used = [], set()
    for record in sorted(records, key=key, reverse=descending):
        if record["tile"] in used:
            continue
        chosen.append(record)
        used.add(record["tile"])
        if len(chosen) == k:
            break
    return chosen


@torch.no_grad()
def render_examples(model, dataset, records, device, threshold, title, out_path):
    if not records:
        print(f"  No candidates for '{title}'; skipping.")
        return
    figure, axes = plt.subplots(len(records), 4, figsize=(14, 3.4 * len(records)), squeeze=False)
    model.eval()
    for row, record in enumerate(records):
        image, mask = dataset[record["idx"]]
        prob = torch.sigmoid(model(image.unsqueeze(0).to(device))["out"].float())[0, 0].cpu().numpy()
        prediction, truth = prob >= threshold, mask[0].numpy() > 0.5
        rgb = np.clip(image.numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)
        error = np.zeros((*truth.shape, 3))
        error[truth & prediction] = (0.2, 0.7, 0.2)   # true positive
        error[~truth & prediction] = (0.9, 0.1, 0.1)  # false positive
        error[truth & ~prediction] = (0.1, 0.3, 0.9)  # false negative
        panels = [(rgb, "Aerial patch"), (truth, "Ground truth"), (prediction, "Prediction"), (error, "Error map")]
        for column, (data, name) in enumerate(panels):
            axes[row, column].imshow(data, cmap=None if data.ndim == 3 else "gray")
            axes[row, column].axis("off")
            axes[row, column].set_title(name if column else f"{name}\n{record['tile']} @({record['x']},{record['y']}) IoU={record['iou']:.3f}", fontsize=9)
    figure.legend(handles=[Patch(color=(0.2, 0.7, 0.2), label="TP"), Patch(color=(0.9, 0.1, 0.1), label="FP"), Patch(color=(0.1, 0.3, 0.9), label="FN")],
                  loc="lower center", ncol=3, frameon=False)
    figure.suptitle(f"{title} (test city, threshold={threshold:.2f})")
    figure.tight_layout(rect=(0, 0.03, 1, 0.98))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"  Saved {out_path}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SegFormer-B3 checkpoint")
    parser.add_argument("--data-root", type=Path, default=Path("AerialImageDataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_city_split_segformer"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to <out-dir>/best_val_iou.pt")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pos-weight", type=float, default=None, help="BCE positive weight used only for the reported loss; estimated from train if omitted")
    parser.add_argument("--boundary-dilation", type=int, default=3, help="Boundary IoU band width in pixels (0.3 m/px)")
    parser.add_argument("--boundary-tolerance", type=int, default=3, help="Boundary F1 matching tolerance in pixels")
    parser.add_argument("--train-max-patches", type=int, default=0, help="Evaluate a strided subset of train patches (0 = all)")
    parser.add_argument("--max-pixels-per-split", type=int, default=2_000_000, help="Pixel subsample size for ROC/PR")
    parser.add_argument("--num-examples", type=int, default=4, help="Rows per qualitative figure")
    parser.add_argument("--min-fg", type=float, default=0.05, help="Min building fraction for a patch to count as a success/failure candidate")
    args = parser.parse_args()

    pipeline.set_seed(pipeline.SEED)
    out_dir = args.out_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or (args.out_dir / "best_val_iou.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\nLoading checkpoint: {checkpoint_path}")

    # pretrained=False only builds the architecture; the trained weights come from the checkpoint.
    model = seg.build_model(pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    best_epoch = checkpoint.get("epoch", "unknown")
    print(f"Checkpoint epoch (selected on validation IoU): {best_epoch}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} ({total / 1e6:.2f} M)")

    # ---- Splits and leakage audit -------------------------------------------------
    ids_by_split = {
        "train": pipeline.tile_ids_for_cities(args.data_root, pipeline.TRAIN_CITIES),
        "val": pipeline.tile_ids_for_cities(args.data_root, (pipeline.VAL_CITY,)),
        "test": pipeline.tile_ids_for_cities(args.data_root, (pipeline.TEST_CITY,)),
    }
    check_tile_sizes(args.data_root, [t for ids in ids_by_split.values() for t in ids])
    datasets = {
        name: pipeline.InriaPatches(pipeline.build_patch_index(ids, args.data_root, args.patch_size), args.data_root, args.patch_size)
        for name, ids in ids_by_split.items()
    }
    audit_split(ids_by_split, datasets, args.patch_size, out_dir / "split_manifest.json")
    print("\n=== Dataset / patch sizes ===")
    for name in SPLITS:
        print(f"{name:<6} tiles={len(ids_by_split[name]):>4}  patches={len(datasets[name]):>6}")
    if args.train_max_patches:
        stride = max(1, len(datasets["train"]) // args.train_max_patches)
        datasets["train"].patch_index = datasets["train"].patch_index[::stride]
        print(f"Train evaluation subsampled with stride {stride}: {len(datasets['train'])} patches")

    # ---- Loss function (for reporting the threshold-independent loss) -------------
    if args.pos_weight is None:
        import random
        sample = random.sample(range(len(datasets["train"])), min(300, len(datasets["train"])))
        fg = float(np.mean([datasets["train"][i][1].mean().item() for i in sample]))
        pos_weight = min((1 - fg) / max(fg, 1e-6), 10.0)
        print(f"\nEDA (train): foreground={fg:.3f}; BCE positive-class weight={pos_weight:.2f}")
    else:
        pos_weight = args.pos_weight
    loss_fn = pipeline.DiceWeightedBCELoss(pos_weight).to(device)

    # ---- Phase A: choose the threshold on the validation city only ----------------
    grid = np.round(np.arange(0.05, 0.951, 0.05), 2).tolist()
    print("\nPhase A: threshold search on validation city...")
    sweep = evaluate_split(model, datasets["val"], args, device, loss_fn, grid, with_boundary=False)
    sweep_stats = aggregate(sweep["tile_stats"])
    val_ious = [summarize(row)["iou"] for row in sweep_stats]
    best_thr = grid[int(np.argmax(val_ious))]
    print(f"Best threshold (max validation IoU): {best_thr:.2f} (IoU={max(val_ious):.4f})")
    plot_threshold_sweep(grid, sweep_stats, best_thr, out_dir / "threshold_sweep_val.png")

    # ---- Phase B: full evaluation at thr=0.5 and thr=best -------------------------
    thresholds = sorted({0.5, best_thr})
    best_index, default_index = thresholds.index(best_thr), thresholds.index(0.5)
    results = {}
    for split in SPLITS:
        print(f"\nPhase B: evaluating '{split}' ({len(datasets[split])} patches)...")
        results[split] = evaluate_split(
            model, datasets[split], args, device, loss_fn, thresholds, with_boundary=True,
            sample_pixels=args.max_pixels_per_split, record_thr=best_index if split == "test" else None,
        )

    print("\n=== Loss (threshold-independent) ===")
    for split in SPLITS:
        print(f"{split:<6} loss={results[split]['loss']:.4f}")

    csv_rows = []
    for label, index in (("thr=0.5", default_index), (f"thr={best_thr:.2f} (val-tuned)", best_index)):
        table = []
        for split in SPLITS:
            stats = results[split]["tile_stats"]
            table.append((split, summarize(aggregate(stats)[index])))
            csv_rows.append({"split": split, "scope": "overall", "threshold": thresholds[index], **table[-1][1]})
            for city in sorted({city_of(t) for t in stats}):
                metrics = summarize(aggregate(stats, lambda t, c=city: city_of(t) == c)[index])
                table.append((f"  {split}/{city}", metrics))
                csv_rows.append({"split": split, "scope": city, "threshold": thresholds[index], **metrics})
        print_table(f"Results at {label}", table)
        counts = {s: aggregate(results[s]["tile_stats"])[index][:4] for s in SPLITS}
        plot_confusion_matrices(counts, thresholds[index], out_dir / f"confusion_matrices_thr{thresholds[index]:.2f}.png")

    fieldnames = ["split", "scope", "threshold"] + METRIC_COLS + ["boundary_precision", "boundary_recall", "fg_fraction"]
    with open(out_dir / "metrics_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    # ---- Per-tile test results (variance, not just a pooled number) ---------------
    per_tile = []
    for tile, stats in sorted(results["test"]["tile_stats"].items()):
        per_tile.append({"tile": tile, **summarize(stats[best_index])})
    with open(out_dir / "per_tile_test_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tile", "fg_fraction"] + METRIC_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_tile)
    print(f"\n=== Test tiles at thr={best_thr:.2f}: macro mean +/- std over {len(per_tile)} tiles ===")
    for key in ("iou", "dice", "boundary_iou", "boundary_f1"):
        values = np.array([row[key] for row in per_tile])
        print(f"  {key:<14} {values.mean():.4f} +/- {values.std():.4f}  (min {values.min():.4f}, max {values.max():.4f})")

    print("\n=== Foreground fraction per city (context for cross-city differences) ===")
    for split in SPLITS:
        stats = results[split]["tile_stats"]
        for city in sorted({city_of(t) for t in stats}):
            print(f"  {city:<10} {summarize(aggregate(stats, lambda t, c=city: city_of(t) == c)[default_index])['fg_fraction']:.3f}")

    # ---- Curves --------------------------------------------------------------------
    print("\nComputing ROC / PR curves...")
    auc_summary = plot_roc_pr({s: results[s]["samples"] for s in SPLITS}, out_dir)
    print("=== AUC summary (pixel-level, subsampled) ===")
    for split, values in auc_summary.items():
        print(f"{split:<6} ROC-AUC={values['roc_auc']:.4f}  PR-AUC={values['pr_auc']:.4f}")
    plot_learning_curves(args.out_dir, best_epoch, out_dir / "learning_curves.png")

    # ---- Qualitative analysis on the held-out test city ----------------------------
    print("\nRendering qualitative panels...")
    records = [r for r in results["test"]["records"] if r["valid_frac"] == 1.0]
    candidates = [r for r in records if r["gt_frac"] >= args.min_fg]
    k = args.num_examples
    render_examples(model, datasets["test"], pick(candidates, lambda r: r["iou"], k, True), device, best_thr,
                    "Successful extractions", out_dir / "qualitative_success.png")
    render_examples(model, datasets["test"], pick(candidates, lambda r: r["iou"], k, False), device, best_thr,
                    "Failure cases: lowest IoU", out_dir / "qualitative_failure.png")
    render_examples(model, datasets["test"], pick([r for r in records if r["gt_frac"] < 0.005], lambda r: r["fp"], k, True), device, best_thr,
                    "False alarms: predicted buildings where almost none exist", out_dir / "qualitative_false_alarm.png")

    print(f"\nAll evaluation artifacts saved under {out_dir}")


if __name__ == "__main__":
    main()