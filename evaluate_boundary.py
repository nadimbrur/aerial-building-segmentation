"""Compare checkpoints and inference modes on whole tiles (baseline vs boundary-aware, overlap, TTA).

For every (model, config) the script
  1. predicts each validation (Vienna) and test (West Tyrol) tile as a whole (no patch-border seams),
  2. picks the decision threshold on the VALIDATION city only (max IoU by default),
  3. reports validation and test metrics at that threshold: IoU, Dice, precision, recall,
     Boundary IoU, Boundary F1, plus mean +/- std of per-tile IoU / Boundary F1 on test.

Configs:  s512 = non-overlapping windows (matches the original patch inference, but scored on whole tiles)
          s256 = 50% overlap with smooth blending
          s256+tta4 / s256+tta8 = overlap + flip/rotation test-time augmentation (4 or 8 views)

Run:
    python evaluate_boundary.py --models baseline=outputs_city_split_segformer/best_val_iou.pt \
        boundary=outputs_city_split_segformer_boundary/best_boundary.pt
Every finished tile is cached under <out-dir>/cache/, so re-running the same command after a crash or
an added config only computes what is missing (use --recompute to force a fresh run).
Runtime note: s256+tta4 costs ~16x the s512 pass. Use --configs s512 s256 first for a quick run.
"""

import argparse
import csv
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

import train_base as pipeline
import train_segformer as seg
from boundary_utils import confusion_and_boundary, gt_terms, load_tile, predict_tile, summarize

SPLIT_CITIES = {"val": (pipeline.VAL_CITY,), "test": (pipeline.TEST_CITY,)}


def load_model(path, device):
    model = seg.build_model(pretrained=False).to(device)  # architecture only; weights come from the checkpoint
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint.get("epoch", "?")


def parse_config(name):
    stride_part, _, tta = name.partition("+")
    return int(stride_part.lstrip("s")), tta or "none"


def cache_key(name, path, args, thresholds):
    """Cache folder id: changes whenever the checkpoint file or any metric setting changes."""
    info = os.stat(path)
    text = f"{name}|{Path(path).resolve()}|{info.st_size}|{info.st_mtime_ns}|{args.patch_size}|{args.boundary_dilation}|{args.boundary_tolerance}|{thresholds}"
    return hashlib.md5(text.encode()).hexdigest()[:10]


@torch.no_grad()
def evaluate_tiles(model, tile_ids, args, thresholds, stride, tta, device, image_dir, mask_dir, cache_dir):
    """Return {tile: (n_thr, 10) stats}. Every finished tile is saved to disk immediately, so an
    interrupted run resumes from the last finished tile instead of starting over."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats, reused = {}, 0
    for n, tile_id in enumerate(tile_ids, start=1):
        cache_file = cache_dir / f"{tile_id}.npy"
        if cache_file.exists() and not args.recompute:
            try:
                stats[tile_id] = np.load(cache_file)
                reused += 1
                print(f"\r    tile {n}/{len(tile_ids)} (cached: {reused})", end="", flush=True)
                continue
            except Exception:
                print(f"\n    cache file {cache_file.name} unreadable; recomputing")
        image, mask = load_tile(image_dir, mask_dir, tile_id)
        prob = predict_tile(model, image, device, args.patch_size, stride, tta, args.window_batch)
        masks = torch.from_numpy(mask).to(device).view(1, 1, *mask.shape)
        gt = gt_terms(masks, args.boundary_dilation, args.boundary_tolerance)
        rows = []
        for threshold in thresholds:
            preds = (prob >= threshold).float().view(1, 1, *mask.shape)
            rows.append(confusion_and_boundary(preds, masks, None, args.boundary_dilation, args.boundary_tolerance, gt)[0].cpu().numpy())
        stats[tile_id] = np.stack(rows).astype(np.float64)
        temporary = cache_dir / f"{tile_id}.tmp.npy"
        np.save(temporary, stats[tile_id])
        os.replace(temporary, cache_file)  # atomic: a power cut never leaves a half-written file
        print(f"\r    tile {n}/{len(tile_ids)} (cached: {reused})", end="", flush=True)
    print()
    return stats


def save_csvs(summary_rows, tile_rows, out_dir, columns):
    fields = ["model", "config", "split", "threshold"] + columns + ["pixel_accuracy", "boundary_precision", "boundary_recall"]
    with open(out_dir / "comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(out_dir / "per_tile_test.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "config", "tile", "threshold", "iou", "dice", "boundary_iou", "boundary_f1"])
        writer.writeheader()
        writer.writerows(tile_rows)


def main():
    parser = argparse.ArgumentParser(description="Tile-level comparison of models and inference modes")
    parser.add_argument("--models", nargs="+", required=True, help="name=checkpoint_path pairs")
    parser.add_argument("--configs", nargs="+", default=["s512", "s256", "s256+tta4"])
    parser.add_argument("--data-root", type=Path, default=Path("AerialImageDataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_boundary_comparison"))
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--window-batch", type=int, default=32, help="Windows per forward pass")
    parser.add_argument("--boundary-dilation", type=int, default=3)
    parser.add_argument("--boundary-tolerance", type=int, default=3)
    parser.add_argument("--recompute", action="store_true", help="Ignore cached per-tile results and recompute everything")
    parser.add_argument("--select-metric", choices=["iou", "boundary_f1"], default="iou", help="Validation metric used to choose the threshold")
    args = parser.parse_args()

    specs = []
    for spec in args.models:
        name, _, path = spec.partition("=")
        if not path or not Path(path).exists():
            parser.error(f"Checkpoint for '{name}' not found: {path!r} (cwd: {Path.cwd()}). Check the path before a long run.")
        specs.append((name, Path(path)))

    pipeline.set_seed(pipeline.SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = np.round(np.arange(0.30, 0.951, 0.05), 2).tolist()
    image_dir, mask_dir = args.data_root / "train" / "images", args.data_root / "train" / "gt"
    tiles = {split: pipeline.tile_ids_for_cities(args.data_root, cities) for split, cities in SPLIT_CITIES.items()}
    print(f"Device: {device} | val tiles={len(tiles['val'])} test tiles={len(tiles['test'])} | thresholds {thresholds[0]}..{thresholds[-1]}")

    columns = ["iou", "dice", "precision", "recall", "boundary_iou", "boundary_f1"]
    summary_rows, tile_rows = [], []
    for name, path in specs:
        model, epoch = load_model(path, device)
        model_key = cache_key(name, path, args, thresholds)
        print(f"\n##### Model '{name}' (checkpoint epoch {epoch}, cache id {model_key})")
        for config in args.configs:
            stride, tta = parse_config(config)
            print(f"  Config {config}: stride={stride}, tta={tta}")
            per_split = {}
            for split in ("val", "test"):
                print(f"   {split}:")
                cache_dir = args.out_dir / "cache" / f"{name}_{model_key}" / config / split
                per_split[split] = evaluate_tiles(model, tiles[split], args, thresholds, stride, tta, device, image_dir, mask_dir, cache_dir)

            val_total = np.sum(list(per_split["val"].values()), axis=0)
            best_t = int(np.argmax([summarize(row)[args.select_metric] for row in val_total]))
            threshold = thresholds[best_t]
            for split in ("val", "test"):
                metrics = summarize(np.sum(list(per_split[split].values()), axis=0)[best_t])
                summary_rows.append({"model": name, "config": config, "split": split, "threshold": threshold, **metrics})
            for tile, stats in per_split["test"].items():
                tile_rows.append({"model": name, "config": config, "tile": tile, "threshold": threshold,
                                  **{k: v for k, v in summarize(stats[best_t]).items() if k in ("iou", "dice", "boundary_iou", "boundary_f1")}})
            save_csvs(summary_rows, tile_rows, args.out_dir, columns)  # partial results survive a crash
            last = [r for r in summary_rows if r["model"] == name and r["config"] == config]
            print("    -> " + " | ".join(f"{r['split']} thr={r['threshold']:.2f} IoU={r['iou']:.4f} BF1={r['boundary_f1']:.4f}" for r in last))

    # ---- report ---------------------------------------------------------------------
    print("\n=== Comparison (threshold chosen on validation city only) ===")
    print(f"{'model':<12}{'config':<12}{'split':<6}{'thr':>6}" + "".join(f"{c:>15}" for c in columns))
    for row in summary_rows:
        print(f"{row['model']:<12}{row['config']:<12}{row['split']:<6}{row['threshold']:>6.2f}" + "".join(f"{row[c]:>15.4f}" for c in columns))

    print("\n=== Test tiles: per-tile mean +/- std ===")
    for model_name in dict.fromkeys(r["model"] for r in tile_rows):
        for config in dict.fromkeys(r["config"] for r in tile_rows):
            group = [r for r in tile_rows if r["model"] == model_name and r["config"] == config]
            if group:
                iou, bf1 = np.array([r["iou"] for r in group]), np.array([r["boundary_f1"] for r in group])
                print(f"{model_name:<12}{config:<12} IoU {iou.mean():.4f} +/- {iou.std():.4f} | Boundary F1 {bf1.mean():.4f} +/- {bf1.std():.4f}")

    print(f"\nSaved comparison.csv and per_tile_test.csv to {args.out_dir}")


if __name__ == "__main__":
    main()