"""Shared helpers for boundary-aware fine-tuning and evaluation.

  - erode / dilate / boundary_band : morphology with a square structuring element (GPU, max-pool based)
  - BoundaryAwareLoss              : existing Dice+weighted-BCE loss + BCE restricted to an edge zone
  - confusion_and_boundary         : TP/FP/FN/TN + Boundary IoU / Boundary F1 counts
  - summarize                      : counts -> IoU, Dice, precision, recall, boundary metrics
  - predict_tile                   : full-tile inference with overlapping windows, blending and TTA
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TILE_SIZE = 5000
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
COLS = ["tp", "fp", "fn", "tn", "b_inter", "b_union", "bp_hit", "bp_sum", "br_hit", "br_sum"]


# ---------------------------------------------------------------- morphology ----------
def erode(mask, radius):
    return -F.max_pool2d(-mask, 2 * radius + 1, 1, radius)


def dilate(mask, radius):
    return F.max_pool2d(mask, 2 * radius + 1, 1, radius)


def boundary_band(mask, radius):
    """Mask pixels within `radius` px of the mask edge."""
    return mask * (1.0 - erode(mask, radius))


def edge_zone(mask, radius):
    """Pixels within `radius` px of the contour on BOTH sides (inside and outside)."""
    return (dilate(mask, radius) - erode(mask, radius)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------- loss ----------
class BoundaryAwareLoss(nn.Module):
    """base_loss + edge_weight * BCE averaged over the edge zone of the ground-truth mask."""

    def __init__(self, base_loss, edge_weight=0.5, radius=3):
        super().__init__()
        self.base_loss, self.edge_weight, self.radius = base_loss, edge_weight, radius

    def forward(self, logits, targets):
        base = self.base_loss(logits, targets)
        with torch.no_grad():
            zone = edge_zone(targets, self.radius)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        edge = (bce * zone).sum() / (zone.sum() + 1.0)
        return base + self.edge_weight * edge


# ------------------------------------------------------------------- metrics ----------
class IndexedPatches(torch.utils.data.Dataset):
    """Wraps a patch dataset so each item also returns its index."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, mask = self.base[index]
        return image, mask, index


def valid_mask(patch_index, indices, patch_size, device):
    """1 for pixels inside the real 5000x5000 tile, 0 for reflect/zero padding."""
    valid = torch.zeros(len(indices), 1, patch_size, patch_size, device=device)
    for k, i in enumerate(indices.tolist()):
        _, x, y = patch_index[i]
        valid[k, :, : max(0, min(patch_size, TILE_SIZE - y)), : max(0, min(patch_size, TILE_SIZE - x))] = 1.0
    return valid


def gt_terms(masks, dilation=3, tolerance=3):
    """Threshold-independent ground-truth boundary terms; compute once, reuse per threshold."""
    band, contour = boundary_band(masks, dilation), boundary_band(masks, 1)
    return band, contour, dilate(contour, tolerance)


def confusion_and_boundary(preds, masks, valid=None, dilation=3, tolerance=3, gt=None):
    """Return a (B, 10) tensor with the columns in COLS. preds/masks: (B,1,H,W) in {0,1}."""
    if valid is not None:
        preds = preds * valid
    gt_band, gt_contour, gt_contour_dilated = gt if gt is not None else gt_terms(masks, dilation, tolerance)
    dims = (1, 2, 3)
    tp = (preds * masks).sum(dims)
    fp = (preds * (1 - masks)).sum(dims)
    fn = ((1 - preds) * masks).sum(dims)
    negatives = (1 - preds) * (1 - masks)
    tn = (negatives * valid).sum(dims) if valid is not None else negatives.sum(dims)
    pred_band, pred_contour = boundary_band(preds, dilation), boundary_band(preds, 1)
    inter = (gt_band * pred_band).sum(dims)
    union = (gt_band + pred_band - gt_band * pred_band).sum(dims)
    return torch.stack([
        tp, fp, fn, tn, inter, union,
        (pred_contour * gt_contour_dilated).sum(dims), pred_contour.sum(dims),
        (gt_contour * dilate(pred_contour, tolerance)).sum(dims), gt_contour.sum(dims),
    ], dim=1)


def summarize(row, eps=1e-7):
    tp, fp, fn, tn, b_inter, b_union, bp_hit, bp_sum, br_hit, br_sum = [float(v) for v in row]
    b_precision, b_recall = bp_hit / (bp_sum + eps), br_hit / (br_sum + eps)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
        "boundary_iou": b_inter / (b_union + eps),
        "boundary_f1": 2 * b_precision * b_recall / (b_precision + b_recall + eps),
        "boundary_precision": b_precision,
        "boundary_recall": b_recall,
    }


# ----------------------------------------------------------------- inference ----------
def load_tile(image_dir, mask_dir, tile_id):
    image = cv2.cvtColor(cv2.imread(str(image_dir / f"{tile_id}.tif"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_dir / f"{tile_id}.tif"), cv2.IMREAD_GRAYSCALE)
    return image, (mask > 127).astype(np.float32)


def _views(tta):
    if tta == "none":
        return [(0, False)]
    if tta == "tta4":   # identity, h-flip, v-flip, 180-degree rotation
        return [(0, False), (0, True), (2, False), (2, True)]
    if tta == "tta8":   # full dihedral group
        return [(k, flip) for k in range(4) for flip in (False, True)]
    raise ValueError(f"Unknown TTA mode: {tta}")


def _predict_batch(model, batch, tta, device):
    total = 0.0
    views = _views(tta)
    for k, flip in views:
        x = torch.flip(batch, dims=(3,)) if flip else batch
        x = torch.rot90(x, k, dims=(2, 3))
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x)["out"]
        y = torch.rot90(torch.sigmoid(logits.float()), -k, dims=(2, 3))
        total = total + (torch.flip(y, dims=(3,)) if flip else y)
    return total / len(views)


@torch.no_grad()
def predict_tile(model, image, device, patch=512, stride=256, tta="none", batch_size=8):
    """Probability map (H, W) for a whole tile using overlapping windows with smooth blending."""
    model.eval()
    height, width = image.shape[:2]
    n_y = 1 if height <= patch else -(-(height - patch) // stride) + 1
    n_x = 1 if width <= patch else -(-(width - patch) // stride) + 1
    padded_h, padded_w = (n_y - 1) * stride + patch, (n_x - 1) * stride + patch
    if padded_h > height or padded_w > width:
        image = cv2.copyMakeBorder(image, 0, padded_h - height, 0, padded_w - width, cv2.BORDER_REFLECT_101)

    ramp = 0.05 + 0.95 * torch.sin(torch.pi * (torch.arange(patch, device=device) + 0.5) / patch)
    window = ramp[:, None] * ramp[None, :]
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    prob_sum = torch.zeros(padded_h, padded_w, device=device)
    weight_sum = torch.zeros(padded_h, padded_w, device=device)

    coords = [(y * stride, x * stride) for y in range(n_y) for x in range(n_x)]
    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        crops = np.stack([image[y:y + patch, x:x + patch] for y, x in chunk])
        batch = torch.from_numpy(crops).to(device).permute(0, 3, 1, 2).float() / 255.0
        probs = _predict_batch(model, (batch - mean) / std, tta, device)
        for k, (y, x) in enumerate(chunk):
            prob_sum[y:y + patch, x:x + patch] += probs[k, 0] * window
            weight_sum[y:y + patch, x:x + patch] += window
    return (prob_sum / weight_sum)[:height, :width]
