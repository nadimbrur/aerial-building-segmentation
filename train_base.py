"""Building-footprint segmentation with city-disjoint train/validation/test splits.

The labelled Inria training tiles are divided geographically, not by random
patches: Austin, Chicago, and Kitsap train the model; Vienna selects the best
checkpoint; and Tyrol-west is held out for a single final test evaluation.
The official AerialImageDataset/test directory has no labels, so it cannot
provide IoU and is intentionally not used for quantitative testing.

Run:
    python train.py
Resume from the last epoch automatically:
    python train.py --epochs 100
Start a new experiment:
    python train.py --no-resume
"""

import argparse
import os
import random
from collections import OrderedDict
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 99
TRAIN_CITIES = ("austin", "chicago", "kitsap")
VAL_CITY = "vienna"
TEST_CITY = "tyrol-w"
ALL_CITIES = (*TRAIN_CITIES, VAL_CITY, TEST_CITY)


class TileCache:
    def __init__(self, max_items=4):
        self.max_items = max_items
        self.items = OrderedDict()

    def get(self, key, loader):
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        value = loader()
        self.items[key] = value
        if len(self.items) > self.max_items:
            self.items.popitem(last=False)
        return value


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def tile_ids_for_cities(data_root, cities):
    images_dir = data_root / "train" / "images"
    ids = []
    for city in cities:
        city_ids = sorted(path.stem for path in images_dir.glob(f"{city}*.tif"))
        if not city_ids:
            raise FileNotFoundError(f"No labelled image tiles found for {city!r} in {images_dir}")
        ids.extend(city_ids)
    return ids


def build_patch_index(tile_ids, data_root, patch_size):
    """Create a padded, non-overlapping grid that covers every tile pixel."""
    from PIL import Image

    image_dir = data_root / "train" / "images"
    index = []
    for tile_id in tile_ids:
        with Image.open(image_dir / f"{tile_id}.tif") as image:
            width, height = image.size
        padded_width = ((width + patch_size - 1) // patch_size) * patch_size
        padded_height = ((height + patch_size - 1) // patch_size) * patch_size
        for y in range(0, padded_height, patch_size):
            for x in range(0, padded_width, patch_size):
                index.append((tile_id, x, y))
    return index


class InriaPatches(Dataset):
    def __init__(self, patch_index, data_root, patch_size, augment=False):
        self.patch_index = patch_index
        self.image_dir = data_root / "train" / "images"
        self.mask_dir = data_root / "train" / "gt"
        self.patch_size = patch_size
        self.augment = augment
        self.cache = TileCache()

    def __len__(self):
        return len(self.patch_index)

    def _read_tile(self, tile_id):
        def load():
            image = cv2.imread(str(self.image_dir / f"{tile_id}.tif"), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(self.mask_dir / f"{tile_id}.tif"), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise FileNotFoundError(f"Could not read image/mask for {tile_id}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask
        return self.cache.get(tile_id, load)

    def __getitem__(self, index):
        tile_id, x, y = self.patch_index[index]
        image, mask = self._read_tile(tile_id)
        patch = self.patch_size
        pad_bottom = max(0, y + patch - image.shape[0])
        pad_right = max(0, x + patch - image.shape[1])
        if pad_bottom or pad_right:
            image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)

        image = image[y:y + patch, x:x + patch].copy()
        mask = (mask[y:y + patch, x:x + patch] > 127).astype(np.float32)
        if self.augment:
            image, mask = self._augment(image, mask)

        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask).unsqueeze(0)

    @staticmethod
    def _augment(image, mask):
        if random.random() < 0.5:
            image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
        if random.random() < 0.5:
            image, mask = image[::-1].copy(), mask[::-1].copy()
        rotations = random.randrange(4)
        if rotations:
            image, mask = np.rot90(image, rotations).copy(), np.rot90(mask, rotations).copy()
        if random.random() < 0.3:
            image = cv2.convertScaleAbs(image, alpha=random.uniform(0.8, 1.2), beta=random.uniform(-20, 20))
        if random.random() < 0.2:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        return image, mask


def build_model(pretrained):
    if pretrained:
        try:
            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
            print("Using COCO-pretrained DeepLabV3-ResNet50 weights.")
        except Exception as error:
            print(f"Could not obtain pretrained weights ({error}); training from random initialization.")
            model = deeplabv3_resnet50(weights=None, weights_backbone=None)
    else:
        model = deeplabv3_resnet50(weights=None, weights_backbone=None)
    model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
    return model


class DiceWeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
        return 0.5 * bce + 0.5 * dice_loss.mean()


def cutmix(images, masks, probability):
    if images.shape[0] < 2 or random.random() >= probability:
        return images, masks
    permutation = torch.randperm(images.shape[0], device=images.device)
    height, width = images.shape[-2:]
    ratio = random.uniform(0.25, 0.5)
    cut_height, cut_width = int(height * ratio), int(width * ratio)
    center_y = random.randrange(height)
    center_x = random.randrange(width)
    y0, y1 = max(0, center_y - cut_height // 2), min(height, center_y + cut_height // 2)
    x0, x1 = max(0, center_x - cut_width // 2), min(width, center_x + cut_width // 2)
    images[:, :, y0:y1, x0:x1] = images[permutation, :, y0:y1, x0:x1]
    masks[:, :, y0:y1, x0:x1] = masks[permutation, :, y0:y1, x0:x1]
    return images, masks


def metric_counts(logits, masks):
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    return ((predictions * masks).sum().item(),(predictions * (1 - masks)).sum().item(),
        ((1 - predictions) * masks).sum().item(), ((1 - predictions) * (1 - masks)).sum().item(), )


def metrics_from_counts(true_positive, false_positive, false_negative, true_negative, epsilon=1e-7):
    iou = true_positive / (true_positive + false_positive + false_negative + epsilon)
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative + epsilon)
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative + epsilon)
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "pixel_accuracy": accuracy}


def run_epoch(model, loader, loss_fn, device, optimizer=None, cutmix_probability=0.0):
    training = optimizer is not None
    model.train(training)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total_loss = 0.0
    counts = np.zeros(4, dtype=np.float64)
    phase = "train" if training else "eval"

    for batch_number, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            images, masks = cutmix(images, masks, cutmix_probability)

        with torch.set_grad_enabled(training), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(images)
            logits = outputs["out"]
            loss = loss_fn(logits, masks)
            if training and outputs.get("aux") is not None:
                loss = loss + 0.4 * loss_fn(outputs["aux"], masks)

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
        counts += metric_counts(logits.detach().float(), masks)
        print(f"\r  [{phase}] batch {batch_number}/{len(loader)} loss={loss.item():.4f}", end="", flush=True)
    print()
    return total_loss / len(loader.dataset), metrics_from_counts(*counts)


@torch.no_grad()
def save_test_visualization(model, dataset, device, output_path, examples=6):
    model.eval()
    choices = random.sample(range(len(dataset)), min(examples, len(dataset)))
    figure, axes = plt.subplots(len(choices), 3, figsize=(10, 3 * len(choices)), squeeze=False)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for row, index in enumerate(choices):
        image, mask = dataset[index]
        logits = model(image.unsqueeze(0).to(device))["out"]
        prediction = (torch.sigmoid(logits)[0, 0] >= 0.5).cpu().numpy()
        display_image = np.clip(image.numpy().transpose(1, 2, 0) * std + mean, 0, 1)
        for axis, data, title in zip(axes[row], (display_image, mask[0], prediction), ("Aerial image", "Ground truth", "Prediction")):
            axis.imshow(data, cmap=None if data.ndim == 3 else "gray")
            axis.set_title(title)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="City-disjoint Inria building segmentation")
    parser.add_argument("--data-root", type=Path, default=Path("AerialImageDataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_city_split"))
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--cutmix-probability", type=float, default=0.0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train cities: {', '.join(TRAIN_CITIES)} | Validation city: {VAL_CITY} | Test city: {TEST_CITY}")

    train_ids = tile_ids_for_cities(args.data_root, TRAIN_CITIES)
    val_ids = tile_ids_for_cities(args.data_root, (VAL_CITY,))
    test_ids = tile_ids_for_cities(args.data_root, (TEST_CITY,))
    assert not (set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids))
    print(f"Tiles: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    train_dataset = InriaPatches(build_patch_index(train_ids, args.data_root, args.patch_size), args.data_root, args.patch_size, augment=True)
    val_dataset = InriaPatches(build_patch_index(val_ids, args.data_root, args.patch_size), args.data_root, args.patch_size)
    test_dataset = InriaPatches(build_patch_index(test_ids, args.data_root, args.patch_size), args.data_root, args.patch_size)
    print(f"Patches: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    sample_count = min(300, len(train_dataset))
    foreground_fraction = float(np.mean([train_dataset[index][1].mean().item() for index in random.sample(range(len(train_dataset)), sample_count)]))
    pos_weight = min((1 - foreground_fraction) / max(foreground_fraction, 1e-6), 10.0)
    print(f"EDA: foreground={foreground_fraction:.3f}; BCE positive-class weight={pos_weight:.2f}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=args.num_workers > 0)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    loss_fn = DiceWeightedBCELoss(pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    best_path = args.out_dir / "best_val_iou.pt"
    last_path = args.out_dir / "last_checkpoint.pt"
    best_iou, start_epoch, history = -1.0, 1, []

    if last_path.exists() and not args.no_resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_iou = checkpoint["best_iou"]
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming at epoch {start_epoch}; current best validation IoU={best_iou:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer, args.cutmix_probability)
        val_loss, val_metrics = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_metrics["iou"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train_iou": train_metrics["iou"], "val_iou": val_metrics["iou"]})


        test_loss, test_metrics = run_epoch(model, test_loader, loss_fn, device)
        print("  Final test-city result (Tyrol-west; not used to select the model):")
        print(f"  loss={test_loss:.4f}, " + ", ".join(f"{name}={value:.4f}" for name, value in test_metrics.items()))
        

        print(f"Epoch {epoch}/{args.epochs}: train IoU={train_metrics['iou']:.4f}, val IoU={val_metrics['iou']:.4f}, val Dice={val_metrics['dice']:.4f}")


        
        
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, best_path)
            print(f"  Saved new best model using validation IoU={best_iou:.4f}")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best_iou": best_iou, "history": history}, last_path)

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_loss, test_metrics = run_epoch(model, test_loader, loss_fn, device)
    print("\nFinal test-city result (Tyrol-west; not used to select the model):")
    print(f"  loss={test_loss:.4f}, " + ", ".join(f"{name}={value:.4f}" for name, value in test_metrics.items()))

    save_test_visualization(model, test_dataset, device, args.out_dir / "test_predictions.png")
    print(f"Saved best model, latest checkpoint, and test visualizations in {args.out_dir}")


if __name__ == "__main__":
    main()
