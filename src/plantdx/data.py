"""Dataset preparation, transforms and loaders.

Instead of copying 54k images into train/val/test folders, the split is saved
as CSV "manifests" (path, label). Splitting is stratified, so every class keeps
the same proportions in each split.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------- preparing splits ----------------
def _class_dirs(folder: Path) -> list[Path]:
    return sorted(d for d in folder.iterdir()
                  if d.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in d.iterdir()))


def find_class_root(src: Path) -> Path:
    """Find the folder whose sub-folders are the classes.

    Handles common download layouts such as
    'plantvillage dataset/color/<class>/*.jpg' by preferring a 'color' folder.
    """
    src = Path(src)
    candidates = [src] + sorted(p for p in src.rglob("*") if p.is_dir())
    color = [c for c in candidates if c.name.lower() == "color" and len(_class_dirs(c)) >= 2]
    if color:
        return color[0]
    for c in candidates:
        if len(_class_dirs(c)) >= 2:
            return c
    raise FileNotFoundError(f"No folder with >=2 class sub-folders of images found under {src}")


def split_dataset(src: Path, out: Path, val_frac: float = 0.15, test_frac: float = 0.15,
                  max_per_class: int | None = None, seed: int = 42) -> dict:
    """Write train.csv / val.csv / test.csv and classes.json to `out`. Returns a summary."""
    root = find_class_root(src)
    classes = [d.name for d in _class_dirs(root)]
    rng = random.Random(seed)
    splits: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    for label, cls in enumerate(classes):
        files = sorted(str(f.resolve()) for f in (root / cls).iterdir() if f.suffix.lower() in IMAGE_EXTS)
        rng.shuffle(files)
        if max_per_class:
            files = files[:max_per_class]
        n_val = max(1, round(len(files) * val_frac))
        n_test = max(1, round(len(files) * test_frac))
        splits["val"] += [(f, label) for f in files[:n_val]]
        splits["test"] += [(f, label) for f in files[n_val:n_val + n_test]]
        splits["train"] += [(f, label) for f in files[n_val + n_test:]]

    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "label"])
            w.writerows(rows)
    (out / "classes.json").write_text(json.dumps(classes, indent=1), encoding="utf-8")
    return {"root": str(root), "classes": len(classes), **{k: len(v) for k, v in splits.items()}}


def read_manifest(path: Path) -> list[tuple[str, int]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [(row["path"], int(row["label"])) for row in csv.DictReader(fh)]


def load_classes(splits_dir: Path) -> list[str]:
    return json.loads((Path(splits_dir) / "classes.json").read_text(encoding="utf-8"))


# ---------------- transforms & datasets ----------------
def build_transforms(img_size: int, train: bool):
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        # Augmentation simulates real-world variety: framing, orientation, lighting.
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            *norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        *norm,
    ])


class ManifestDataset(Dataset):
    def __init__(self, rows: list[tuple[str, int]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        path, label = self.rows[idx]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), label


def class_weights(rows: list[tuple[str, int]], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights so rare classes aren't ignored by the loss."""
    counts = Counter(label for _, label in rows)
    total = sum(counts.values())
    return torch.tensor([total / (num_classes * max(counts.get(i, 0), 1)) for i in range(num_classes)],
                        dtype=torch.float32)


def make_loaders(splits_dir: Path, img_size: int, batch_size: int, num_workers: int):
    splits_dir = Path(splits_dir)
    rows = {s: read_manifest(splits_dir / f"{s}.csv") for s in ("train", "val", "test")}
    loaders = {
        s: DataLoader(ManifestDataset(r, build_transforms(img_size, train=(s == "train"))),
                      batch_size=batch_size, shuffle=(s == "train"), num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())
        for s, r in rows.items()
    }
    return loaders, rows


def split_summary(splits_dir: Path) -> dict[str, dict[int, int]]:
    out = {}
    for s in ("train", "val", "test"):
        per = defaultdict(int)
        for _, label in read_manifest(Path(splits_dir) / f"{s}.csv"):
            per[label] += 1
        out[s] = dict(per)
    return out
