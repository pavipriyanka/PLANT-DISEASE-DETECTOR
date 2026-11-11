"""Generate a tiny synthetic 'leaf' dataset (4 classes) to test the pipeline end to end
without downloading anything. Not for real accuracy, just to check that everything runs.

Usage:  python scripts/make_demo_dataset.py --out data/demo --per-class 30
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = {
    "Tomato___healthy": None,
    "Tomato___Early_blight": (110, 70, 30),
    "Tomato___Leaf_Mold": (200, 200, 90),
    "Potato___Late_blight": (40, 50, 40),
}


def make_leaf(spot_color, rng: np.random.Generator, size: int = 96) -> Image.Image:
    img = Image.new("RGB", (size, size), tuple(int(v) for v in rng.integers(180, 230, 3)))
    d = ImageDraw.Draw(img)
    green = (int(rng.integers(40, 80)), int(rng.integers(120, 170)), int(rng.integers(30, 70)))
    m = int(rng.integers(6, 14))
    d.ellipse([m, m + 10, size - m, size - m - 10], fill=green)
    if spot_color:
        for _ in range(int(rng.integers(4, 9))):
            x, y = rng.integers(25, size - 25, 2)
            r = int(rng.integers(4, 9))
            d.ellipse([x - r, y - r, x + r, y + r], fill=spot_color)
    noise = rng.normal(0, 8, (size, size, 3))
    return Image.fromarray(np.clip(np.asarray(img, dtype=np.float32) + noise, 0, 255).astype("uint8"))


def main(out: Path = Path("data/demo"), per_class: int = 30, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    for cls, spot in CLASSES.items():
        folder = out / "color" / cls
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            make_leaf(spot, rng).save(folder / f"{i:03d}.jpg", quality=92)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/demo"))
    ap.add_argument("--per-class", type=int, default=30)
    args = ap.parse_args()
    print(f"Wrote demo dataset to {main(args.out, args.per_class)}")
