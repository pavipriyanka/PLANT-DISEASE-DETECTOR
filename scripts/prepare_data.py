"""Create stratified train/val/test splits (CSV manifests) from a folder of class sub-folders.

Usage:
  python scripts/prepare_data.py --src "path/to/plantvillage dataset"
  python scripts/prepare_data.py --src data/raw --max-per-class 150     # small subset for CPU
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
from plantdx.data import split_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="dataset folder (a 'color' sub-folder is auto-detected)")
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--max-per-class", type=int, default=None, help="limit images per class (quick runs)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s = split_dataset(args.src, args.out, args.val_frac, args.test_frac, args.max_per_class, args.seed)
    print(f"Class folder : {s['root']}")
    print(f"Classes      : {s['classes']}")
    print(f"Train/val/test: {s['train']} / {s['val']} / {s['test']} images")
    print(f"Saved splits to {args.out}")


if __name__ == "__main__":
    main()
