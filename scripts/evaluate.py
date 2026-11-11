"""Evaluate the trained model on the held-out test set.

Usage:
  python scripts/evaluate.py [--checkpoint models/best_model.pt] [--manifest data/splits/test.csv]
Writes reports/metrics.json, classification_report.txt and confusion_matrix.png.
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
from plantdx.config import resolve_device
from plantdx.evaluate import evaluate
from plantdx.model import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pt"))
    ap.add_argument("--manifest", type=Path, default=Path("data/splits/test.csv"))
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    m = evaluate(model, ckpt, args.manifest, args.reports_dir, device)
    print(m["report"])
    print(f"Test images : {m['test_images']}")
    print(f"Accuracy    : {m['accuracy']:.4f}")
    print(f"Macro F1    : {m['macro_f1']:.4f}")
    if m["most_confused"]:
        print("Most confused pairs:")
        for p in m["most_confused"]:
            print(f"  {p['true']}  →  {p['predicted']}  ({p['count']})")
    print(f"\nReports saved to {args.reports_dir}/")


if __name__ == "__main__":
    main()
