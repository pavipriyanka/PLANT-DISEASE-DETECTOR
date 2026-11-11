"""Predict the disease for one leaf image and save a Grad-CAM heatmap.

Usage:
  python scripts/predict.py path/to/leaf.jpg [--checkpoint models/best_model.pt]
"""
import argparse
from pathlib import Path

from PIL import Image

import _path  # noqa: F401
from plantdx import Predictor, disease_info, pretty_label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pt"))
    ap.add_argument("--out", type=Path, default=None, help="where to save the heatmap (default: <image>_gradcam.png)")
    args = ap.parse_args()

    predictor = Predictor(args.checkpoint)
    img = Image.open(args.image)
    preds = predictor.predict(img, top_k=3)
    for cls, p in preds:
        print(f"{p:6.1%}  {pretty_label(cls)}")
    info = disease_info(preds[0][0])
    print(f"\nType : {info['type']}\nAbout: {info['about']}")
    for a in info["actions"]:
        print(f"  - {a}")

    cam, _ = predictor.gradcam(img)
    out = args.out or args.image.with_name(args.image.stem + "_gradcam.png")
    predictor.overlay(img, cam).save(out)
    print(f"\nGrad-CAM heatmap saved to {out}")


if __name__ == "__main__":
    main()
