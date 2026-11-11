"""Train the classifier. Every TrainConfig field is a flag.

Usage:
  python scripts/train.py                                   # defaults (MobileNetV2, 3 + 5 epochs)
  python scripts/train.py --arch efficientnet_b0 --epochs-finetune 8
  python scripts/train.py --img-size 160 --epochs-head 2 --epochs-finetune 2   # faster on CPU
"""
import _path  # noqa: F401
from plantdx.config import TrainConfig
from plantdx.train import train

if __name__ == "__main__":
    train(TrainConfig.from_cli())
