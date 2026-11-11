"""Training configuration. Every field can be overridden from the command line."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    splits_dir: Path = Path("data/splits")     # output of scripts/prepare_data.py
    out_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    arch: str = "mobilenet_v2"                  # mobilenet_v2 | efficientnet_b0 | resnet18
    pretrained: bool = True
    img_size: int = 224
    batch_size: int = 32
    epochs_head: int = 3                        # phase 1: train only the new classifier head
    epochs_finetune: int = 5                    # phase 2: also train the last backbone blocks
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    unfreeze_blocks: int = 3                    # how many final backbone blocks to unfreeze
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    patience: int = 3                           # early stopping on validation loss
    num_workers: int = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
    seed: int = 42
    device: str = "auto"                        # auto | cpu | cuda | mps

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "TrainConfig":
        """Build a config from --flags; each dataclass field becomes a flag."""
        import argparse

        defaults = cls()
        ap = argparse.ArgumentParser(description="Train a plant disease classifier.")
        for f in fields(cls):
            value = getattr(defaults, f.name)
            flag = "--" + f.name.replace("_", "-")
            if isinstance(value, bool):
                ap.add_argument(flag, type=lambda s: s.lower() in {"1", "true", "yes"}, default=value,
                                metavar="true|false")
            else:
                ap.add_argument(flag, type=type(value), default=value)
        return cls(**vars(ap.parse_args(argv)))


def resolve_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
