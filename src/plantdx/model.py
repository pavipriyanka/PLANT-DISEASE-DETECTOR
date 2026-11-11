"""Model construction, freezing/unfreezing and checkpoint I/O."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

ARCHS = ("mobilenet_v2", "efficientnet_b0", "resnet18")


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained backbone with a new classification head for our classes."""
    if arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {ARCHS}")
    return m


def head_params(model: nn.Module, arch: str):
    return (model.fc if arch == "resnet18" else model.classifier).parameters()


def backbone_blocks(model: nn.Module, arch: str) -> list[nn.Module]:
    if arch == "resnet18":
        return [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]
    return list(model.features)


def freeze_backbone(model: nn.Module, arch: str) -> None:
    """Phase 1: only the new head learns; pretrained features stay fixed."""
    for p in model.parameters():
        p.requires_grad = False
    for p in head_params(model, arch):
        p.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, arch: str, n: int) -> None:
    """Phase 2: also fine-tune the last n backbone blocks (the most task-specific features)."""
    for block in backbone_blocks(model, arch)[-n:]:
        for p in block.parameters():
            p.requires_grad = True


def gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    """The last convolutional block, which Grad-CAM visualises."""
    return model.layer4[-1] if arch == "resnet18" else model.features[-1]


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------- checkpoints ----------------
def save_checkpoint(path: Path, model: nn.Module, arch: str, classes: list[str], img_size: int,
                    metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": arch, "classes": classes,
                "img_size": img_size, "metrics": metrics}, path)


def load_checkpoint(path: Path, device="cpu") -> tuple[nn.Module, dict]:
    # weights_only=True: only tensors and plain Python types are loaded (safe).
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
