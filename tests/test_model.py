from pathlib import Path

import pytest
import torch

from plantdx.config import TrainConfig
from plantdx.model import (ARCHS, build_model, count_params, freeze_backbone, load_checkpoint,
                           save_checkpoint, unfreeze_last_blocks)


@pytest.mark.parametrize("arch", ARCHS)
def test_build_and_freeze(arch):
    model = build_model(arch, num_classes=5, pretrained=False)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 5)

    freeze_backbone(model, arch)
    _, head_only = count_params(model)
    unfreeze_last_blocks(model, arch, 2)
    total, more = count_params(model)
    assert 0 < head_only < more < total


def test_unknown_arch():
    with pytest.raises(ValueError):
        build_model("vgg99", 3, pretrained=False)


def test_checkpoint_roundtrip(tmp_path):
    model = build_model("mobilenet_v2", 3, pretrained=False).eval()
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, "mobilenet_v2", ["a", "b", "c"], 64, {"val_acc": 0.5})
    loaded, ckpt = load_checkpoint(path)
    x = torch.randn(1, 3, 64, 64)
    assert torch.allclose(model(x), loaded(x), atol=1e-6)
    assert ckpt["classes"] == ["a", "b", "c"] and ckpt["img_size"] == 64


def test_config_from_cli():
    cfg = TrainConfig.from_cli(["--arch", "resnet18", "--pretrained", "false", "--img-size", "160",
                                "--lr-head", "0.01", "--splits-dir", "x/y"])
    assert cfg.arch == "resnet18" and cfg.pretrained is False
    assert cfg.img_size == 160 and cfg.lr_head == 0.01 and cfg.splits_dir == Path("x/y")