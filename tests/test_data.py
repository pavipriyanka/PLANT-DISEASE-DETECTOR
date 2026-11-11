from collections import Counter

import pytest
import torch

from plantdx.data import (build_transforms, class_weights, find_class_root, load_classes, read_manifest,
                          split_dataset)


def test_find_class_root_prefers_color_folder(demo_dir):
    assert find_class_root(demo_dir).name == "color"


def test_find_class_root_errors_on_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_class_root(tmp_path)


def test_split_is_stratified_and_disjoint(splits_dir):
    classes = load_classes(splits_dir)
    rows = {s: read_manifest(splits_dir / f"{s}.csv") for s in ("train", "val", "test")}
    assert len(classes) == 4
    for s in ("val", "test"):
        assert set(Counter(label for _, label in rows[s]).values()) == {3}   # 12 * 0.25 per class
    paths = [p for r in rows.values() for p, _ in r]
    assert len(paths) == len(set(paths)) == 48


def test_max_per_class(demo_dir, tmp_path):
    summary = split_dataset(demo_dir, tmp_path, max_per_class=8)
    assert summary["train"] + summary["val"] + summary["test"] == 4 * 8


def test_class_weights_favour_rare_classes():
    w = class_weights([("a", 0)] * 30 + [("b", 1)] * 10, num_classes=2)
    assert w[1] > w[0]
    assert torch.isclose(w[1] / w[0], torch.tensor(3.0))


def test_transforms_output_shape():
    from PIL import Image

    img = Image.new("RGB", (300, 200))
    for train in (True, False):
        assert build_transforms(128, train)(img).shape == (3, 128, 128)
