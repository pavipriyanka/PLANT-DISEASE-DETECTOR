import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plantdx.config import TrainConfig  # noqa: E402
from plantdx.data import split_dataset  # noqa: E402
from plantdx.train import train  # noqa: E402


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory) -> Path:
    import make_demo_dataset

    return make_demo_dataset.main(tmp_path_factory.mktemp("raw"), per_class=12)


@pytest.fixture(scope="session")
def splits_dir(demo_dir, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("splits")
    split_dataset(demo_dir, out, val_frac=0.25, test_frac=0.25)
    return out


@pytest.fixture(scope="session")
def trained(splits_dir, tmp_path_factory) -> dict:
    """A tiny model trained for 1+1 epochs at 64px. No pretrained download needed."""
    work = tmp_path_factory.mktemp("run")
    cfg = TrainConfig(splits_dir=splits_dir, out_dir=work / "models", reports_dir=work / "reports",
                      pretrained=False, img_size=64, batch_size=8, epochs_head=1, epochs_finetune=1,
                      num_workers=0, device="cpu")
    result = train(cfg)
    return {"cfg": cfg, "result": result, "checkpoint": Path(result["checkpoint"])}
