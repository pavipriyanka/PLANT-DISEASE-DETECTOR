import json

import numpy as np
from PIL import Image

from plantdx import Predictor
from plantdx.evaluate import evaluate, most_confused
from plantdx.model import load_checkpoint


def test_training_outputs(trained):
    reports = trained["cfg"].reports_dir
    assert trained["checkpoint"].exists()
    history = json.loads((reports / "history.json").read_text())
    assert [h["phase"] for h in history] == ["head", "finetune"]
    assert (reports / "training_curves.png").stat().st_size > 0


def test_evaluate_writes_reports(trained, splits_dir, tmp_path):
    model, ckpt = load_checkpoint(trained["checkpoint"])
    m = evaluate(model, ckpt, splits_dir / "test.csv", tmp_path, device="cpu")
    assert m["test_images"] == 12 and 0 <= m["accuracy"] <= 1
    for name in ("metrics.json", "classification_report.txt", "confusion_matrix.png"):
        assert (tmp_path / name).exists()


def test_most_confused():
    cm = np.array([[5, 2, 0], [0, 4, 1], [3, 0, 2]])
    top = most_confused(cm, ["a", "b", "c"], top=2)
    assert top[0] == {"true": "c", "predicted": "a", "count": 3}
    assert top[1]["count"] == 2


def test_predictor_and_gradcam(trained, demo_dir):
    p = Predictor(trained["checkpoint"], device="cpu")
    img = Image.open(next((demo_dir / "color" / "Tomato___Leaf_Mold").iterdir()))
    preds = p.predict(img, top_k=3)
    assert len(preds) == 3 and preds[0][1] >= preds[1][1]
    assert sum(prob for _, prob in preds) <= 1.0001

    cam, idx = p.gradcam(img)
    assert cam.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1
    assert p.classes[idx] == preds[0][0]
    assert p.overlay(img, cam).size == (64, 64)


def test_app_without_and_with_model(trained, monkeypatch):
    import importlib

    monkeypatch.setenv("MODEL_PATH", str(trained["checkpoint"]))
    import app

    importlib.reload(app)
    labels, heat, md = app.analyse(Image.new("RGB", (120, 120), (60, 140, 50)))
    assert len(labels) == 3 and heat.size == (64, 64) and md.startswith("## ")
