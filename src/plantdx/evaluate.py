"""Test-set evaluation: accuracy, macro F1, per-class report and confusion matrix."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .data import ManifestDataset, build_transforms, read_manifest
from .labels import pretty_label


@torch.no_grad()
def predict_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def evaluate(model, ckpt: dict, manifest: Path, reports_dir: Path, device, batch_size: int = 64,
             num_workers: int = 0) -> dict:
    rows = read_manifest(manifest)
    loader = DataLoader(ManifestDataset(rows, build_transforms(ckpt["img_size"], train=False)),
                        batch_size=batch_size, num_workers=num_workers)
    y_pred, y_true = predict_loader(model, loader, device)
    classes = ckpt["classes"]
    names = [pretty_label(c) for c in classes]
    labels = list(range(len(classes)))

    metrics = {
        "test_images": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)), 4),
    }
    report = classification_report(y_true, y_pred, labels=labels, target_names=names, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["most_confused"] = most_confused(cm, names)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (reports_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion(cm, names, reports_dir / "confusion_matrix.png")
    return {**metrics, "report": report}


def most_confused(cm: np.ndarray, names: list[str], top: int = 5) -> list[dict]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0]
    return [{"true": names[i], "predicted": names[j], "count": int(off[i, j])}
            for i, j in pairs[:top] if off[i, j] > 0]


def plot_confusion(cm: np.ndarray, names: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)   # row-normalised = recall per class
    size = max(6, 0.32 * len(names))
    fig, ax = plt.subplots(figsize=(size + 2, size))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
