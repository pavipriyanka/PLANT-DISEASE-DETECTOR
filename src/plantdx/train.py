"""Two-phase transfer-learning trainer.

Phase 1 (head):      backbone frozen, train the new classifier at a high LR.
Phase 2 (finetune):  unfreeze the last N backbone blocks, train at a low LR
                     so the pretrained features adapt without being destroyed.
Both phases use a cosine LR schedule; early stopping watches validation loss.
The best model (lowest val loss) is saved with its classes and settings.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .config import TrainConfig, resolve_device
from .data import class_weights, load_classes, make_loaders
from .model import (build_model, count_params, freeze_backbone, save_checkpoint,
                    unfreeze_last_blocks)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, desc=""):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:      # mixed precision on GPU: faster, less memory
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
    return total_loss / seen, correct / seen


def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    classes = load_classes(cfg.splits_dir)
    loaders, rows = make_loaders(cfg.splits_dir, cfg.img_size, cfg.batch_size, cfg.num_workers)
    print(f"Device: {device} | classes: {len(classes)} | "
          f"train/val/test: {len(rows['train'])}/{len(rows['val'])}/{len(rows['test'])}")

    model = build_model(cfg.arch, len(classes), cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(rows["train"], len(classes)).to(device),
                                    label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    ckpt_path = Path(cfg.out_dir) / "best_model.pt"
    history, best_val, bad_epochs, epoch = [], float("inf"), 0, 0
    start = time.time()

    phases = [("head", cfg.epochs_head, cfg.lr_head), ("finetune", cfg.epochs_finetune, cfg.lr_finetune)]
    for phase, n_epochs, lr in phases:
        if n_epochs <= 0:
            continue
        freeze_backbone(model, cfg.arch)
        if phase == "finetune":
            unfreeze_last_blocks(model, cfg.arch, cfg.unfreeze_blocks)
        total, trainable = count_params(model)
        print(f"\n== Phase '{phase}': {trainable:,}/{total:,} trainable params, lr={lr}")
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        bad_epochs = 0
        for _ in range(n_epochs):
            epoch += 1
            tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler,
                                        desc=f"epoch {epoch} train")
            va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device, desc=f"epoch {epoch} val")
            scheduler.step()
            history.append({"epoch": epoch, "phase": phase, "train_loss": tr_loss, "train_acc": tr_acc,
                            "val_loss": va_loss, "val_acc": va_acc})
            improved = va_loss < best_val
            print(f"epoch {epoch:>2} [{phase}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f}{'  ✓ saved' if improved else ''}")
            if improved:
                best_val, bad_epochs = va_loss, 0
                save_checkpoint(ckpt_path, model, cfg.arch, classes, cfg.img_size,
                                {"val_loss": va_loss, "val_acc": va_acc, "epoch": epoch})
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    print(f"Early stopping: no val-loss improvement for {cfg.patience} epochs.")
                    break

    reports = Path(cfg.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    plot_history(history, reports / "training_curves.png")
    minutes = (time.time() - start) / 60
    print(f"\nDone in {minutes:.1f} min. Best val loss {best_val:.4f}. Saved {ckpt_path}")
    return {"checkpoint": str(ckpt_path), "best_val_loss": best_val, "history": history, "minutes": minutes}


def plot_history(history: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for key, ax, title in (("loss", a1, "Loss"), ("acc", a2, "Accuracy")):
        ax.plot(ep, [h[f"train_{key}"] for h in history], "o-", label="train")
        ax.plot(ep, [h[f"val_{key}"] for h in history], "o-", label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    switch = next((h["epoch"] for h in history if h["phase"] == "finetune"), None)
    if switch:
        for ax in (a1, a2):
            ax.axvline(switch - 0.5, color="gray", ls="--", lw=1)
            ax.text(switch - 0.4, ax.get_ylim()[1], " fine-tune", va="top", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
