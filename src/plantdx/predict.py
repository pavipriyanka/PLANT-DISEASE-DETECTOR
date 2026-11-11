"""Inference with Grad-CAM explanations.

Grad-CAM: take the gradients of the predicted class score with respect to the
last conv layer's feature maps, average them per channel to get channel
importances, then sum the weighted feature maps. Positive regions are the parts
of the image that pushed the model towards its prediction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import resolve_device
from .data import build_transforms
from .model import gradcam_layer, load_checkpoint


class Predictor:
    def __init__(self, checkpoint: Path, device: str = "auto"):
        self.device = resolve_device(device)
        self.model, ckpt = load_checkpoint(Path(checkpoint), self.device)
        self.arch, self.classes, self.img_size = ckpt["arch"], ckpt["classes"], ckpt["img_size"]
        self.metrics = ckpt.get("metrics", {})
        self.transform = build_transforms(self.img_size, train=False)

    def _tensor(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, img: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
        probs = F.softmax(self.model(self._tensor(img))[0], dim=0)
        values, idx = probs.topk(min(top_k, len(self.classes)))
        return [(self.classes[i], float(v)) for v, i in zip(values.cpu(), idx.cpu())]

    def gradcam(self, img: Image.Image, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """Return a heatmap (img_size x img_size, values 0..1) and the explained class index."""
        layer = gradcam_layer(self.model, self.arch)
        store = {}

        def save_activation(module, inputs, output):
            # A tensor hook (instead of a module backward hook) works even with
            # in-place activations such as MobileNetV2's ReLU6.
            store["act"] = output
            output.register_hook(lambda grad: store.__setitem__("grad", grad))

        handle = layer.register_forward_hook(save_activation)
        try:
            x = self._tensor(img).requires_grad_(True)
            logits = self.model(x)
            idx = int(logits.argmax(1)) if class_idx is None else class_idx
            self.model.zero_grad(set_to_none=True)
            logits[0, idx].backward()
        finally:
            handle.remove()
        weights = store["grad"].mean(dim=(2, 3), keepdim=True)           # channel importance
        cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))   # weighted feature maps
        cam = F.interpolate(cam, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, idx

    def overlay(self, img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
        """Blend the heatmap over the (resized, centre-cropped) input image."""
        import matplotlib

        size = self.img_size
        base = img.convert("RGB")
        scale = int(size * 1.14) / min(base.size)
        base = base.resize((max(size, round(base.width * scale)), max(size, round(base.height * scale))))
        left, top = (base.width - size) // 2, (base.height - size) // 2
        base = np.asarray(base.crop((left, top, left + size, top + size)), dtype=np.float32) / 255
        heat = matplotlib.colormaps["jet"](cam)[..., :3]
        return Image.fromarray(np.uint8(255 * ((1 - alpha) * base + alpha * heat)))
