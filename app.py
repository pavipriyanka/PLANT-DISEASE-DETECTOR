"""Gradio web app: upload a leaf photo → disease prediction, Grad-CAM heatmap and care tips.

Run locally:  python app.py   (then open http://127.0.0.1:7860)
Also works as-is on Hugging Face Spaces (Gradio SDK).
"""
import os
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from plantdx import Predictor, disease_info, pretty_label  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "best_model.pt"))
LOW_CONFIDENCE = 0.5

predictor = Predictor(MODEL_PATH) if MODEL_PATH.exists() else None


def model_card() -> str:
    if predictor is None:
        return (f"⚠️ **No trained model found at `{MODEL_PATH}`.** Train one first "
                "(see README → *Train on Kaggle*), then put `best_model.pt` in the `models/` folder.")
    m = predictor.metrics
    acc = f" · val accuracy **{m['val_acc']:.1%}**" if "val_acc" in m else ""
    return f"Model: **{predictor.arch}** · {len(predictor.classes)} classes{acc}"


def analyse(image):
    if predictor is None:
        raise gr.Error("No trained model found. See the README to train one.")
    if image is None:
        raise gr.Error("Please upload a leaf image first.")
    preds = predictor.predict(image, top_k=3)
    top_cls, top_p = preds[0]
    cam, _ = predictor.gradcam(image)
    heatmap = predictor.overlay(image, cam)

    info = disease_info(top_cls)
    lines = [f"## {pretty_label(top_cls)}", f"**Confidence:** {top_p:.1%} · **Type:** {info['type']}", "",
             info["about"], "", "**What to do:**"] + [f"- {a}" for a in info["actions"]]
    if top_p < LOW_CONFIDENCE:
        lines.insert(2, "> ⚠️ Low confidence. Try a clearer photo of a single leaf on a plain background.")
    lines += ["", "<sub>For guidance only. Confirm with a local agricultural expert before treating crops.</sub>"]
    return {pretty_label(c): p for c, p in preds}, heatmap, "\n".join(lines)


with gr.Blocks(title="Plant Disease Detector") as demo:
    gr.Markdown("# 🌿 Plant Disease Detector\nUpload a photo of a leaf to identify the disease, "
                "see **where** the model looked (Grad-CAM), and get care tips.")
    gr.Markdown(model_card())
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Leaf photo", height=320)
            button = gr.Button("Analyse leaf", variant="primary")
            gr.Markdown("<sub>Works best with one leaf, well lit, plain background. "
                        "Trained on the PlantVillage dataset (14 crops, 38 classes).</sub>")
        with gr.Column():
            label_out = gr.Label(num_top_classes=3, label="Top predictions")
            heat_out = gr.Image(label="Grad-CAM: regions that drove the prediction", height=320)
    info_out = gr.Markdown()
    button.click(analyse, inputs=image_in, outputs=[label_out, heat_out, info_out])

    examples = sorted((ROOT / "examples").glob("*.jpg")) if (ROOT / "examples").exists() else []
    if examples:
        gr.Examples([[str(p)] for p in examples], inputs=image_in)


if __name__ == "__main__":
    demo.launch()
