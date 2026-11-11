# 🌿 Plant Disease Detector: Transfer Learning + Grad-CAM

Identify **38 plant diseases across 14 crops** from a leaf photo. A pretrained CNN (MobileNetV2 by default) is fine-tuned in two phases on the PlantVillage dataset (54,303 images). The app also shows **where the model looked** with Grad-CAM and gives **care tips** for each disease.

<!-- After training, replace with your numbers: -->
> **Results:** Test accuracy **XX.X%** · Macro F1 **0.XXX** on a held-out 15% test split (see `reports/`)

## Features
- 🧠 **Transfer learning** with 3 backbones: `mobilenet_v2` (default), `efficientnet_b0`, `resnet18`
- 🪜 **Two-phase training**: (1) train only the new head, (2) unfreeze the last blocks and fine-tune at a lower LR
- 📉 Cosine LR schedule, **early stopping**, label smoothing, class-weighted loss, **mixed precision** on GPU
- 🎲 **Augmentation**: random crops, flips, rotation, colour jitter
- 📊 **Evaluation**: accuracy, macro F1, per-class report, **confusion matrix**, most-confused pairs, training curves
- 🔥 **Grad-CAM heatmaps** to explain each prediction
- 🌱 **Care tips** for every disease class (type, description, recommended actions)
- 🖥️ **Gradio app**, deployable to Hugging Face Spaces as-is
- ☁️ **Kaggle notebook** to train free on a GPU
- ✅ **20 unit tests** (run offline on a synthetic dataset)

## How it works
```
PlantVillage images ──► prepare_data.py ──► stratified 70/15/15 split (CSV manifests)
                                              │
                    ┌─────────────────────────┴──────────────────────────┐
                    ▼                                                    │
     ImageNet-pretrained MobileNetV2                                     │
     + new Linear head (38 classes)                                      │
                    │                                                    │
   Phase 1: backbone frozen, lr 1e-3  (head learns to use features)      │
   Phase 2: last 3 blocks unfrozen, lr 1e-4  (features adapt to leaves)  │
                    │   best checkpoint by val loss, early stopping      │
                    ▼                                                    ▼
             best_model.pt ──► evaluate.py (test set) ──► metrics, confusion matrix
                    │
                    └──► app.py: prediction + Grad-CAM + care tips
```

## Project structure
```
plant-disease-detector/
├── app.py                       # Gradio web app
├── src/plantdx/
│   ├── config.py                # TrainConfig (every field is a CLI flag)
│   ├── data.py                  # dataset discovery, stratified split, transforms, loaders
│   ├── model.py                 # backbones, freeze/unfreeze, checkpoints
│   ├── train.py                 # two-phase trainer, early stopping, curves
│   ├── evaluate.py              # metrics, classification report, confusion matrix
│   ├── predict.py               # Predictor + Grad-CAM
│   └── labels.py                # readable labels + care info lookup
├── scripts/                     # prepare_data · train · evaluate · predict · make_demo_dataset
├── notebooks/train_on_kaggle.ipynb
├── data/disease_info.json       # care tips for all disease classes
├── models/                      # best_model.pt goes here after training
├── reports/                     # metrics.json, confusion_matrix.png, training_curves.png
└── tests/
```

## Setup
Python 3.11 or 3.12 recommended.
```bash
python -m venv .venv
.venv\Scripts\activate             # Mac/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                             # 20 tests, about 10 seconds
```

## Train on Kaggle (recommended: free GPU, ~20–30 min)
1. Push this repo to your GitHub.
2. On kaggle.com: **Create → New Notebook → File → Import Notebook** → upload `notebooks/train_on_kaggle.ipynb`.
3. Turn on **GPU** and **Internet** in the notebook settings, and **Add Input → "PlantVillage Dataset"**.
4. Put your GitHub username in the first cell, then **Run All**.
5. Download `results.zip` from the Output panel and unzip it into this folder. It contains `models/best_model.pt` and `reports/`.

## Train locally (CPU, small subset)
Download the PlantVillage dataset from Kaggle into `data/raw/`, then:
```bash
python scripts/prepare_data.py --src data/raw --max-per-class 100
python scripts/train.py --img-size 160 --epochs-head 2 --epochs-finetune 2
python scripts/evaluate.py
```
No dataset yet? Check that the whole pipeline runs on synthetic images:
```bash
python scripts/make_demo_dataset.py --out data/demo
python scripts/prepare_data.py --src data/demo
python scripts/train.py --pretrained false --img-size 96 --epochs-head 1 --epochs-finetune 1
```

## Use the model
```bash
python app.py                                   # web app at http://127.0.0.1:7860
python scripts/predict.py path/to/leaf.jpg      # terminal prediction + Grad-CAM image
```
Tip: copy a few test images into an `examples/` folder and the app shows them as clickable examples.

## Deploy free on Hugging Face Spaces
Create a new Space (SDK: **Gradio**), then upload `app.py`, `requirements.txt`, the `src/` folder, `data/disease_info.json` and `models/best_model.pt`. The Space builds and gives you a public link to put on your resume.

## Key design decisions
| Decision | Why |
|---|---|
| Transfer learning instead of training from scratch | ImageNet features (edges, textures, shapes) transfer well to leaves; trains in minutes with high accuracy |
| MobileNetV2 default | ~3.5M parameters and fast on CPU, so it's cheap to host and could run on a phone in the field |
| Two-phase training | A randomly initialised head sends large, noisy gradients; training it first protects the pretrained weights during fine-tuning |
| Lower LR when fine-tuning | Small updates adapt the features without "forgetting" ImageNet knowledge |
| Stratified split, separate test set | Every class is represented in each split; the test set is never used for model selection |
| Macro F1 alongside accuracy | Accuracy can hide poor performance on small classes; macro F1 weights every class equally |
| Class-weighted loss + label smoothing | Handles class imbalance; reduces over-confidence |
| CSV manifests instead of copying images | No duplicated 2 GB folders; splits are reproducible and easy to inspect |
| Grad-CAM | Shows whether the model looks at the lesions or at the background (a sanity check for shortcut learning) |

## Limitations
PlantVillage photos are single leaves on plain backgrounds in controlled lighting. Accuracy on real field photos (cluttered backgrounds, several leaves) is lower. Grad-CAM helps reveal this. Next steps: fine-tune on field datasets such as PlantDoc, add leaf segmentation, export to ONNX/TFLite for mobile.

## Dataset
Hughes & Salathé, *An open access repository of images on plant health to enable the development of mobile disease diagnostics* (2015), arXiv:1511.08060. Care tips are general guidance only; consult local agricultural experts before treating crops.

## Tech stack
Python · PyTorch · torchvision · scikit-learn · matplotlib · Gradio · Kaggle GPU · pytest
