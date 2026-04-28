# Pneumonia Detection — Multi-Model XAI System

> **Core question:** Which model truly detects pneumonia — and which relies on shortcuts?

This project trains four CNN architectures on chest X-ray images, evaluates them under identical conditions, and applies three interpretability methods to determine whether each model focuses on the lungs or on irrelevant background features.

---

## Models

| Model | Architecture | Params |
|-------|-------------|--------|
| DenseNet-121 | Dense skip connections, ImageNet pretrained | ~7 M |
| ResNet-18 | Residual connections, ImageNet pretrained | ~11 M |
| EfficientNet-B0 | Compound scaling, ImageNet pretrained | ~5 M |
| MobileNet-V2 | Inverted residuals, ImageNet pretrained | ~3 M |

All models share: the same data, same transforms, same optimizer (Adam), same scheduler (StepLR), same loss (BCEWithLogitsLoss with class weighting), and the same decision threshold (0.3, tuned for recall).

---

## Interpretability

Three methods reveal *where* each model looks:

| Method | What it shows |
|--------|--------------|
| **Grad-CAM** | Gradients from the last conv layer — which spatial regions are most discriminative |
| **LRP** | Layer-wise relevance backpropagation — pixel-level relevance scores (requires `captum`) |
| **Occlusion Sensitivity** | Sliding 50×50 window — probability drop when a patch is hidden |

After generating maps, `src/analysis.py` scores each model by computing what fraction of attribution mass falls inside the lung region of the image:

- **≥ 0.60** → `GOOD — focuses on lungs`
- **< 0.40** → `BAD — focuses on background`
- **otherwise** → `MIXED`

---

## Quick Start

```bash
# 1. Install dependencies
make install
# pip install captum   # optional, required for LRP

# 2. Download dataset (Kaggle chest-xray-pneumonia)
# Place images in:  data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/

# 3. Populate fixed evaluation subset (5 NORMAL + 5 PNEUMONIA images)
make setup-eval

# 4. Run full pipeline
make all
```

---

## Makefile Targets

| Command | Description |
|---------|-------------|
| `make all` | Full pipeline: train → evaluate → interpret → analyze |
| `make train` | Train all 4 models, save `checkpoints/{model}.pt` |
| `make evaluate` | Evaluate all models, save per-model metrics + `final_comparison.csv` |
| `make interpret` | Run Grad-CAM, LRP, Occlusion on eval subset |
| `make analyze` | Score lung focus, update `final_comparison.csv` with verdicts |
| `make setup-eval` | Copy 5+5 images to `data/eval_subset/` |
| `make demo` | Launch Streamlit web demo |
| `make test` | Run pytest unit tests |
| `make clean` | Delete checkpoints and results |

---

## Output Structure

```
checkpoints/
  densenet121.pt
  resnet18.pt
  efficientnet_b0.pt
  mobilenet_v2.pt

results/
  {model}_metrics.json          # ROC-AUC, PR-AUC, F1, Recall, Accuracy @ threshold=0.3
  final_comparison.csv          # All models + interpretability verdicts
  analysis.json                 # Lung-focus scores per model and per method
  {model}_training_plot.png     # Loss / accuracy curves
  interpretability/
    {model}/
      {image_name}/
        gradcam.png             # Original | heatmap | overlay
        lrp.png
        occlusion.png
        gradcam.npy             # Raw (H, W) float32 attribution array
        lrp.npy
        occlusion.npy
        metadata.json           # Prediction probability, true/predicted class
```

---

## Evaluation Protocol

- **Test set:** 624 images (234 NORMAL + 390 PNEUMONIA) — standard Kaggle split
- **Fixed eval subset:** 5 NORMAL + 5 PNEUMONIA for interpretability (same images for all models)
- **Threshold:** 0.3 — prioritizes recall; missing pneumonia is more costly than a false alarm
- **Metrics:** ROC-AUC, PR-AUC, F1-score, Recall, Accuracy
- **Seed:** 42 for all random operations

---

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) — Paul Mooney, Kaggle.

```
data/chest_xray/
  train/  NORMAL (1 341)  PNEUMONIA (3 875)
  val/    NORMAL (8)       PNEUMONIA (8)
  test/   NORMAL (234)     PNEUMONIA (390)
```

---

## Requirements

```
torch torchvision numpy pandas matplotlib scikit-learn
pillow tqdm PyYAML streamlit
captum   # optional, for LRP
```

Install: `make install`

---

## Project Structure

```
src/
  models.py                get_model(name) — 4 pretrained architectures
  train.py                 Unified training loop  (--model flag)
  evaluate.py              Per-model metrics + final_comparison.csv
  run_interpretability.py  Grad-CAM + LRP + Occlusion for all models
  analysis.py              Lung-focus scoring and verdict assignment
  interpretability/
    gradcam.py             Grad-CAM implementation
    lrp.py                 LRP via Captum
    occlusion.py           Sliding-window occlusion sensitivity
  data.py                  DataLoaders + transforms
  utils.py                 Seed, checkpointing, logging

configs/
  config.yaml              Single config for all models

scripts/
  train_all.sh             Loop: train all 4 models
  setup_eval_subset.sh     Populate data/eval_subset/
```
