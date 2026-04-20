# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All `src/` scripts must be run from the project root (they use relative imports resolved via `sys.path.insert`).

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Train:**
```bash
python src/train.py --config configs/config.yaml
```

**Evaluate:**
```bash
python src/evaluate.py --config configs/config.yaml --model checkpoints/best_model.pt
```

**Single-image inference:**
```bash
python src/infer.py --image path/to/xray.jpg --model checkpoints/best_model.pt
```

**Streamlit web demo:**
```bash
streamlit run app.py
```

**Run all tests:**
```bash
python -m pytest tests/ -v
```

**Run a single test file:**
```bash
python -m pytest tests/test_models.py -v
```

## Dataset Setup

Download the [Chest X-Ray Pneumonia dataset from Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it at `data/chest_xray/` with `train/`, `val/`, and `test/` subdirectories, each containing `NORMAL/` and `PNEUMONIA/` subfolders (standard `ImageFolder` layout).

## Architecture

This is a binary medical image classifier (Normal vs Pneumonia) built on PyTorch.

**Data flow:** `configs/config.yaml` → `src/train.py` or `src/evaluate.py` → `src/data.py` (dataloaders + class weights) → `src/models.py` (DenseNet121) → `checkpoints/best_model.pt`

**Key design decisions:**
- Model outputs a **single raw logit** (not probability). `torch.sigmoid` is applied at inference time. The classifier head is a single `nn.Linear` replacing DenseNet's default classifier.
- **Inference threshold is 0.3** (not 0.5). This is intentional — in medical diagnosis, false negatives (missing pneumonia) are more costly than false positives. The threshold is set in `configs/config.yaml` and read by both `evaluate.py` and `infer.py`.
- **Class imbalance** is handled via `BCEWithLogitsLoss(pos_weight=...)`, where `pos_weight = num_normal / num_pneumonia` computed from the training set at runtime in `data.compute_class_weights`.
- **Experiment tracking** is minimal but built-in: each training run auto-creates a numbered `results/run_N/` directory containing a copy of the config, training history JSON, and plots. The best checkpoint is always saved to `checkpoints/best_model.pt`.

**Module responsibilities:**
- `src/data.py` — transforms (train augmentation vs test resize-only), dataloader creation, class weight computation
- `src/models.py` — single function `get_model(freeze_backbone)` returning the DenseNet121 with replaced classifier
- `src/train.py` — full training loop with early stopping, LR scheduling, history tracking
- `src/evaluate.py` — test-set evaluation, confusion matrix/ROC plots, error image extraction, threshold sweep
- `src/utils.py` — seed setting, model save/load, device selection, logger, config loader, `EarlyStopping`, run directory management
- `src/infer.py` — standalone CLI for single-image prediction
- `app.py` — Streamlit web UI wrapping `infer.predict`

**Config:** All hyperparameters (lr, batch size, epochs, threshold, paths) live in `configs/config.yaml`. `configs/config_baseline.yaml` is a reference baseline config.
