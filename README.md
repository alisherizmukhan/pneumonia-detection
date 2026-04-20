# Medical Image Classification: Pneumonia Detection

Binary classification of chest X-ray images into **Normal** vs **Pneumonia** using deep learning with PyTorch. The project implements three models — Baseline CNN, ResNet-18, and DenseNet-121 — to demonstrate progressive improvement through transfer learning.

## Dataset Download

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
mv data/chest_xray data/chest_xray_tmp 2>/dev/null; mv data/chest-xray-pneumonia/chest_xray data/chest_xray 2>/dev/null || true
rm -rf data/chest-xray-pneumonia chest-xray-pneumonia.zip
```

Requires the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) and your `~/.kaggle/kaggle.json` credentials. After running, the dataset will be at `data/chest_xray/` with `train/`, `val/`, and `test/` subdirectories.

## Project Structure

```
pneumonia-detection/
├── src/
│   ├── data.py           # Data loading, augmentation, class weight computation
│   ├── models.py         # All 3 model architectures (BaselineCNN, ResNet-18, DenseNet-121)
│   ├── train.py          # Training pipeline (early stopping, LR scheduling, experiment tracking)
│   ├── evaluate.py       # Evaluation, threshold tuning, error analysis, model comparison
│   ├── infer.py          # Single-image CLI inference
│   └── utils.py          # Seed, save/load, logging, config, experiment tracking
├── app.py                # Streamlit web demo (DenseNet-121)
├── configs/
│   ├── config.yaml             # DenseNet-121 config (primary)
│   ├── config_resnet18.yaml    # ResNet-18 config
│   └── config_baseline.yaml    # Baseline CNN config
├── scripts/
│   ├── train.sh          # Train all 3 models
│   └── evaluate.sh       # Evaluate primary model + comparison
├── tests/                # Unit tests for models and data
├── results/              # Metrics, plots, error analysis, model comparison
├── checkpoints/          # Saved model weights
├── data/chest_xray/      # Dataset (train/val/test splits)
├── Makefile              # One-command build targets
├── requirements.txt      # Python dependencies
├── MODEL_COMPARISON.md   # Model evolution narrative and comparison
├── PROJECT_EXPLANATION.md # Detailed project explanation
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or using Make:

```bash
make install
```

### 2. Download the dataset

Download the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract it:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

The dataset should be at `data/chest_xray/` with `train/`, `val/`, and `test/` subdirectories.

### 3. Verify setup

```bash
make test
```

## Quick Start (One Command)

Train all 3 models and evaluate:

```bash
make train-all && make evaluate
```

## Training

Train individual models:

```bash
make train-baseline      # Baseline CNN
make train-resnet18      # ResNet-18
make train-densenet121   # DenseNet-121 (primary)
```

Or train all at once:

```bash
make train-all
```

Each training run:
- Saves the best checkpoint to `checkpoints/best_model_{name}.pt`
- Creates a numbered run directory under `results/run_N/` with config, history, and plots
- Uses early stopping (patience=3) and StepLR scheduling
- Applies class-weight correction for the imbalanced dataset
- Seeds all RNGs with seed=42 for reproducibility

## Evaluation

```bash
make evaluate
```

This evaluates DenseNet-121 on the test set and runs a 3-model comparison.

### Output Artifacts

After training and evaluation, the `results/` directory contains:

| File / Directory | Description |
|-----------------|-------------|
| `results/metrics.json` | Test-set metrics for the primary model (DenseNet-121): accuracy, precision, recall, F1, ROC-AUC |
| `results/comparison.json` | Side-by-side metrics for all 3 models, used to populate [MODEL_COMPARISON.md](MODEL_COMPARISON.md) |
| `results/best_threshold.json` | Threshold tuning results — sweeps [0.3–0.7] and reports the threshold maximizing F1 (found: 0.3) |
| `results/confusion_matrix.png` | Confusion matrix heatmap for the primary model on the test set |
| `results/roc_curve.png` | ROC curve with AUC score for the primary model |
| `results/training_plot.png` | Loss/accuracy curves for the last training run |
| `results/history.json` | Per-epoch training history (loss, accuracy, LR) for the last training run |
| `results/errors/` | Misclassified images — `false_positive/` and `false_negative/` subdirectories for error analysis |
| `results/run_N/` | Per-run archives: each training run saves its config, history, training plot, and model metadata |

## Inference

Single image:

```bash
python src/infer.py --image path/to/xray.jpg
```

Web demo:

```bash
make demo
```

## Models

| Model | Type | Description |
|-------|------|-------------|
| Baseline CNN | From scratch | 3-layer CNN with dropout |
| ResNet-18 | Transfer learning | Pretrained ImageNet backbone, fine-tuned |
| DenseNet-121 | Transfer learning | Pretrained ImageNet backbone, fine-tuned (primary) |

See [MODEL_COMPARISON.md](MODEL_COMPARISON.md) for the full evolution narrative and results comparison.

## Configuration

All configs share the same structure. Key parameters in `configs/config.yaml`:

```yaml
model: densenet121          # Model architecture
batch_size: 32
learning_rate: 0.001
epochs: 10
weight_decay: 0.0001        # L2 regularization
lr_step_size: 5             # StepLR schedule
lr_gamma: 0.1
early_stopping_patience: 3
threshold: 0.3              # Decision threshold (tuned for recall)
seed: 42                    # Reproducibility
```

## Tests

```bash
make test
```

Runs 12 tests covering all 3 model architectures (forward pass, frozen backbone, factory dispatch) and data loading.
## Results

### Model Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1 (0.5) | F1 (0.3) | ROC-AUC |
|-------|----------|-----------|--------|----------|----------|---------|
| Baseline CNN | 0.8686 | 0.9254 | 0.8590 | 0.8910 | 0.9010 | 0.9415 |
| ResNet-18 | 0.8750 | 0.9727 | 0.8231 | 0.8917 | 0.9272 | 0.9743 |
| **DenseNet-121** | **0.9071** | **0.9536** | **0.8949** | **0.9233** | **0.9506** | **0.9763** |

DenseNet-121 is the best model across all metrics. After threshold tuning (0.3), recall improves to 0.9615 with F1 = 0.9506.

> A threshold of 0.3 is used instead of the default 0.5 to improve recall — critical in medical diagnosis where missing a pneumonia case (false negative) is more costly than a false alarm.

See [MODEL_COMPARISON.md](MODEL_COMPARISON.md) for the full evolution narrative.

