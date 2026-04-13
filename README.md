# Medical Image Classification: Pneumonia Detection

A deep learning system for binary classification of chest X-ray images into **Normal** vs **Pneumonia** using a fine-tuned DenseNet-121 model.

## Project Structure

```
pneumonia-detection/
├── src/
│   ├── data.py           # Data loading, transforms, class weights
│   ├── models.py         # DenseNet121 model definition
│   ├── train.py          # Training pipeline with early stopping
│   ├── evaluate.py       # Evaluation, error analysis, threshold tuning
│   ├── infer.py          # Single-image inference CLI
│   └── utils.py          # Utilities (seed, save/load, logging, config)
├── app.py                # Streamlit web demo
├── configs/
│   └── config.yaml       # Training and inference configuration
├── scripts/
│   ├── train.sh          # Training launch script
│   └── evaluate.sh       # Evaluation launch script
├── tests/                # Unit tests
├── results/              # Metrics, plots, error analysis
├── checkpoints/          # Saved model weights
├── data/chest_xray/      # Dataset (train/val/test splits)
├── requirements.txt
└── README.md
```

## Model

**DenseNet-121** pretrained on ImageNet, with the classifier layer replaced for binary output. Trained using:

- **Loss**: BCEWithLogitsLoss with class-weight correction for imbalanced data
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: StepLR (step_size=5, gamma=0.1)
- **Early Stopping**: patience=3 on validation loss
- **Threshold**: 0.3 (optimized via F1-score sweep)

## Setup

```bash
pip install -r requirements.txt
```

### Dataset

Download the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it at `data/chest_xray/` with `train/`, `val/`, and `test/` subdirectories.

## Training

```bash
python src/train.py --config configs/config.yaml
```

## Evaluation

```bash
python src/evaluate.py --config configs/config.yaml --model checkpoints/best_model.pt
```

Generates: `metrics.json`, `confusion_matrix.png`, `roc_curve.png`, threshold analysis, and error analysis (false positive/negative images).

## Inference

Single image:

```bash
python src/infer.py --image path/to/xray.jpg
```

Web demo:

```bash
streamlit run app.py
```

## Tests

```bash
python -m pytest tests/ -v
```
## Results

Final model performance (DenseNet121):

- Accuracy: 0.9071
- Precision: 0.9536
- Recall: 0.8949
- F1-score: 0.9233
- ROC-AUC: 0.9763

After threshold tuning (0.3):

- Accuracy: 0.9375
- Recall: 0.9615
- F1-score: 0.9506

> Note: A threshold of 0.3 is used instead of the default 0.5 to improve recall, which is critical in medical diagnosis tasks.

## Motivation

In medical diagnosis, false negatives (missing a disease) are more critical than false positives. 
Therefore, the model is optimized to maximize recall.

