# Project Explanation: Pneumonia Detection from Chest X-Rays

## 1. Problem Statement

Pneumonia is a leading cause of mortality worldwide, particularly among children under five and the elderly. Chest X-ray imaging is the most common diagnostic tool, but interpretation requires trained radiologists whose availability is limited in many regions. This project builds an automated binary classifier that distinguishes **Normal** from **Pneumonia** chest X-rays using deep learning, aiming to assist clinicians with faster, more consistent screening.

## 2. Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle, originally curated from Guangzhou Women and Children's Medical Center.

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

**Key characteristics:**
- **Class imbalance**: The training set has approximately a 1:2.9 ratio (Normal:Pneumonia), which is addressed through weighted loss.
- **Small validation set**: Only 16 images, which makes validation metrics noisy. Early stopping is used with patience to mitigate this.
- **Image format**: Grayscale X-rays converted to RGB (3-channel) for compatibility with ImageNet-pretrained models.

## 3. Model Architecture

The model is **DenseNet-121** (Densely Connected Convolutional Network) pretrained on ImageNet.

**Why DenseNet-121:**
- Dense connectivity encourages feature reuse and reduces parameter count compared to similarly deep architectures.
- Strong performance on medical imaging benchmarks due to rich multi-scale feature extraction.
- The 121-layer variant balances depth and computational cost.

**Modifications:**
- The original 1000-class ImageNet classifier head is replaced with a single linear layer: `nn.Linear(1024, 1)`, producing a raw logit for binary classification.
- All pretrained feature layers are fine-tuned (not frozen by default), allowing the model to adapt low-level features to grayscale X-ray patterns.

## 4. Training Pipeline

### Data Preprocessing
- **Training augmentation**: `RandomResizedCrop(224)`, `RandomHorizontalFlip`, `RandomRotation(10°)`, `ColorJitter(brightness=0.2, contrast=0.2)`
- **Test/Val transform**: `Resize(224×224)`
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Loss Function
`BCEWithLogitsLoss` with `pos_weight` computed from training class distribution. The positive weight equals `num_normal / num_pneumonia ≈ 0.346`, which down-weights the majority pneumonia class to prevent the model from trivially predicting pneumonia for everything.

### Optimizer and Scheduler
- **Adam** optimizer with learning rate 0.001 and weight decay 1e-4 for L2 regularization.
- **StepLR** scheduler reducing the learning rate by a factor of 0.1 every 5 epochs.

### Early Stopping
Training halts if validation loss does not improve for 3 consecutive epochs, preventing overfitting.

### Experiment Tracking
Each training run creates a numbered directory under `results/` containing:
- A copy of the configuration used
- Training history (loss/accuracy per epoch)
- Training curves plot
- Model metadata (parameter counts, class distribution)

## 5. Evaluation Pipeline

### Metrics
- **Accuracy**: Overall correctness
- **Precision**: Of predicted pneumonia cases, how many are correct
- **Recall (Sensitivity)**: Of actual pneumonia cases, how many are detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

### Threshold Tuning
The default sigmoid threshold of 0.5 is not necessarily optimal, especially with class imbalance. The evaluation pipeline sweeps thresholds [0.3, 0.4, 0.5, 0.6, 0.7] and selects the one maximizing F1-score. A lower threshold (0.3) tends to improve recall at a modest precision cost — appropriate for medical screening where missing a pneumonia case (false negative) is more harmful than a false alarm.

### Error Analysis
Misclassified images are extracted and saved:
- **False Positives**: Normal X-rays incorrectly classified as pneumonia
- **False Negatives**: Pneumonia X-rays missed by the model

This allows visual inspection of failure modes to guide future improvements.

## 6. Design Decisions

| Decision | Rationale |
|----------|-----------|
| DenseNet-121 over simpler CNNs | Transfer learning from ImageNet provides strong feature initialization; dense connections improve gradient flow |
| BCEWithLogitsLoss over CrossEntropyLoss | Numerically more stable for binary classification; supports pos_weight for class imbalance |
| Threshold 0.3 instead of 0.5 | Prioritizes recall (catching more pneumonia cases) in a medical screening context |
| ImageNet normalization on grayscale X-rays | The pretrained weights expect ImageNet-normalized inputs; converting to RGB preserves compatibility |
| Adam over SGD | Faster convergence with adaptive learning rates; weight decay provides regularization |
| Early stopping with patience 3 | Prevents overfitting while allowing the model time to recover from validation noise |

## 7. Project Structure

```
src/
├── data.py       — Dataset loading, augmentation, class weight computation
├── models.py     — DenseNet-121 model definition
├── train.py      — Full training loop with scheduling and early stopping
├── evaluate.py   — Metrics, plots, threshold tuning, error analysis
├── infer.py      — Single-image CLI inference
└── utils.py      — Seed, save/load, logging, config, experiment tracking

app.py            — Streamlit web interface for interactive demo
configs/config.yaml — Centralized hyperparameters and paths
```

## 8. How to Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset at data/chest_xray/ (train/val/test splits)

# 3. Train
python src/train.py --config configs/config.yaml

# 4. Evaluate
python src/evaluate.py --config configs/config.yaml --model checkpoints/best_model.pt

# 5. Run inference on a single image
python src/infer.py --image path/to/xray.jpg

# 6. Launch web demo
streamlit run app.py

# 7. Run tests
python -m pytest tests/ -v
```

## 9. Limitations and Future Work

**Current limitations:**
- **Small validation set** (16 images) makes early stopping and hyperparameter selection unreliable. A k-fold cross-validation approach would be more robust.
- **Binary classification only** — does not distinguish between bacterial and viral pneumonia, which have different treatment protocols.
- **Single dataset** — the model has not been validated on external datasets from different hospitals or imaging equipment, limiting generalizability.

**Potential improvements:**
- **Grad-CAM visualization**: Generate attention heatmaps to show which regions of the X-ray the model focuses on, improving interpretability for clinicians.
- **Multi-class extension**: Classify into Normal / Bacterial Pneumonia / Viral Pneumonia.
- **Ensemble methods**: Combine predictions from multiple architectures for improved robustness.
- **External validation**: Test on datasets from different institutions to assess generalization.
- **Data augmentation expansion**: Add elastic deformations and mixup augmentation strategies.
