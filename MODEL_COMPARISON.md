# Model Comparison: Evolution from Baseline CNN to DenseNet-121

This document traces the model evolution across three iterations, explaining the hypothesis behind each transition and comparing results.

## 1. Baseline CNN — Training from Scratch

**Architecture:** A simple 3-layer CNN with 32→64→128 filters, max pooling, dropout (0.5), and a fully connected classifier.

**Hypothesis:** A lightweight CNN trained from scratch should learn basic pneumonia patterns (opacities, consolidation) directly from the X-ray data, establishing a performance floor.

**Results:**

| Metric | Value |
|--------|-------|
| Accuracy | 0.8686 |
| Precision | 0.9254 |
| Recall | 0.8590 |
| F1 (threshold=0.5) | 0.8910 |
| F1 (threshold=0.3) | 0.9010 |
| ROC-AUC | 0.9415 |

**Observations:** The baseline CNN learns coarse spatial patterns but has limited capacity to capture the subtle texture differences between normal and pneumonia X-rays. With only ~25M parameters (most in the flattened FC layer), it relies on raw pixel patterns rather than hierarchical features.

---

## 2. ResNet-18 — Transfer Learning (Iteration 1)

**Hypothesis:** Pretrained ImageNet features (edges, textures, shapes) transfer well to medical imaging even though X-rays look different from natural photos. The hierarchical feature representations learned from millions of images should provide a stronger starting point than random initialization.

**Architecture:** ResNet-18 pretrained on ImageNet, with `fc` layer replaced by `nn.Linear(512, 1)`. All layers fine-tuned.

**Results:**

| Metric | Value |
|--------|-------|
| Accuracy | 0.8750 |
| Precision | 0.9727 |
| Recall | 0.8231 |
| F1 (threshold=0.5) | 0.8917 |
| F1 (threshold=0.3) | 0.9272 |
| ROC-AUC | 0.9743 |

**Observations:** Transfer learning provides a significant improvement over training from scratch. The pretrained low-level features (edge detectors, texture filters) adapt well to X-ray patterns with fine-tuning. However, ResNet-18's skip connections, while effective for gradient flow, do not explicitly encourage feature reuse across layers.

---

## 3. DenseNet-121 — Transfer Learning (Iteration 2, Final Model)

**Why not stop at ResNet-18?** While ResNet-18 improved over the baseline, we hypothesized that DenseNet-121's dense connectivity would further improve performance for two reasons:
1. **Feature reuse:** Each layer receives feature maps from all preceding layers, allowing the network to combine low-level texture features with high-level semantic features — important for detecting subtle radiographic patterns.
2. **Established medical imaging performance:** The CheXNet paper (Rajpurkar et al., 2017) demonstrated DenseNet-121's effectiveness on chest X-ray classification, achieving radiologist-level performance on 14 pathologies.

**Architecture:** DenseNet-121 pretrained on ImageNet, with `classifier` layer replaced by `nn.Linear(1024, 1)`. All layers fine-tuned.

**Results:**

| Metric | Value |
|--------|-------|
| Accuracy | 0.9071 |
| Precision | 0.9536 |
| Recall | 0.8949 |
| F1 (threshold=0.5) | 0.9233 |
| F1 (threshold=0.3) | 0.9506 |
| ROC-AUC | 0.9763 |

---

## 4. Comparison Table

| Model | Accuracy | Precision | Recall | F1 (0.5) | F1 (0.3) | ROC-AUC |
|-------|----------|-----------|--------|----------|----------|---------|
| Baseline CNN | 0.8686 | 0.9254 | 0.8590 | 0.8910 | 0.9010 | 0.9415 |
| ResNet-18 | 0.8750 | 0.9727 | 0.8231 | 0.8917 | 0.9272 | 0.9743 |
| DenseNet-121 | 0.9071 | 0.9536 | 0.8949 | 0.9233 | 0.9506 | 0.9763 |

## 5. Key Takeaways

- **Transfer learning matters:** Moving from a from-scratch CNN to pretrained models provides a large performance jump, confirming that ImageNet features transfer to medical imaging.
- **Architecture choice matters:** DenseNet-121's dense connectivity provides richer feature representations than ResNet-18's residual connections for this task.
- **Threshold tuning matters:** Lowering the decision threshold from 0.5 to 0.3 trades a small amount of precision for a larger recall gain — appropriate for medical screening where missing a pneumonia case is more costly than a false alarm.

## 6. Known Limitations and Mitigations

- **Small validation set (16 images):** Early stopping responds to noise rather than genuine overfitting. Each misclassification shifts validation accuracy by 6.25%. This is a known methodological weakness of the Kaggle dataset split. **Mitigation:** Stratified 5-fold cross-validation on the full 5,216-image training set is implemented in `src/kfold.py` — each fold uses ~1,043 images as validation (65× the official split), producing mean ± std estimates that are statistically meaningful. Run with `bash scripts/kfold.sh`.
- **Explainability:** Grad-CAM (Selvaraju et al., ICCV 2017) is implemented in `src/gradcam.py` and integrated into the Streamlit app. Heatmaps verify whether the model activates on the lung parenchyma or on spurious artifacts (scanner borders, text annotations). Run `bash scripts/gradcam.sh` to generate visualisation grids for all three models.

## 7. References

- Rajpurkar et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*. arXiv:1711.05225.
- Huang et al. (2017). *Densely Connected Convolutional Networks*. CVPR 2017.
- He et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
- Kohavi (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection*. IJCAI.
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV 2017.
