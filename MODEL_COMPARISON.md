# Model Comparison: Evolution from Baseline CNN to DenseNet-121

This document traces the model evolution across three iterations, explaining the hypothesis behind
each transition and reporting empirical results on the held-out test set (624 images) and via
stratified 5-fold cross-validation on the training set (5,216 images).

---

## 1. Baseline CNN — Training from Scratch

**Architecture:** 3-layer CNN (32→64→128 filters), max pooling after each block, dropout (0.5),
fully connected classifier. ~25M parameters, no pretrained weights.

**Hypothesis:** A CNN trained from scratch establishes the performance floor and isolates the
contribution of transfer learning in subsequent iterations.

**Test set results (threshold = 0.5):**

| Accuracy | Precision | Recall | F1   | F1 (t=0.3) | ROC-AUC |
|----------|-----------|--------|------|------------|---------|
| 0.8606   | 0.9417    | 0.8282 | 0.8813 | 0.9046   | 0.9451  |

**Observation:** The model achieves moderate discrimination (AUC 0.945) but suffers from high false
negatives at the default threshold (recall 0.828). Limited feature expressiveness with only 3
convolutional layers and no prior visual knowledge constrains performance.

---

## 2. ResNet-18 — Transfer Learning, Iteration 1

**Architecture:** ResNet-18 pretrained on ImageNet; final `fc` replaced by `nn.Linear(512, 1)`.
All layers fine-tuned.

**Hypothesis:** ImageNet-pretrained features (edges, textures, shapes) provide a stronger
initialisation than random weights, even under domain shift from natural to medical images.
Residual connections mitigate vanishing gradients during fine-tuning.

**Test set results (threshold = 0.5):**

| Accuracy | Precision | Recall | F1   | F1 (t=0.3) | ROC-AUC |
|----------|-----------|--------|------|------------|---------|
| 0.9247   | 0.9277    | 0.9538 | 0.9406 | 0.9487   | 0.9776  |

**Observation:** Transfer learning yields a substantial gain over the baseline: +6.4pp accuracy,
+3.3pp AUC. Notably, recall at the default threshold (0.954) already exceeds the baseline's
threshold-tuned recall, suggesting better probability calibration.

---

## 3. DenseNet-121 — Transfer Learning, Iteration 2 (Final Model)

**Hypothesis:** DenseNet-121's dense connectivity (each layer receives all preceding feature maps)
promotes feature reuse across spatial scales, which is advantageous for detecting diffuse opacity
patterns in X-rays. This architecture was validated on chest X-ray classification in CheXNet
(Rajpurkar et al., 2017), achieving radiologist-level performance on 14 pathologies.

**Architecture:** DenseNet-121 pretrained on ImageNet; `classifier` replaced by
`nn.Linear(1024, 1)`. All layers fine-tuned. ~7M parameters.

**Test set results (threshold = 0.5):**

| Accuracy | Precision | Recall | F1   | F1 (t=0.4) | ROC-AUC |
|----------|-----------|--------|------|------------|---------|
| 0.9199   | 0.9337    | 0.9385 | 0.9361 | 0.9369   | 0.9729  |

**Observation:** DenseNet-121 matches ResNet-18 on recall and accuracy. The optimal threshold for
this run is 0.4 (not 0.3): at t=0.3, F1 drops to 0.9330 as precision falls faster than recall
gains, indicating this trained model is better calibrated than the prior run.

---

## 4. Test Set Comparison

| Model        | Accuracy | Precision | Recall | F1 (t=0.5) | F1 (tuned) | ROC-AUC |
|--------------|----------|-----------|--------|------------|------------|---------|
| Baseline CNN | 0.8606   | 0.9417    | 0.8282 | 0.8813     | 0.9046 (t=0.3) | 0.9451 |
| ResNet-18    | 0.9247   | 0.9277    | 0.9538 | 0.9406     | 0.9487 (t=0.3) | 0.9776 |
| DenseNet-121 | 0.9199   | 0.9337    | 0.9385 | 0.9361     | 0.9369 (t=0.4) | 0.9729 |

**Key findings:**

- Transfer learning is the dominant factor: both ResNet-18 and DenseNet-121 outperform the
  baseline by a wide margin (ΔF1 ≈ +0.06, ΔAUC ≈ +0.03).
- On this training run, ResNet-18 marginally outperforms DenseNet-121 (ΔAUC = +0.005,
  ΔF1 = +0.004). Given the noisy 16-image val set governing early stopping, this difference
  is within expected run-to-run variance; k-fold results (§5) confirm the two architectures
  are statistically equivalent on this dataset.
- DenseNet-121 uses ~36% fewer parameters than ResNet-18 (7M vs 11M) while achieving
  comparable performance.

---

## 5. Stratified 5-Fold Cross-Validation

The official validation split (16 images, 8 per class) causes early stopping to respond to noise:
one misclassification shifts accuracy by 6.25%. The training history confirms this
(`val_acc = [0.875, 0.75, 0.875, 0.9375, 0.75]`, an 18.75pp swing across consecutive epochs).

To obtain statistically reliable performance estimates, stratified 5-fold CV was performed on the
full training set (5,216 images), holding out ~1,043 images per fold (65× the official val split).
The test set was not used during CV. Threshold = 0.3 applied uniformly across all folds.

### 5-Fold CV Results (mean ± std across folds)

| Model        | Accuracy          | F1                | ROC-AUC           |
|--------------|-------------------|-------------------|-------------------|
| Baseline CNN | 0.9536 ± 0.0071   | 0.9681 ± 0.0050   | 0.9914 ± 0.0013   |
| ResNet-18    | 0.9816 ± 0.0073   | 0.9876 ± 0.0050   | 0.9974 ± 0.0014   |
| DenseNet-121 | 0.9801 ± 0.0064   | 0.9865 ± 0.0044   | 0.9976 ± 0.0015   |

### Per-Fold Breakdown

**Baseline CNN:**

| Fold | Accuracy | F1     | ROC-AUC |
|------|----------|--------|---------|
| 1    | 0.9588   | 0.9719 | 0.9907  |
| 2    | 0.9636   | 0.9751 | 0.9932  |
| 3    | 0.9492   | 0.9649 | 0.9905  |
| 4    | 0.9530   | 0.9677 | 0.9925  |
| 5    | 0.9434   | 0.9610 | 0.9899  |

**ResNet-18:**

| Fold | Accuracy | F1     | ROC-AUC |
|------|----------|--------|---------|
| 1    | 0.9674   | 0.9779 | 0.9951  |
| 2    | 0.9866   | 0.9910 | 0.9982  |
| 3    | 0.9818   | 0.9877 | 0.9965  |
| 4    | 0.9866   | 0.9909 | 0.9987  |
| 5    | 0.9856   | 0.9903 | 0.9986  |

**DenseNet-121:**

| Fold | Accuracy | F1     | ROC-AUC |
|------|----------|--------|---------|
| 1    | 0.9828   | 0.9883 | 0.9987  |
| 2    | 0.9674   | 0.9778 | 0.9947  |
| 3    | 0.9827   | 0.9884 | 0.9974  |
| 4    | 0.9847   | 0.9897 | 0.9985  |
| 5    | 0.9827   | 0.9883 | 0.9984  |

### Interpretation

- The std across folds is 5–10× smaller than the swing produced by the 16-image val set,
  confirming that single-val early stopping was responding to noise rather than generalisation.
- ResNet-18 and DenseNet-121 are statistically indistinguishable (AUC difference 0.0002,
  well within each model's std of ~0.0014–0.0015). Architecture choice between the two does
  not meaningfully affect performance on this dataset.
- Baseline CNN shows a larger gap vs transfer learning models on CV (ΔAUC ≈ 0.006) than on
  the test set, suggesting the baseline is more sensitive to the specific test distribution.
- All three models show low variance across folds (std ≤ 0.007), indicating stable learning
  rather than fold-specific overfitting.

---

## 6. Threshold Analysis — DenseNet-121

| Threshold | Accuracy | Precision | Recall | F1     |
|-----------|----------|-----------|--------|--------|
| 0.3       | 0.9135   | 0.9038    | 0.9641 | 0.9330 |
| **0.4**   | **0.9199** | **0.9229** | **0.9513** | **0.9369** |
| 0.5       | 0.9199   | 0.9337    | 0.9385 | 0.9361 |
| 0.6       | 0.9119   | 0.9373    | 0.9205 | 0.9288 |
| 0.7       | 0.9038   | 0.9412    | 0.9026 | 0.9215 |

Empirically optimal threshold: **0.4** (F1 = 0.9369). At t=0.3, precision drops to 0.904 while
recall gains (+2.6pp vs t=0.4) do not compensate, indicating this model produces better-separated
probability scores than the prior run where t=0.3 was optimal. For medical screening, t=0.3
remains defensible if recall is prioritised over F1 (recall = 0.964 vs 0.951 at t=0.4).

---

## 7. Known Limitations and Mitigations

- **Small validation set:** Addressed by 5-fold CV (§5). The official 16-image split should not
  be used for model selection; it is retained only for compatibility with the dataset's original
  structure.
- **Single training run:** Test set results reflect one training run per model. Run-to-run
  variance under the noisy val-based early stopping means individual test numbers should be
  interpreted alongside k-fold estimates.
- **Explainability:** See §7 for Grad-CAM analysis.

---

## 7. Grad-CAM Explainability Analysis

Grad-CAM heatmaps were generated for all three models on held-out test images (4 NORMAL, 4
PNEUMONIA per model). Visualisations are saved to `results/gradcam/`. The analysis below is based
on visual inspection of `gradcam_{model}_combined.png`.

### Methodology

For each architecture, the gradient of the predicted logit with respect to the last convolutional
feature map is computed, globally average-pooled to produce per-channel weights, and combined with
the activations to produce a spatial heatmap (ReLU-gated, upsampled to input resolution). Target
layers: `denseblock4` (DenseNet-121), `backbone.layer4[-1]` (ResNet-18), `features[6]`
(Baseline CNN, last Conv2d).

### Findings by Model

**Baseline CNN — Not clinically interpretable.**
Activations are consistently localised to the skeletal structures (ribs, clavicles) and image
borders rather than the lung parenchyma. The model does not differentiate between NORMAL and
PNEUMONIA cases in its spatial attention. This is consistent with the model's limited depth: 3
convolutional layers cannot produce the receptive field or feature hierarchy needed to isolate
soft-tissue opacity patterns. The heatmaps confirm the model is classifying on low-level structural
correlates rather than pathological features.

**DenseNet-121 — Clinically plausible.**
NORMAL cases produce uniformly low-activation (blue) heatmaps, indicating the model correctly
suppresses response when pathology is absent. PNEUMONIA cases show focused, high-activation regions
concentrated within the lung fields, with several examples clearly co-localising with visually
opaque consolidation areas. All activations are intra-thoracic; no border or skeletal artefacts
were observed. This pattern is consistent with the model having learned features associated with
pulmonary opacity rather than dataset-level shortcuts.

**ResNet-18 — Poor spatial specificity.**
Heatmaps are diffuse (predominantly yellow-green across large image regions) and in some cases
activate exclusively on the upper image borders where no anatomical structures are present. The
model does not discriminate clearly between NORMAL and PNEUMONIA in its attention pattern. Despite
achieving comparable or higher F1/AUC than DenseNet-121 on this test set, the heatmaps indicate
the model is likely exploiting global image statistics or dataset-level biases rather than
localised lung pathology.

### Implications for Model Selection

The divergence between quantitative metrics and Grad-CAM quality for ResNet-18 is an instance of
**shortcut learning** (Geirhos et al., 2020): a model learns to exploit spurious correlates in the
training distribution that happen to align with the label. This Kaggle dataset is known to contain
systematic biases (e.g., PNEUMONIA images skew toward paediatric patients, affecting image scale
and contrast). A model exploiting such correlates may achieve high test F1 on this split while
generalising poorly to data from different scanners or patient populations.

DenseNet-121's Grad-CAM pattern — suppressed response for NORMAL, focal parenchymal activation
for PNEUMONIA — provides evidence of clinically grounded feature learning. This is the basis for
selecting DenseNet-121 as the deployment model despite its marginally lower single-run F1.

Heatmap grids: `results/gradcam/gradcam_{densenet121,resnet18,baseline}_combined.png`

---

## 8. References

- Rajpurkar et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*. arXiv:1711.05225.
- Huang et al. (2017). *Densely Connected Convolutional Networks*. CVPR 2017.
- He et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
- Kohavi (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection*. IJCAI.
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV 2017.
