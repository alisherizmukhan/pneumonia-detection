# Project Feedback

Overall this is a well-structured project with real understanding behind the implementation choices. The pipeline is end-to-end, the code is clean, and the final model performs strongly. The items below are what separates a good project from an excellent one.

---

## 1. The model evolution is not traceable — this is the biggest gap

The project clearly went through at least three iterations: **Baseline CNN → ResNet-18 → DenseNet-121**. But the final repository only contains DenseNet-121 code. To get full marks the evolution needs to be fully preserved and narrated.

### What to do

**Keep all model code in the repository.** Add the Baseline CNN and ResNet-18 implementations back into `src/models.py` (or separate files), dispatch on the `model` key already present in `config_baseline.yaml`. The evolution should be rerunnable, not just described.

**Add a model comparison document** (e.g. `MODEL_COMPARISON.md` or a dedicated section in `PROJECT_OVERVIEW.md`) that follows this structure:

1. **Baseline CNN results** — already recorded in `PROJECT_OVERVIEW.md`, so this part exists
2. **Why ResNet-18 was tried** — hypothesis: pretrained ImageNet features transfer to X-rays better than training from scratch
3. **ResNet-18 results** — already recorded in `PROJECT_OVERVIEW.md`
4. **Why ResNet-18 was not the final answer** — this is currently missing. What specific gap motivated the switch? (e.g. "ResNet-18 achieved 0.974 AUC but we hypothesized DenseNet-121's dense connectivity would improve gradient flow and feature reuse, particularly for subtle X-ray texture patterns, as shown in the CheXNet paper")
5. **DenseNet-121 results** — currently only in `README.md`; needs to be added to the comparison table alongside the others

**Add DenseNet-121 to the comparison table in `PROJECT_OVERVIEW.md`.** Currently the table ends at ResNet-18. The table should look like:

| Model | Accuracy | F1 (0.5) | F1 (tuned) | ROC-AUC |
|-------|----------|----------|------------|---------|
| Baseline CNN | 0.8686 | 0.8910 | — | 0.9415 |
| ResNet-18 | 0.8750 | 0.8917 | 0.9272 | 0.9743 |
| DenseNet-121 | 0.9071 | 0.9233 | 0.9506 | 0.9763 |

Without this, a reader cannot see whether DenseNet-121 was actually an improvement, or whether the refactor was just cosmetic.

---

## 2. `config_baseline.yaml` is broken

`config_baseline.yaml` contains `model: baseline`, but `get_model()` in `src/models.py` only accepts `freeze_backbone` — there is no model type dispatch. Running a training with this config would silently train a DenseNet-121, not a Baseline CNN. Once the Baseline CNN code is restored (see point 1), `get_model()` needs to accept the `model` key from the config and route accordingly.

---

## 3. `PROJECT_OVERVIEW.md` contradicts the actual code

`PROJECT_OVERVIEW.md` describes ResNet-18 as the project model, with architecture details, a threshold sweep table, and error analysis numbers specific to ResNet-18. The actual code and final results use DenseNet-121. A reader following the overview document would be confused about what the project actually does. This document needs to be updated to reflect the final model (or restructured as a historical comparison, which is actually the better choice given point 1).

---

## 4. Validation set makes early stopping unreliable — acknowledge it more concretely

The validation set has 16 images. Each misclassification moves accuracy by 6.25%. The saved training history confirms this: `val_acc = [0.875, 0.75, 0.875, 0.9375, 0.75]`. Early stopping triggered at epoch 5, but it was responding to noise, not genuine overfitting. The train/val loss gap at stopping (`0.061` vs `0.208`) also signals that training ended prematurely.

This is a known limitation of this dataset. The fix is to do **k-fold cross-validation on the training set** (ignoring the official 16-image val split for model selection) and only use the test set for final reporting. The project mentions k-fold as "future work" but it should be framed as a known methodological weakness of the current approach, not a nice-to-have.

---

## 5. No explainability

For a medical AI project, showing *what the model looks at* matters as much as accuracy. Grad-CAM heatmaps overlaid on the X-ray images would show whether the model activates on lung regions or on spurious artifacts (scanner borders, text annotations, etc.) — a real concern with this dataset. Currently the project mentions Grad-CAM as future work. Adding even a basic Grad-CAM visualization script would meaningfully strengthen the project.

---

## What is already correct and should not change

- **`BCEWithLogitsLoss` with `pos_weight`** — correctly computed and applied. The weight of 0.346 equalizes the loss contribution from both classes given the 2.9:1 imbalance.
- **Threshold 0.3** — empirically validated via the F1 sweep, and the medical justification (prioritizing recall) is well-articulated.
- **Single logit output + sigmoid at inference** — correct pattern; `BCEWithLogitsLoss` requires raw logits.
- **Full fine-tuning (not frozen backbone)** — correct for this domain; X-ray texture patterns differ enough from ImageNet that fine-tuning all layers outperforms head-only training.
- **Experiment tracking** (`results/run_N/`) — clean and lightweight.
- **Streamlit app** — `@st.cache_resource`, config-driven threshold, two-tab interface — all correctly implemented.
- **Tests** — dataloader tests use a proper temporary dataset fixture with teardown; model tests cover forward pass, frozen backbone, and output shape.
