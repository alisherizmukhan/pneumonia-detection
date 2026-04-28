#!/bin/bash
# Master script — runs the full pipeline on a 2-GPU machine.
#
# Order:
#   1. Train all 3 models        (Baseline+ResNet-18 parallel on GPU0/1, then DenseNet-121)
#   2. Evaluate all 3 models     (sequential, fast)
#   3. K-fold CV                 (Baseline+ResNet-18 parallel on GPU0/1, then DenseNet-121)
#   4. Grad-CAM grids            (sequential, CPU-light, requires trained checkpoints)
#   5. Collect results → CSV     (instant)
#
# After this script completes, everything you need for presentation is in results/:
#   results/all_results.csv      — flat table of all metrics
#   results/kfold_summary.csv    — k-fold mean ± std per model
#   results/threshold_sweep.csv  — threshold sweep for DenseNet-121
#   results/gradcam/             — heatmap grids per model
#   results/comparison.json      — raw comparison JSON
set -e

cd "$(dirname "$0")/.."
FOLDS=${1:-5}

echo "========================================"
echo " STEP 1: Training"
echo "========================================"
bash scripts/train.sh

echo ""
echo "========================================"
echo " STEP 2: Evaluation"
echo "========================================"
bash scripts/evaluate.sh

echo ""
echo "========================================"
echo " STEP 3: K-Fold CV (k=${FOLDS})"
echo "========================================"
bash scripts/kfold.sh "$FOLDS"

echo ""
echo "========================================"
echo " STEP 4: Grad-CAM"
echo "========================================"
bash scripts/gradcam.sh

echo ""
echo "========================================"
echo " STEP 5: Collect results → CSV"
echo "========================================"
python src/collect_results.py --results-dir results

echo ""
echo "======================================== "
echo " All done. Results in results/"
echo "========================================"
