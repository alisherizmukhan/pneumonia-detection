#!/bin/bash
# Run stratified 5-fold CV for all three models.
# With two GPUs, run ResNet-18 and Baseline in parallel on separate GPUs,
# then DenseNet-121 on its own.
set -e

cd "$(dirname "$0")/.."
FOLDS=${1:-5}

echo "=== K-Fold CV (k=${FOLDS}) — Baseline CNN ==="
python src/kfold.py --config configs/config_baseline.yaml --folds "$FOLDS"

echo "=== K-Fold CV (k=${FOLDS}) — ResNet-18 ==="
python src/kfold.py --config configs/config_resnet18.yaml --folds "$FOLDS"

echo "=== K-Fold CV (k=${FOLDS}) — DenseNet-121 ==="
python src/kfold.py --config configs/config.yaml --folds "$FOLDS"

echo "=== K-Fold CV complete. Results in results/kfold/ ==="
