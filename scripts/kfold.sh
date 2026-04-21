#!/bin/bash
# Stratified k-fold CV for all three models.
# Baseline and ResNet-18 run in parallel on GPU 0/1; DenseNet-121 after.
set -e

cd "$(dirname "$0")/.."
FOLDS=${1:-5}

echo "=== K-Fold CV (k=${FOLDS}) — Baseline CNN (GPU 0) + ResNet-18 (GPU 1) in parallel ==="
CUDA_VISIBLE_DEVICES=0 python src/kfold.py --config configs/config_baseline.yaml --folds "$FOLDS" &
PID_BASELINE=$!
CUDA_VISIBLE_DEVICES=1 python src/kfold.py --config configs/config_resnet18.yaml --folds "$FOLDS" &
PID_RESNET=$!

wait $PID_BASELINE && echo "=== Baseline k-fold done ===" || { echo "Baseline k-fold failed"; exit 1; }
wait $PID_RESNET   && echo "=== ResNet-18 k-fold done ===" || { echo "ResNet-18 k-fold failed"; exit 1; }

echo ""
echo "=== K-Fold CV (k=${FOLDS}) — DenseNet-121 (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 python src/kfold.py --config configs/config.yaml --folds "$FOLDS"
echo "=== DenseNet-121 k-fold done ==="

echo ""
echo "=== K-Fold CV complete. Results in results/kfold/ ==="
