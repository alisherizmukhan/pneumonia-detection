#!/bin/bash
# Trains all three models. Baseline and ResNet-18 run in parallel on GPU 0/1;
# DenseNet-121 runs after on whichever GPU is free first (defaults to GPU 0).
set -e

cd "$(dirname "$0")/.."

echo "=== Training Baseline CNN (GPU 0) and ResNet-18 (GPU 1) in parallel ==="
CUDA_VISIBLE_DEVICES=0 python src/train.py --config configs/config_baseline.yaml &
PID_BASELINE=$!
CUDA_VISIBLE_DEVICES=1 python src/train.py --config configs/config_resnet18.yaml &
PID_RESNET=$!

wait $PID_BASELINE && echo "=== Baseline CNN done ===" || { echo "Baseline training failed"; exit 1; }
wait $PID_RESNET   && echo "=== ResNet-18 done ===" || { echo "ResNet-18 training failed"; exit 1; }

echo ""
echo "=== Training DenseNet-121 (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 python src/train.py --config configs/config.yaml
echo "=== DenseNet-121 done ==="

echo ""
echo "=== All models trained ==="
