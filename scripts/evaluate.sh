#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "=== Evaluating Baseline CNN ==="
python src/evaluate.py --config configs/config_baseline.yaml \
    --model checkpoints/best_model_baseline.pt

echo ""
echo "=== Evaluating ResNet-18 ==="
python src/evaluate.py --config configs/config_resnet18.yaml \
    --model checkpoints/best_model_resnet18.pt

echo ""
echo "=== Evaluating DenseNet-121 ==="
python src/evaluate.py --config configs/config.yaml \
    --model checkpoints/best_model_densenet121.pt

echo "=== All evaluations complete ==="
