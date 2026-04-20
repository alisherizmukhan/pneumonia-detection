#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "=== Evaluating DenseNet-121 (primary model) ==="
python src/evaluate.py --config configs/config.yaml --model checkpoints/best_model_densenet121.pt
