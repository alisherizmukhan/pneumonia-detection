#!/bin/bash
# Train all four models sequentially with the same config.
set -e

CONFIG="${1:-configs/config.yaml}"

for model in densenet121 resnet18 efficientnet_b0 mobilenet_v2; do
    echo "========================================"
    echo "Training: $model"
    echo "========================================"
    python src/train.py --model "$model" --config "$CONFIG"
done

echo "All models trained."
