#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "=== Training Baseline CNN ==="
python src/train.py --config configs/config_baseline.yaml

echo "=== Training ResNet-18 ==="
python src/train.py --config configs/config_resnet18.yaml

echo "=== Training DenseNet-121 ==="
python src/train.py --config configs/config.yaml

echo "=== All models trained ==="
