#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "=== Training DenseNet121 ==="
python src/train.py --config configs/config.yaml
