#!/bin/bash
set -e

cd "$(dirname "$0")/.."
python src/evaluate.py --config configs/config.yaml --model checkpoints/best_model.pt
