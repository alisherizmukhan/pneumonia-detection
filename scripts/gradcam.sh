#!/bin/bash
# Generate Grad-CAM grids for all trained models.
# Run this AFTER scripts/train.sh has completed.
set -e

cd "$(dirname "$0")/.."

echo "=== Grad-CAM — DenseNet-121 ==="
python src/gradcam.py \
    --config configs/config.yaml \
    --model checkpoints/best_model_densenet121.pt \
    --num-images 8 \
    --output results/gradcam

echo "=== Grad-CAM — ResNet-18 ==="
python src/gradcam.py \
    --config configs/config_resnet18.yaml \
    --model checkpoints/best_model_resnet18.pt \
    --num-images 8 \
    --output results/gradcam

echo "=== Grad-CAM — Baseline CNN ==="
python src/gradcam.py \
    --config configs/config_baseline.yaml \
    --model checkpoints/best_model_baseline.pt \
    --num-images 8 \
    --output results/gradcam

echo "=== Grad-CAM grids saved to results/gradcam/ ==="
