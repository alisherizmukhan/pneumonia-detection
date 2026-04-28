#!/bin/bash
# Populate data/eval_subset/ with 5 NORMAL + 5 PNEUMONIA images from the test set.
# Usage: bash scripts/setup_eval_subset.sh [data_dir]
set -e

DATA_DIR="${1:-data/chest_xray}"
EVAL_DIR="data/eval_subset"
N=5

mkdir -p "$EVAL_DIR/NORMAL" "$EVAL_DIR/PNEUMONIA"

for class in NORMAL PNEUMONIA; do
    count=0
    src_dir="$DATA_DIR/test/$class"
    if [ ! -d "$src_dir" ]; then
        echo "Warning: $src_dir not found — skipping $class"
        continue
    fi
    for img in "$src_dir"/*.jpeg "$src_dir"/*.jpg "$src_dir"/*.png; do
        [ -f "$img" ] || continue
        cp "$img" "$EVAL_DIR/$class/"
        count=$((count + 1))
        [ "$count" -ge "$N" ] && break
    done
    echo "Copied $count $class images → $EVAL_DIR/$class/"
done

echo "Eval subset ready: $EVAL_DIR"
