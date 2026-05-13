#!/usr/bin/env bash
# Compare gradient accumulation: batch=32 accum=1 vs batch=2 accum=16.
#
# Uses the mask-wearing-608pr dataset (face detection, 2 classes: mask / no-mask).
# Fine-tunes from COCO-pretrained YOLOX weights.
#
# Usage:
#   bash scripts/compare_grad_accum.sh
#   bash scripts/compare_grad_accum.sh --device cuda --epochs 30
#
# All extra args are forwarded to compare_grad_accum.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv/bin/python"
DATASET_DIR="$HOME/datasets/mask-wearing"
DATA_YAML="$DATASET_DIR/data.yaml"

# ── 1. Download dataset if needed ────────────────────────────────────────────
if [ -f "$DATA_YAML" ]; then
    echo "Dataset already present at $DATASET_DIR — skipping download."
else
    echo "Downloading mask-wearing-608pr face detection dataset..."
    "$VENV" "$REPO_ROOT/scripts/download_mask_wearing.py" --dest "$DATASET_DIR"
fi

# ── 2. Run comparison ────────────────────────────────────────────────────────
echo ""
echo "Starting gradient accumulation comparison (fine-tune from pretrained COCO weights)..."
echo "  batch=32 accum=1  vs  batch=2 accum=16"
echo ""

"$VENV" "$REPO_ROOT/scripts/compare_grad_accum.py" \
    --data          "$DATA_YAML" \
    --epochs        50 \
    --eval-interval 1 \
    "$@" 2>&1 | tee "$REPO_ROOT/test_gradient_acc.txt"
