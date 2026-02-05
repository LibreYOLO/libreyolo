#!/bin/bash
# RF5 - Run all 15 models
# Usage: cd <repo_root> && bash tests/e2e/run_all.sh

set -o pipefail

MODELS=(
    "yolox.yaml nano"
    "yolox.yaml tiny"
    "yolox.yaml s"
    "yolox.yaml m"
    "yolox.yaml l"
    "yolox.yaml x"
    "yolo9.yaml t"
    "yolo9.yaml s"
    "yolo9.yaml m"
    "yolo9.yaml c"
    "rfdetr.yaml nano"
    "rfdetr.yaml small"
    "rfdetr.yaml base"
    "rfdetr.yaml medium"
    "rfdetr.yaml large"
)

PASSED=0
FAILED=0
FAILED_LIST=""

for entry in "${MODELS[@]}"; do
    config=$(echo "$entry" | awk '{print $1}')
    size=$(echo "$entry" | awk '{print $2}')
    echo ""
    echo "========================================================"
    echo "  STARTING: $config --size $size"
    echo "  $(date)"
    echo "========================================================"

    python -m tests.e2e.test_rf5_training --config "$config" --size "$size"

    if [ $? -eq 0 ]; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST="$FAILED_LIST  - $config $size\n"
        echo "  FAILED: $config $size (continuing...)"
    fi
done

echo ""
echo "========================================================"
echo "  RF5 ALL MODELS SUMMARY"
echo "  $(date)"
echo "========================================================"
echo "  Passed: $PASSED / ${#MODELS[@]}"
echo "  Failed: $FAILED / ${#MODELS[@]}"
if [ -n "$FAILED_LIST" ]; then
    echo "  Failed models:"
    echo -e "$FAILED_LIST"
fi
