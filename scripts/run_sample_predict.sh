#!/bin/bash
# Run all models on sample images and save annotated results.
# Pass --augment to enable TTA.
#
# Usage:
#   ./vision_analysis_benchmark/run_sample_predict.sh
#   ./vision_analysis_benchmark/run_sample_predict.sh --augment

echo "Running sample prediction check..."

python scripts/run_sample_predict.py "$@"
