#!/bin/bash
# Quick benchmark script - Just the small models for testing

echo "Running quick benchmark (small models only)..."

python vision_analysis_benchmark/benchmark_coco.py \
    --coco-yaml vision_analysis_benchmark/coco-val-only \
    --output-dir ./benchmark_results \
    --models yolov9t yoloxnano rfdetrn \
    --batch-size 16

echo ""
echo "Results saved to: ./benchmark_results/"
