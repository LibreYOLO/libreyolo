#!/bin/bash
# Full benchmark script - ALL 15 models

echo "Running FULL benchmark (all 19 models)..."
echo "This will take a while..."

python vision_analysis_benchmark/benchmark_coco.py \
    --coco-yaml vision_analysis_benchmark/coco_benchmark.yaml \
    --output-dir ./benchmark_results \
    --batch-size 16

echo ""
echo "Results saved to: ./benchmark_results/"
echo "  - Individual JSONs: ./benchmark_results/<model>_benchmark.json"
echo "  - Summary CSV: ./benchmark_results/benchmark_summary.csv"
