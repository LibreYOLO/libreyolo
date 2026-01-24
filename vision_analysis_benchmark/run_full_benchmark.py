#!/usr/bin/env python3
"""
Run full benchmark of all LibreYOLO models on COCO val2017.

This will benchmark all 19 model variants:
- YOLOv8: n, s, m, l, x (5 models)
- YOLOv9: t, s, m, c (4 models)
- YOLOv11: n, s, m, l, x (5 models)
- YOLOX: nano, tiny, s, m, l, x (6 models)

Results will be saved to: ./benchmark_results/
"""

import subprocess
import sys
from pathlib import Path

# Configuration
COCO_YAML = "vision_analysis_benchmark/coco_benchmark.yaml"
OUTPUT_DIR = "./benchmark_results"
BATCH_SIZE = 16  # Use batch size 16 for faster processing

# Models to benchmark (leave empty to benchmark ALL)
# To benchmark specific models, add them like: ["yolov8n", "yolov11s"]
MODELS_TO_BENCHMARK = []  # Empty = all models

def main():
    print("="*80)
    print("LibreYOLO Full Benchmark Script")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  COCO YAML: {COCO_YAML}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Batch size: {BATCH_SIZE}")

    if MODELS_TO_BENCHMARK:
        print(f"  Models: {', '.join(MODELS_TO_BENCHMARK)}")
    else:
        print(f"  Models: ALL (19 models)")

    # Check if COCO yaml exists
    if not Path(COCO_YAML).exists():
        print(f"\n❌ Error: COCO YAML not found at {COCO_YAML}")
        print("   Please create it or update the path in this script.")
        sys.exit(1)

    # Build command
    cmd = [
        "python",
        "vision_analysis_benchmark/benchmark_coco.py",
        "--coco-yaml", COCO_YAML,
        "--output-dir", OUTPUT_DIR,
        "--batch-size", str(BATCH_SIZE),
    ]

    # Add specific models if requested
    if MODELS_TO_BENCHMARK:
        cmd.extend(["--models"] + MODELS_TO_BENCHMARK)

    print(f"\n{'='*80}")
    print("Running benchmark...")
    print(f"{'='*80}")
    print(f"\nCommand: {' '.join(cmd)}\n")

    # Run the benchmark
    try:
        subprocess.run(cmd, check=True)

        print(f"\n{'='*80}")
        print("✅ Benchmark Complete!")
        print(f"{'='*80}")
        print(f"\nResults saved to: {OUTPUT_DIR}/")
        print(f"  - Individual JSON files: {OUTPUT_DIR}/<model>_benchmark.json")
        print(f"  - Summary CSV: {OUTPUT_DIR}/benchmark_summary.csv")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Benchmark failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    main()
