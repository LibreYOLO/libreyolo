#!/usr/bin/env python3
"""
Benchmark only YOLOv9 models - quick script to avoid re-running all models
"""

import sys
from pathlib import Path

# Import the main benchmark script
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_coco import main, LIBREYOLO_MODELS
import argparse

if __name__ == '__main__':
    # Override to only benchmark YOLOv9
    original_models = LIBREYOLO_MODELS.copy()

    # Keep only YOLO9
    LIBREYOLO_MODELS.clear()
    LIBREYOLO_MODELS['yolo9'] = original_models['yolo9']

    print("=" * 80)
    print("YOLOv9-ONLY BENCHMARK")
    print("Will only benchmark YOLOv9 variants: t, s, m, c")
    print("=" * 80)

    # Run main benchmark
    main()
