#!/usr/bin/env python3
"""
Test using the proper model.val() API instead of manual predict loop.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from libreyolo import LIBREYOLO
from libreyolo.validation import ValidationConfig

def test_yolox_s_val():
    print("Testing YOLOX-s with proper .val() API...")

    # Paths
    weights_path = "weights/libreyoloXs.pt"
    coco_path = Path("/home/jovyan/datasets/coco")

    print(f"\n1. Loading model from {weights_path}...")
    model = LIBREYOLO(model_path=weights_path, device='auto')
    print(f"   ✓ Model loaded on {model.device}")

    # Count parameters
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"   ✓ Parameters: {params:.2f}M")

    # Run validation
    print(f"\n2. Running validation on COCO val2017...")
    yaml_path = Path("vision_analysis_benchmark/coco_benchmark.yaml")

    results = model.val(
        data=str(yaml_path),
        batch=16,
        imgsz=640,
        conf=0.001,
        iou=0.6,
        verbose=True,
    )

    # Print results (model.val() returns a dict)
    print(f"\n4. Results:")
    print(f"   mAP@50-95: {results.get('metrics/mAP50-95', 0):.4f}")
    print(f"   mAP@50:    {results.get('metrics/mAP50', 0):.4f}")
    print(f"   Precision: {results.get('metrics/precision', 0):.4f}")
    print(f"   Recall:    {results.get('metrics/recall', 0):.4f}")

    print("\n✓ Validation complete using proper .val() API!")
    print(f"\nAll metrics:")
    for key, value in results.items():
        if 'metrics' in key:
            print(f"   {key}: {value:.4f}")

    return {
        'params_millions': params,
        'map_50_95': float(results.get('metrics/mAP50-95', 0)),
        'map_50': float(results.get('metrics/mAP50', 0)),
        'precision': float(results.get('metrics/precision', 0)),
        'recall': float(results.get('metrics/recall', 0)),
    }

if __name__ == '__main__':
    results = test_yolox_s_val()
    print("\n" + "="*60)
    print("SUMMARY:")
    print(json.dumps(results, indent=2))
