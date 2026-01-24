#!/usr/bin/env python3
"""
Simple test of the benchmark with just 10 images to verify functionality.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from libreyolo import LIBREYOLO

# CUDA Timer
class CUDATimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        torch.cuda.synchronize()
        self.start_event.record()

    def stop(self) -> float:
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)

# Simple test
def test_yolov8n():
    print("Testing YOLOv8n benchmark...")

    # Paths
    weights_path = "../weights/libreyolo8n.pt"
    coco_path = Path("/home/jovyan/datasets/coco")
    ann_file = coco_path / "annotations" / "instances_val2017.json"
    img_dir = coco_path / "images" / "val2017"

    print(f"\n1. Loading model from {weights_path}...")
    model = LIBREYOLO(model_path=weights_path, size='n', device='auto')
    print(f"   ✓ Model loaded on {model.device}")

    # Count parameters
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"   ✓ Parameters: {params:.2f}M")

    # Warmup
    print("\n2. Warming up (10 iterations)...")
    dummy = torch.randn(1, 3, 640, 640).to(model.device)
    for _ in range(10):
        with torch.no_grad():
            _ = model._forward(dummy)
    torch.cuda.synchronize()
    print("   ✓ Warmup complete")

    # Load COCO
    print(f"\n3. Loading COCO from {ann_file}...")
    coco_gt = COCO(str(ann_file))
    img_ids = sorted(coco_gt.getImgIds())  # All 5000 images
    print(f"   ✓ Loaded {len(img_ids)} images")

    # Run inference
    print("\n4. Running inference...")
    timings = []
    predictions = []

    for img_id in tqdm(img_ids, desc="Processing"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_dir / img_info['file_name']

        # Time it
        timer = CUDATimer()
        timer.start()
        results = model.predict(str(img_path), save=False, conf_thres=0.001, iou_thres=0.6)
        elapsed = timer.stop()

        timings.append(elapsed)

        # Convert to COCO format
        if results['num_detections'] > 0:
            boxes = results['boxes']
            scores = results['scores']
            classes = results['classes']

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                predictions.append({
                    'image_id': img_id,
                    'category_id': int(cls) + 1,
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score),
                })

    # Timing stats
    timings = np.array(timings)
    print(f"\n5. Timing Results:")
    print(f"   Mean:  {np.mean(timings):.2f}ms")
    print(f"   Std:   {np.std(timings):.2f}ms")
    print(f"   p50:   {np.percentile(timings, 50):.2f}ms")
    print(f"   p95:   {np.percentile(timings, 95):.2f}ms")
    print(f"   p99:   {np.percentile(timings, 99):.2f}ms")
    print(f"   FPS:   {1000.0 / np.mean(timings):.1f}")

    # Evaluate
    print(f"\n6. COCO Evaluation ({len(predictions)} detections)...")
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = img_ids  # Only evaluate on our test images
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        print(f"\n7. Accuracy Results:")
        print(f"   mAP@50-95: {coco_eval.stats[0]:.4f}")
        print(f"   mAP@50:    {coco_eval.stats[1]:.4f}")
        print(f"   mAP small: {coco_eval.stats[3]:.4f}")
        print(f"   mAP med:   {coco_eval.stats[4]:.4f}")
        print(f"   mAP large: {coco_eval.stats[5]:.4f}")
    else:
        print("   No detections made!")

    print("\n✓ Test complete! All measurements working.")

    return {
        'params_millions': params,
        'timing_mean_ms': float(np.mean(timings)),
        'timing_p50_ms': float(np.percentile(timings, 50)),
        'fps': float(1000.0 / np.mean(timings)),
        'map_50_95': float(coco_eval.stats[0]) if predictions else 0.0,
        'map_50': float(coco_eval.stats[1]) if predictions else 0.0,
    }

if __name__ == '__main__':
    results = test_yolov8n()
    print("\n" + "="*60)
    print("SUMMARY:")
    print(json.dumps(results, indent=2))
