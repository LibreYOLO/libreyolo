"""
COCO evaluator for LibreYOLO.

Provides COCO-standard evaluation metrics (AP, AR, etc.) for object detection.
Works with both COCO format and Ultralytics YOLO format datasets via YOLOCocoAPI.
"""

from pathlib import Path
from typing import Dict, List, Optional
import json

import numpy as np
import torch


class COCOEvaluator:
    """
    COCO evaluation wrapper for LibreYOLO.

    Provides standard COCO metrics:
    - AP (mAP@[0.5:0.95]): Mean Average Precision at IoU 0.5 to 0.95
    - AP50 (mAP@0.5): Mean Average Precision at IoU 0.5
    - AP75 (mAP@0.75): Mean Average Precision at IoU 0.75
    - AP_small/medium/large: AP for different object sizes
    - AR1/10/100: Average Recall with 1/10/100 detections
    - AR_small/medium/large: AR for different object sizes

    Example:
        >>> from libreyolo.data import create_yolo_coco_api
        >>> coco_gt = create_yolo_coco_api("data.yaml", split="val")
        >>> evaluator = COCOEvaluator(coco_gt)
        >>>
        >>> # After inference
        >>> for img_id, predictions in enumerate(all_predictions):
        ...     evaluator.update(predictions, img_id)
        >>>
        >>> metrics = evaluator.compute()
        >>> print(f"mAP50-95: {metrics['mAP']:.3f}")
    """

    def __init__(self, coco_gt, iou_type: str = 'bbox'):
        """
        Initialize COCO evaluator.

        Args:
            coco_gt: Ground truth COCO API (YOLOCocoAPI or pycocotools COCO)
            iou_type: Type of IoU to compute ('bbox', 'segm', or 'keypoints')
        """
        self.coco_gt = coco_gt
        self.iou_type = iou_type
        self.results = []
        self._img_ids = set()

    def update(self, predictions: Dict, image_id: int):
        """
        Add predictions for an image.

        Args:
            predictions: Dictionary with keys:
                - 'boxes': List or tensor of boxes in xyxy format [[x1, y1, x2, y2], ...]
                - 'scores': List or tensor of confidence scores
                - 'classes': List or tensor of class IDs (0-indexed)
            image_id: Image ID (must match COCO API image IDs)
        """
        boxes = predictions['boxes']
        scores = predictions['scores']
        classes = predictions['classes']

        # Convert to numpy if tensors
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(classes, torch.Tensor):
            classes = classes.cpu().numpy()

        # Convert to lists
        boxes = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes
        scores = np.array(scores) if not isinstance(scores, np.ndarray) else scores
        classes = np.array(classes) if not isinstance(classes, np.ndarray) else classes

        # Add each detection
        for box, score, label in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            # COCO format: [x, y, width, height]
            self.results.append({
                'image_id': int(image_id),
                'category_id': int(label),  # COCO uses 0-indexed like we do
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'score': float(score)
            })

        self._img_ids.add(image_id)

    def compute(self, save_json: Optional[str] = None) -> Dict[str, float]:
        """
        Compute COCO metrics.

        Args:
            save_json: Optional path to save predictions in COCO JSON format

        Returns:
            Dictionary with metrics:
                - mAP: Mean AP @[0.5:0.95]
                - mAP50: Mean AP @0.5
                - mAP75: Mean AP @0.75
                - mAP_small: Mean AP for small objects
                - mAP_medium: Mean AP for medium objects
                - mAP_large: Mean AP for large objects
                - AR1: Average Recall with 1 det per image
                - AR10: Average Recall with 10 dets per image
                - AR100: Average Recall with 100 dets per image
                - AR_small: AR for small objects
                - AR_medium: AR for medium objects
                - AR_large: AR for large objects
        """
        if len(self.results) == 0:
            print("Warning: No predictions to evaluate")
            return self._empty_metrics()

        # Save predictions if requested
        if save_json:
            with open(save_json, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Saved predictions to {save_json}")

        # Use pycocotools for evaluation
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            raise ImportError(
                "pycocotools not installed. Install with: pip install pycocotools"
            )

        # Load predictions as COCO results
        coco_dt = self.coco_gt.loadRes(self.results)

        # Run COCO evaluation
        coco_eval = COCOeval(self.coco_gt, coco_dt, self.iou_type)

        # Optionally restrict to images we have predictions for
        # coco_eval.params.imgIds = sorted(list(self._img_ids))

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics from COCO stats
        # stats[0]: AP @[0.5:0.95]
        # stats[1]: AP @0.5
        # stats[2]: AP @0.75
        # stats[3]: AP small
        # stats[4]: AP medium
        # stats[5]: AP large
        # stats[6]: AR maxDets=1
        # stats[7]: AR maxDets=10
        # stats[8]: AR maxDets=100
        # stats[9]: AR small
        # stats[10]: AR medium
        # stats[11]: AR large

        return {
            'mAP': float(coco_eval.stats[0]),           # mAP@[0.5:0.95]
            'mAP50': float(coco_eval.stats[1]),         # mAP@0.5
            'mAP75': float(coco_eval.stats[2]),         # mAP@0.75
            'mAP_small': float(coco_eval.stats[3]),     # mAP for small objects
            'mAP_medium': float(coco_eval.stats[4]),    # mAP for medium objects
            'mAP_large': float(coco_eval.stats[5]),     # mAP for large objects
            'AR1': float(coco_eval.stats[6]),           # AR with 1 det per image
            'AR10': float(coco_eval.stats[7]),          # AR with 10 dets per image
            'AR100': float(coco_eval.stats[8]),         # AR with 100 dets per image
            'AR_small': float(coco_eval.stats[9]),      # AR for small objects
            'AR_medium': float(coco_eval.stats[10]),    # AR for medium objects
            'AR_large': float(coco_eval.stats[11]),     # AR for large objects
        }

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict (all zeros)."""
        return {
            'mAP': 0.0,
            'mAP50': 0.0,
            'mAP75': 0.0,
            'mAP_small': 0.0,
            'mAP_medium': 0.0,
            'mAP_large': 0.0,
            'AR1': 0.0,
            'AR10': 0.0,
            'AR100': 0.0,
            'AR_small': 0.0,
            'AR_medium': 0.0,
            'AR_large': 0.0,
        }

    def reset(self):
        """Clear all accumulated results."""
        self.results = []
        self._img_ids = set()
