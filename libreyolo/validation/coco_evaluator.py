"""COCO evaluator for LibreYOLO."""

from typing import Dict, Optional
import json

import numpy as np
import torch


class COCOEvaluator:
    """
    COCO evaluation wrapper.

    Computes standard COCO metrics: AP (mAP@[0.5:0.95]), AP50, AP75,
    AP/AR by object size, and AR at different maxDets.
    """

    def __init__(self, coco_gt, iou_type: str = "bbox"):
        self.coco_gt = coco_gt
        self.iou_type = iou_type
        self.results = []
        self._img_ids = set()

    def update(self, predictions: Dict, image_id: int):
        """
        Add predictions for an image.

        Args:
            predictions: Dict with boxes (xyxy), scores, classes.
            image_id: Image ID matching COCO API.
        """
        boxes = predictions["boxes"]
        scores = predictions["scores"]
        classes = predictions["classes"]

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(classes, torch.Tensor):
            classes = classes.cpu().numpy()

        boxes = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes
        scores = np.array(scores) if not isinstance(scores, np.ndarray) else scores
        classes = np.array(classes) if not isinstance(classes, np.ndarray) else classes

        for box, score, label in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            self.results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(w), float(h)],  # COCO xywh
                    "score": float(score),
                }
            )

        self._img_ids.add(image_id)

    def compute(self, save_json: Optional[str] = None) -> Dict[str, float]:
        """
        Run COCO evaluation and return 12 standard metrics.

        Args:
            save_json: Optional path to save predictions in COCO JSON format.
        """
        if len(self.results) == 0:
            print("Warning: No predictions to evaluate")
            return self._empty_metrics()

        if save_json:
            with open(save_json, "w") as f:
                json.dump(self.results, f, indent=2)
            print(f"Saved predictions to {save_json}")

        try:
            from pycocotools.coco import COCO  # noqa: F401
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            raise ImportError(
                "pycocotools not installed. Install with: pip install pycocotools"
            )

        coco_dt = self.coco_gt.loadRes(self.results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, self.iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # stats layout: [mAP, mAP50, mAP75, AP_s, AP_m, AP_l,
        #                AR1, AR10, AR100, AR_s, AR_m, AR_l]
        return {
            "mAP": float(coco_eval.stats[0]),
            "mAP50": float(coco_eval.stats[1]),
            "mAP75": float(coco_eval.stats[2]),
            "mAP_small": float(coco_eval.stats[3]),
            "mAP_medium": float(coco_eval.stats[4]),
            "mAP_large": float(coco_eval.stats[5]),
            "AR1": float(coco_eval.stats[6]),
            "AR10": float(coco_eval.stats[7]),
            "AR100": float(coco_eval.stats[8]),
            "AR_small": float(coco_eval.stats[9]),
            "AR_medium": float(coco_eval.stats[10]),
            "AR_large": float(coco_eval.stats[11]),
        }

    def _empty_metrics(self) -> Dict[str, float]:
        """Return all-zero metrics dict."""
        return {
            "mAP": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
            "AR1": 0.0,
            "AR10": 0.0,
            "AR100": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0,
        }

    def reset(self):
        """Clear all accumulated results."""
        self.results = []
        self._img_ids = set()
