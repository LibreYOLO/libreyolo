"""COCO evaluator for LibreYOLO."""

import logging
from typing import Dict, Mapping, Optional
import json

import numpy as np
import torch

logger = logging.getLogger(__name__)


class COCOEvaluator:
    """
    COCO evaluation wrapper.

    Computes standard COCO metrics: AP (mAP@[0.5:0.95]), AP50, AP75,
    AP/AR by object size, and AR at different maxDets.
    """

    def __init__(
        self,
        coco_gt,
        iou_type: str = "bbox",
        label_to_category_id: Optional[Mapping[int, int]] = None,
    ):
        self.coco_gt = coco_gt
        self.iou_type = iou_type
        self.label_to_category_id = (
            {int(k): int(v) for k, v in label_to_category_id.items()}
            if label_to_category_id is not None
            else None
        )
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
        masks = predictions.get("masks")

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(classes, torch.Tensor):
            classes = classes.cpu().numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        boxes = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes
        scores = np.array(scores) if not isinstance(scores, np.ndarray) else scores
        classes = np.array(classes) if not isinstance(classes, np.ndarray) else classes
        masks = np.array(masks) if masks is not None and not isinstance(masks, np.ndarray) else masks

        if self.iou_type == "segm" and masks is None:
            self._img_ids.add(image_id)
            return

        for idx, (box, score, label) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            label = int(label)
            category_id = (
                self.label_to_category_id.get(label, label)
                if self.label_to_category_id is not None
                else label
            )

            result = {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [float(x1), float(y1), float(w), float(h)],  # COCO xywh
                "score": float(score),
            }
            if self.iou_type == "segm":
                mask = masks[idx]
                result["segmentation"] = self._encode_mask(mask)
                result["area"] = float((mask > 0).sum())
            self.results.append(result)

        self._img_ids.add(image_id)

    @staticmethod
    def _encode_mask(mask: np.ndarray) -> dict:
        """Encode a binary mask to JSON-safe COCO RLE."""
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            raise ImportError(
                "pycocotools not installed. Install with: pip install pycocotools"
            )

        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask for COCO RLE, got shape {mask.shape}")
        mask = (mask > 0).astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mask))
        counts = rle.get("counts")
        if isinstance(counts, bytes):
            rle["counts"] = counts.decode("ascii")
        rle["size"] = [int(mask.shape[0]), int(mask.shape[1])]
        return rle

    def compute(self, save_json: Optional[str] = None) -> Dict[str, float]:
        """
        Run COCO evaluation and return 12 standard metrics.

        Args:
            save_json: Optional path to save predictions in COCO JSON format.
        """
        if len(self.results) == 0:
            logger.warning("No predictions to evaluate")
            return self._empty_metrics()

        if save_json:
            with open(save_json, "w") as f:
                json.dump(self.results, f, indent=2)
            logger.info("Saved predictions to %s", save_json)

        try:
            from pycocotools.coco import COCO  # noqa: F401
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            raise ImportError(
                "pycocotools not installed. Install with: pip install pycocotools"
            )

        coco_dt = self.coco_gt.loadRes(self.results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, self.iou_type)
        if self._img_ids:
            coco_eval.params.imgIds = sorted(self._img_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # stats layout: [mAP, mAP50, mAP75, AP_s, AP_m, AP_l,
        #                AR1, AR10, AR100, AR_s, AR_m, AR_l]
        precision = self._mean_valid(coco_eval.eval["precision"][:, :, :, 0, -1])
        recall = self._mean_valid(coco_eval.eval["recall"][:, :, 0, -1])
        return {
            "precision": precision,
            "recall": recall,
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
            "precision": 0.0,
            "recall": 0.0,
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

    @staticmethod
    def _mean_valid(values: np.ndarray) -> float:
        """Mean over COCOeval arrays while ignoring absent -1 entries."""
        valid = values[values > -1]
        if valid.size == 0:
            return 0.0
        return float(valid.mean())
