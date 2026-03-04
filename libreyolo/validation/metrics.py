"""Detection metrics for LibreYOLO validation."""

from typing import Dict, List, Optional, Tuple

import numpy as np


class DetMetrics:
    """
    Detection metrics calculator.

    Computes precision, recall, and Average Precision (AP) curves
    following COCO-style 101-point interpolation.
    """

    def __init__(
        self,
        nc: int = 80,
        conf: float = 0.25,
        iou_thresholds: Optional[Tuple[float, ...]] = None,
    ) -> None:
        self.nc = nc
        self.conf = conf
        self.iou_thresholds = iou_thresholds or (
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
        )
        self.niou = len(self.iou_thresholds)

        self.stats: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        self.ap: Optional[np.ndarray] = None  # (nc, niou)
        self.ap_class: Optional[np.ndarray] = None  # classes with GT samples
        self.precision: Optional[np.ndarray] = None  # (nc,) at conf threshold
        self.recall: Optional[np.ndarray] = None  # (nc,) at conf threshold

    def update(
        self,
        correct: np.ndarray,
        conf: np.ndarray,
        pred_cls: np.ndarray,
        target_cls: np.ndarray,
    ) -> None:
        """
        Accumulate batch results.

        Args:
            correct: (N_pred, N_iou_thresholds) boolean array indicating TP.
            conf: (N_pred,) confidence scores.
            pred_cls: (N_pred,) predicted class indices.
            target_cls: (N_gt,) ground truth class indices.
        """
        self.stats.append((correct, conf, pred_cls, target_cls))

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics from accumulated stats.

        Returns:
            Dictionary with precision, recall, mAP50, mAP50-95.
        """
        if not self.stats:
            return {
                "metrics/precision": 0.0,
                "metrics/recall": 0.0,
                "metrics/mAP50": 0.0,
                "metrics/mAP50-95": 0.0,
            }

        correct = np.concatenate([s[0] for s in self.stats], axis=0)
        conf = np.concatenate([s[1] for s in self.stats], axis=0)
        pred_cls = np.concatenate([s[2] for s in self.stats], axis=0)
        target_cls = np.concatenate([s[3] for s in self.stats], axis=0)

        ap, p, r, unique_classes = self._compute_ap_per_class(
            correct, conf, pred_cls, target_cls
        )

        self.ap = ap
        self.ap_class = unique_classes
        self.precision = p
        self.recall = r

        map50 = ap[:, 0].mean() if len(ap) > 0 else 0.0
        map50_95 = ap.mean() if len(ap) > 0 else 0.0
        mp = p.mean() if len(p) > 0 else 0.0
        mr = r.mean() if len(r) > 0 else 0.0

        return {
            "metrics/precision": float(mp),
            "metrics/recall": float(mr),
            "metrics/mAP50": float(map50),
            "metrics/mAP50-95": float(map50_95),
        }

    def _compute_ap_per_class(
        self,
        correct: np.ndarray,
        conf: np.ndarray,
        pred_cls: np.ndarray,
        target_cls: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute AP for each class.

        Args:
            correct: (N, niou) boolean array.
            conf: (N,) confidence scores.
            pred_cls: (N,) predicted classes.
            target_cls: (M,) target classes.

        Returns:
            Tuple of (ap, precision, recall, unique_classes).
        """
        # Sort by confidence (descending)
        i = np.argsort(-conf)
        correct, conf, pred_cls = correct[i], conf[i], pred_cls[i]

        unique_classes = np.unique(target_cls)
        nc = len(unique_classes)

        ap = np.zeros((nc, self.niou))
        precision_at_conf = np.zeros(nc)
        recall_at_conf = np.zeros(nc)

        for ci, c in enumerate(unique_classes):
            pred_mask = pred_cls == c
            n_gt = (target_cls == c).sum()
            n_pred = pred_mask.sum()

            if n_pred == 0 or n_gt == 0:
                continue

            fpc = (1 - correct[pred_mask]).cumsum(axis=0)
            tpc = correct[pred_mask].cumsum(axis=0)

            recall = tpc / n_gt  # TP / (TP + FN)
            precision = tpc / (tpc + fpc)  # TP / (TP + FP)

            for iou_idx in range(self.niou):
                ap[ci, iou_idx] = self._compute_ap(
                    recall[:, iou_idx], precision[:, iou_idx]
                )

            # Precision/recall at conf threshold (using IoU=0.50)
            conf_mask = conf[pred_mask] >= self.conf
            if conf_mask.any():
                idx = conf_mask.sum() - 1
                precision_at_conf[ci] = precision[idx, 0]
                recall_at_conf[ci] = recall[idx, 0]

        return ap, precision_at_conf, recall_at_conf, unique_classes

    @staticmethod
    def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
        """Compute AP using COCO-style 101-point interpolation."""
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Make precision monotonically decreasing (right to left)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        x = np.linspace(0, 1, 101)
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        ap = _trapz(np.interp(x, mrec, mpre), x)

        return float(ap)

    def ap_per_class_values(self) -> Optional[np.ndarray]:
        """Get per-class AP values: (nc, niou) array, or None if not computed."""
        return self.ap

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.stats = []
        self.ap = None
        self.ap_class = None
        self.precision = None
        self.recall = None
