"""
Detection metrics for LibreYOLO validation.

Provides classes for computing mAP, precision, recall, and confusion matrix.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class DetMetrics:
    """
    Detection metrics calculator.

    Computes precision, recall, and Average Precision (AP) curves
    for object detection evaluation following COCO-style metrics.

    Attributes:
        nc: Number of classes.
        conf: Confidence threshold for precision/recall reporting.
        iou_thresholds: IoU thresholds for AP calculation.
    """

    def __init__(
        self,
        nc: int = 80,
        conf: float = 0.25,
        iou_thresholds: Optional[Tuple[float, ...]] = None,
    ) -> None:
        """
        Initialize detection metrics.

        Args:
            nc: Number of classes.
            conf: Confidence threshold for precision/recall.
            iou_thresholds: IoU thresholds for AP calculation.
        """
        self.nc = nc
        self.conf = conf
        self.iou_thresholds = iou_thresholds or (
            0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
        )
        self.niou = len(self.iou_thresholds)

        # Accumulated statistics: list of (correct, conf, pred_cls, target_cls)
        self.stats: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        # Computed metrics
        self.ap: Optional[np.ndarray] = None  # (nc, niou)
        self.ap_class: Optional[np.ndarray] = None  # Classes with GT samples
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
        Update metrics with batch results.

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
            Dictionary with:
                - metrics/precision: Mean precision at conf threshold
                - metrics/recall: Mean recall at conf threshold
                - metrics/mAP50: Mean AP at IoU=0.50
                - metrics/mAP50-95: Mean AP across IoU 0.50-0.95
        """
        if not self.stats:
            return {
                "metrics/precision": 0.0,
                "metrics/recall": 0.0,
                "metrics/mAP50": 0.0,
                "metrics/mAP50-95": 0.0,
            }

        # Concatenate all stats
        correct = np.concatenate([s[0] for s in self.stats], axis=0)
        conf = np.concatenate([s[1] for s in self.stats], axis=0)
        pred_cls = np.concatenate([s[2] for s in self.stats], axis=0)
        target_cls = np.concatenate([s[3] for s in self.stats], axis=0)

        # Compute AP per class
        ap, p, r, unique_classes = self._compute_ap_per_class(
            correct, conf, pred_cls, target_cls
        )

        self.ap = ap
        self.ap_class = unique_classes
        self.precision = p
        self.recall = r

        # Compute mean metrics
        # mAP50 is AP at first IoU threshold (0.50)
        map50 = ap[:, 0].mean() if len(ap) > 0 else 0.0
        # mAP50-95 is mean over all IoU thresholds
        map50_95 = ap.mean() if len(ap) > 0 else 0.0
        # Mean precision and recall at conf threshold
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

        # Find unique classes with ground truth
        unique_classes = np.unique(target_cls)
        nc = len(unique_classes)

        # Initialize outputs
        ap = np.zeros((nc, self.niou))
        precision_at_conf = np.zeros(nc)
        recall_at_conf = np.zeros(nc)

        for ci, c in enumerate(unique_classes):
            # Predictions and targets for this class
            pred_mask = pred_cls == c
            n_gt = (target_cls == c).sum()
            n_pred = pred_mask.sum()

            if n_pred == 0 or n_gt == 0:
                continue

            # Accumulate FPs and TPs for this class
            fpc = (1 - correct[pred_mask]).cumsum(axis=0)
            tpc = correct[pred_mask].cumsum(axis=0)

            # Recall = TP / (TP + FN) = TP / n_gt
            recall = tpc / n_gt

            # Precision = TP / (TP + FP)
            precision = tpc / (tpc + fpc)

            # AP for each IoU threshold
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
        """
        Compute Average Precision using 101-point interpolation (COCO-style).

        Args:
            recall: Recall values (cumulative, ascending).
            precision: Precision values (cumulative).

        Returns:
            Average Precision value.
        """
        # Prepend/append sentinel values
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Make precision monotonically decreasing (from right to left)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # 101-point interpolation
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)

        return float(ap)

    def ap_per_class_values(self) -> Optional[np.ndarray]:
        """
        Get per-class AP values.

        Returns:
            Array of shape (nc, niou) with AP per class and IoU threshold,
            or None if not computed yet.
        """
        return self.ap

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.stats = []
        self.ap = None
        self.ap_class = None
        self.precision = None
        self.recall = None


class ConfusionMatrix:
    """
    Confusion matrix for object detection evaluation.

    Tracks TP, FP, and FN across classes for visualization and analysis.
    The matrix has shape (nc+1, nc+1) where the last row/column represents
    background (no detection / false positive).
    """

    def __init__(
        self,
        nc: int,
        conf: float = 0.25,
        iou_thres: float = 0.5,
    ) -> None:
        """
        Initialize confusion matrix.

        Args:
            nc: Number of classes.
            conf: Confidence threshold.
            iou_thres: IoU threshold for matching.
        """
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres
        # Matrix shape: (nc+1, nc+1) for background class
        self.matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)

    def update(
        self,
        pred_boxes: torch.Tensor,
        pred_classes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_classes: torch.Tensor,
        iou_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update confusion matrix with predictions and ground truth.

        Args:
            pred_boxes: (N, 4) predicted boxes in xyxy format.
            pred_classes: (N,) predicted class indices.
            pred_scores: (N,) confidence scores.
            gt_boxes: (M, 4) ground truth boxes in xyxy format.
            gt_classes: (M,) ground truth class indices.
            iou_matrix: Optional precomputed IoU matrix (N, M).
        """
        # Filter by confidence
        conf_mask = pred_scores >= self.conf
        pred_boxes = pred_boxes[conf_mask]
        pred_classes = pred_classes[conf_mask]
        pred_scores = pred_scores[conf_mask]

        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)

        if n_gt == 0:
            # All predictions are false positives
            for pc in pred_classes:
                self.matrix[self.nc, int(pc)] += 1  # FP: background predicted as class
            return

        if n_pred == 0:
            # All ground truths are missed (false negatives)
            for gc in gt_classes:
                self.matrix[int(gc), self.nc] += 1  # FN: class missed
            return

        # Compute IoU if not provided
        if iou_matrix is None:
            from libreyolo.common.postprocessing import box_iou
            iou_matrix = box_iou(pred_boxes, gt_boxes)

        # Match predictions to ground truth
        gt_matched = torch.zeros(n_gt, dtype=torch.bool, device=gt_boxes.device)

        # Sort predictions by confidence
        sorted_idx = torch.argsort(pred_scores, descending=True)

        for pred_idx in sorted_idx:
            pred_cls = int(pred_classes[pred_idx])

            # Find best matching GT
            ious = iou_matrix[pred_idx]
            ious[gt_matched] = 0  # Mask already matched

            if ious.max() >= self.iou_thres:
                gt_idx = ious.argmax()
                gt_cls = int(gt_classes[gt_idx])
                gt_matched[gt_idx] = True

                # Update matrix: row=GT, col=pred
                self.matrix[gt_cls, pred_cls] += 1
            else:
                # False positive: no matching GT
                self.matrix[self.nc, pred_cls] += 1

        # Count unmatched GT as false negatives
        for gt_idx in range(n_gt):
            if not gt_matched[gt_idx]:
                gt_cls = int(gt_classes[gt_idx])
                self.matrix[gt_cls, self.nc] += 1

    def matrix_normalized(self) -> np.ndarray:
        """
        Get row-normalized confusion matrix.

        Returns:
            Normalized matrix where each row sums to 1.
        """
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        return self.matrix / row_sums

    def tp_fp_fn(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get true positives, false positives, and false negatives per class.

        Returns:
            Tuple of (tp, fp, fn) arrays of shape (nc,).
        """
        # TP: diagonal elements (excluding background)
        tp = np.diag(self.matrix)[:self.nc]

        # FP: sum of column (predictions for class) minus TP
        fp = self.matrix[:, :self.nc].sum(axis=0) - tp

        # FN: sum of row (GT for class) minus TP
        fn = self.matrix[:self.nc, :].sum(axis=1) - tp

        return tp, fp, fn

    def plot(
        self,
        save_path: Optional[Union[str, Path]] = None,
        names: Optional[List[str]] = None,
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10),
    ) -> Optional[np.ndarray]:
        """
        Plot confusion matrix.

        Args:
            save_path: Path to save the plot.
            names: Class names.
            normalize: Whether to normalize values.
            figsize: Figure size.

        Returns:
            Plot as numpy array if not saved, else None.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting. Install with: pip install matplotlib")
            return None

        matrix = self.matrix_normalized() if normalize else self.matrix.astype(float)

        # Create labels
        if names is None:
            names = [str(i) for i in range(self.nc)]
        names = list(names) + ["background"]

        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(matrix, cmap="Blues")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Proportion" if normalize else "Count", rotation=-90, va="bottom")

        # Set ticks
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)

        # Add text annotations
        thresh = matrix.max() / 2.0
        for i in range(len(names)):
            for j in range(len(names)):
                value = matrix[i, j]
                text = f"{value:.2f}" if normalize else f"{int(value)}"
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color="white" if value > thresh else "black",
                    fontsize=8,
                )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            return None
        else:
            # Convert to numpy array
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data

    def reset(self) -> None:
        """Reset confusion matrix."""
        self.matrix = np.zeros((self.nc + 1, self.nc + 1), dtype=np.int64)
