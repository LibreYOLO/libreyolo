"""Detection metrics for LibreYOLO validation."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


# =============================================================================
# Confusion matrix (object detection)
# =============================================================================


class ConfusionMatrix:
    """Per-class confusion matrix for object detection.

    Tracks an `(nc + 1) x (nc + 1)` matrix where the last row/column is
    "background":

        matrix[true_cls, pred_cls]    -> count of preds matched to a GT box
                                         (above conf, IoU >= iou_thresh)
        matrix[bg, pred_cls]          -> false positives (no matching GT)
        matrix[true_cls, bg]          -> false negatives (no matching pred)

    Matching uses greedy IoU assignment (each GT can match at most one pred,
    sorted by descending pred conf) — same convention as YOLOv5/v8.

    Args:
        nc: number of (foreground) classes
        conf: predictions below this confidence are dropped before matching
        iou_thresh: IoU at which a pred is considered to match a GT
    """

    def __init__(self, nc: int, conf: float = 0.25, iou_thresh: float = 0.45) -> None:
        if nc < 1:
            raise ValueError(f"nc must be >= 1, got {nc}")
        self.nc = nc
        self.conf = conf
        self.iou_thresh = iou_thresh
        self.matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)

    def reset(self) -> None:
        self.matrix.fill(0)

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """IoU matrix for two sets of xyxy boxes. Shape (len(a), len(b))."""
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)), dtype=np.float64)
        inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
        inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
        inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
        inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / np.maximum(union, 1e-9)

    def update(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_classes: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
    ) -> None:
        """Update confusion matrix with one image's predictions and ground truth.

        All inputs are numpy arrays. Boxes are xyxy in the same coordinate frame.
        Predictions below `self.conf` are dropped before matching.
        """
        # Filter low-conf predictions
        if len(pred_scores) > 0:
            keep = pred_scores >= self.conf
            pred_boxes = pred_boxes[keep]
            pred_classes = pred_classes[keep]
            pred_scores = pred_scores[keep]

        bg = self.nc

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            return
        if len(gt_boxes) == 0:
            for c in pred_classes.astype(np.int64):
                if 0 <= c < self.nc:
                    self.matrix[bg, c] += 1
            return
        if len(pred_boxes) == 0:
            for c in gt_classes.astype(np.int64):
                if 0 <= c < self.nc:
                    self.matrix[c, bg] += 1
            return

        ious = self._box_iou(pred_boxes, gt_boxes)
        # Greedy assignment by descending pred confidence
        order = np.argsort(-pred_scores)
        gt_taken = np.zeros(len(gt_boxes), dtype=bool)
        pred_matched = np.full(len(pred_boxes), -1, dtype=np.int64)

        for i in order:
            cand = ious[i].copy()
            cand[gt_taken] = -1
            j = int(np.argmax(cand))
            if cand[j] >= self.iou_thresh:
                pred_matched[i] = j
                gt_taken[j] = True

        # Matched preds: increment matrix[true, pred]
        for i, j in enumerate(pred_matched):
            pc = int(pred_classes[i])
            if not (0 <= pc < self.nc):
                continue
            if j >= 0:
                tc = int(gt_classes[j])
                if 0 <= tc < self.nc:
                    self.matrix[tc, pc] += 1
            else:
                # Unmatched pred = false positive against background
                self.matrix[bg, pc] += 1

        # Unmatched GTs = false negatives (model missed)
        for j in range(len(gt_boxes)):
            if gt_taken[j]:
                continue
            tc = int(gt_classes[j])
            if 0 <= tc < self.nc:
                self.matrix[tc, bg] += 1

    def normalize(self, axis: str = "row") -> np.ndarray:
        """Return a row-normalized (or column-normalized) copy of the matrix.

        `axis="row"` divides each row by its sum (per-true-class confusion shape);
        `axis="col"` divides each column (per-prediction-class composition).
        """
        m = self.matrix.astype(np.float64)
        if axis == "row":
            denom = m.sum(axis=1, keepdims=True)
        elif axis == "col":
            denom = m.sum(axis=0, keepdims=True)
        elif axis == "none":
            return m
        else:
            raise ValueError(f"axis must be 'row', 'col', or 'none'; got {axis!r}")
        denom = np.where(denom > 0, denom, 1)
        return m / denom

    def plot(
        self,
        save_path: str | Path,
        names: Optional[Sequence[str]] = None,
        normalize: str = "row",
        dpi: int = 150,
    ) -> Path:
        """Save a confusion-matrix heatmap to `save_path` (PNG). Returns the path.

        Set `normalize="row"` for the standard recall-style view, `"col"` for
        precision-style, or `"none"` for raw counts.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(names) if names is not None else [str(i) for i in range(self.nc)]
        if len(names) != self.nc:
            raise ValueError(f"names has {len(names)} entries; expected {self.nc}")
        labels = names + ["background"]
        m = self.normalize(normalize)

        fig_w = max(8, 0.45 * (self.nc + 1) + 4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85), dpi=dpi)
        cmap = plt.cm.Blues if normalize != "none" else plt.cm.viridis
        im = ax.imshow(m, cmap=cmap, aspect="auto", vmin=0,
                       vmax=1.0 if normalize != "none" else None)
        ax.set_xticks(range(self.nc + 1))
        ax.set_yticks(range(self.nc + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        title = "Confusion matrix"
        if normalize == "row":
            title += " (row-normalized = recall view)"
        elif normalize == "col":
            title += " (column-normalized = precision view)"
        else:
            title += " (raw counts)"
        ax.set_title(title)

        # Annotate cells with their value
        thresh = m.max() / 2.0 if m.max() > 0 else 0.5
        for i in range(self.nc + 1):
            for j in range(self.nc + 1):
                val = m[i, j]
                if normalize == "none":
                    txt = f"{int(val)}"
                else:
                    txt = f"{val:.2f}" if val > 0.005 else ""
                if txt:
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                            color="white" if val > thresh else "black")

        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path


def plot_per_class_ap(
    ap_per_class: np.ndarray,
    classes: np.ndarray,
    nc: int,
    save_path: str | Path,
    names: Optional[Sequence[str]] = None,
    iou_index: Tuple[int, ...] = (0, -1),  # (mAP50, mAP50-95)
    dpi: int = 150,
) -> Path:
    """Save a horizontal bar chart of per-class AP to `save_path`.

    Args:
        ap_per_class: (n_classes_with_gt, n_iou_thresholds) AP values from DetMetrics.
        classes: (n_classes_with_gt,) class indices that had GT samples.
        nc: total number of foreground classes.
        save_path: PNG output path.
        names: optional class names. Defaults to numeric IDs.
        iou_index: which IoU columns to plot. Default plots AP50 (col 0) and
                   mAP across all thresholds (col -1 = AP at IoU=0.95).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(names) if names is not None else [str(i) for i in range(nc)]
    if len(names) != nc:
        raise ValueError(f"names has {len(names)} entries; expected {nc}")

    full_ap = np.zeros((nc, ap_per_class.shape[1])) if ap_per_class.ndim == 2 else np.zeros((nc, 1))
    if ap_per_class.size > 0 and ap_per_class.ndim == 2:
        for ci, c in enumerate(classes.astype(np.int64)):
            if 0 <= c < nc:
                full_ap[c] = ap_per_class[ci]

    ap50 = full_ap[:, iou_index[0]]
    ap_mean = full_ap.mean(axis=1)

    order = np.argsort(ap_mean)
    fig_h = max(4, 0.35 * nc + 2)
    fig, ax = plt.subplots(figsize=(8, fig_h), dpi=dpi)
    y = np.arange(nc)
    ax.barh(y - 0.18, ap50[order], height=0.36, label="AP@0.5", color="#3b82f6")
    ax.barh(y + 0.18, ap_mean[order], height=0.36, label="mAP@0.5:0.95", color="#10b981")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel("AP")
    ax.set_title(f"Per-class AP ({nc} classes)")
    ax.set_xlim(0, max(1.0, max(ap50.max(), ap_mean.max()) * 1.05))
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path
