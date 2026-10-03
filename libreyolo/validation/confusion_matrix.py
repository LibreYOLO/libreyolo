"""Confusion matrix for detection and classification validation.

``val()`` returns it as ``results.confusion_matrix``. The public layout
follows the ecosystem: ``matrix[predicted, true]``, with an extra
``background`` row and column for detection (a missed ground truth is
predicted as background; an unmatched prediction has background as its truth).
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BACKGROUND = "background"
#: Tasks the matrix is defined for.
CONFUSION_MATRIX_TASKS = ("detect", "classify")
#: Confidence at or above which a detection is counted, unless the run's
#: ``conf`` is higher (the ``visualize`` and ``box.image_metrics`` rule).
DEFAULT_CONF_THRES = 0.25
#: IoU at or above which a detection can be paired with a ground-truth box.
DEFAULT_IOU_THRES = 0.5
#: Widest classification head the dense matrix is built for (0.8 GB of int64
#: at this width; ImageNet-21k would need 3.8 GB). Wider heads are read
#: through ``nonzero()``, ``tp_fp()`` and ``class_accuracy()``.
MAX_DENSE_CLASSES = 10_000


def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Pairwise xyxy IoU, shape ``(len(boxes1), len(boxes2))``."""
    a1 = (boxes1[:, 2] - boxes1[:, 0]).clip(0) * (boxes1[:, 3] - boxes1[:, 1]).clip(0)
    a2 = (boxes2[:, 2] - boxes2[:, 0]).clip(0) * (boxes2[:, 3] - boxes2[:, 1]).clip(0)
    ix1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    iy1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    ix2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    iy2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    union = a1[:, None] + a2[None, :] - inter
    return inter / np.maximum(union, 1e-7)


def _as_index_array(values: Any) -> np.ndarray:
    """Flatten tensors, arrays or (lists of) either into one int64 vector."""
    if isinstance(values, (list, tuple)):
        if not values:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([_as_index_array(v) for v in values])
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1).astype(np.int64)


class ConfusionMatrix:
    """Counts of predicted class against true class.

    Detection (``task="detect"``): predictions at or above ``conf_thres`` are
    paired one to one with ground-truth boxes, highest IoU first, whatever
    their classes, when the IoU reaches ``iou_thres``. A pair counts at
    ``[predicted class, true class]``, so a right box with the wrong class
    lands off the diagonal. An unpaired ground truth counts in the
    ``background`` row (missed), an unpaired prediction in the ``background``
    column (false alarm). The matrix is ``(nc + 1, nc + 1)``.

    Classification (``task="classify"``): one count per image at
    ``[top-1 class, label]``. The matrix is ``(nc, nc)``. Only the pairs that
    occur are stored, so accumulating stays cheap for very wide heads
    (ImageNet-21k); ``matrix`` builds the dense array when it is read, up to
    ``MAX_DENSE_CLASSES`` classes. ``nonzero()``, ``tp_fp()`` and
    ``class_accuracy()`` never build it.

    Attributes:
        matrix: Integer counts, indexed ``[predicted, true]``.
        nc: Number of classes.
        names: Class names by index.
        task: ``"detect"`` or ``"classify"``.
        conf_thres: Detection confidence threshold.
        iou_thres: Detection pairing IoU threshold.
    """

    def __init__(
        self,
        nc: int,
        iou_thres: float = DEFAULT_IOU_THRES,
        conf_thres: float = DEFAULT_CONF_THRES,
        *,
        names: Optional[Sequence[str] | Mapping[int, str]] = None,
        task: str = "detect",
    ) -> None:
        if task not in CONFUSION_MATRIX_TASKS:
            raise ValueError(
                f"ConfusionMatrix task must be one of {CONFUSION_MATRIX_TASKS}, "
                f"got {task!r}"
            )
        self.nc = int(nc)
        if self.nc < 1:
            raise ValueError(f"ConfusionMatrix needs at least one class, got {nc}")
        self.task = task
        self.iou_thres = float(iou_thres)
        self.conf_thres = float(conf_thres)
        self.names: Dict[int, str] = self._normalize_names(names, self.nc)
        # Detection keeps the dense (nc + 1, nc + 1) counts; classification
        # keeps {predicted * nc + true: count} for the pairs that occur.
        self._counts: Optional[np.ndarray] = (
            np.zeros((self.nc + 1, self.nc + 1), dtype=np.int64)
            if task == "detect"
            else None
        )
        self._pairs: Dict[int, int] = {}

    @staticmethod
    def _normalize_names(names, nc: int) -> Dict[int, str]:
        if isinstance(names, Mapping):
            lookup = {int(k): str(v) for k, v in names.items()}
        elif names is not None:
            lookup = {i: str(v) for i, v in enumerate(names)}
        else:
            lookup = {}
        return {i: lookup.get(i, str(i)) for i in range(nc)}

    @property
    def matrix(self) -> np.ndarray:
        """Integer counts indexed ``[predicted, true]``."""
        if self._counts is not None:
            return self._counts
        if self.nc > MAX_DENSE_CLASSES:
            raise ValueError(
                f"The dense confusion matrix of a {self.nc}-class head is too "
                f"large to build (limit {MAX_DENSE_CLASSES} classes). Use "
                "nonzero(), tp_fp() or class_accuracy(), which read the "
                "stored counts directly."
            )
        dense = np.zeros((self.nc, self.nc), dtype=np.int64)
        if self._pairs:
            keys = np.fromiter(self._pairs.keys(), dtype=np.int64)
            dense[keys // self.nc, keys % self.nc] = np.fromiter(
                self._pairs.values(), dtype=np.int64
            )
        return dense

    def nonzero(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The cells that hold a count, as aligned ``(predicted, true, count)``.

        Sorted by predicted then true index. For detection the background
        index is ``nc``. Does not build the dense classification matrix.
        """
        if self._counts is not None:
            predicted, true = np.nonzero(self._counts)
            return predicted, true, self._counts[predicted, true]
        keys = np.sort(np.fromiter(self._pairs.keys(), dtype=np.int64))
        counts = np.asarray([self._pairs[int(k)] for k in keys], dtype=np.int64)
        return keys // self.nc, keys % self.nc, counts

    def _marginals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Diagonal, predicted-row totals and true-column totals.

        Computed without building the dense classification matrix.
        """
        if self._counts is not None:
            return (
                np.diag(self._counts).copy(),
                self._counts.sum(axis=1),
                self._counts.sum(axis=0),
            )
        diagonal = np.zeros(self.nc, dtype=np.int64)
        rows = np.zeros(self.nc, dtype=np.int64)
        columns = np.zeros(self.nc, dtype=np.int64)
        for key, count in self._pairs.items():
            pred, true = divmod(key, self.nc)
            rows[pred] += count
            columns[true] += count
            if pred == true:
                diagonal[pred] += count
        return diagonal, rows, columns

    # ------------------------------------------------------------------ #
    # Accumulation
    # ------------------------------------------------------------------ #

    def process_image(
        self,
        pred_boxes: np.ndarray,
        pred_classes: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
    ) -> None:
        """Add one image's detections (xyxy pixels) to a detection matrix."""
        if self.task != "detect":
            raise ValueError("process_image applies to a detection confusion matrix")
        pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
        pred_classes = np.asarray(pred_classes).reshape(-1).astype(np.int64)
        pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
        gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
        gt_classes = np.asarray(gt_classes).reshape(-1).astype(np.int64)

        # Classes outside [0, nc) cannot be placed in the matrix.
        keep = (
            (pred_scores >= self.conf_thres)
            & (pred_classes >= 0)
            & (pred_classes < self.nc)
        )
        pred_boxes, pred_classes = pred_boxes[keep], pred_classes[keep]
        gt_keep = (gt_classes >= 0) & (gt_classes < self.nc)
        gt_boxes, gt_classes = gt_boxes[gt_keep], gt_classes[gt_keep]

        background = self.nc
        gt_paired = np.zeros(len(gt_boxes), dtype=bool)
        pred_paired = np.zeros(len(pred_boxes), dtype=bool)
        if len(gt_boxes) and len(pred_boxes):
            iou = _box_iou(gt_boxes, pred_boxes)  # (num_gt, num_pred)
            gt_idxs, pred_idxs = np.where(iou >= self.iou_thres)
            order = np.argsort(-iou[gt_idxs, pred_idxs], kind="stable")
            for gi, pi in zip(gt_idxs[order], pred_idxs[order]):
                if gt_paired[gi] or pred_paired[pi]:
                    continue
                gt_paired[gi] = True
                pred_paired[pi] = True
                self._counts[pred_classes[pi], gt_classes[gi]] += 1

        np.add.at(self._counts, (background, gt_classes[~gt_paired]), 1)
        np.add.at(self._counts, (pred_classes[~pred_paired], background), 1)

    def process_cls_preds(self, preds: Any, targets: Any) -> None:
        """Add top-1 class indices against label indices to a classify matrix.

        ``preds`` and ``targets`` are aligned vectors of class indices, or
        lists of such vectors (one per batch).
        """
        if self.task != "classify":
            raise ValueError(
                "process_cls_preds applies to a classification confusion matrix"
            )
        pred_idx = _as_index_array(preds)
        target_idx = _as_index_array(targets)
        if pred_idx.shape != target_idx.shape:
            raise ValueError(
                "preds and targets must hold the same number of class indices, "
                f"got {pred_idx.size} and {target_idx.size}"
            )
        valid = (
            (pred_idx >= 0)
            & (pred_idx < self.nc)
            & (target_idx >= 0)
            & (target_idx < self.nc)
        )
        keys, counts = np.unique(
            pred_idx[valid] * self.nc + target_idx[valid], return_counts=True
        )
        for key, count in zip(keys.tolist(), counts.tolist()):
            self._pairs[key] = self._pairs.get(key, 0) + count

    # ------------------------------------------------------------------ #
    # Derived values
    # ------------------------------------------------------------------ #

    @property
    def labels(self) -> List[str]:
        """Row/column labels: the class names, plus ``background`` for detect."""
        labels = [self.names[i] for i in range(self.nc)]
        if len(set(labels)) != len(labels) or BACKGROUND in labels:
            # Labels key the summary rows; keep them unique.
            labels = [f"{i}: {name}" for i, name in enumerate(labels)]
        if self.task == "detect":
            labels.append(BACKGROUND)
        return labels

    def normalized(self) -> np.ndarray:
        """The matrix with every true-class column summing to 1 (0 if empty)."""
        matrix = self.matrix
        totals = matrix.sum(axis=0, keepdims=True).astype(np.float64)
        return np.divide(
            matrix,
            totals,
            out=np.zeros(matrix.shape, dtype=np.float64),
            where=totals > 0,
        )

    def tp_fp(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-class true positives and false positives (background excluded).

        A true positive is a diagonal count. A false positive is any other
        count in the class's predicted row: a wrong class, or for detection an
        unmatched prediction.
        """
        diagonal, rows, _ = self._marginals()
        tp = diagonal[: self.nc]
        return tp, rows[: self.nc] - tp

    def class_accuracy(self) -> Dict[str, float]:
        """Per class, how often its true instances get that class.

        Diagonal count over the true-class column. For detection the column
        counts only ground-truth boxes the model localized (paired with a
        prediction of any class), so this is the top-1 class accuracy among
        found objects and a missed box does not lower it. Classes with no
        such instance are left out. Keyed like :attr:`labels`.
        """
        diagonal, _, columns = self._marginals()
        totals = columns[: self.nc].copy()
        if self._counts is not None:
            # Leave out the misses: the background row of each true column.
            totals -= self._counts[self.nc, : self.nc]
        labels = self.labels
        return {
            labels[i]: float(diagonal[i] / totals[i])
            for i in range(self.nc)
            if totals[i] > 0
        }

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def summary(self, normalize: bool = False, decimals: int = 5) -> List[Dict[str, Any]]:
        """One row per predicted class: ``{"Predicted": name, <true>: value}``.

        Values are counts, or with ``normalize`` the share of each true class
        (rounded to ``decimals``).
        """
        labels = self.labels
        values = self.normalized() if normalize else self.matrix
        rows = []
        for i, label in enumerate(labels):
            row: Dict[str, Any] = {"Predicted": label}
            for j, true_label in enumerate(labels):
                value = values[i, j]
                row[true_label] = (
                    round(float(value), decimals) if normalize else int(value)
                )
            rows.append(row)
        return rows

    def to_json(self, normalize: bool = False, decimals: int = 5) -> str:
        """The summary as a JSON string."""
        return json.dumps(self.summary(normalize=normalize, decimals=decimals))

    def to_csv(self, normalize: bool = False, decimals: int = 5) -> str:
        """The summary as CSV text."""
        rows = self.summary(normalize=normalize, decimals=decimals)
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer, fieldnames=["Predicted", *self.labels], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()

    def to_df(self, normalize: bool = False, decimals: int = 5):
        """The summary as a Polars DataFrame (needs ``pip install polars``)."""
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "ConfusionMatrix.to_df() needs polars. Install it with "
                "`pip install polars`, or use summary(), to_csv() or to_json()."
            ) from exc
        return pl.DataFrame(self.summary(normalize=normalize, decimals=decimals))

    def plot(
        self,
        normalize: bool = True,
        save_dir: str | Path = "",
    ) -> Path:
        """Save the heatmap and return its path.

        Writes ``confusion_matrix_normalized.png`` when ``normalize`` is set
        and ``confusion_matrix.png`` (raw counts) otherwise.
        """
        from .val_plotter import ValPlotter  # noqa: PLC0415

        name = "confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"
        save_path = Path(save_dir) / name
        # The plotter draws true classes as rows.
        ValPlotter.plot_confusion_matrix(
            self.matrix.T,
            [self.names[i] for i in range(self.nc)],
            save_path,
            normalize=normalize,
            background=self.task == "detect",
        )
        return save_path

    def __repr__(self) -> str:
        return (
            f"ConfusionMatrix(task={self.task!r}, nc={self.nc}, "
            f"total={int(self._marginals()[1].sum())})"
        )
