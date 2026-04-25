"""Unit tests for ConfusionMatrix + plot_per_class_ap (DX issue #46)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from libreyolo.validation import ConfusionMatrix, plot_per_class_ap


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ConfusionMatrix
# ---------------------------------------------------------------------------


def _box(x1, y1, x2, y2):
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def test_perfect_match_diagonal():
    cm = ConfusionMatrix(nc=3, conf=0.0, iou_thresh=0.5)
    boxes = np.array([_box(0, 0, 10, 10), _box(20, 20, 30, 30), _box(50, 50, 60, 60)])
    classes = np.array([0, 1, 2])
    scores = np.array([0.9, 0.9, 0.9])
    cm.update(boxes, scores, classes, boxes.copy(), classes.copy())
    # Diagonal == 1, everything else == 0
    expected = np.zeros((4, 4), dtype=np.int64)
    expected[0, 0] = expected[1, 1] = expected[2, 2] = 1
    assert (cm.matrix == expected).all()


def test_class_misclassification_off_diagonal():
    cm = ConfusionMatrix(nc=3, conf=0.0, iou_thresh=0.5)
    box = _box(0, 0, 10, 10)
    cm.update(
        pred_boxes=np.array([box]),
        pred_scores=np.array([0.9]),
        pred_classes=np.array([1]),  # predicts cat as dog
        gt_boxes=np.array([box]),
        gt_classes=np.array([0]),
    )
    # matrix[true=0, pred=1] should be 1
    assert cm.matrix[0, 1] == 1
    assert cm.matrix.sum() == 1


def test_unmatched_pred_is_false_positive_against_background():
    cm = ConfusionMatrix(nc=3, conf=0.0, iou_thresh=0.5)
    cm.update(
        pred_boxes=np.array([_box(0, 0, 10, 10)]),
        pred_scores=np.array([0.9]),
        pred_classes=np.array([1]),
        gt_boxes=np.zeros((0, 4)),
        gt_classes=np.zeros(0),
    )
    # background (row 3) misclassified as pred class 1
    assert cm.matrix[3, 1] == 1
    assert cm.matrix.sum() == 1


def test_unmatched_gt_is_false_negative():
    cm = ConfusionMatrix(nc=3, conf=0.0, iou_thresh=0.5)
    cm.update(
        pred_boxes=np.zeros((0, 4)),
        pred_scores=np.zeros(0),
        pred_classes=np.zeros(0),
        gt_boxes=np.array([_box(0, 0, 10, 10)]),
        gt_classes=np.array([2]),
    )
    # true class 2, predicted as background (col 3)
    assert cm.matrix[2, 3] == 1
    assert cm.matrix.sum() == 1


def test_low_conf_predictions_filtered():
    cm = ConfusionMatrix(nc=2, conf=0.5, iou_thresh=0.5)
    box = _box(0, 0, 10, 10)
    cm.update(
        pred_boxes=np.array([box]),
        pred_scores=np.array([0.3]),  # below conf
        pred_classes=np.array([0]),
        gt_boxes=np.array([box]),
        gt_classes=np.array([0]),
    )
    # pred dropped, GT becomes a false negative
    assert cm.matrix[0, 2] == 1
    assert cm.matrix[0, 0] == 0


def test_iou_below_threshold_treats_pred_as_background():
    cm = ConfusionMatrix(nc=2, conf=0.0, iou_thresh=0.9)
    cm.update(
        pred_boxes=np.array([_box(0, 0, 10, 10)]),
        pred_scores=np.array([0.9]),
        pred_classes=np.array([0]),
        gt_boxes=np.array([_box(0, 0, 12, 12)]),  # IoU ~0.69
        gt_classes=np.array([0]),
    )
    # Pred unmatched: matrix[bg=2, pred=0] += 1; GT unmatched: matrix[true=0, bg=2] += 1
    assert cm.matrix[2, 0] == 1
    assert cm.matrix[0, 2] == 1


def test_normalize_row_yields_recall_view():
    cm = ConfusionMatrix(nc=2, conf=0.0, iou_thresh=0.5)
    cm.matrix[0, 0] = 8
    cm.matrix[0, 1] = 2  # 8/10 = 0.8 recall for class 0
    norm = cm.normalize("row")
    assert abs(norm[0, 0] - 0.8) < 1e-6
    assert abs(norm[0, 1] - 0.2) < 1e-6


def test_normalize_zero_row_does_not_divide_by_zero():
    cm = ConfusionMatrix(nc=2)
    norm = cm.normalize("row")
    assert np.isfinite(norm).all()
    assert (norm == 0).all()


def test_plot_writes_png(tmp_path: Path):
    cm = ConfusionMatrix(nc=3)
    cm.matrix[0, 0] = 5
    cm.matrix[1, 0] = 1
    cm.matrix[2, 2] = 3
    cm.matrix[3, 0] = 2  # FP
    out = cm.plot(tmp_path / "cm.png", names=["cat", "dog", "bird"])
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG


def test_plot_uses_class_indices_when_names_omitted(tmp_path: Path):
    cm = ConfusionMatrix(nc=2)
    cm.matrix[0, 0] = 1
    out = cm.plot(tmp_path / "cm.png")
    assert out.exists()


def test_plot_rejects_wrong_name_count(tmp_path: Path):
    cm = ConfusionMatrix(nc=3)
    cm.matrix[0, 0] = 1
    with pytest.raises(ValueError):
        cm.plot(tmp_path / "cm.png", names=["cat", "dog"])


def test_init_rejects_zero_nc():
    with pytest.raises(ValueError):
        ConfusionMatrix(nc=0)


# ---------------------------------------------------------------------------
# plot_per_class_ap
# ---------------------------------------------------------------------------


def test_per_class_ap_plot_writes_png(tmp_path: Path):
    nc = 4
    classes_with_gt = np.array([0, 2, 3])
    ap = np.array([
        [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
        [0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
    ])
    out = plot_per_class_ap(ap, classes_with_gt, nc, tmp_path / "ap.png",
                            names=["a", "b", "c", "d"])
    assert out.exists()
    assert out.stat().st_size > 1000


def test_per_class_ap_handles_empty_ap(tmp_path: Path):
    out = plot_per_class_ap(
        np.zeros((0, 10)), np.array([], dtype=np.int64), 3, tmp_path / "ap.png",
        names=["x", "y", "z"],
    )
    assert out.exists()
