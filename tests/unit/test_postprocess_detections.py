"""Unit tests for postprocess_detections in libreyolo.utils.general."""

from __future__ import annotations

import math

import pytest
import torch

from libreyolo.utils.general import postprocess_detections

pytestmark = pytest.mark.unit


def _make(boxes, scores, classes):
    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        torch.tensor(classes, dtype=torch.int64),
    )


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty():
    boxes = torch.zeros((0, 4))
    scores = torch.zeros(0)
    class_ids = torch.zeros(0, dtype=torch.int64)
    result = postprocess_detections(boxes, scores, class_ids)
    assert result["num_detections"] == 0
    assert result["boxes"] == []


def test_single_box_kept():
    boxes, scores, class_ids = _make([[10, 10, 50, 50]], [0.9], [0])
    result = postprocess_detections(boxes, scores, class_ids, conf_thres=0.0)
    assert result["num_detections"] == 1
    assert result["scores"][0] == pytest.approx(0.9)


def test_high_overlap_same_class_suppressed():
    # Two nearly identical boxes for the same class — lower score must be dropped.
    boxes, scores, class_ids = _make(
        [[0, 0, 100, 100], [1, 1, 101, 101]],
        [0.9, 0.7],
        [0, 0],
    )
    result = postprocess_detections(boxes, scores, class_ids, iou_thres=0.5)
    assert result["num_detections"] == 1
    assert result["scores"][0] == pytest.approx(0.9)


def test_high_overlap_different_classes_both_kept():
    # Same boxes, different classes — batched_nms must not suppress across classes.
    boxes, scores, class_ids = _make(
        [[0, 0, 100, 100], [0, 0, 100, 100]],
        [0.9, 0.8],
        [0, 1],
    )
    result = postprocess_detections(boxes, scores, class_ids, iou_thres=0.5)
    assert result["num_detections"] == 2


def test_multi_class_independence():
    """Suppression within each class is independent; results are score-sorted."""
    boxes, scores, class_ids = _make(
        [
            [0, 0, 10, 10],   # cls 0, score 0.9 — kept
            [0, 0, 10, 10],   # cls 0, score 0.5 — suppressed (same box)
            [50, 50, 60, 60], # cls 1, score 0.8 — kept (different class + location)
            [50, 50, 60, 60], # cls 1, score 0.3 — suppressed (same box)
        ],
        [0.9, 0.5, 0.8, 0.3],
        [0, 0, 1, 1],
    )
    result = postprocess_detections(boxes, scores, class_ids, iou_thres=0.5)
    assert result["num_detections"] == 2
    # Results should be sorted by descending score.
    assert result["scores"][0] >= result["scores"][1]


def test_output_sorted_by_descending_score():
    boxes, scores, class_ids = _make(
        [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
        [0.5, 0.9, 0.7],
        [0, 1, 2],
    )
    result = postprocess_detections(boxes, scores, class_ids)
    s = result["scores"]
    assert s == sorted(s, reverse=True), "scores must be in descending order"


# ---------------------------------------------------------------------------
# max_det
# ---------------------------------------------------------------------------


def test_max_det_limits_output():
    n = 20
    boxes = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)
    # Make valid (x1<x2, y1<y2) by offsetting columns
    boxes[:, 2] += 100
    boxes[:, 3] += 100
    scores = torch.linspace(0.1, 0.9, n)
    class_ids = torch.arange(n, dtype=torch.int64)  # all different — none suppressed

    result = postprocess_detections(boxes, scores, class_ids, max_det=5)
    assert result["num_detections"] == 5
    # Should be the top-5 by score.
    assert min(result["scores"]) > scores[n - 6].item() - 1e-5


# ---------------------------------------------------------------------------
# Finite-values guard
# ---------------------------------------------------------------------------


def test_nan_boxes_dropped():
    boxes, scores, class_ids = _make(
        [[float("nan"), 0, 10, 10], [0, 0, 10, 10]],
        [0.9, 0.8],
        [0, 1],
    )
    result = postprocess_detections(boxes, scores, class_ids)
    # NaN box must be dropped; only the clean box survives.
    assert result["num_detections"] == 1
    assert result["scores"][0] == pytest.approx(0.8)


def test_inf_boxes_dropped():
    boxes, scores, class_ids = _make(
        [[0, 0, float("inf"), 10], [20, 20, 30, 30]],
        [0.95, 0.6],
        [0, 0],
    )
    result = postprocess_detections(boxes, scores, class_ids)
    assert result["num_detections"] == 1
    assert result["scores"][0] == pytest.approx(0.6)


def test_nan_score_dropped():
    boxes, scores, class_ids = _make(
        [[0, 0, 10, 10], [20, 20, 30, 30]],
        [float("nan"), 0.7],
        [0, 1],
    )
    result = postprocess_detections(boxes, scores, class_ids)
    assert result["num_detections"] == 1
    assert result["scores"][0] == pytest.approx(0.7)


def test_all_nan_returns_empty():
    boxes, scores, class_ids = _make(
        [[float("nan"), 0, 10, 10]],
        [0.9],
        [0],
    )
    result = postprocess_detections(boxes, scores, class_ids)
    assert result["num_detections"] == 0


# ---------------------------------------------------------------------------
# Scaling / original_size
# ---------------------------------------------------------------------------


def test_boxes_scaled_to_original_size():
    # Box covers half the 640-input image; after scaling to 1280×960 it should double.
    boxes, scores, class_ids = _make([[0, 0, 320, 320]], [0.9], [0])
    result = postprocess_detections(
        boxes, scores, class_ids,
        input_size=640,
        original_size=(1280, 960),
    )
    b = result["boxes"][0]
    assert b[2] == pytest.approx(640.0)   # 320 * (1280/640)
    assert b[3] == pytest.approx(480.0)   # 320 * (960/640)
