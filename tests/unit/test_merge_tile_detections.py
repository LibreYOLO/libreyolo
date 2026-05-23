"""Unit tests for InferenceRunner._merge_tile_detections.

Locks down the batched_nms substitution and the finite-values guard at
the tile-merge call site. Without these tests the guards can drift away
silently — the original PR cherry-pick deleted them and we restored them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from libreyolo.models.base.inference import InferenceRunner

pytestmark = pytest.mark.unit


def _runner(device: str = "cpu") -> InferenceRunner:
    """Build a runner without instantiating a real model."""
    runner = object.__new__(InferenceRunner)
    runner.model = SimpleNamespace(device=torch.device(device))
    return runner


def test_empty_input_returns_empty_lists():
    out = _runner()._merge_tile_detections([], [], [], iou_thres=0.5)
    assert out == ([], [], [])


def test_classwise_nms_suppresses_within_class_but_not_across():
    # Two identical boxes in same class → one suppressed.
    # Same boxes in different class → both kept.
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]]
    scores = [0.9, 0.7, 0.8]
    classes = [0, 0, 1]

    out_boxes, out_scores, out_classes = _runner()._merge_tile_detections(
        boxes, scores, classes, iou_thres=0.5
    )

    assert len(out_boxes) == 2
    # Highest-score box per class kept (paired by index).
    paired = sorted(zip(out_classes, out_scores))
    assert paired[0][0] == 0 and paired[0][1] == pytest.approx(0.9)
    assert paired[1][0] == 1 and paired[1][1] == pytest.approx(0.8)


def test_nan_box_dropped_at_merge():
    # The deleted nms() filtered NaN rows. batched_nms does not — restored
    # via an explicit finite-values guard at this site.
    boxes = [
        [float("nan"), 0, 10, 10],  # NaN row — must be dropped
        [0, 0, 10, 10],             # clean row
    ]
    scores = [0.99, 0.8]
    classes = [0, 1]

    out_boxes, out_scores, out_classes = _runner()._merge_tile_detections(
        boxes, scores, classes, iou_thres=0.5
    )

    assert len(out_boxes) == 1
    assert out_scores == [pytest.approx(0.8)]
    assert out_classes == [1]


def test_nan_score_dropped_at_merge():
    boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
    scores = [float("nan"), 0.7]
    classes = [0, 1]

    out_boxes, out_scores, out_classes = _runner()._merge_tile_detections(
        boxes, scores, classes, iou_thres=0.5
    )

    assert len(out_boxes) == 1
    assert out_scores == [pytest.approx(0.7)]
    assert out_classes == [1]


def test_all_nonfinite_returns_empty():
    boxes = [[float("inf"), 0, 10, 10], [float("nan"), 0, 10, 10]]
    scores = [0.9, 0.8]
    classes = [0, 1]

    out = _runner()._merge_tile_detections(boxes, scores, classes, iou_thres=0.5)
    assert out == ([], [], [])
