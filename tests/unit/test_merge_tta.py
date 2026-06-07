"""Unit tests for BaseModel._merge_tta.

Locks the batched_nms substitution and the finite-values guard at the TTA
merge call site. _merge_tile_detections is tested separately — this file
covers the symmetric path for test-time augmentation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from libreyolo.models.base.model import BaseModel

pytestmark = pytest.mark.unit


def _model():
    """Minimal stand-in: _merge_tta only reads self.names."""
    return SimpleNamespace(names={0: "a", 1: "b"})


def test_obb_rejects_tta_before_axis_aligned_merge():
    model = SimpleNamespace(task="obb")

    with pytest.raises(ValueError, match="oriented boxes"):
        BaseModel._predict_augment(model, image=None)


def test_pose_rejects_tta_before_keypoint_merge():
    model = SimpleNamespace(task="pose")

    with pytest.raises(ValueError, match="pose keypoints"):
        BaseModel._predict_augment(model, image=None)


def _det(boxes, scores, classes):
    return {
        "boxes": boxes,
        "scores": scores,
        "classes": classes,
        "num_detections": len(boxes),
    }


def test_empty_aug_dets_returns_empty_results():
    result = BaseModel._merge_tta(
        _model(),
        aug_dets=[],
        iou_thres=0.5,
        image_path="img.jpg",
        original_size=(100, 50),
    )
    assert len(result) == 0


def test_classwise_nms_across_aug_views():
    # Two aug views each contribute a near-duplicate box for class 0; one of them
    # also has a separate class-1 box. Expected: one class-0 box (NMS), one class-1.
    aug_dets = [
        (_det([[0, 0, 10, 10]], [0.9], [0]), (100, 50), False, 1.0),
        (_det([[1, 1, 11, 11], [50, 5, 60, 15]], [0.7, 0.8], [0, 1]),
            (100, 50), False, 1.0),
    ]

    result = BaseModel._merge_tta(
        _model(),
        aug_dets=aug_dets,
        iou_thres=0.5,
        image_path="img.jpg",
        original_size=(100, 50),
    )

    assert len(result) == 2
    classes = result.boxes.cls.tolist()
    scores = result.boxes.conf.tolist()
    assert sorted(classes) == [0.0, 1.0]
    # Class-0 kept the higher-score (0.9) of the duplicates.
    cls0_score = next(s for s, c in zip(scores, classes) if c == 0.0)
    assert cls0_score == pytest.approx(0.9)


def test_nan_box_dropped_at_tta_merge():
    # The deleted nms() filtered NaN rows. batched_nms doesn't — guard restored.
    aug_dets = [
        (_det([[float("nan"), 0, 10, 10]], [0.99], [0]), (100, 50), False, 1.0),
        (_det([[20, 20, 30, 30]], [0.7], [1]), (100, 50), False, 1.0),
    ]

    result = BaseModel._merge_tta(
        _model(),
        aug_dets=aug_dets,
        iou_thres=0.5,
        image_path="img.jpg",
        original_size=(100, 50),
    )

    assert len(result) == 1
    assert result.boxes.cls.tolist() == [1.0]
    assert result.boxes.conf.tolist()[0] == pytest.approx(0.7)


def test_nan_score_dropped_at_tta_merge():
    aug_dets = [
        (_det([[0, 0, 10, 10]], [float("nan")], [0]), (100, 50), False, 1.0),
        (_det([[20, 20, 30, 30]], [0.7], [1]), (100, 50), False, 1.0),
    ]

    result = BaseModel._merge_tta(
        _model(),
        aug_dets=aug_dets,
        iou_thres=0.5,
        image_path="img.jpg",
        original_size=(100, 50),
    )

    assert len(result) == 1
    assert result.boxes.cls.tolist() == [1.0]


def test_all_nonfinite_returns_empty_results():
    aug_dets = [
        (_det([[float("inf"), 0, 10, 10]], [0.9], [0]), (100, 50), False, 1.0),
        (_det([[float("nan"), 0, 10, 10]], [0.8], [1]), (100, 50), False, 1.0),
    ]

    result = BaseModel._merge_tta(
        _model(),
        aug_dets=aug_dets,
        iou_thres=0.5,
        image_path="img.jpg",
        original_size=(100, 50),
    )

    assert len(result) == 0


def test_classes_filter_restricts_output():
    aug_dets = [
        (_det([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.8], [0, 1]),
            (100, 50), False, 1.0),
    ]

    result = BaseModel._merge_tta(
        _model(),
        aug_dets=aug_dets,
        iou_thres=0.5,
        image_path="img.jpg",
        original_size=(100, 50),
        classes=[1],  # only keep class 1
    )

    assert len(result) == 1
    assert result.boxes.cls.tolist() == [1.0]
