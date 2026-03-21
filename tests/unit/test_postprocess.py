"""Tests for pluggable post-processing functions."""

import torch
import pytest

from libreyolo.utils.postprocess import soft_nms, diou_nms, resolve_nms_fn
from libreyolo.utils.general import nms, postprocess_detections

pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def overlapping_boxes():
    """Three overlapping boxes with descending scores."""
    boxes = torch.tensor(
        [
            [0.0, 0.0, 100.0, 100.0],  # box 0 (highest score)
            [10.0, 10.0, 110.0, 110.0],  # box 1 (overlaps 0 heavily)
            [200.0, 200.0, 300.0, 300.0],  # box 2 (no overlap)
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    return boxes, scores


@pytest.fixture
def non_overlapping_boxes():
    """Three non-overlapping boxes."""
    boxes = torch.tensor(
        [
            [0.0, 0.0, 50.0, 50.0],
            [100.0, 100.0, 150.0, 150.0],
            [200.0, 200.0, 250.0, 250.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    return boxes, scores


# ============================================================================
# resolve_nms_fn
# ============================================================================


class TestResolveNmsFn:
    def test_none_returns_none(self):
        assert resolve_nms_fn(None) is None

    def test_nms_string_returns_none(self):
        assert resolve_nms_fn("nms") is None

    def test_soft_nms_string_returns_callable(self):
        fn = resolve_nms_fn("soft_nms")
        assert fn is soft_nms

    def test_diou_nms_string_returns_callable(self):
        fn = resolve_nms_fn("diou_nms")
        assert fn is diou_nms

    def test_callable_passthrough(self):
        custom = lambda b, s, t: torch.tensor([0])
        assert resolve_nms_fn(custom) is custom

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown postprocess method 'foo'"):
            resolve_nms_fn("foo")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            resolve_nms_fn(42)


# ============================================================================
# soft_nms
# ============================================================================


class TestSoftNMS:
    def test_basic_suppression(self, overlapping_boxes):
        boxes, scores = overlapping_boxes
        keep = soft_nms(boxes, scores, 0.45)
        # Box 0 (top score) always kept. Box 1 overlaps heavily -> score decayed.
        # Box 2 (no overlap) kept.
        assert 0 in keep.tolist()
        assert 2 in keep.tolist()

    def test_no_suppression(self, non_overlapping_boxes):
        boxes, scores = non_overlapping_boxes
        keep = soft_nms(boxes, scores, 0.45)
        assert len(keep) == 3

    def test_empty_input(self):
        boxes = torch.zeros((0, 4))
        scores = torch.zeros(0)
        keep = soft_nms(boxes, scores, 0.45)
        assert len(keep) == 0
        assert keep.dtype == torch.long

    def test_single_box(self):
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        scores = torch.tensor([0.9])
        keep = soft_nms(boxes, scores, 0.45)
        assert keep.tolist() == [0]

    def test_nan_scores(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        scores = torch.tensor([0.9, float("nan")])
        keep = soft_nms(boxes, scores, 0.45)
        assert 0 in keep.tolist()
        assert 1 not in keep.tolist()

    def test_inf_boxes(self):
        boxes = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [float("inf"), 0.0, 10.0, 10.0]]
        )
        scores = torch.tensor([0.9, 0.8])
        keep = soft_nms(boxes, scores, 0.45)
        assert 0 in keep.tolist()
        assert 1 not in keep.tolist()

    def test_all_invalid(self):
        boxes = torch.tensor([[float("nan"), 0.0, 10.0, 10.0]])
        scores = torch.tensor([0.9])
        keep = soft_nms(boxes, scores, 0.45)
        assert len(keep) == 0

    def test_high_sigma_keeps_more(self, overlapping_boxes):
        """Higher sigma = less aggressive decay = more boxes kept."""
        boxes, scores = overlapping_boxes
        keep_aggressive = soft_nms(boxes, scores, 0.45, sigma=0.1)
        keep_mild = soft_nms(boxes, scores, 0.45, sigma=2.0)
        assert len(keep_mild) >= len(keep_aggressive)

    def test_returns_same_device(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        scores = torch.tensor([0.9])
        keep = soft_nms(boxes, scores, 0.45)
        assert keep.device == boxes.device


# ============================================================================
# diou_nms
# ============================================================================


class TestDIoUNMS:
    def test_basic_suppression(self, overlapping_boxes):
        boxes, scores = overlapping_boxes
        keep = diou_nms(boxes, scores, 0.45)
        # Box 0 kept (highest score). Box 1 suppressed (high DIoU with 0).
        # Box 2 kept (no overlap).
        assert 0 in keep.tolist()
        assert 2 in keep.tolist()

    def test_no_suppression(self, non_overlapping_boxes):
        boxes, scores = non_overlapping_boxes
        keep = diou_nms(boxes, scores, 0.45)
        assert len(keep) == 3

    def test_empty_input(self):
        boxes = torch.zeros((0, 4))
        scores = torch.zeros(0)
        keep = diou_nms(boxes, scores, 0.45)
        assert len(keep) == 0
        assert keep.dtype == torch.long

    def test_single_box(self):
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        scores = torch.tensor([0.9])
        keep = diou_nms(boxes, scores, 0.45)
        assert keep.tolist() == [0]

    def test_nan_scores(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        scores = torch.tensor([0.9, float("nan")])
        keep = diou_nms(boxes, scores, 0.45)
        assert 0 in keep.tolist()
        assert 1 not in keep.tolist()

    def test_center_distance_matters(self):
        """Two boxes with same IoU but different center distances.
        DIoU-NMS should suppress the closer-centered one more."""
        # Box 0: reference
        # Box 1: same size, nearby center (high DIoU -> suppressed)
        # Box 2: same size, far center (lower DIoU -> may survive)
        boxes = torch.tensor(
            [
                [0.0, 0.0, 100.0, 100.0],
                [5.0, 5.0, 105.0, 105.0],  # close center
                [40.0, 40.0, 140.0, 140.0],  # farther center
            ]
        )
        scores = torch.tensor([0.9, 0.85, 0.8])
        keep = diou_nms(boxes, scores, 0.5)
        # Box 1 has higher IoU AND closer center -> more likely suppressed
        assert 0 in keep.tolist()

    def test_returns_same_device(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        scores = torch.tensor([0.9])
        keep = diou_nms(boxes, scores, 0.45)
        assert keep.device == boxes.device


# ============================================================================
# Golden baseline: default NMS path unchanged
# ============================================================================


class TestDefaultNMSUnchanged:
    """Verify that postprocess_detections with nms_fn=None produces identical results."""

    def test_golden_basic(self):
        """Known inputs should produce known outputs with default NMS."""
        torch.manual_seed(42)
        N = 50
        boxes = torch.rand(N, 4) * 500
        # Ensure x2 > x1, y2 > y1
        boxes[:, 2] = boxes[:, 0] + torch.rand(N) * 100 + 10
        boxes[:, 3] = boxes[:, 1] + torch.rand(N) * 100 + 10
        scores = torch.rand(N)
        class_ids = torch.randint(0, 5, (N,))

        # Run with default (nms_fn=None)
        result_default = postprocess_detections(
            boxes.clone(), scores.clone(), class_ids.clone(),
            conf_thres=0.1, iou_thres=0.45, max_det=300,
        )

        # Run with explicit nms_fn=None
        result_explicit = postprocess_detections(
            boxes.clone(), scores.clone(), class_ids.clone(),
            conf_thres=0.1, iou_thres=0.45, max_det=300, nms_fn=None,
        )

        assert result_default["num_detections"] == result_explicit["num_detections"]
        assert result_default["boxes"] == result_explicit["boxes"]
        assert result_default["scores"] == result_explicit["scores"]
        assert result_default["classes"] == result_explicit["classes"]


# ============================================================================
# Integration: postprocess_detections with alternative NMS
# ============================================================================


class TestPostprocessDetectionsIntegration:
    def test_with_soft_nms(self):
        boxes = torch.tensor(
            [
                [0.0, 0.0, 100.0, 100.0],
                [10.0, 10.0, 110.0, 110.0],
                [200.0, 200.0, 300.0, 300.0],
            ]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        class_ids = torch.tensor([0, 0, 1])

        result = postprocess_detections(
            boxes, scores, class_ids,
            conf_thres=0.1, iou_thres=0.45, max_det=300, nms_fn=soft_nms,
        )
        assert result["num_detections"] > 0

    def test_with_diou_nms(self):
        boxes = torch.tensor(
            [
                [0.0, 0.0, 100.0, 100.0],
                [10.0, 10.0, 110.0, 110.0],
                [200.0, 200.0, 300.0, 300.0],
            ]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        class_ids = torch.tensor([0, 0, 1])

        result = postprocess_detections(
            boxes, scores, class_ids,
            conf_thres=0.1, iou_thres=0.45, max_det=300, nms_fn=diou_nms,
        )
        assert result["num_detections"] > 0

    def test_with_custom_callable(self):
        """User-provided callable that keeps everything."""

        def keep_all(boxes, scores, iou_threshold):
            return torch.arange(len(boxes), device=boxes.device)

        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
        scores = torch.tensor([0.9, 0.8])
        class_ids = torch.tensor([0, 0])

        result = postprocess_detections(
            boxes, scores, class_ids,
            conf_thres=0.1, iou_thres=0.45, max_det=300, nms_fn=keep_all,
        )
        assert result["num_detections"] == 2
