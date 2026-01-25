"""
Unit tests for validation metrics.

Tests the DetMetrics class along with utility functions for prediction matching.
"""

import numpy as np
import pytest
import torch

from libreyolo.validation import (
    DetMetrics,
    match_predictions_to_gt,
    process_batch,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


class TestDetMetrics:
    """Tests for DetMetrics class."""

    def test_initialization(self):
        """Test default initialization."""
        metrics = DetMetrics(nc=80)
        assert metrics.nc == 80
        assert metrics.conf == 0.25
        assert len(metrics.iou_thresholds) == 10
        assert metrics.stats == []

    def test_perfect_detection(self):
        """Test with perfect predictions (all TP)."""
        metrics = DetMetrics(nc=3)

        # Simulate perfect detection: all predictions correct at all IoU thresholds
        correct = np.ones((5, 10), dtype=bool)  # 5 predictions, 10 IoU thresholds
        conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        pred_cls = np.array([0, 1, 2, 0, 1])
        target_cls = np.array([0, 1, 2, 0, 1])

        metrics.update(correct, conf, pred_cls, target_cls)
        results = metrics.compute()

        assert results["metrics/mAP50"] == pytest.approx(1.0, abs=0.01)
        assert results["metrics/mAP50-95"] == pytest.approx(1.0, abs=0.01)

    def test_no_detections(self):
        """Test with no predictions (0 recall)."""
        metrics = DetMetrics(nc=3)

        # No predictions
        correct = np.zeros((0, 10), dtype=bool)
        conf = np.array([])
        pred_cls = np.array([])
        target_cls = np.array([0, 1, 2])  # 3 ground truths

        metrics.update(correct, conf, pred_cls, target_cls)
        results = metrics.compute()

        assert results["metrics/recall"] == 0.0
        assert results["metrics/mAP50"] == 0.0

    def test_false_positives(self):
        """Test that false positives reduce precision."""
        metrics = DetMetrics(nc=2)

        # 2 TP and 3 FP
        correct = np.array([
            [True] * 10,   # TP
            [True] * 10,   # TP
            [False] * 10,  # FP
            [False] * 10,  # FP
            [False] * 10,  # FP
        ])
        conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        pred_cls = np.array([0, 0, 0, 0, 0])
        target_cls = np.array([0, 0])  # 2 GT

        metrics.update(correct, conf, pred_cls, target_cls)
        results = metrics.compute()

        # Precision should be 2/5 = 0.4 at conf=0.25
        assert results["metrics/precision"] < 1.0
        assert results["metrics/recall"] == pytest.approx(1.0, abs=0.01)

    def test_multiple_iou_thresholds(self):
        """Test that mAP50 >= mAP50-95."""
        metrics = DetMetrics(nc=2)

        # Predictions match GT at IoU=0.5 but not at higher thresholds
        correct = np.array([
            [True, True, True, False, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False, False, False],
        ])
        conf = np.array([0.9, 0.8])
        pred_cls = np.array([0, 1])
        target_cls = np.array([0, 1])

        metrics.update(correct, conf, pred_cls, target_cls)
        results = metrics.compute()

        # mAP50 should be higher than mAP50-95
        assert results["metrics/mAP50"] >= results["metrics/mAP50-95"]

    def test_reset(self):
        """Test that reset clears accumulated stats."""
        metrics = DetMetrics(nc=2)

        correct = np.ones((2, 10), dtype=bool)
        conf = np.array([0.9, 0.8])
        pred_cls = np.array([0, 1])
        target_cls = np.array([0, 1])

        metrics.update(correct, conf, pred_cls, target_cls)
        assert len(metrics.stats) == 1

        metrics.reset()
        assert len(metrics.stats) == 0
        assert metrics.ap is None


class TestMatchPredictionsToGT:
    """Tests for prediction matching utility."""

    def test_exact_match(self):
        """Test that identical boxes have IoU=1."""
        pred_boxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        pred_classes = torch.tensor([0])
        gt_boxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        gt_classes = torch.tensor([0])
        iou_thresholds = torch.tensor([0.5, 0.75, 0.95])

        correct, iou_values = match_predictions_to_gt(
            pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresholds
        )

        # Should be correct at all thresholds
        assert correct[0].all()
        assert iou_values[0] == pytest.approx(1.0, abs=0.001)

    def test_no_overlap(self):
        """Test that non-overlapping boxes don't match."""
        pred_boxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        pred_classes = torch.tensor([0])
        gt_boxes = torch.tensor([[100, 100, 150, 150]], dtype=torch.float32)
        gt_classes = torch.tensor([0])
        iou_thresholds = torch.tensor([0.5])

        correct, iou_values = match_predictions_to_gt(
            pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresholds
        )

        assert not correct[0, 0]
        assert iou_values[0] == 0.0

    def test_class_filtering(self):
        """Test that different classes don't match."""
        pred_boxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        pred_classes = torch.tensor([0])
        gt_boxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        gt_classes = torch.tensor([1])  # Different class
        iou_thresholds = torch.tensor([0.5])

        correct, iou_values = match_predictions_to_gt(
            pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresholds
        )

        # Should not match due to different classes
        assert not correct[0, 0]

    def test_empty_predictions(self):
        """Test with no predictions."""
        pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
        pred_classes = torch.zeros(0, dtype=torch.int64)
        gt_boxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        gt_classes = torch.tensor([0])
        iou_thresholds = torch.tensor([0.5])

        correct, iou_values = match_predictions_to_gt(
            pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresholds
        )

        assert correct.shape == (0, 1)
        assert iou_values.shape == (0,)

    def test_empty_ground_truth(self):
        """Test with no ground truth."""
        pred_boxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        pred_classes = torch.tensor([0])
        gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
        gt_classes = torch.zeros(0, dtype=torch.int64)
        iou_thresholds = torch.tensor([0.5])

        correct, iou_values = match_predictions_to_gt(
            pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresholds
        )

        # All predictions should be FP
        assert not correct.any()


class TestBoxConversions:
    """Tests for box format conversion utilities."""

    def test_xywh_to_xyxy(self):
        """Test xywh to xyxy conversion."""
        xywh = torch.tensor([[50, 50, 20, 30]])  # cx=50, cy=50, w=20, h=30
        xyxy = xywh_to_xyxy(xywh)

        expected = torch.tensor([[40, 35, 60, 65]])  # x1=40, y1=35, x2=60, y2=65
        assert torch.allclose(xyxy, expected)

    def test_xyxy_to_xywh(self):
        """Test xyxy to xywh conversion."""
        xyxy = torch.tensor([[40, 35, 60, 65]])  # x1=40, y1=35, x2=60, y2=65
        xywh = xyxy_to_xywh(xyxy)

        expected = torch.tensor([[50, 50, 20, 30]])  # cx=50, cy=50, w=20, h=30
        assert torch.allclose(xywh, expected)

    def test_round_trip_conversion(self):
        """Test that conversion is reversible."""
        original = torch.tensor([[10, 20, 100, 200], [50, 50, 80, 80]], dtype=torch.float32)
        converted = xyxy_to_xywh(original)
        recovered = xywh_to_xyxy(converted)

        assert torch.allclose(original, recovered)


class TestProcessBatch:
    """Tests for process_batch utility."""

    def test_basic_processing(self):
        """Test basic batch processing."""
        pred_boxes = torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float32)
        pred_scores = torch.tensor([0.9, 0.7])
        pred_classes = torch.tensor([0, 1])
        gt_boxes = torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float32)
        gt_classes = torch.tensor([0, 1])
        iou_thresholds = torch.tensor([0.5, 0.75])

        correct, conf, pred_cls, target_cls = process_batch(
            pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_thresholds
        )

        assert correct.shape == (2, 2)
        assert len(conf) == 2
        assert len(pred_cls) == 2
        assert len(target_cls) == 2

        # Should be sorted by confidence
        assert conf[0] >= conf[1]
