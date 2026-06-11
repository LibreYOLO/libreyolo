"""Unit tests for PointValidator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo.models.base.model import BaseModel
from libreyolo.validation import PointValidator, ValidationConfig
from libreyolo.validation.point_validator import (
    _ImageRecord,
    _average_precision_at_threshold,
)

pytestmark = pytest.mark.unit


# ===========================================================================
# Test Helpers
# ===========================================================================

def _make_validator_with_records(records: List[_ImageRecord]) -> PointValidator:
    """Build a PointValidator with pre-seeded records (no model needed)."""
    validator = object.__new__(PointValidator)
    from libreyolo.validation.point_validator import _DEFAULT_DIST_THRESHOLDS

    validator._dist_thresholds = _DEFAULT_DIST_THRESHOLDS
    validator._primary_threshold = _DEFAULT_DIST_THRESHOLDS[0]
    validator._records = records
    validator.nc = 1
    validator.config = type("_Cfg", (), {"verbose": False})()
    validator.seen = sum(1 for _ in records)
    return validator


def _make_validator_for_parsing(imgsz: int = 100) -> PointValidator:
    """Build a PointValidator configured for ground-truth parsing tests."""
    v = object.__new__(PointValidator)
    from libreyolo.validation.point_validator import _DEFAULT_DIST_THRESHOLDS

    v._dist_thresholds = _DEFAULT_DIST_THRESHOLDS
    v._primary_threshold = _DEFAULT_DIST_THRESHOLDS[0]
    v._records = []
    v.nc = 2
    v._actual_imgsz = imgsz
    v.config = type("_Cfg", (), {"verbose": False})()
    v.seen = 0

    class _FakePreproc:
        uses_letterbox = False

    v.val_preproc = _FakePreproc()
    return v


class _DummyPointModel:
    nb_classes = 1
    names = {0: "point_class"}
    size = "t"
    task = "point"

    def __init__(self):
        self.device = torch.device("cpu")
        self.model = torch.nn.Identity()

    def _get_input_size(self):
        return 64

    def _get_model_name(self):
        return "DummyPointModel"

    def _get_val_preprocessor(self, img_size=None):
        from libreyolo.validation.preprocessors import YOLO9ValPreprocessor
        img_size = img_size or 64
        return YOLO9ValPreprocessor(img_size=(img_size, img_size))

    def _forward(self, images):
        return images

    def _postprocess(self, *args, **kwargs):
        return {
            "points": [[32.0, 32.0, 0.0, 0.99]],
        }


def _write_point_dataset(root: Path) -> Path:
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump({
            "path": str(root).replace("\\", "/"),
            "val": "images/val",
            "nc": 1,
            "names": {0: "point_class"},
        }),
        encoding="utf-8",
    )
    return data_yaml


# ===========================================================================
# Math Verification (101-point Interpolated AP)
# ===========================================================================

def test_point_validator_ap_perfect_is_one():
    """Verify that AP is exactly 1.0 when all predictions are correct."""
    scores = np.array([0.9, 0.8, 0.7])
    matched = np.array([True, True, True])
    ap = _average_precision_at_threshold(scores, matched, n_gt_total=3)
    assert ap == pytest.approx(1.0, abs=1e-3)


def test_point_validator_ap_tp_ordering_matters():
    """Verify that AP is sensitive to confidence score ordering (ranking)."""
    n_gt = 2
    ap_a = _average_precision_at_threshold(
        np.array([0.9, 0.3]), np.array([True, False]), n_gt_total=n_gt
    )
    ap_b = _average_precision_at_threshold(
        np.array([0.9, 0.3]), np.array([False, True]), n_gt_total=n_gt
    )
    assert ap_a > ap_b


def test_point_validator_ap_zero_when_no_gt():
    """Verify that AP is 0.0 when there are no ground-truth points."""
    scores = np.array([0.9, 0.8])
    matched = np.array([False, False])
    ap = _average_precision_at_threshold(scores, matched, n_gt_total=0)
    assert ap == 0.0


# ===========================================================================
# Metric Correctness
# ===========================================================================

def test_point_validator_metrics_perfect_localisation():
    pts = np.array([[0.1, 0.2], [0.5, 0.5]], dtype=np.float64)
    rec = _ImageRecord(
        pred_xy=pts,
        pred_scores=np.array([0.9, 0.8], dtype=np.float32),
        pred_classes=np.zeros(2, dtype=np.int64),
        gt_xy=pts,
        gt_classes=np.zeros(2, dtype=np.int64),
    )
    v = _make_validator_with_records([rec])
    m = v._compute_metrics()
    assert m["metrics/precision"] == pytest.approx(1.0)
    assert m["metrics/recall"] == pytest.approx(1.0)
    assert m["metrics/f1"] == pytest.approx(1.0)
    assert m["metrics/MLE"] == pytest.approx(0.0, abs=1e-9)
    assert m["metrics/MAE"] == pytest.approx(0.0)
    assert m["metrics/RMSE"] == pytest.approx(0.0)


def test_point_validator_metrics_no_predictions_all_fn():
    gt = np.array([[0.2, 0.3], [0.7, 0.8]], dtype=np.float64)
    rec = _ImageRecord(
        pred_xy=np.zeros((0, 2), dtype=np.float64),
        pred_scores=np.zeros(0, dtype=np.float32),
        pred_classes=np.zeros(0, dtype=np.int64),
        gt_xy=gt,
        gt_classes=np.zeros(2, dtype=np.int64),
    )
    v = _make_validator_with_records([rec])
    m = v._compute_metrics()
    assert m["metrics/precision"] == pytest.approx(0.0)
    assert m["metrics/recall"] == pytest.approx(0.0)
    assert m["metrics/f1"] == pytest.approx(0.0)
    assert m["metrics/MAE"] == pytest.approx(2.0)


def test_point_validator_metrics_no_gt_all_fp():
    pred = np.array([[0.4, 0.5], [0.6, 0.7]], dtype=np.float64)
    rec = _ImageRecord(
        pred_xy=pred,
        pred_scores=np.array([0.7, 0.6], dtype=np.float32),
        pred_classes=np.zeros(2, dtype=np.int64),
        gt_xy=np.zeros((0, 2), dtype=np.float64),
        gt_classes=np.zeros(0, dtype=np.int64),
    )
    v = _make_validator_with_records([rec])
    m = v._compute_metrics()
    assert m["metrics/precision"] == pytest.approx(0.0)
    assert m["metrics/recall"] == pytest.approx(0.0)
    assert m["metrics/MAE"] == pytest.approx(2.0)


def test_point_validator_metrics_partial_tp_fp_fn():
    pred_xy = np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float64)
    gt_xy = np.array([[0.1, 0.1], [0.5, 0.5]], dtype=np.float64)
    rec = _ImageRecord(
        pred_xy=pred_xy,
        pred_scores=np.array([0.9, 0.8], dtype=np.float32),
        pred_classes=np.zeros(2, dtype=np.int64),
        gt_xy=gt_xy,
        gt_classes=np.zeros(2, dtype=np.int64),
    )
    v = _make_validator_with_records([rec])
    m = v._compute_metrics()
    assert m["metrics/precision"] == pytest.approx(0.5)
    assert m["metrics/recall"] == pytest.approx(0.5)
    assert m["metrics/f1"] == pytest.approx(0.5)


def test_point_validator_metrics_mle_average_tp_distances():
    pred_xy = np.array([[0.1 + 0.005, 0.1], [0.9, 0.9]], dtype=np.float64)
    gt_xy = np.array([[0.1, 0.1], [0.5, 0.5]], dtype=np.float64)
    rec = _ImageRecord(
        pred_xy=pred_xy,
        pred_scores=np.array([0.95, 0.85], dtype=np.float32),
        pred_classes=np.zeros(2, dtype=np.int64),
        gt_xy=gt_xy,
        gt_classes=np.zeros(2, dtype=np.int64),
    )
    v = _make_validator_with_records([rec])
    m = v._compute_metrics()
    assert m["metrics/MLE"] == pytest.approx(0.005, abs=1e-7)


def test_point_validator_metrics_counting_mae_rmse():
    rec1 = _ImageRecord(
        pred_xy=np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], np.float64),
        pred_scores=np.ones(3, np.float32),
        pred_classes=np.zeros(3, np.int64),
        gt_xy=np.array([[0.1, 0.1], [0.5, 0.5]], np.float64),
        gt_classes=np.zeros(2, np.int64),
    )
    rec2 = _ImageRecord(
        pred_xy=np.array([[0.2, 0.2]], np.float64),
        pred_scores=np.ones(1, np.float32),
        pred_classes=np.zeros(1, np.int64),
        gt_xy=np.array([[0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]], np.float64),
        gt_classes=np.zeros(4, np.int64),
    )
    v = _make_validator_with_records([rec1, rec2])
    m = v._compute_metrics()
    assert m["metrics/MAE"] == pytest.approx(2.0)
    assert m["metrics/RMSE"] == pytest.approx(np.sqrt(5.0), abs=1e-6)


def test_point_validator_metrics_map_sweep_range():
    pts = np.array([[0.2, 0.3]], np.float64)
    rec = _ImageRecord(
        pred_xy=pts,
        pred_scores=np.array([0.8], np.float32),
        pred_classes=np.zeros(1, np.int64),
        gt_xy=pts,
        gt_classes=np.zeros(1, np.int64),
    )
    v = _make_validator_with_records([rec])
    m = v._compute_metrics()
    assert 0.0 <= m[v._map_sweep_key()] <= 1.0


def test_point_validator_metrics_empty_records_returns_zeros():
    v = _make_validator_with_records([])
    m = v._compute_metrics()
    assert m["metrics/precision"] == pytest.approx(0.0)
    assert m["metrics/recall"] == pytest.approx(0.0)
    assert m[v._map_sweep_key()] == pytest.approx(0.0)


def test_point_validator_metrics_fitness_equals_map_sweep():
    pts = np.array([[0.5, 0.5]], np.float64)
    rec = _ImageRecord(
        pred_xy=pts,
        pred_scores=np.array([0.9], np.float32),
        pred_classes=np.zeros(1, np.int64),
        gt_xy=pts,
        gt_classes=np.zeros(1, np.int64),
    )
    v = _make_validator_with_records([rec])
    m = v._compute_metrics()
    assert m["fitness"] == m[v._map_sweep_key()]


def test_point_validator_metrics_multi_image_ap_rollup():
    rec1 = _ImageRecord(
        pred_xy=np.array([[0.1, 0.1], [0.5, 0.5]], np.float64),
        pred_scores=np.array([0.9, 0.8], np.float32),
        pred_classes=np.zeros(2, np.int64),
        gt_xy=np.array([[0.1, 0.1], [0.5, 0.5]], np.float64),
        gt_classes=np.zeros(2, np.int64),
    )
    rec2 = _ImageRecord(
        pred_xy=np.array([[0.9, 0.9]], np.float64),
        pred_scores=np.array([0.7], np.float32),
        pred_classes=np.zeros(1, np.int64),
        gt_xy=np.array([[0.1, 0.1]], np.float64),
        gt_classes=np.zeros(1, np.int64),
    )
    v = _make_validator_with_records([rec1, rec2])
    m = v._compute_metrics()
    assert m["metrics/precision"] == pytest.approx(2 / 3, abs=1e-6)
    assert m["metrics/recall"]    == pytest.approx(2 / 3, abs=1e-6)
    assert m["metrics/f1"]        == pytest.approx(2 / 3, abs=1e-6)
    primary_key = f"metrics/mAP@{v._primary_threshold:.2f}"
    assert m[primary_key] > 0.0
    assert m[primary_key] <= 1.0


# ===========================================================================
# Edge Cases
# ===========================================================================

def test_point_validator_edge_many_preds_zero_gt():
    rng = np.random.default_rng(42)
    pred = rng.random((50, 2)).astype(np.float64)
    rec = _ImageRecord(
        pred_xy=pred,
        pred_scores=rng.random(50).astype(np.float32),
        pred_classes=np.zeros(50, np.int64),
        gt_xy=np.zeros((0, 2), np.float64),
        gt_classes=np.zeros(0, np.int64),
    )
    v = _make_validator_with_records([rec])
    m = v._compute_metrics()
    assert m["metrics/precision"] == pytest.approx(0.0)
    assert m["metrics/recall"] == pytest.approx(0.0)
    assert m["metrics/MAE"] == pytest.approx(50.0)


def test_point_validator_edge_loose_threshold_more_tp():
    from libreyolo.validation.point_validator import _euclidean_distance_matrix, _hungarian_match

    pred = np.array([[0.10, 0.10]], np.float64)
    gt = np.array([[0.10 + 0.08, 0.10]], np.float64)

    tp5, fp5, fn5 = _hungarian_match(
        _euclidean_distance_matrix(pred, gt), threshold=0.05
    )
    tp10, fp10, fn10 = _hungarian_match(
        _euclidean_distance_matrix(pred, gt), threshold=0.10
    )
    assert len(tp5) == 0
    assert len(tp10) == 1


def test_point_validator_edge_custom_thresholds_used():
    v = object.__new__(PointValidator)
    v._dist_thresholds = (0.05, 0.10)
    v._primary_threshold = 0.05
    v._records = []
    v.nc = 1
    v.config = type("_Cfg", (), {"verbose": False})()
    v.seen = 0

    m = v._compute_metrics()
    assert m["metrics/precision"] == 0.0
    assert "metrics/mAP@0.05" in m
    assert v._map_sweep_key() == "metrics/mAP@[0.05:0.10]"


def test_point_validator_edge_custom_threshold_changes_values():
    pred_xy = np.array([[0.10, 0.10]], np.float64)
    gt_xy   = np.array([[0.16, 0.10]], np.float64)  # distance = 0.06

    rec = _ImageRecord(
        pred_xy=pred_xy,
        pred_scores=np.array([0.9], np.float32),
        pred_classes=np.zeros(1, np.int64),
        gt_xy=gt_xy,
        gt_classes=np.zeros(1, np.int64),
    )

    v_tight = object.__new__(PointValidator)
    v_tight._dist_thresholds = (0.05,)
    v_tight._primary_threshold = 0.05
    v_tight._records = [rec]
    v_tight.nc = 1
    v_tight.config = type("_Cfg", (), {"verbose": False})()
    v_tight.seen = 1

    v_loose = object.__new__(PointValidator)
    v_loose._dist_thresholds = (0.10,)
    v_loose._primary_threshold = 0.10
    v_loose._records = [rec]
    v_loose.nc = 1
    v_loose.config = type("_Cfg", (), {"verbose": False})()
    v_loose.seen = 1

    m_tight = v_tight._compute_metrics()
    m_loose = v_loose._compute_metrics()

    assert m_tight["metrics/recall"] == pytest.approx(0.0)
    assert m_loose["metrics/recall"] == pytest.approx(1.0)
    assert m_tight["metrics/mAP@0.05"] < 1 / 101 + 1e-6
    assert m_loose["metrics/mAP@0.10"] > 0.0


# ===========================================================================
# Coordinate Normalization & Target Parsing
# ===========================================================================

def test_point_validator_parse_gt_yolo_normalised():
    row = np.array(
        [
            [0.0, 0.25, 0.75, 0.10, 0.10],
            [1.0, 0.80, 0.20, 0.05, 0.05],
            [0.0, 0.00, 0.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    v = _make_validator_for_parsing()
    xy, cls = v._parse_gt_points(row, orig_h=640, orig_w=640)

    assert xy.shape == (2, 2)
    np.testing.assert_allclose(xy[0], [0.25, 0.75], atol=1e-6)
    np.testing.assert_allclose(xy[1], [0.80, 0.20], atol=1e-6)
    assert list(cls) == [0, 1]


def test_point_validator_parse_gt_clipped_coords():
    row = np.array([[0.0, 1.2, -0.1, 0.1, 0.1]], dtype=np.float32)
    v = _make_validator_for_parsing()
    xy, cls = v._parse_gt_points(row, orig_h=640, orig_w=640)
    assert xy.shape == (1, 2)
    assert 0.0 <= xy[0, 0] <= 1.0
    assert 0.0 <= xy[0, 1] <= 1.0


def test_point_validator_parse_gt_pixel_scaled_no_letterbox():
    orig_h, orig_w, imgsz = 200, 400, 100
    x1s, y1s, x2s, y2s = 40.0, 40.0, 60.0, 60.0
    row = np.array([[x1s, y1s, x2s, y2s, 0.0]], dtype=np.float32)

    v = _make_validator_for_parsing(imgsz=imgsz)
    xy, cls = v._parse_gt_points(row, orig_h=orig_h, orig_w=orig_w)

    assert xy.shape == (1, 2)
    np.testing.assert_allclose(xy[0], [0.5, 0.5], atol=1e-6)


def test_point_validator_parse_gt_empty_padded_yolo_row():
    row = np.zeros((5, 5), dtype=np.float32)
    v = _make_validator_for_parsing()
    xy, cls = v._parse_gt_points(row, orig_h=640, orig_w=640)
    assert xy.shape == (0, 2)
    assert len(cls) == 0


def test_point_validator_parse_gt_single_valid_with_padding():
    row = np.array(
        [
            [0.0, 0.5, 0.5, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    v = _make_validator_for_parsing()
    xy, cls = v._parse_gt_points(row, orig_h=640, orig_w=640)
    assert xy.shape == (1, 2)
    np.testing.assert_allclose(xy[0], [0.5, 0.5], atol=1e-6)


def test_point_validator_parse_gt_1d_row():
    row = np.array([0.0, 0.3, 0.7, 0.1, 0.1], dtype=np.float32)
    v = _make_validator_for_parsing()
    xy, cls = v._parse_gt_points(row, orig_h=640, orig_w=640)
    assert xy.shape == (1, 2)
    np.testing.assert_allclose(xy[0], [0.3, 0.7], atol=1e-6)
    assert cls[0] == 0


def test_point_validator_parse_gt_letterbox():
    orig_h = orig_w = 200
    imgsz = 100
    r = imgsz / orig_w
    off_x = off_y = 0.0

    v = object.__new__(PointValidator)
    from libreyolo.validation.point_validator import _DEFAULT_DIST_THRESHOLDS
    v._dist_thresholds = _DEFAULT_DIST_THRESHOLDS
    v._primary_threshold = _DEFAULT_DIST_THRESHOLDS[0]
    v._records = []
    v.nc = 1
    v._actual_imgsz = imgsz
    v.config = type("_Cfg", (), {"verbose": False})()
    v.seen = 0

    class _LetterboxPreproc:
        uses_letterbox = True

        def letterbox_scale(self, h, w, sz):
            return r, off_x, off_y

    v.val_preproc = _LetterboxPreproc()

    row = np.array([[40.0, 40.0, 60.0, 60.0, 0.0]], dtype=np.float32)
    xy, cls_ids = v._parse_gt_points(row, orig_h=orig_h, orig_w=orig_w)

    assert xy.shape == (1, 2)
    np.testing.assert_allclose(xy[0], [0.5, 0.5], atol=1e-6)
    assert cls_ids[0] == 0


# ===========================================================================
# Class-Aware Matching
# ===========================================================================

def test_point_validator_class_aware_cross_class_fp():
    pred_xy = np.array([[0.5, 0.5]], np.float64)
    gt_xy   = np.array([[0.5, 0.5]], np.float64)

    rec = _ImageRecord(
        pred_xy=pred_xy,
        pred_scores=np.array([0.9], np.float32),
        pred_classes=np.array([0], np.int64),
        gt_xy=gt_xy,
        gt_classes=np.array([1], np.int64),
    )
    v = _make_validator_with_records([rec])
    m = v._compute_metrics()
    assert m["metrics/precision"] == pytest.approx(0.0)
    assert m["metrics/recall"]    == pytest.approx(0.0)
    assert m["metrics/f1"]        == pytest.approx(0.0)


def test_point_validator_class_aware_two_classes_tp():
    rec = _ImageRecord(
        pred_xy=np.array([[0.1, 0.1], [0.8, 0.8]], np.float64),
        pred_scores=np.array([0.9, 0.8], np.float32),
        pred_classes=np.array([0, 1], np.int64),
        gt_xy=np.array([[0.1, 0.1], [0.8, 0.8]], np.float64),
        gt_classes=np.array([0, 1], np.int64),
    )
    v = _make_validator_with_records([rec])
    m = v._compute_metrics()
    assert m["metrics/precision"] == pytest.approx(1.0)
    assert m["metrics/recall"]    == pytest.approx(1.0)
    assert m["metrics/f1"]        == pytest.approx(1.0)


def test_point_validator_class_aware_same_loc_diff_class():
    xy = np.array([[0.3, 0.7]], np.float64)
    rec = _ImageRecord(
        pred_xy=np.tile(xy, (2, 1)),
        pred_scores=np.array([0.9, 0.8], np.float32),
        pred_classes=np.array([0, 1], np.int64),
        gt_xy=np.tile(xy, (2, 1)),
        gt_classes=np.array([0, 1], np.int64),
    )
    v = _make_validator_with_records([rec])
    m = v._compute_metrics()
    assert m["metrics/precision"] == pytest.approx(1.0)
    assert m["metrics/recall"]    == pytest.approx(1.0)
    assert m["metrics/MLE"]       == pytest.approx(0.0, abs=1e-9)
    assert m["metrics/MAE"]       == pytest.approx(0.0)


def test_point_validator_class_aware_partial_overlap():
    rec = _ImageRecord(
        pred_xy=np.array([[0.1, 0.1], [0.5, 0.5]], np.float64),
        pred_scores=np.array([0.9, 0.8], np.float32),
        pred_classes=np.array([0, 1], np.int64),
        gt_xy=np.array([[0.1, 0.1], [0.9, 0.9]], np.float64),
        gt_classes=np.array([0, 2], np.int64),
    )
    v = _make_validator_with_records([rec])
    m = v._compute_metrics()
    assert m["metrics/precision"] == pytest.approx(1 / 3)
    assert m["metrics/recall"]    == pytest.approx(1 / 3)
    assert m["metrics/f1"]        == pytest.approx(1 / 3)


def test_point_validator_class_aware_map_macro_averaging():
    n = 10
    xy_cls0 = np.column_stack([
        np.linspace(0.1, 0.9, n),
        np.linspace(0.1, 0.9, n),
    ])
    gt_cls1 = np.array([[0.5, 0.5]], np.float64)

    rec = _ImageRecord(
        pred_xy=xy_cls0,
        pred_scores=np.ones(n, np.float32),
        pred_classes=np.zeros(n, np.int64),
        gt_xy=np.vstack([xy_cls0, gt_cls1]),
        gt_classes=np.array([0] * n + [1], np.int64),
    )
    v = _make_validator_with_records([rec])
    m = v._compute_metrics()

    primary_key = f"metrics/mAP@{v._primary_threshold:.2f}"
    assert m[primary_key] == pytest.approx(0.5, abs=1e-6)
    assert m[primary_key] < 0.7


# ===========================================================================
# Smoke & Module Exports Tests
# ===========================================================================

def test_point_validator_task_attribute():
    assert PointValidator.task == "point"


def test_point_validator_in_module_exports():
    from libreyolo.validation import PointValidator as PV

    assert PV is PointValidator


def test_point_validator_default_dist_thresholds_cover_expected_range():
    from libreyolo.validation.point_validator import _DEFAULT_DIST_THRESHOLDS

    assert len(_DEFAULT_DIST_THRESHOLDS) == 10
    assert _DEFAULT_DIST_THRESHOLDS[0] == pytest.approx(0.01, abs=1e-6)
    assert _DEFAULT_DIST_THRESHOLDS[-1] == pytest.approx(0.10, abs=1e-6)
    for a, b in zip(_DEFAULT_DIST_THRESHOLDS, _DEFAULT_DIST_THRESHOLDS[1:]):
        assert a < b


def test_point_validator_update_metrics_numpy():
    v = object.__new__(PointValidator)
    v._records = []
    v.nc = 2
    v._actual_imgsz = 640
    v.val_preproc = type("_FakePreproc", (), {"uses_letterbox": False})()

    preds = [
        {
            "xy_norm": np.array([[0.5, 0.5]], np.float64),
            "scores": np.array([0.9], np.float32),
            "classes": np.array([0], np.int64),
        }
    ]
    targets = np.array([[[0.0, 0.5, 0.5, 0.1, 0.1]]], np.float32)
    img_info = [(640, 640)]

    v._update_metrics(preds, targets, img_info)
    assert len(v._records) == 1
    rec = v._records[0]
    np.testing.assert_allclose(rec.pred_xy, [[0.5, 0.5]])
    np.testing.assert_allclose(rec.gt_xy, [[0.5, 0.5]])
    assert rec.gt_classes[0] == 0


def test_point_validator_update_metrics_torch_tensor():
    import torch

    v = object.__new__(PointValidator)
    v._records = []
    v.nc = 2
    v._actual_imgsz = 640
    v.val_preproc = type("_FakePreproc", (), {"uses_letterbox": False})()

    preds = [
        {
            "xy_norm": np.array([[0.2, 0.8]], np.float64),
            "scores": np.array([0.85], np.float32),
            "classes": np.array([1], np.int64),
        }
    ]
    targets = torch.tensor([[[1.0, 0.2, 0.8, 0.1, 0.1]]], dtype=torch.float32)
    img_info = [(640, 640)]

    v._update_metrics(preds, targets, img_info)
    assert len(v._records) == 1
    rec = v._records[0]
    np.testing.assert_allclose(rec.pred_xy, [[0.2, 0.8]])
    np.testing.assert_allclose(rec.gt_xy, [[0.2, 0.8]])
    assert rec.gt_classes[0] == 1


# ===========================================================================
# End-to-End Integration Flow
# ===========================================================================

def test_point_validator_runs_on_yolo_dataset(tmp_path):
    data_yaml = _write_point_dataset(tmp_path)
    config = ValidationConfig(
        data=str(data_yaml),
        batch_size=1,
        imgsz=64,
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "val_run"),
    )

    validator = PointValidator(_DummyPointModel(), config)
    metrics = validator.run()

    assert metrics["metrics/precision"] == pytest.approx(1.0)
    assert metrics["metrics/recall"] == pytest.approx(1.0)
    assert metrics["metrics/f1"] == pytest.approx(1.0)
    assert metrics["speed/images_seen"] == 1
    assert "speed/preprocess_ms" in metrics
    assert "speed/inference_ms" in metrics
    assert "speed/postprocess_ms" in metrics
    assert "speed/total_ms" in metrics


def test_point_validation_rejects_augmented_validation():
    with pytest.raises(ValueError, match="Augmented validation"):
        BaseModel.val(_DummyPointModel(), data="unused.yaml", imgsz=64, augment=True)


def test_point_validator_honors_explicit_imgsz():
    validator = object.__new__(PointValidator)
    validator.config = ValidationConfig(data="unused.yaml", imgsz=1280)
    validator.model = _DummyPointModel()
    assert validator._resolve_imgsz() == 1280


def test_point_validator_uses_model_imgsz_when_unspecified():
    validator = object.__new__(PointValidator)
    validator.config = ValidationConfig(data="unused.yaml", imgsz=None)
    validator.model = _DummyPointModel()
    assert validator._resolve_imgsz() == 64


def test_point_validator_requires_data_config():
    with pytest.raises(ValueError, match="PointValidator requires data=.* or data_dir="):
        validator = PointValidator(_DummyPointModel(), ValidationConfig(data=None, data_dir=None, keypoints_json="unused.json"))
        validator._setup_dataloader()


def test_point_validator_hungarian_match_threshold_priority():
    """Verify that Hungarian matching prioritizes maximizing valid matches under threshold."""
    from libreyolo.validation.point_validator import _hungarian_match

    dist_mat = np.array([[0.01, 0.04], [0.04, 0.051]], dtype=np.float64)
    tp_pairs, fp_idx, fn_idx = _hungarian_match(dist_mat, threshold=0.05)
    
    assert len(tp_pairs) == 2
    assert set(tuple(p) for p in tp_pairs.tolist()) == {(0, 1), (1, 0)}


def test_point_validator_ap_zero_when_all_predictions_false():
    """Verify that AP is exactly 0.0 when all predictions are false positives."""
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    matched = np.array([False, False, False], dtype=bool)
    ap = _average_precision_at_threshold(scores, matched, n_gt_total=3)
    assert ap == 0.0


def test_point_validator_postprocess_standard_payload_shape():
    """Verify that postprocessing correctly parses standard (N, 4) points payload format."""
    v = object.__new__(PointValidator)
    v.config = type("_Cfg", (), {"conf_thres": 0.25, "iou_thres": 0.45, "max_det": 100})()
    
    class _FakeModel:
        task = "point"
        def _postprocess(self, *args, **kwargs):
            return {
                "points": [[32.0, 16.0, 1.0, 0.95], [48.0, 48.0, 0.0, 0.82]],
            }
    v.model = _FakeModel()
    v._actual_imgsz = 64
    v.val_preproc = type("_FakePreproc", (), {"uses_letterbox": False})()
    
    batch = (None, None, [(64, 64)], None)
    results = v._postprocess_predictions(None, batch)
    
    assert len(results) == 1
    res = results[0]
    np.testing.assert_allclose(res["xy_norm"], [[0.5, 0.25], [0.75, 0.75]])
    assert list(res["classes"]) == [1, 0]
    np.testing.assert_allclose(res["scores"], [0.95, 0.82], atol=1e-6)


def test_point_validator_fp_for_class_without_gt():
    """Verify that prediction-only classes (without ground truth in split) accumulate FPs and are penalized."""
    rec = _ImageRecord(
        pred_xy=np.array([[0.5, 0.5]], np.float64),
        pred_scores=np.array([0.9], np.float32),
        pred_classes=np.array([1], np.int64),
        gt_xy=np.array([[0.5, 0.5]], np.float64),
        gt_classes=np.array([0], np.int64),
    )
    v = _make_validator_with_records([rec])
    v.nc = 2
    m = v._compute_metrics()
    
    assert m["metrics/precision"] == pytest.approx(0.0)
    assert m["metrics/recall"]    == pytest.approx(0.0)
    assert m["metrics/f1"]        == pytest.approx(0.0)


def test_point_validator_parse_gt_delegates_to_model():
    """Verify that _parse_gt_points delegates to the model if it implements it."""
    v = _make_validator_for_parsing()
    
    class _CustomModel:
        def _parse_gt_points(self, gt_row, orig_h, orig_w, validator):
            return np.array([[0.123, 0.456]], np.float64), np.array([42], np.int64)
            
    v.model = _CustomModel()
    xy, cls = v._parse_gt_points(None, orig_h=100, orig_w=100)
    
    np.testing.assert_allclose(xy, [[0.123, 0.456]])
    assert list(cls) == [42]
