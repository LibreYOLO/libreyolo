"""
Unit tests for the best-confidence-threshold sweep (issue #768 item 3.3).

The sweep reads the per-image match results of a finished COCO evaluation
(IoU 0.50 matching) and reports the confidence threshold that maximizes F1,
per class and micro-averaged globally. Synthetic fixtures with known optima
verify the sweep; a regression test proves existing metric keys and values
are untouched.
"""

import math
from types import SimpleNamespace

import pytest

pytest.importorskip("pycocotools")

NAN_OK = math.isnan

BACKENDS = ["pycocotools", "faster"]


@pytest.fixture(autouse=True)
def _no_backend_env_override(monkeypatch):
    monkeypatch.delenv("LIBREYOLO_FASTER_COCO_EVAL", raising=False)


def _require_backend(backend):
    if backend == "faster":
        pytest.importorskip("faster_coco_eval")


def _make_coco(images, annotations, categories):
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco.createIndex()
    return coco


def _gt(ann_id, image_id, category_id, x, y, w=10.0, h=10.0, iscrowd=0):
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, w, h],
        "area": w * h,
        "iscrowd": iscrowd,
    }


def _pred(x, y, score, label, w=10.0, h=10.0):
    return [x, y, x + w, y + h], score, label


def _run_evaluator(
    annotations, preds, categories, label_map, image_id=1, backend="pycocotools"
):
    """Feed one 200x200 image through COCOEvaluator and return it."""
    from libreyolo.validation import COCOEvaluator

    coco = _make_coco(
        [{"id": image_id, "file_name": "img.jpg", "width": 200, "height": 200}],
        [dict(a) for a in annotations],
        [dict(c) for c in categories],
    )
    evaluator = COCOEvaluator(
        coco,
        iou_type="bbox",
        label_to_category_id=label_map,
        faster_coco_eval=(backend == "faster"),
    )
    boxes, scores, classes = [], [], []
    for box, score, label in preds:
        boxes.append(box)
        scores.append(score)
        classes.append(label)
    evaluator.update(
        {"boxes": boxes, "scores": scores, "classes": classes}, image_id=image_id
    )
    evaluator.compute()
    if backend == "faster":
        assert evaluator.last_backend.startswith("faster-coco-eval")
    return evaluator


_TWO_CLASS_CATEGORIES = [
    {"id": 1, "name": "cat"},
    {"id": 2, "name": "dog"},
]
_TWO_CLASS_LABEL_MAP = {0: 1, 1: 2}


def _two_class_fixture():
    """Known optima: cat best at 0.7 (F1 1.0), dog at 0.4 (F1 0.8).

    Global micro-average pools 6 TP/FP detections over 5 GT boxes and peaks
    at threshold 0.4 with F1 = 10/11.
    """
    annotations = [
        _gt(1, 1, 1, 0, 0),
        _gt(2, 1, 1, 20, 0),
        _gt(3, 1, 1, 40, 0),
        _gt(4, 1, 2, 0, 50),
        _gt(5, 1, 2, 20, 50),
    ]
    preds = [
        _pred(0, 0, 0.9, 0),     # cat TP
        _pred(20, 0, 0.8, 0),    # cat TP
        _pred(40, 0, 0.7, 0),    # cat TP
        _pred(100, 100, 0.3, 0), # cat FP
        _pred(120, 100, 0.2, 0), # cat FP
        _pred(0, 50, 0.6, 1),    # dog TP
        _pred(100, 150, 0.5, 1), # dog FP
        _pred(20, 50, 0.4, 1),   # dog TP
    ]
    return annotations, preds


@pytest.mark.unit
@pytest.mark.parametrize("backend", BACKENDS)
def test_best_conf_sweep_finds_known_optimum_per_class_and_global(backend):
    _require_backend(backend)
    annotations, preds = _two_class_fixture()
    evaluator = _run_evaluator(
        annotations, preds, _TWO_CLASS_CATEGORIES, _TWO_CLASS_LABEL_MAP,
        backend=backend,
    )

    sweep = evaluator.best_conf_thresholds()

    assert sweep is not None
    cat_thr, cat_f1 = sweep["per_class"][0]
    dog_thr, dog_f1 = sweep["per_class"][1]
    assert cat_thr == pytest.approx(0.7)
    assert cat_f1 == pytest.approx(1.0)
    assert dog_thr == pytest.approx(0.4)
    assert dog_f1 == pytest.approx(0.8)
    global_thr, global_f1 = sweep["global"]
    assert global_thr == pytest.approx(0.4)
    assert global_f1 == pytest.approx(10.0 / 11.0)


@pytest.mark.unit
@pytest.mark.parametrize("backend", BACKENDS)
def test_best_conf_tie_group_is_included_or_excluded_as_a_whole(backend):
    _require_backend(backend)
    # Two GT; TP@0.8 alone gives F1 2/3, the 0.5 tie group (TP + FP
    # together) lifts it to 0.8. Cutting inside the tie group is illegal.
    annotations = [_gt(1, 1, 1, 0, 0), _gt(2, 1, 1, 20, 0)]
    preds = [
        _pred(0, 0, 0.8, 0),     # TP
        _pred(20, 0, 0.5, 0),    # TP, tied score
        _pred(100, 100, 0.5, 0), # FP, tied score
    ]
    evaluator = _run_evaluator(
        annotations, preds, [{"id": 1, "name": "cat"}], {0: 1}, backend=backend
    )

    sweep = evaluator.best_conf_thresholds()

    thr, f1 = sweep["per_class"][0]
    assert thr == pytest.approx(0.5)
    assert f1 == pytest.approx(0.8)


@pytest.mark.unit
@pytest.mark.parametrize("backend", BACKENDS)
def test_best_conf_f1_tie_resolves_to_higher_threshold(backend):
    _require_backend(backend)
    # Two GT; F1 is 2/3 both at 0.8 (1 TP, 0 FP) and at 0.2 (2 TP, 2 FP).
    # The sweep must report the higher threshold.
    annotations = [_gt(1, 1, 1, 0, 0), _gt(2, 1, 1, 20, 0)]
    preds = [
        _pred(0, 0, 0.8, 0),     # TP
        _pred(100, 100, 0.6, 0), # FP
        _pred(120, 100, 0.4, 0), # FP
        _pred(20, 0, 0.2, 0),    # TP
    ]
    evaluator = _run_evaluator(
        annotations, preds, [{"id": 1, "name": "cat"}], {0: 1}, backend=backend
    )

    thr, f1 = evaluator.best_conf_thresholds()["per_class"][0]
    assert thr == pytest.approx(0.8)
    assert f1 == pytest.approx(2.0 / 3.0)


@pytest.mark.unit
@pytest.mark.parametrize("backend", BACKENDS)
def test_best_conf_single_prediction(backend):
    _require_backend(backend)
    annotations = [_gt(1, 1, 1, 0, 0)]
    preds = [_pred(0, 0, 0.55, 0)]
    evaluator = _run_evaluator(
        annotations, preds, [{"id": 1, "name": "cat"}], {0: 1}, backend=backend
    )

    sweep = evaluator.best_conf_thresholds()

    assert sweep["per_class"][0] == pytest.approx((0.55, 1.0))
    assert sweep["global"] == pytest.approx((0.55, 1.0))


@pytest.mark.unit
@pytest.mark.parametrize("backend", BACKENDS)
def test_best_conf_all_fp_and_no_prediction_classes_are_nan(backend):
    _require_backend(backend)
    # cat: predictions but every one a false positive (GT exists elsewhere
    # for dog only). dog: ground truth but no predictions. Both NaN; the
    # pooled global sweep never reaches F1 > 0 either.
    annotations = [_gt(1, 1, 2, 0, 50)]
    preds = [
        _pred(100, 100, 0.9, 0),
        _pred(120, 100, 0.4, 0),
    ]
    evaluator = _run_evaluator(
        annotations, preds, _TWO_CLASS_CATEGORIES, _TWO_CLASS_LABEL_MAP,
        backend=backend,
    )

    sweep = evaluator.best_conf_thresholds()

    assert all(NAN_OK(v) for v in sweep["per_class"][0])
    assert all(NAN_OK(v) for v in sweep["per_class"][1])
    assert all(NAN_OK(v) for v in sweep["global"])


@pytest.mark.unit
@pytest.mark.parametrize("backend", BACKENDS)
def test_best_conf_respects_iscrowd_ignore_regions(backend):
    _require_backend(backend)
    # Two normal GT matched at 0.9 and 0.5, plus a detection at 0.7 sitting
    # exactly on a crowd region. The crowd-matched detection must be neither
    # TP nor FP and the crowd GT must not count toward recall, so the sweep
    # still reaches a perfect F1 at threshold 0.5.
    annotations = [
        _gt(1, 1, 1, 0, 0),
        _gt(2, 1, 1, 20, 0),
        _gt(3, 1, 1, 100, 100, w=40.0, h=40.0, iscrowd=1),
    ]
    preds = [
        _pred(0, 0, 0.9, 0),                     # TP
        _pred(100, 100, 0.7, 0, w=40.0, h=40.0), # on the crowd region
        _pred(20, 0, 0.5, 0),                    # TP
    ]
    evaluator = _run_evaluator(
        annotations, preds, [{"id": 1, "name": "cat"}], {0: 1}, backend=backend
    )

    sweep = evaluator.best_conf_thresholds()

    thr, f1 = sweep["per_class"][0]
    assert thr == pytest.approx(0.5)
    assert f1 == pytest.approx(1.0)
    assert sweep["global"] == pytest.approx((0.5, 1.0))


@pytest.mark.unit
def test_best_conf_backend_parity_on_randomized_fixture():
    """The faster-coco-eval fallback must reproduce the pycocotools sweep.

    Randomized boxes with duplicate scores, crowds, misses and false
    positives; both backends must agree on every per-class pair and the
    global pair (NaN included).
    """
    pytest.importorskip("faster_coco_eval")
    import numpy as np

    rng = np.random.default_rng(768)
    annotations = []
    preds = []
    ann_id = 1
    score_pool = np.round(rng.uniform(0.05, 0.95, size=40), 2)  # forces ties
    for cat_label, cat_id in ((0, 1), (1, 2)):
        for i in range(12):
            x = float((i % 6) * 30)
            y = float((i // 6) * 30 + cat_label * 90)
            iscrowd = 1 if i % 5 == 4 else 0
            annotations.append(
                _gt(ann_id, 1, cat_id, x, y, w=20.0, h=20.0, iscrowd=iscrowd)
            )
            ann_id += 1
            roll = rng.uniform()
            if roll < 0.7:  # jittered detection on this GT
                dx, dy = rng.uniform(-6, 6, size=2)
                preds.append(
                    _pred(
                        x + float(dx),
                        y + float(dy),
                        float(rng.choice(score_pool)),
                        cat_label,
                        w=20.0,
                        h=20.0,
                    )
                )
            # some GTs get a second, weaker detection
            if roll > 0.55:
                preds.append(
                    _pred(
                        x + 2.0,
                        y + 2.0,
                        float(rng.choice(score_pool)),
                        cat_label,
                        w=20.0,
                        h=20.0,
                    )
                )
        # pure false positives far from every GT
        for j in range(4):
            preds.append(
                _pred(
                    150.0 + j * 10.0,
                    180.0,
                    float(rng.choice(score_pool)),
                    cat_label,
                    w=8.0,
                    h=8.0,
                )
            )

    sweeps = {}
    for backend in BACKENDS:
        evaluator = _run_evaluator(
            annotations, preds, _TWO_CLASS_CATEGORIES, _TWO_CLASS_LABEL_MAP,
            backend=backend,
        )
        sweeps[backend] = evaluator.best_conf_thresholds()

    reference, fallback = sweeps["pycocotools"], sweeps["faster"]
    assert reference is not None and fallback is not None
    assert set(reference["per_class"]) == set(fallback["per_class"])
    for label, pair in reference["per_class"].items():
        assert fallback["per_class"][label] == pytest.approx(pair, nan_ok=True)
    assert fallback["global"] == pytest.approx(reference["global"], nan_ok=True)
    # The randomized fixture must exercise a real optimum, not the NaN path.
    assert not math.isnan(reference["global"][0])


def _detection_validator(evaluator, class_names):
    from libreyolo.validation.detection_validator import DetectionValidator

    validator = DetectionValidator.__new__(DetectionValidator)
    validator.config = SimpleNamespace(verbose=False, save_json=False)
    validator.save_dir = None
    validator.coco_evaluator = evaluator
    validator.class_names = class_names
    return validator


@pytest.mark.unit
def test_detection_validator_emits_best_conf_keys_with_class_names():
    annotations, preds = _two_class_fixture()
    evaluator = _run_evaluator(
        annotations, preds, _TWO_CLASS_CATEGORIES, _TWO_CLASS_LABEL_MAP
    )
    validator = _detection_validator(evaluator, ["cat", "dog"])

    metrics = validator._compute_metrics()

    assert metrics["metrics/best_conf"] == pytest.approx(0.4)
    assert metrics["metrics/best_conf_f1"] == pytest.approx(10.0 / 11.0)
    # Per-class thresholds live on the result object, not in the flat mapping.
    assert "metrics/best_conf_per_class" not in metrics
    per_class = validator.best_conf_per_class
    assert per_class["cat"] == pytest.approx(0.7)
    assert per_class["dog"] == pytest.approx(0.4)

    from libreyolo.validation.base import with_image_metrics

    validator.image_metrics = {}
    results = with_image_metrics(metrics, validator)
    assert results.box.best_conf_per_class == per_class


@pytest.mark.unit
def test_detection_validator_print_results_handles_best_conf_table():
    annotations, preds = _two_class_fixture()
    evaluator = _run_evaluator(
        annotations, preds, _TWO_CLASS_CATEGORIES, _TWO_CLASS_LABEL_MAP
    )
    validator = _detection_validator(evaluator, ["cat", "dog"])
    validator.seen = 1
    validator.speed = {"total": 0.0}

    metrics = validator._compute_metrics()

    validator._print_results(metrics)  # must not raise


@pytest.mark.unit
def test_detection_validator_best_conf_zero_without_sweep_source():
    """Evaluator stubs without match results degrade to 0.0, never crash."""

    class _DummyEvaluator:
        def compute(self, save_json=None):
            return {
                "precision": 0.201,
                "recall": 0.202,
                "mAP": 0.2,
                "mAP50": 0.21,
                "mAP75": 0.22,
                "mAP_small": 0.23,
                "mAP_medium": 0.24,
                "mAP_large": 0.25,
                "AR1": 0.26,
                "AR10": 0.27,
                "AR100": 0.28,
                "AR_max_det": 0.28,
                "max_det": 100.0,
                "AR_small": 0.29,
                "AR_medium": 0.30,
                "AR_large": 0.31,
            }

    validator = _detection_validator(_DummyEvaluator(), None)

    metrics = validator._compute_metrics()

    # Existing keys keep their exact values.
    assert metrics["metrics/precision"] == pytest.approx(0.201)
    assert metrics["metrics/recall"] == pytest.approx(0.202)
    assert metrics["metrics/mAP50"] == pytest.approx(0.21)
    assert metrics["metrics/mAP50-95"] == pytest.approx(0.2)
    # New keys degrade to finite values.
    assert metrics["metrics/best_conf"] == 0.0
    assert metrics["metrics/best_conf_f1"] == 0.0
    assert validator.best_conf_per_class == {}


@pytest.mark.unit
def test_best_conf_is_additive_existing_metrics_bit_exact():
    """Regression: pre-existing keys and values match stock pycocotools."""
    from pycocotools.cocoeval import COCOeval

    annotations, preds = _two_class_fixture()
    evaluator = _run_evaluator(
        annotations, preds, _TWO_CLASS_CATEGORIES, _TWO_CLASS_LABEL_MAP
    )
    validator = _detection_validator(evaluator, ["cat", "dog"])
    metrics = validator._compute_metrics()

    coco = _make_coco(
        [{"id": 1, "file_name": "img.jpg", "width": 200, "height": 200}],
        [dict(a) for a in annotations],
        [dict(c) for c in _TWO_CLASS_CATEGORIES],
    )
    results = [
        {
            "image_id": 1,
            "category_id": _TWO_CLASS_LABEL_MAP[label],
            "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
            "score": score,
        }
        for box, score, label in preds
    ]
    reference = COCOeval(coco, coco.loadRes(results), "bbox")
    reference.params.imgIds = [1]
    reference.evaluate()
    reference.accumulate()
    reference.summarize()

    assert metrics["metrics/mAP50-95"] == float(reference.stats[0])
    assert metrics["metrics/mAP50"] == float(reference.stats[1])
    assert metrics["metrics/mAP75"] == float(reference.stats[2])
    assert metrics["metrics/AR100"] == float(reference.stats[8])
    expected_existing = {
        "metrics/precision",
        "metrics/recall",
        "metrics/mAP50-95",
        "metrics/mAP50",
        "metrics/mAP75",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/mAP_small",
        "metrics/mAP_medium",
        "metrics/mAP_large",
        "metrics/AR1",
        "metrics/AR10",
        "metrics/AR100",
        "metrics/AR_max_det",
        "metrics/max_det",
        "metrics/AR_small",
        "metrics/AR_medium",
        "metrics/AR_large",
    }
    new_keys = {
        "metrics/best_conf",
        "metrics/best_conf_f1",
    }
    assert set(metrics) == expected_existing | new_keys


def _assert_flat_finite(metrics):
    """The val() contract: a flat mapping of finite real numbers."""
    import json
    import numbers

    for key, value in metrics.items():
        assert isinstance(value, numbers.Real) and not isinstance(value, bool), (
            key,
            value,
        )
        assert math.isfinite(value), (key, value)
    {k: float(v) for k, v in metrics.items()}
    json.dumps(metrics, allow_nan=False)


@pytest.mark.unit
def test_detection_validator_metrics_are_flat_finite_without_any_threshold():
    """All-FP and no-prediction classes: 0.0 thresholds, never NaN."""
    from libreyolo.validation.base import with_image_metrics

    annotations = [_gt(1, 1, 2, 0, 50)]
    preds = [
        _pred(100, 100, 0.9, 0),
        _pred(120, 100, 0.4, 0),
    ]
    evaluator = _run_evaluator(
        annotations, preds, _TWO_CLASS_CATEGORIES, _TWO_CLASS_LABEL_MAP
    )
    validator = _detection_validator(evaluator, ["cat", "dog"])
    validator.image_metrics = {}

    results = with_image_metrics(validator._compute_metrics(), validator)

    _assert_flat_finite(results)
    assert results["metrics/best_conf"] == 0.0
    assert results["metrics/best_conf_f1"] == 0.0
    assert results.box.best_conf_per_class == {"cat": 0.0, "dog": 0.0}
    import json

    json.dumps(results.box.best_conf_per_class, allow_nan=False)
    validator.seen = 1
    validator.speed = {"total": 0.0}
    validator._print_results(results)


@pytest.mark.unit
def test_detection_validator_metrics_are_flat_finite():
    annotations, preds = _two_class_fixture()
    evaluator = _run_evaluator(
        annotations, preds, _TWO_CLASS_CATEGORIES, _TWO_CLASS_LABEL_MAP
    )
    validator = _detection_validator(evaluator, ["cat", "dog"])

    _assert_flat_finite(validator._compute_metrics())


@pytest.mark.unit
def test_segmentation_validator_metrics_are_flat_finite():
    from libreyolo.validation.base import with_image_metrics
    from libreyolo.validation.detection_validator import SegmentationValidator

    annotations, preds = _two_class_fixture()
    validator = SegmentationValidator.__new__(SegmentationValidator)
    validator.config = SimpleNamespace(verbose=False, save_json=False)
    validator.save_dir = None
    validator.class_names = ["cat", "dog"]
    validator.image_metrics = {}
    validator.bbox_evaluator = _run_evaluator(
        annotations, preds, _TWO_CLASS_CATEGORIES, _TWO_CLASS_LABEL_MAP
    )
    validator.mask_evaluator = _run_evaluator(
        annotations, preds, _TWO_CLASS_CATEGORIES, _TWO_CLASS_LABEL_MAP
    )

    results = with_image_metrics(validator._compute_metrics(), validator)

    _assert_flat_finite(results)
    assert isinstance(results.box.best_conf_per_class, dict)


if __name__ == "__main__":
    # Windows spawn-safe entry point for ad-hoc runs.
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
