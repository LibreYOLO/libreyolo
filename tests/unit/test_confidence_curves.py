"""Per-class precision/recall/F1 across confidence thresholds (#928).

``val()`` exposes them as ``results.box.p_curve`` / ``r_curve`` / ``f1_curve``
over ``results.box.px``, one row per class in ``results.box.ap_class_index``.
They are read from the finished COCO matching at IoU 0.50, the same source as
the best-confidence thresholds.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from libreyolo.validation import COCOEvaluator
from libreyolo.validation.base import BoxImageMetrics, with_image_metrics

pytestmark = pytest.mark.unit

BACKENDS = ["pycocotools", "faster"]
CATEGORIES = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}, {"id": 3, "name": "fox"}]
LABEL_MAP = {0: 1, 1: 2, 2: 3}


@pytest.fixture(autouse=True)
def _no_backend_env_override(monkeypatch):
    monkeypatch.delenv("LIBREYOLO_FASTER_COCO_EVAL", raising=False)


def _gt(ann_id, category_id, x, y, image_id=1, iscrowd=0):
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, 10.0, 10.0],
        "area": 100.0,
        "iscrowd": iscrowd,
    }


def _pred(x, y, score, label):
    return [x, y, x + 10.0, y + 10.0], score, label


def _evaluate(annotations, preds, backend="pycocotools", label_map=LABEL_MAP):
    from pycocotools.coco import COCO

    if backend == "faster":
        pytest.importorskip("faster_coco_eval")
    coco = COCO()
    coco.dataset = {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "file_name": "img.jpg", "width": 200, "height": 200}],
        "annotations": [dict(a) for a in annotations],
        "categories": [dict(c) for c in CATEGORIES],
    }
    coco.createIndex()
    evaluator = COCOEvaluator(
        coco,
        iou_type="bbox",
        label_to_category_id=label_map,
        faster_coco_eval=(backend == "faster"),
    )
    evaluator.update(
        {
            "boxes": [p[0] for p in preds],
            "scores": [p[1] for p in preds],
            "classes": [p[2] for p in preds],
        },
        image_id=1,
    )
    evaluator.compute()
    return evaluator


def _fixture():
    """cat: TP 0.9, TP 0.8, FP 0.3, one cat never found. dog: FP 0.6, TP 0.4."""
    annotations = [
        _gt(1, 1, 0, 0),
        _gt(2, 1, 20, 0),
        _gt(3, 1, 40, 0),
        _gt(4, 2, 0, 50),
    ]
    preds = [
        _pred(0, 0, 0.9, 0),
        _pred(20, 0, 0.8, 0),
        _pred(100, 100, 0.3, 0),
        _pred(100, 150, 0.6, 1),
        _pred(0, 50, 0.4, 1),
    ]
    return annotations, preds


def _at(curves, name, row, conf):
    """Curve value at the grid point nearest to ``conf``."""
    return float(curves[name][row, int(np.abs(curves["px"] - conf).argmin())])


@pytest.mark.parametrize("backend", BACKENDS)
def test_curves_match_hand_computed_values(backend):
    curves = _evaluate(*_fixture(), backend=backend).confidence_curves()

    assert curves["px"].shape == (1000,)
    assert curves["px"][0] == 0.0 and curves["px"][-1] == 1.0
    # fox has no ground truth: no row.
    assert curves["labels"].tolist() == [0, 1]
    assert curves["p"].shape == curves["r"].shape == curves["f1"].shape == (2, 1000)

    # cat, 3 ground truths. At 0.1 all three detections are kept (2 TP, 1 FP).
    assert _at(curves, "p", 0, 0.1) == pytest.approx(2 / 3)
    assert _at(curves, "r", 0, 0.1) == pytest.approx(2 / 3)
    assert _at(curves, "f1", 0, 0.1) == pytest.approx(2 / 3)
    # At 0.5 the false positive is gone.
    assert _at(curves, "p", 0, 0.5) == pytest.approx(1.0)
    assert _at(curves, "r", 0, 0.5) == pytest.approx(2 / 3)
    assert _at(curves, "f1", 0, 0.5) == pytest.approx(0.8)
    # At 0.85 one true positive is left.
    assert _at(curves, "r", 0, 0.85) == pytest.approx(1 / 3)
    assert _at(curves, "f1", 0, 0.85) == pytest.approx(0.5)

    # dog, 1 ground truth. At 0.2: FP 0.6 and TP 0.4 kept.
    assert _at(curves, "p", 1, 0.2) == pytest.approx(0.5)
    assert _at(curves, "r", 1, 0.2) == pytest.approx(1.0)
    # At 0.5 only the false positive is kept.
    assert _at(curves, "p", 1, 0.5) == pytest.approx(0.0)
    assert _at(curves, "r", 1, 0.5) == pytest.approx(0.0)
    assert _at(curves, "f1", 1, 0.5) == pytest.approx(0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_surviving_detection_means_precision_one_recall_zero(backend):
    curves = _evaluate(*_fixture(), backend=backend).confidence_curves()

    assert _at(curves, "p", 0, 0.95) == 1.0
    assert _at(curves, "r", 0, 0.95) == 0.0
    assert _at(curves, "f1", 0, 0.95) == 0.0


def test_curve_shapes_are_monotone_where_they_must_be():
    curves = _evaluate(*_fixture()).confidence_curves()

    # Raising the threshold can only lose detections, so recall never rises.
    assert (np.diff(curves["r"], axis=1) <= 1e-12).all()
    for name in ("p", "r", "f1"):
        assert np.isfinite(curves[name]).all()
        assert (curves[name] >= 0).all() and (curves[name] <= 1).all()


def test_f1_is_the_harmonic_mean_of_the_other_two():
    curves = _evaluate(*_fixture()).confidence_curves()
    p, r = curves["p"], curves["r"]
    expected = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    assert curves["f1"] == pytest.approx(expected)


def test_a_detection_scoring_exactly_the_threshold_is_kept():
    annotations = [_gt(1, 1, 0, 0)]
    curves = _evaluate(annotations, [_pred(0, 0, 1.0, 0)]).confidence_curves()
    # px ends at exactly 1.0; a score of 1.0 still counts there.
    assert curves["r"][0, -1] == 1.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_curve_peak_agrees_with_the_best_conf_sweep(backend):
    evaluator = _evaluate(*_fixture(), backend=backend)
    curves = evaluator.confidence_curves()
    sweep = evaluator.best_conf_thresholds()

    for row, label in enumerate(curves["labels"].tolist()):
        _, best_f1 = sweep["per_class"][label]
        assert curves["f1"][row].max() == pytest.approx(best_f1)


def test_backends_produce_identical_curves():
    pytest.importorskip("faster_coco_eval")
    rng = np.random.default_rng(928)
    annotations, preds = [], []
    for i in range(40):
        category = int(rng.integers(1, 4))
        x, y = float(rng.uniform(0, 180)), float(rng.uniform(0, 180))
        annotations.append(_gt(i + 1, category, x, y, iscrowd=int(i % 13 == 0)))
        if rng.random() < 0.8:
            jitter = float(rng.uniform(-4, 4))
            preds.append(
                _pred(x + jitter, y, float(rng.uniform(0.05, 1)), int(rng.integers(0, 3)))
            )
    for _ in range(25):
        preds.append(
            _pred(
                float(rng.uniform(0, 180)),
                float(rng.uniform(0, 180)),
                float(rng.uniform(0.05, 1)),
                int(rng.integers(0, 3)),
            )
        )
    stock = _evaluate(annotations, preds, backend="pycocotools").confidence_curves()
    faster = _evaluate(annotations, preds, backend="faster").confidence_curves()

    assert stock["labels"].tolist() == faster["labels"].tolist()
    for name in ("px", "p", "r", "f1"):
        assert np.array_equal(stock[name], faster[name]), name


def test_category_ids_are_mapped_back_to_model_labels():
    curves = _evaluate(
        [_gt(1, 3, 0, 0)], [_pred(0, 0, 0.9, 2)]
    ).confidence_curves()
    # COCO category 3 is model label 2.
    assert curves["labels"].tolist() == [2]


def test_no_evaluation_gives_none():
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = {"images": [], "annotations": [], "categories": []}
    coco.createIndex()
    assert COCOEvaluator(coco).confidence_curves() is None


def test_points_argument_sets_the_grid():
    curves = _evaluate(*_fixture()).confidence_curves(points=11)
    assert curves["px"].tolist() == pytest.approx(np.linspace(0, 1, 11).tolist())
    assert curves["f1"].shape == (2, 11)


# ---------------------------------------------------------------------------
# results.box
# ---------------------------------------------------------------------------


def test_box_exposes_the_ecosystem_attribute_names():
    curves = _evaluate(*_fixture()).confidence_curves()
    box = BoxImageMetrics({}, None, curves)

    assert box.px is not None and box.px.shape == (1000,)
    assert box.ap_class_index.tolist() == [0, 1]
    assert box.p_curve.shape == box.r_curve.shape == box.f1_curve.shape == (2, 1000)
    assert np.array_equal(box.f1_curve, curves["f1"])


def test_box_p_r_f1_are_read_at_the_best_mean_f1_confidence():
    curves = _evaluate(*_fixture()).confidence_curves()
    box = BoxImageMetrics({}, None, curves)

    best = int(curves["f1"].mean(axis=0).argmax())
    assert box.p.tolist() == curves["p"][:, best].tolist()
    assert box.r.tolist() == curves["r"][:, best].tolist()
    assert box.f1.tolist() == curves["f1"][:, best].tolist()
    # Mean F1 peaks where cat keeps its 2 TP and dog keeps FP + TP.
    assert box.f1.tolist() == pytest.approx([0.8, 2 / 3])


def test_box_is_empty_without_curves():
    box = BoxImageMetrics({})
    assert box.px.shape == (0,)
    assert box.ap_class_index.shape == (0,)
    assert box.p_curve.shape == (0, 0)
    assert box.p.shape == box.r.shape == box.f1.shape == (0,)


def _validator(evaluator):
    from libreyolo.validation.detection_validator import DetectionValidator

    validator = DetectionValidator.__new__(DetectionValidator)
    validator.config = SimpleNamespace(verbose=False, save_json=False)
    validator.save_dir = None
    validator.coco_evaluator = evaluator
    validator.class_names = ["cat", "dog", "fox"]
    validator.image_metrics = {}
    return validator


def test_detection_validator_puts_curves_on_the_results():
    validator = _validator(_evaluate(*_fixture()))
    metrics = validator._compute_metrics()
    results = with_image_metrics(metrics, validator)

    assert results.box.f1_curve.shape == (2, 1000)
    assert results.box.ap_class_index.tolist() == [0, 1]
    # The curves are attributes, never metric keys: the dict stays flat.
    assert all(isinstance(v, float) for v in results.values())


def test_curves_do_not_change_the_metrics():
    evaluator = _evaluate(*_fixture())
    with_curves = _validator(evaluator)._compute_metrics()

    class _NoCurves:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "confidence_curves":
                raise AttributeError(name)
            return getattr(self._inner, name)

    validator = _validator(_NoCurves(_evaluate(*_fixture())))
    without = validator._compute_metrics()

    assert with_curves == without
    assert validator.confidence_curves is None


def test_f1_plot_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    from libreyolo.validation.val_plotter import ValPlotter

    curves = _evaluate(*_fixture()).confidence_curves()
    path = tmp_path / "plots" / "f1_conf_box.png"
    ValPlotter.plot_f1_curve(curves, ["cat", "dog", "fox"], path)
    assert path.stat().st_size > 0
