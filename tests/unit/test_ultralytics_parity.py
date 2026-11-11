"""Code written against the Ultralytics API does the expected thing.

Covers ``results[0]`` on a single-image Results, ``Results.save/show``,
``val()`` metric names (``box.map`` ...), ``add_callback`` and export format
hints. Behavior was learned from public documentation only.
"""

from __future__ import annotations

import types
import warnings

import numpy as np
import pytest
import torch

from libreyolo.utils import results as results_module
from libreyolo.utils.results import Boxes, Probs, Results
from libreyolo.utils.ultralytics_hints import UltralyticsNameError

pytestmark = pytest.mark.unit


def _detections(n: int = 3) -> Results:
    boxes = torch.tensor([[10.0 * i, 10.0, 10.0 * i + 5, 20.0] for i in range(n)])
    return Results(
        boxes=Boxes(boxes, torch.linspace(0.9, 0.5, n), torch.arange(n).float()),
        orig_shape=(64, 64),
        names={i: f"c{i}" for i in range(n)},
    )


@pytest.fixture
def fresh_warning(monkeypatch):
    """Reset the once-per-process notice so each test sees it."""
    monkeypatch.setattr(results_module, "_image_indexing_warned", False)


# ---------------------------------------------------------------------------
# results[0]
# ---------------------------------------------------------------------------


def test_index_zero_returns_the_whole_image_result(fresh_warning):
    result = _detections(3)
    with pytest.warns(DeprecationWarning, match="whole result"):
        first = result[0]
    assert first is result
    assert first.boxes.xyxy.shape == (3, 4)


def test_ultralytics_style_code_gets_every_detection(fresh_warning):
    result = _detections(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        r = result[0]
        assert len(r.boxes) == 4
        assert result[-1] is result
        seen = [len(r.boxes) for r in result]
    assert seen == [4]


def test_image_indexing_notice_is_shown_once(fresh_warning):
    result = _detections(3)
    with pytest.warns(DeprecationWarning):
        result[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert result[0] is result
        assert list(result) == [result]


def test_no_notice_when_the_meaning_did_not_change(fresh_warning):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        single = _detections(1)
        assert single[0] is single
        probs = Results(boxes=None, orig_shape=(1, 1), probs=Probs(torch.tensor([0.2, 0.8])))
        assert probs[0] is probs
        assert list(probs) == [probs]


def test_iterating_a_dense_result_terminates():
    # Before, iteration went through __getitem__ and never stopped for
    # whole-image payloads, whose indexing is a no-op.
    probs = Results(boxes=None, orig_shape=(1, 1), probs=Probs(torch.tensor([0.2, 0.8])))
    assert len(list(probs)) == 1


def test_slices_masks_and_arrays_still_select_detections():
    result = _detections(4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(result[:1]) == 1
        assert torch.equal(result[:1].boxes.xyxy, result.boxes.xyxy[:1])
        assert len(result[result.boxes.conf > 0.7]) == 2
        assert len(result[torch.tensor([1, 3])]) == 2
        assert len(result[np.array([0, 2])]) == 2
        assert len(result) == 4


@pytest.mark.parametrize("index", [1, 2, np.int64(1)])
def test_other_integers_select_one_detection_with_a_warning(index):
    result = _detections(3)
    with pytest.warns(DeprecationWarning, match=r"results\[1:2\]|results\[2:3\]"):
        picked = result[index]
    assert len(picked) == 1
    assert torch.equal(picked.boxes.xyxy, result.boxes.xyxy[int(index)][None])


def test_bools_are_not_image_indices():
    assert results_module._is_scalar_int(0)
    assert results_module._is_scalar_int(np.int32(0))
    assert not results_module._is_scalar_int(True)
    assert not results_module._is_scalar_int(np.bool_(False))
    assert not results_module._is_scalar_int(torch.tensor(0))


# ---------------------------------------------------------------------------
# Results.save / show
# ---------------------------------------------------------------------------


def _with_image(tmp_path) -> Results:
    from PIL import Image

    image = tmp_path / "scene.jpg"
    Image.new("RGB", (64, 64), "white").save(image)
    result = _detections(2)
    result.path = str(image)
    return result


def test_save_writes_the_annotated_image(tmp_path):
    result = _with_image(tmp_path)
    out = tmp_path / "annotated.jpg"
    assert result.save(filename=str(out)) == str(out)
    assert out.is_file()
    other = tmp_path / "positional.png"
    assert result.save(str(other)) == str(other)
    assert other.is_file()


def test_save_without_a_name_uses_results_prefix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _with_image(tmp_path)
    assert result.save() == "results_scene.jpg"
    assert (tmp_path / "results_scene.jpg").is_file()


def test_save_rejects_two_different_names(tmp_path):
    with pytest.raises(TypeError, match="path or filename"):
        _with_image(tmp_path).save("a.jpg", filename="b.jpg")


def test_show_displays_the_plot(tmp_path, monkeypatch):
    shown = []
    from PIL import Image

    monkeypatch.setattr(Image.Image, "show", lambda self, *a, **k: shown.append(self.size))
    assert _with_image(tmp_path).show() is None
    assert shown == [(64, 64)]


# ---------------------------------------------------------------------------
# val(): metrics.box.map and friends
# ---------------------------------------------------------------------------


def test_detection_metrics_have_ultralytics_names():
    from libreyolo.validation.base import ValidationMetrics

    metrics = ValidationMetrics(
        {
            "metrics/precision": 0.6,
            "metrics/recall": 0.5,
            "metrics/mAP50-95": 0.4,
            "metrics/mAP50": 0.7,
            "metrics/mAP75": 0.45,
            "metrics/mAP50-95(B)": 0.4,
            "metrics/mAP50(B)": 0.7,
        },
        image_metrics={},
        per_class_ap={0: 0.3, 2: 0.5},
        nc=4,
    )
    box = metrics.box
    assert (box.map, box.map50, box.map75, box.mp, box.mr) == (0.4, 0.7, 0.45, 0.6, 0.5)
    np.testing.assert_allclose(box.maps, [0.3, 0.4, 0.5, 0.4])
    assert not hasattr(metrics, "seg")


def test_segmentation_metrics_split_box_and_mask():
    from libreyolo.validation.base import ValidationMetrics

    metrics = ValidationMetrics(
        {
            "metrics/mAP50-95": 0.2,  # mask value, unsuffixed
            "metrics/mAP75": 0.15,
            "metrics/mAP50-95(B)": 0.4,
            "metrics/mAP50(B)": 0.6,
            "metrics/mAP75(B)": 0.35,
            "metrics/mAP50-95(M)": 0.2,
            "metrics/mAP50(M)": 0.5,
            "metrics/mAP75(M)": 0.15,
            "metrics/precision(B)": 0.7,
        },
        image_metrics={},
        per_class_ap={0: 0.4},
        mask_per_class_ap={0: 0.2},
        nc=1,
    )
    assert (metrics.box.map, metrics.box.map50, metrics.box.map75) == (0.4, 0.6, 0.35)
    assert (metrics.seg.map, metrics.seg.map50, metrics.seg.map75) == (0.2, 0.5, 0.15)
    assert metrics.box.mp == 0.7
    assert np.isnan(metrics.seg.mp)  # no (M) precision: never the box value
    np.testing.assert_allclose(metrics.seg.maps, [0.2])


def test_with_image_metrics_passes_per_class_ap_and_class_count():
    from libreyolo.validation.base import with_image_metrics

    validator = types.SimpleNamespace(
        image_metrics={},
        per_class_ap={1: 0.9},
        nc=3,
        confusion_matrix=None,
    )
    metrics = with_image_metrics({"metrics/mAP50-95": 0.6}, validator)
    np.testing.assert_allclose(metrics.box.maps, [0.6, 0.9, 0.6])
    assert dict(metrics) == {"metrics/mAP50-95": 0.6}


def test_per_class_ap_reads_the_accumulated_precision():
    from libreyolo.validation.coco_evaluator import COCOEvaluator

    evaluator = COCOEvaluator.__new__(COCOEvaluator)
    evaluator.label_to_category_id = {0: 1, 1: 3}
    precision = -np.ones((2, 3, 2, 4, 3))  # (T, R, K, A, M)
    precision[:, :, 0, 0, -1] = 0.5
    precision[0, :, 0, 0, -1] = 0.7  # class 0 mean = (0.7 + 0.5) / 2
    # class 1 has no ground truth: stays -1 and is left out
    evaluator._last_coco_eval = types.SimpleNamespace(
        eval={"precision": precision},
        params=types.SimpleNamespace(
            catIds=[1, 3], areaRngLbl=["all", "small", "medium", "large"]
        ),
    )
    assert evaluator.per_class_ap() == {0: pytest.approx(0.6)}
    evaluator._last_coco_eval = None
    assert evaluator.per_class_ap() is None


# ---------------------------------------------------------------------------
# add_callback and export formats
# ---------------------------------------------------------------------------


def test_add_callback_names_train_callbacks():
    from libreyolo import LibreYOLO9

    model = object.__new__(LibreYOLO9)  # no weights needed for attribute access
    with pytest.raises(UltralyticsNameError, match=r"callbacks=\[fn\]"):
        model.add_callback("on_train_epoch_end", print)
    assert hasattr(model, "add_callback") is False
    assert getattr(model, "add_callback", None) is None


@pytest.mark.parametrize(
    "fmt, expected",
    [("saved_model", "tflite"), ("pb", "tflite"), ("tfjs", "onnxruntime-web"), ("edgetpu", "tflite")],
)
def test_unsupported_ultralytics_export_formats_name_an_alternative(fmt, expected):
    from libreyolo.export import BaseExporter

    with pytest.raises(ValueError, match="Must be one of") as info:
        BaseExporter.create(fmt, None)
    assert expected in str(info.value)


def test_engine_is_tensorrt():
    from libreyolo.export import BaseExporter

    assert BaseExporter._aliases["engine"] == "tensorrt"
