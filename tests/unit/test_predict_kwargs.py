"""Tests for predict keyword compatibility policy."""

import pytest

from libreyolo.utils.predict_args import (
    check_agnostic_nms,
    normalize_classes,
    normalize_predict_kwargs,
    postprocess_max_det,
)

pytestmark = pytest.mark.unit


def test_noop_predict_kwargs_warn_and_are_removed():
    with pytest.warns(UserWarning, match="no-op"):
        remaining = normalize_predict_kwargs({"boxes": True})
    assert remaining == {}


@pytest.mark.parametrize(
    "key",
    [
        "boxes",
        "dnn",
        "half",
        "line_width",
        "retina_masks",
        "show_conf",
        "show_labels",
        "verbose",
    ],
)
def test_supported_noop_predict_kwargs_warn_and_are_removed(key):
    with pytest.warns(UserWarning, match="no-op"):
        remaining = normalize_predict_kwargs({key: True})
    assert remaining == {}


def test_agnostic_nms_off_is_dropped_silently():
    """A path without agnostic NMS still accepts the option switched off."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert normalize_predict_kwargs({"agnostic_nms": False}) == {}


def test_agnostic_nms_on_fails_where_it_is_not_implemented():
    """It is never accepted and ignored (#928)."""
    with pytest.raises(NotImplementedError, match="agnostic_nms"):
        normalize_predict_kwargs({"agnostic_nms": True})


@pytest.mark.parametrize("task", ["detect", "segment", "pose", "obb"])
def test_check_agnostic_nms_accepts_box_tasks(task):
    assert check_agnostic_nms(True, task) is True
    assert check_agnostic_nms(False, task) is False


@pytest.mark.parametrize("task", ["classify", "semantic", "depth", None])
def test_check_agnostic_nms_rejects_tasks_without_boxes(task):
    assert check_agnostic_nms(False, task) is False
    with pytest.raises(ValueError, match="agnostic_nms"):
        check_agnostic_nms(True, task)


def test_rejected_predict_kwargs_fail_clearly():
    with pytest.raises(NotImplementedError, match="visualize"):
        normalize_predict_kwargs({"visualize": True})


@pytest.mark.parametrize(
    "key,value",
    [
        ("classes", [0]),
        ("conf", 0.25),
        ("device", "cpu"),
        ("imgsz", 640),
        ("iou", 0.45),
        ("max_det", 300),
        ("save", False),
        ("stream", False),
        ("stream_buffer", False),
        ("vid_stride", 1),
    ],
)
def test_supported_predict_kwargs_are_accepted(key, value):
    assert normalize_predict_kwargs({key: value}) == {}


def test_native_passthrough_kwargs_are_forwarded_explicitly():
    assert normalize_predict_kwargs(
        {"num_select": 100}, passthrough={"num_select"}
    ) == {"num_select": 100}


def test_passthrough_kwargs_are_not_silently_accepted_by_default():
    with pytest.raises(TypeError, match="num_select"):
        normalize_predict_kwargs({"num_select": 100})


def test_unknown_predict_kwargs_fail_clearly():
    with pytest.raises(TypeError, match="Unsupported predict option"):
        normalize_predict_kwargs({"unknown": True})


@pytest.mark.parametrize(
    "classes, expected",
    [(0, [0]), (3, [3]), ([0, 2], [0, 2]), ((1,), (1,)), (None, None)],
)
def test_normalize_classes_accepts_a_single_int(classes, expected):
    assert normalize_classes(classes) == expected


def test_postprocess_max_det_widens_only_with_a_class_filter():
    assert postprocess_max_det(1, None) == 1
    assert postprocess_max_det(1, [16]) == 300
    assert postprocess_max_det(1000, [16]) == 1000

