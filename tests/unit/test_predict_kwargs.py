"""Tests for predict keyword compatibility policy."""

import pytest

from libreyolo.utils.predict_args import normalize_predict_kwargs

pytestmark = pytest.mark.unit


def test_noop_predict_kwargs_warn_and_are_removed():
    with pytest.warns(UserWarning, match="no-op"):
        remaining = normalize_predict_kwargs({"augment": True})
    assert remaining == {}


@pytest.mark.parametrize(
    "key",
    [
        "agnostic_nms",
        "augment",
        "boxes",
        "dnn",
        "half",
        "line_width",
        "retina_masks",
        "show_conf",
        "show_labels",
        "stream_buffer",
        "verbose",
    ],
)
def test_ultralytics_compat_noop_kwargs_warn_and_are_removed(key):
    with pytest.warns(UserWarning, match="no-op"):
        remaining = normalize_predict_kwargs({key: True})
    assert remaining == {}


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
        ("vid_stride", 1),
    ],
)
def test_supported_predict_kwargs_are_accepted(key, value):
    assert normalize_predict_kwargs({key: value}) == {}


def test_native_passthrough_kwargs_are_forwarded_explicitly():
    assert normalize_predict_kwargs({"num_select": 100}, passthrough={"num_select"}) == {
        "num_select": 100
    }


def test_passthrough_kwargs_are_not_silently_accepted_by_default():
    with pytest.raises(TypeError, match="num_select"):
        normalize_predict_kwargs({"num_select": 100})


def test_unknown_predict_kwargs_fail_clearly():
    with pytest.raises(TypeError, match="Unsupported predict option"):
        normalize_predict_kwargs({"unknown": True})
