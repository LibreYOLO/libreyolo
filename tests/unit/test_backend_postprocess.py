from __future__ import annotations

import numpy as np
import pytest

import libreyolo.backends.base as backend_base
from libreyolo.backends.base import BaseBackend

pytestmark = pytest.mark.unit


class _DummyBackend(BaseBackend):
    def __init__(
        self,
        model_family: str,
        task: str | None = None,
        supported_tasks=("detect",),
        model_size: str | None = None,
    ):
        super().__init__(
            model_path="dummy",
            nb_classes=2,
            device="cpu",
            imgsz=640,
            model_family=model_family,
            model_size=model_size,
            names={0: "class_0", 1: "class_1"},
            task=task,
            supported_tasks=supported_tasks,
        )

    def _run_inference(self, blob: np.ndarray) -> list:
        raise NotImplementedError


def test_dfine_backend_skips_generic_nms():
    backend = _DummyBackend("dfine")

    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int64)

    result = backend._build_result(
        boxes,
        scores,
        classes,
        orig_shape=(10, 10),
        image_path=None,
        iou=0.45,
        classes=None,
        max_det=300,
    )

    assert len(result.boxes) == 2


def test_rfdetr_backend_skips_generic_nms():
    backend = _DummyBackend("rfdetr")

    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int64)

    result = backend._build_result(
        boxes,
        scores,
        classes,
        orig_shape=(10, 10),
        image_path=None,
        iou=0.45,
        classes=None,
        max_det=300,
    )

    assert len(result.boxes) == 2


def test_rfdetr_backend_uses_topk_over_queries_and_classes():
    backend = _DummyBackend("rfdetr")

    boxes = np.array(
        [[[0.5, 0.5, 0.25, 0.25], [0.25, 0.25, 0.1, 0.1]]],
        dtype=np.float32,
    )
    logits = np.array([[[10.0, 9.0], [-10.0, -10.0]]], dtype=np.float32)

    parsed_boxes, scores, classes, masks = backend._parse_rfdetr(
        [boxes, logits],
        orig_w=100,
        orig_h=100,
        conf=0.5,
    )

    assert masks is None
    assert len(parsed_boxes) == 2
    assert classes.tolist() == [0, 1]
    assert scores[0] > scores[1] > 0.5
    np.testing.assert_allclose(parsed_boxes[0], [37.5, 37.5, 62.5, 62.5])
    np.testing.assert_allclose(parsed_boxes[1], [37.5, 37.5, 62.5, 62.5])


def test_rfdetr_seg_backend_uses_variant_num_select():
    backend = _DummyBackend(
        "rfdetr",
        task="segment",
        supported_tasks=("segment",),
        model_size="n",
    )
    num_queries = 150
    boxes = np.tile(
        np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32),
        (1, num_queries, 1),
    )
    logits = np.linspace(10.0, 1.0, num_queries, dtype=np.float32).reshape(
        1, num_queries, 1
    )
    masks = np.ones((1, num_queries, 4, 4), dtype=np.float32)

    parsed_boxes, scores, classes, parsed_masks = backend._parse_rfdetr(
        [boxes, logits, masks],
        orig_w=16,
        orig_h=16,
        conf=0.5,
    )

    assert len(parsed_boxes) == 100
    assert len(scores) == 100
    assert classes.tolist() == [0] * 100
    assert parsed_masks.shape == (100, 16, 16)


def test_rfdetr_seg_backend_uses_detected_size_for_num_select_without_metadata():
    backend = _DummyBackend(
        "rfdetr",
        task="segment",
        supported_tasks=("segment",),
        model_size=None,
    )
    backend.size = "n"
    num_queries = 150
    boxes = np.tile(
        np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32),
        (1, num_queries, 1),
    )
    logits = np.linspace(10.0, 1.0, num_queries, dtype=np.float32).reshape(
        1, num_queries, 1
    )
    masks = np.ones((1, num_queries, 4, 4), dtype=np.float32)

    parsed_boxes, scores, classes, parsed_masks = backend._parse_rfdetr(
        [boxes, logits, masks],
        orig_w=16,
        orig_h=16,
        conf=0.5,
    )

    assert len(parsed_boxes) == 100
    assert len(scores) == 100
    assert classes.tolist() == [0] * 100
    assert parsed_masks.shape == (100, 16, 16)


def test_yolo_backend_still_applies_nms():
    backend = _DummyBackend("yolo9")

    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 0], dtype=np.int64)

    result = backend._build_result(
        boxes,
        scores,
        classes,
        orig_shape=(10, 10),
        image_path=None,
        iou=0.45,
        classes=None,
        max_det=300,
    )

    assert len(result.boxes) == 1


def test_yolo9_backend_parse_uses_letterbox_inverse():
    backend = _DummyBackend("yolo9")
    pred = np.zeros((1, 6, 1), dtype=np.float32)
    pred[0, :4, 0] = [0.0, 0.0, 320.0, 320.0]
    pred[0, 4, 0] = 0.9

    boxes, scores, classes, masks = backend._parse_outputs(
        [pred], 640, (1280, 960), conf=0.25
    )

    assert masks is None
    np.testing.assert_allclose(boxes, [[0.0, 0.0, 640.0, 640.0]])
    np.testing.assert_allclose(scores, [0.9])
    np.testing.assert_array_equal(classes, [0])


def test_yolo9_backend_parse_detection_is_multilabel():
    backend = _DummyBackend("yolo9")
    pred = np.zeros((1, 6, 1), dtype=np.float32)
    pred[0, :4, 0] = [0.0, 0.0, 100.0, 100.0]
    pred[0, 4:, 0] = [0.9, 0.8]

    boxes, scores, classes, masks = backend._parse_outputs(
        [pred], 100, (100, 100), conf=0.25
    )

    assert masks is None
    np.testing.assert_allclose(boxes, [[0.0, 0.0, 100.0, 100.0]] * 2)
    np.testing.assert_allclose(np.sort(scores), [0.8, 0.9])
    np.testing.assert_array_equal(np.sort(classes), [0, 1])


def test_yolo9_backend_parse_caps_multilabel_candidates(monkeypatch):
    monkeypatch.setattr(backend_base, "_YOLO9_MAX_NMS_CANDIDATES", 3)
    backend = _DummyBackend("yolo9")
    pred = np.zeros((1, 6, 4), dtype=np.float32)
    pred[0, :4] = np.array(
        [
            [0.0, 20.0, 40.0, 60.0],
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 30.0, 50.0, 70.0],
            [10.0, 10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    pred[0, 4:] = np.array(
        [[0.1, 0.9, 0.7, 0.5], [0.8, 0.2, 0.6, 0.4]], dtype=np.float32
    )

    boxes, scores, classes, masks = backend._parse_outputs(
        [pred], 80, (80, 80), conf=0.01
    )

    assert masks is None
    assert boxes.shape[0] == 3
    np.testing.assert_allclose(
        np.sort(scores), [0.7, 0.8, 0.9], rtol=0, atol=1e-6
    )
    np.testing.assert_array_equal(classes, [0, 1, 0])


def test_damoyolo_backend_preprocess_uses_stretch_resize():
    from libreyolo.models.damoyolo.utils import preprocess_numpy

    backend = _DummyBackend("damoyolo")
    image = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)

    tensor, _, size, ratio = backend._preprocess(image, 4, "rgb")
    expected, _ = preprocess_numpy(image, 4)

    assert size == (4, 2)
    assert ratio == 1.0
    np.testing.assert_allclose(tensor.numpy()[0], expected)


def test_damoyolo_backend_parse_uses_stretch_inverse():
    backend = _DummyBackend("damoyolo")
    cls_scores = np.array([[[0.9, 0.8]]], dtype=np.float32)
    boxes = np.array([[[10.0, 20.0, 30.0, 40.0]]], dtype=np.float32)

    parsed_boxes, scores, classes, masks = backend._parse_outputs(
        [cls_scores, boxes], 100, (200, 50), conf=0.25
    )

    assert masks is None
    np.testing.assert_allclose(
        parsed_boxes,
        [[20.0, 10.0, 60.0, 20.0], [20.0, 10.0, 60.0, 20.0]],
    )
    np.testing.assert_allclose(scores, [0.9, 0.8])
    np.testing.assert_array_equal(classes, [0, 1])


def test_yolo9_segment_backend_parses_masks():
    backend = _DummyBackend(
        "yolo9", task="segment", supported_tasks=("detect", "segment")
    )

    num_anchors = 4
    num_classes = 2
    num_masks = 32
    pred = np.zeros((1, 4 + num_classes, num_anchors), dtype=np.float32)
    pred[0, :4] = np.array(
        [
            [10, 12, 11, 200],
            [10, 12, 11, 200],
            [50, 60, 55, 240],
            [50, 60, 55, 240],
        ],
        dtype=np.float32,
    )
    pred[0, 4:] = np.array([[0.9, 0.2, 0.95, 0.1], [0.1, 0.8, 0.05, 0.7]])
    proto = np.random.randn(1, num_masks, 16, 16).astype(np.float32)
    coeffs = np.random.randn(1, num_masks, num_anchors).astype(np.float32)

    boxes, scores, classes, masks = backend._parse_outputs(
        [pred, proto, coeffs], 64, (128, 96), conf=0.25
    )

    assert boxes.shape[0] == 4
    assert scores.shape[0] == 4
    assert classes.shape[0] == 4
    assert masks.shape == (4, 96, 128)


def test_backend_call_accepts_device_kwarg(monkeypatch):
    backend = _DummyBackend("yolo9")
    monkeypatch.setattr(backend, "_predict_single", lambda source, **kwargs: "ok")

    assert backend("image.jpg", device="cpu") == "ok"


def test_backend_rejects_unsupported_explicit_task():
    with pytest.raises(ValueError, match="not supported"):
        _DummyBackend("yolo9", task="segment", supported_tasks=("detect",))
