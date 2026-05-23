from __future__ import annotations

import numpy as np
import pytest

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
