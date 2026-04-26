from __future__ import annotations

import numpy as np
import pytest

from libreyolo.backends.base import BaseBackend

pytestmark = pytest.mark.unit


class _DummyBackend(BaseBackend):
    def __init__(self, model_family: str):
        super().__init__(
            model_path="dummy",
            nb_classes=2,
            device="cpu",
            imgsz=640,
            model_family=model_family,
            names={0: "class_0", 1: "class_1"},
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
