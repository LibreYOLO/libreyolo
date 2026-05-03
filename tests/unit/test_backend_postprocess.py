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
    ):
        super().__init__(
            model_path="dummy",
            nb_classes=2,
            device="cpu",
            imgsz=640,
            model_family=model_family,
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


def test_backend_call_accepts_device_kwarg(monkeypatch):
    backend = _DummyBackend("yolo9")
    monkeypatch.setattr(backend, "_predict_single", lambda source, **kwargs: "ok")

    assert backend("image.jpg", device="cpu") == "ok"


def test_backend_rejects_unsupported_explicit_task():
    with pytest.raises(ValueError, match="not supported"):
        _DummyBackend("yolo9", task="segment", supported_tasks=("detect",))
