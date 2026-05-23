from __future__ import annotations

import pytest
import numpy as np
import torch

from libreyolo.backends.tensorrt import TensorRTBackend


pytestmark = pytest.mark.unit


def _bare_tensorrt_backend(path: str, model_family: str | None = None):
    backend = TensorRTBackend.__new__(TensorRTBackend)
    backend.model_path = path
    backend.model_family = model_family
    backend._sidecar_size = None
    backend.output_names = ["pred_logits", "pred_boxes"]
    backend.output_shapes = {
        "pred_logits": (1, 300, 80),
        "pred_boxes": (1, 300, 4),
    }
    return backend


@pytest.mark.parametrize(
    ("path", "family"),
    [
        ("LibreDEIMv2_s.engine", "deimv2"),
        ("LibreEC_s.engine", "ec"),
        ("LibreDFINE_s.engine", "dfine"),
        ("LibreDEIM_s.engine", "deim"),
        ("LibreRTDETR_s.engine", "rtdetr"),
        ("LibreRFDETR_s.engine", "rfdetr"),
    ],
)
def test_tensorrt_sidecarless_detr_family_detection_uses_filename(path, family):
    backend = _bare_tensorrt_backend(path)

    assert backend._detect_model_family() == family


def test_tensorrt_backend_reports_deimv2_model_name():
    backend = _bare_tensorrt_backend("LibreDEIMv2_s.engine", model_family="deimv2")

    assert backend._get_model_name() == "deimv2"


@pytest.mark.parametrize(
    ("path", "expected_size"),
    [
        ("LibreRFDETR_s.engine", "s"),
        ("LibreRFDETRs.engine", "s"),
        ("rfdetr_n_seg.engine", "n"),
        ("rf-detr-seg-xl.engine", "x"),
        ("rf-detr-seg-2xl.engine", "xx"),
        ("LibreDEIMv2_s.engine", "s"),
    ],
)
def test_tensorrt_sidecarless_size_detection_avoids_family_letters(
    path, expected_size
):
    backend = _bare_tensorrt_backend(path, model_family="rfdetr")

    assert backend.size == expected_size


@pytest.mark.parametrize(
    "path",
    [
        "rfdetr_n_seg.engine",
        "LibreRFDETRn-seg.engine",
        "LibreRFDETR_seg_n.engine",
    ],
)
def test_tensorrt_sidecarless_rfdetr_seg_task_detection(path):
    backend = _bare_tensorrt_backend(path, model_family="rfdetr")

    assert backend._detect_task_from_filename() == "segment"


def test_tensorrt_dynamic_max_batch_uses_engine_profile():
    class _Engine:
        def get_tensor_profile_shape(self, name, profile_index):
            assert name == "input"
            assert profile_index == 0
            return (
                (1, 3, 64, 64),
                (4, 3, 64, 64),
                (16, 3, 64, 64),
            )

    backend = _bare_tensorrt_backend("LibreRFDETR_s.engine", model_family="rfdetr")
    backend._dynamic_batch = True
    backend.engine = _Engine()
    backend.input_name = "input"
    backend._metadata = {}

    assert backend._detect_max_batch() == 16


def test_tensorrt_dynamic_max_batch_falls_back_to_metadata():
    class _Engine:
        pass

    backend = _bare_tensorrt_backend("LibreRFDETR_s.engine", model_family="rfdetr")
    backend._dynamic_batch = True
    backend.engine = _Engine()
    backend.input_name = "input"
    backend._metadata = {"trt_max_batch": "12"}

    assert backend._detect_max_batch() == 12


def test_tensorrt_dynamic_batching_caps_requested_batch_to_profile():
    backend = _bare_tensorrt_backend("LibreRFDETR_s.engine", model_family="rfdetr")
    backend._dynamic_batch = True
    backend._max_batch = 2
    backend.imgsz = 64
    backend.output_names = ["dets", "labels"]
    infer_batches = []

    def preprocess(path, imgsz, color_format):
        return (
            torch.zeros(1, 3, imgsz, imgsz),
            np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
            (imgsz, imgsz),
        )

    def infer(batched_input):
        infer_batches.append(batched_input.shape[0])
        return {
            "dets": np.zeros((batched_input.shape[0], 1, 4), dtype=np.float32),
            "labels": np.zeros((batched_input.shape[0], 1, 2), dtype=np.float32),
        }

    def parse_outputs(per_image, imgsz, orig_size, conf, ratio=1.0):
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            None,
        )

    def build_result(
        boxes,
        max_scores,
        class_ids,
        *,
        masks,
        orig_shape,
        image_path,
        iou,
        classes,
        max_det,
    ):
        return image_path

    backend._preprocess = preprocess
    backend._infer = infer
    backend._parse_outputs = parse_outputs
    backend._build_result = build_result

    results = backend._process_in_batches(
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"],
        batch=8,
    )

    assert infer_batches == [2, 2, 1]
    assert results == ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]
