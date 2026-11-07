from __future__ import annotations

import pytest
import numpy as np
import torch

from libreyolo.backends.tensorrt import TensorRTBackend


pytestmark = pytest.mark.unit


def _bare_tensorrt_backend(
    path: str,
    model_family: str | None = None,
    task: str = "detect",
):
    backend = TensorRTBackend.__new__(TensorRTBackend)
    backend.model_path = path
    backend.model_family = model_family
    backend.task = task
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
def test_tensorrt_sidecarless_size_detection_avoids_family_letters(path, expected_size):
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


def test_tensorrt_backend_reads_static_input_imgsz():
    assert TensorRTBackend._read_static_input_imgsz((1, 3, 64, 64)) == 64
    assert TensorRTBackend._read_static_input_imgsz((1, 3, 32, 64)) == (32, 64)
    assert TensorRTBackend._read_static_input_imgsz((-1, 3, -1, -1)) is None


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

    def parse_outputs(
        per_image, imgsz, orig_size, conf, ratio=1.0, iou=0.45, max_det=300
    ):
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
        obb=None,
        keypoints=None,
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


def test_tensorrt_dynamic_batching_preserves_pose_keypoints():
    backend = _bare_tensorrt_backend(
        "LibreYOLO9t-pose.engine", model_family="yolo9", task="pose"
    )
    backend._dynamic_batch = True
    backend._max_batch = 2
    backend.imgsz = 64
    backend.output_names = ["detections", "keypoints"]
    captured = {}

    def preprocess(path, imgsz, color_format):
        return (
            torch.zeros(1, 3, imgsz, imgsz),
            np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
            (imgsz, imgsz),
        )

    def infer(batched_input):
        return {
            "detections": np.zeros((batched_input.shape[0], 1, 5), dtype=np.float32),
            "keypoints": np.ones((batched_input.shape[0], 1, 2, 3), dtype=np.float32),
        }

    def parse_outputs(
        per_image, imgsz, orig_size, conf, ratio=1.0, iou=0.45, max_det=300
    ):
        return (
            np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
            np.array([0.9], dtype=np.float32),
            np.array([0], dtype=np.int64),
            None,
            None,
            np.ones((1, 2, 3), dtype=np.float32),
        )

    def build_result(
        boxes,
        max_scores,
        class_ids,
        *,
        masks,
        obb=None,
        keypoints=None,
        orig_shape,
        image_path,
        iou,
        classes,
        max_det,
    ):
        captured[image_path] = keypoints
        return image_path

    backend._preprocess = preprocess
    backend._infer = infer
    backend._parse_outputs = parse_outputs
    backend._build_result = build_result

    results = backend._process_in_batches(["a.jpg", "b.jpg"], batch=2)

    assert results == ["a.jpg", "b.jpg"]
    assert captured["a.jpg"].shape == (1, 2, 3)
    assert captured["b.jpg"].shape == (1, 2, 3)


@pytest.mark.parametrize(
    ("task", "builder_name"),
    [
        ("restore", "_build_restore_result"),
        ("matte", "_build_matte_result"),
        ("gaze", "_build_gaze_result"),
        ("semantic", "_build_semantic_result"),
        ("point", "_build_point_result"),
    ],
)
def test_tensorrt_dynamic_batching_routes_dense_tasks_to_task_builder(
    task, builder_name
):
    backend = _bare_tensorrt_backend(
        f"LibreModel-{task}.engine",
        model_family="fomo" if task == "point" else "test",
        task=task,
    )
    backend._dynamic_batch = True
    backend._max_batch = 2
    backend.imgsz = 64
    backend.output_names = ["output"]
    calls = []

    def preprocess(path, imgsz, color_format):
        return (
            torch.zeros(1, 3, imgsz, imgsz),
            np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
            (imgsz, imgsz),
            1.0,
        )

    def infer(batched_input):
        return {"output": np.zeros((batched_input.shape[0], 1, 8, 8), dtype=np.float32)}

    def build_result(per_image, **kwargs):
        calls.append((per_image[0].shape, kwargs))
        return kwargs["image_path"]

    backend._preprocess = preprocess
    backend._infer = infer
    backend._parse_outputs = lambda *args, **kwargs: pytest.fail(
        f"{task} must not use the detection parser"
    )
    setattr(backend, builder_name, build_result)

    results = backend._process_in_batches(["a.jpg", "b.jpg"], batch=2)

    assert results == ["a.jpg", "b.jpg"]
    assert [shape for shape, _ in calls] == [(1, 1, 8, 8), (1, 1, 8, 8)]


def test_tensorrt_dynamic_batching_rejects_rectangular_non_yolo9_imgsz():
    backend = _bare_tensorrt_backend("LibreRFDETR_s.engine", model_family="rfdetr")
    backend._dynamic_batch = True
    backend._max_batch = 2
    backend.imgsz = 64

    with pytest.raises(NotImplementedError, match="YOLO9-family"):
        backend._process_in_batches(["a.jpg", "b.jpg"], batch=2, imgsz=(32, 64))


def test_missing_tensorrt_names_python_314(monkeypatch):
    # The tensorrt extra's marker skips tensorrt-cu12 on 3.14 (no bindings),
    # so the error must say why instead of repeating an install command.
    import sys

    import libreyolo.export.tensorrt as trt_export

    monkeypatch.setitem(sys.modules, "tensorrt", None)
    monkeypatch.setattr(trt_export.sys, "version_info", (3, 14, 0))
    for call in (
        trt_export.check_tensorrt_available,
        trt_export.get_calibrator_class,
        lambda: TensorRTBackend("model.engine"),
    ):
        with pytest.raises(ImportError, match="Use Python 3.13 or earlier"):
            call()


def test_missing_tensorrt_keeps_install_hint_on_supported_python(monkeypatch):
    import sys

    import libreyolo.export.tensorrt as trt_export

    monkeypatch.setitem(sys.modules, "tensorrt", None)
    monkeypatch.setattr(trt_export.sys, "version_info", (3, 12, 0))
    with pytest.raises(ImportError, match="pip install tensorrt"):
        trt_export.check_tensorrt_available()
    with pytest.raises(ImportError, match="pip install tensorrt"):
        TensorRTBackend("model.engine")
