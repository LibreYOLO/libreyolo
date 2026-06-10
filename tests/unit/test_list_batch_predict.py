"""Tests for in-memory batch prediction via list/tuple sources (issue #384)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.backends.base import BaseBackend
from libreyolo.backends.tensorrt import TensorRTBackend
from libreyolo.models.base.inference import InferenceRunner
from libreyolo.utils.image_loader import ImageLoader


pytestmark = pytest.mark.unit


class _StubModel:
    """Minimal stand-in for a BaseModel as seen by InferenceRunner."""

    task = "detect"
    TTA_ENABLED = False
    size = "n"
    names = {0: "thing"}
    device = torch.device("cpu")

    def _get_input_size(self):
        return 32

    def _get_model_name(self):
        return "stub"

    def _preprocess(self, image, color_format="auto", input_size=None):
        assert not isinstance(image, (list, tuple))
        pil = ImageLoader.load(image, color_format=color_format)
        return torch.zeros(1, 3, 32, 32), pil, pil.size, 1.0

    def _forward(self, tensor):
        return tensor

    def _postprocess(
        self,
        output,
        conf,
        iou,
        original_size,
        max_det=300,
        ratio=1.0,
        classes=None,
        **kwargs,
    ):
        return {
            "boxes": [[1.0, 1.0, 5.0, 5.0]],
            "scores": [0.9],
            "classes": [0],
            "num_detections": 1,
        }


# =============================================================================
# PyTorch pipeline (InferenceRunner)
# =============================================================================


def test_runner_accepts_list_of_in_memory_images():
    runner = InferenceRunner(_StubModel())
    images = [
        np.zeros((16, 20, 3), dtype=np.uint8),
        np.zeros((16, 20, 3), dtype=np.uint8),
    ]

    results = runner(images)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(r.path is None for r in results)
    assert all(len(r) == 1 for r in results)


def test_runner_accepts_tuple_and_empty_list():
    runner = InferenceRunner(_StubModel())

    assert runner([]) == []

    results = runner((np.zeros((8, 8, 3), dtype=np.uint8),))
    assert len(results) == 1


def test_runner_list_mixes_paths_and_in_memory_images(tmp_path):
    img_file = tmp_path / "photo.png"
    Image.new("RGB", (10, 10)).save(img_file)
    runner = InferenceRunner(_StubModel())

    results = runner([str(img_file), np.zeros((8, 8, 3), dtype=np.uint8)])

    assert results[0].path == str(img_file)
    assert results[1].path is None


def test_runner_list_save_uses_indexed_filenames(tmp_path):
    runner = InferenceRunner(_StubModel())
    out_dir = tmp_path / "out"
    images = [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.uint8),
    ]

    runner(images, save=True, output_path=str(out_dir))

    assert sorted(p.name for p in out_dir.iterdir()) == ["image0.jpg", "image1.jpg"]


# =============================================================================
# Exported-backend pipeline (BaseBackend and TensorRT batching)
# =============================================================================


def _bare_backend():
    backend = TensorRTBackend.__new__(TensorRTBackend)
    backend.model_path = "LibreRFDETR_s.engine"
    backend.model_family = "rfdetr"
    backend._sidecar_size = None
    backend.device = "cpu"
    return backend


def test_backend_call_routes_list_to_process_in_batches():
    backend = _bare_backend()
    seen = {}

    def fake_process(images, **kwargs):
        seen["images"] = images
        seen["batch"] = kwargs.get("batch")
        return ["r"] * len(images)

    backend._process_in_batches = fake_process

    out = backend([np.zeros((8, 8, 3), dtype=np.uint8)] * 3, batch=2)

    assert out == ["r", "r", "r"]
    assert seen["batch"] == 2
    assert len(seen["images"]) == 3


def test_backend_sequential_batches_pass_indexed_save_stems():
    backend = _bare_backend()
    stems = []

    def fake_single(image, **kwargs):
        stems.append(kwargs["save_stem"])
        return "r"

    backend._predict_single = fake_single

    BaseBackend._process_in_batches(
        backend, ["a.jpg", np.zeros((4, 4, 3), dtype=np.uint8)]
    )

    assert stems == [None, "image1"]


def test_tensorrt_batched_in_memory_images_keep_path_none_and_indexed_saves():
    backend = _bare_backend()
    backend._dynamic_batch = True
    backend._max_batch = 2
    backend.imgsz = 64
    backend.output_names = ["dets", "labels"]

    def preprocess(image, imgsz, color_format):
        assert isinstance(image, np.ndarray)
        return (
            torch.zeros(1, 3, imgsz, imgsz),
            np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
            (imgsz, imgsz),
        )

    def infer(batched_input):
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

    built_paths = []

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
        built_paths.append(image_path)
        return image_path

    saved_names = []

    def save_annotated(result, orig_img, image_path, output_path):
        saved_names.append(image_path)

    backend._preprocess = preprocess
    backend._infer = infer
    backend._parse_outputs = parse_outputs
    backend._build_result = build_result
    backend._save_annotated = save_annotated

    images = [np.zeros((8, 8, 3), dtype=np.uint8)] * 3
    results = backend._process_in_batches(images, batch=2, save=True)

    assert len(results) == 3
    assert built_paths == [None, None, None]
    assert saved_names == ["image0", "image1", "image2"]
