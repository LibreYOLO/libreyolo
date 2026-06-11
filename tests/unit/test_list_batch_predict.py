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


# =============================================================================
# True batched inference — PyTorch pipeline mechanics
# =============================================================================


class _BatchedStubModel:
    """Stub with the batched-predict contract: stackable tensors, dict output.

    Each image's preprocessed tensor is filled with the image's mean pixel
    value, so per-image routing through the batched forward can be asserted
    from the Results scores.
    """

    task = "detect"
    TTA_ENABLED = False
    SUPPORTS_BATCHED_PREDICT = True
    names = {0: "thing"}
    device = torch.device("cpu")

    def __init__(self):
        self.forward_shapes = []

    def _get_input_size(self):
        return 32

    def _preprocess(self, image, color_format="auto", input_size=None):
        pil = ImageLoader.load(image, color_format=color_format)
        marker = float(np.asarray(pil, dtype=np.float32).mean())
        return torch.full((1, 3, 32, 32), marker), pil, pil.size, 1.0

    def _forward(self, tensor):
        self.forward_shapes.append(tuple(tensor.shape))
        return {"predictions": tensor, "obb": False}

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
        assert output["predictions"].shape[0] == 1, "expected a batch-1 slice"
        marker = float(output["predictions"].flatten()[0])
        return {
            "boxes": [[0.0, 0.0, 1.0, 1.0]],
            "scores": [marker],
            "classes": [0],
            "num_detections": 1,
        }


def test_batched_runner_stacks_chunks_and_routes_each_image():
    model = _BatchedStubModel()
    runner = InferenceRunner(model)
    images = [
        np.full((16, 24, 3), 10, dtype=np.uint8),
        np.full((20, 12, 3), 60, dtype=np.uint8),
        np.full((8, 8, 3), 200, dtype=np.uint8),
    ]

    results = runner(images, batch=2)

    # One stacked forward per chunk: 2 + 1.
    assert model.forward_shapes == [(2, 3, 32, 32), (1, 3, 32, 32)]
    # Scores carry each image's marker — order and per-image routing held.
    assert [float(r.boxes.conf[0]) for r in results] == [10.0, 60.0, 200.0]
    # Per-image original sizes survived the batched path.
    assert [r.orig_shape for r in results] == [(16, 24), (20, 12), (8, 8)]


def test_batched_runner_keeps_paths_and_indexed_save_names(tmp_path):
    img_file = tmp_path / "photo.png"
    Image.new("RGB", (10, 10), color=(60, 60, 60)).save(img_file)
    out_dir = tmp_path / "out"
    model = _BatchedStubModel()
    runner = InferenceRunner(model)
    images = [
        np.full((16, 16, 3), 10, dtype=np.uint8),
        str(img_file),
        np.full((16, 16, 3), 200, dtype=np.uint8),
    ]

    results = runner(images, batch=3, save=True, output_path=str(out_dir))

    assert model.forward_shapes == [(3, 3, 32, 32)]
    assert results[0].path is None
    assert results[1].path == str(img_file)
    assert results[2].path is None
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "image0.jpg",
        "image2.jpg",
        "photo.jpg",
    ]


def test_runner_without_batched_flag_stays_sequential():
    class _Counting(_StubModel):
        def __init__(self):
            self.forward_shapes = []

        def _forward(self, tensor):
            self.forward_shapes.append(tuple(tensor.shape))
            return tensor

    model = _Counting()
    runner = InferenceRunner(model)

    runner([np.zeros((8, 8, 3), dtype=np.uint8)] * 3, batch=2)

    assert model.forward_shapes == [(1, 3, 32, 32)] * 3


def test_batched_runner_falls_back_when_tensors_not_stackable():
    class _RaggedStub(_BatchedStubModel):
        def _preprocess(self, image, color_format="auto", input_size=None):
            pil = ImageLoader.load(image, color_format=color_format)
            side = 32 if pil.size[0] >= 16 else 16
            return torch.zeros(1, 3, side, side), pil, pil.size, 1.0

    model = _RaggedStub()
    runner = InferenceRunner(model)
    images = [
        np.zeros((8, 24, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    ]

    results = runner(images, batch=2)

    assert len(results) == 2
    # Ragged shapes → chunk re-ran sequentially, one forward per image.
    assert model.forward_shapes == [(1, 3, 32, 32), (1, 3, 16, 16)]


# =============================================================================
# True batched inference — parity against sequential, real models
# =============================================================================


def _rand_images(seed, sizes):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8) for h, w in sizes]


def _assert_results_parity(sequential, batched):
    assert len(sequential) == len(batched)
    for r_seq, r_bat in zip(sequential, batched):
        assert r_seq.orig_shape == r_bat.orig_shape
        assert len(r_seq) == len(r_bat)
        if r_seq.boxes is not None and len(r_seq) > 0:
            torch.testing.assert_close(
                r_seq.boxes.xyxy, r_bat.boxes.xyxy, rtol=1e-3, atol=1e-3
            )
            torch.testing.assert_close(
                r_seq.boxes.conf, r_bat.boxes.conf, rtol=1e-3, atol=1e-3
            )
            torch.testing.assert_close(
                r_seq.boxes.cls, r_bat.boxes.cls, rtol=0.0, atol=0.0
            )
        if r_seq.keypoints is not None and len(r_seq) > 0:
            torch.testing.assert_close(
                r_seq.keypoints.data, r_bat.keypoints.data, rtol=1e-3, atol=1e-3
            )
        if r_seq.masks is not None and len(r_seq) > 0:
            assert r_seq.masks.data.shape == r_bat.masks.data.shape


@pytest.mark.parametrize("task", ["detect", "segment", "pose"])
def test_yolo9_batched_predict_matches_sequential(task):
    from libreyolo import LibreYOLO9

    torch.manual_seed(0)
    model = LibreYOLO9(None, size="t", task=task, device="cpu")
    model.model.eval()  # weightless init leaves train mode; BN must use running stats
    images = _rand_images(7, [(64, 48), (40, 80), (56, 56)])

    sequential = model(images, conf=0.25, batch=1, imgsz=64)
    batched = model(images, conf=0.25, batch=2, imgsz=64)

    _assert_results_parity(sequential, batched)


def test_yolo9_batched_forward_slices_match_single_forwards():
    from libreyolo import LibreYOLO9
    from libreyolo.postprocess.slicing import slice_batch_outputs

    torch.manual_seed(0)
    model = LibreYOLO9(None, size="t", device="cpu")
    model.model.eval()
    first = torch.rand(1, 3, 64, 64)
    second = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        out_batched = model._forward(torch.cat([first, second]))
        out_first = model._forward(first)
        out_second = model._forward(second)

    torch.testing.assert_close(
        slice_batch_outputs(out_batched, 0)["predictions"],
        out_first["predictions"],
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        slice_batch_outputs(out_batched, 1)["predictions"],
        out_second["predictions"],
        rtol=1e-4,
        atol=1e-5,
    )


def test_deimv2_batched_predict_matches_sequential():
    from libreyolo import LibreDEIMv2

    torch.manual_seed(0)
    model = LibreDEIMv2(None, size="atto", device="cpu")
    model.model.eval()
    images = _rand_images(11, [(48, 64), (64, 48), (32, 32)])

    # conf=0.0 + top-k selection → a fixed number of rows per image, so the
    # comparison always has signal even with random weights.
    sequential = model(images, conf=0.0, batch=1, max_det=20)
    batched = model(images, conf=0.0, batch=2, max_det=20)

    _assert_results_parity(sequential, batched)


def test_picodet_batched_predict_matches_sequential():
    from libreyolo import LibrePICODET

    torch.manual_seed(0)
    model = LibrePICODET(None, size="s", device="cpu")
    model.model.eval()
    images = _rand_images(13, [(48, 64), (40, 40)])

    sequential = model(images, conf=0.001, batch=1, max_det=20)
    batched = model(images, conf=0.001, batch=2, max_det=20)

    _assert_results_parity(sequential, batched)


# =============================================================================
# True batched inference — exported-backend pipeline (BaseBackend)
# =============================================================================


def _batchable_backend():
    backend = _bare_backend()
    backend.imgsz = 32
    backend.task = "detect"
    return backend


def _marker_preprocess(image, imgsz, color_format):
    marker = float(np.asarray(image, dtype=np.float32).mean())
    return (
        torch.full((1, 3, imgsz, imgsz), marker),
        np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
        (imgsz, imgsz),
        1.0,
    )


def _marker_parse_outputs(per_image, imgsz, orig_size, conf, ratio=1.0, iou=0.45, max_det=300):
    marker = float(np.asarray(per_image[0]).ravel()[0])
    return (
        np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        np.array([marker], dtype=np.float32),
        np.array([0], dtype=np.int64),
        None,
    )


def _marker_build_result(
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
    return (image_path, float(max_scores[0]))


def test_backend_predict_batch_runs_single_stacked_inference():
    backend = _batchable_backend()
    blob_shapes = []

    def run_inference(blob):
        blob_shapes.append(blob.shape)
        return [blob[:, :1, :1, :1].reshape(blob.shape[0], 1)]

    backend._preprocess = _marker_preprocess
    backend._run_inference = run_inference
    backend._parse_outputs = _marker_parse_outputs
    backend._build_result = _marker_build_result
    backend._supports_batched_inference = lambda: True

    images = [np.full((8, 8, 3), v, dtype=np.uint8) for v in (10, 60, 200)]
    results = BaseBackend._process_in_batches(backend, images, batch=2)

    assert blob_shapes == [(2, 3, 32, 32), (1, 3, 32, 32)]
    assert results == [(None, 10.0), (None, 60.0), (None, 200.0)]


def test_backend_predict_batch_falls_back_when_runtime_rejects_batch():
    backend = _batchable_backend()
    blob_shapes = []

    def run_inference(blob):
        blob_shapes.append(blob.shape)
        if blob.shape[0] != 1:
            raise RuntimeError("static batch-1 graph")
        return [blob[:, :1, :1, :1].reshape(blob.shape[0], 1)]

    backend._preprocess = _marker_preprocess
    backend._run_inference = run_inference
    backend._parse_outputs = _marker_parse_outputs
    backend._build_result = _marker_build_result
    backend._supports_batched_inference = lambda: True

    images = [np.full((8, 8, 3), v, dtype=np.uint8) for v in (10, 60, 200)]
    results = BaseBackend._process_in_batches(backend, images, batch=2)

    # Chunk of 2 fails batched, re-runs per image; final chunk of 1 succeeds.
    assert blob_shapes == [(2, 3, 32, 32), (1, 3, 32, 32), (1, 3, 32, 32), (1, 3, 32, 32)]
    assert [marker for _, marker in results] == [10.0, 60.0, 200.0]


def test_backend_without_batch_support_stays_sequential():
    backend = _batchable_backend()
    blob_shapes = []

    def run_inference(blob):
        blob_shapes.append(blob.shape)
        return [blob[:, :1, :1, :1].reshape(blob.shape[0], 1)]

    backend._preprocess = _marker_preprocess
    backend._run_inference = run_inference
    backend._parse_outputs = _marker_parse_outputs
    backend._build_result = _marker_build_result

    images = [np.full((8, 8, 3), v, dtype=np.uint8) for v in (10, 60)]
    results = BaseBackend._process_in_batches(backend, images, batch=2)

    assert blob_shapes == [(1, 3, 32, 32), (1, 3, 32, 32)]
    assert len(results) == 2


def test_onnx_backend_batched_list_predict_matches_sequential(tmp_path):
    """End to end: dynamic ONNX export of a real model, list predict, batch>1."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from libreyolo import LibreDEIMv2
    from libreyolo.backends.onnx import OnnxBackend

    torch.manual_seed(0)
    model = LibreDEIMv2(None, size="atto", device="cpu")
    out_path = tmp_path / "LibreDEIMv2_atto.onnx"
    model.export("onnx", output_path=str(out_path), simplify=False, dynamic=True)

    backend = OnnxBackend(str(out_path), device="cpu")
    assert backend._supports_batched_inference()

    images = _rand_images(17, [(48, 64), (64, 48), (32, 40)])
    sequential = backend(images, conf=0.0, batch=1, max_det=20)
    batched = backend(images, conf=0.0, batch=2, max_det=20)

    _assert_results_parity(sequential, batched)


def test_onnx_supports_batched_inference_flag():
    from libreyolo.backends.onnx import OnnxBackend

    backend = OnnxBackend.__new__(OnnxBackend)
    backend._dynamic_batch_axis = True
    backend.embedded_nms = False
    assert backend._supports_batched_inference()

    backend.embedded_nms = True
    assert not backend._supports_batched_inference()

    backend.embedded_nms = False
    backend._dynamic_batch_axis = False
    assert not backend._supports_batched_inference()


def test_ncnn_run_inference_rejects_batched_blob():
    from libreyolo.backends.ncnn import NcnnBackend

    backend = NcnnBackend.__new__(NcnnBackend)
    with pytest.raises(ValueError, match="one image per forward pass"):
        backend._run_inference(np.zeros((2, 3, 8, 8), dtype=np.float32))
