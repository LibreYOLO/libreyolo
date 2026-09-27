"""Focused unit coverage for the opaque LibreLaMa inpainting family."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo.models.lama import LibreLaMa
from libreyolo.models.lama import nn as lama_nn
from libreyolo.models.lama.nn import OpaqueLaMaONNX
from libreyolo.models.lama.utils import preprocess_image_and_mask
from libreyolo.models.lama.validator import LaMaValidationDataset
from libreyolo.postprocess.lama import postprocess
from libreyolo.utils.serialization import (
    load_untrusted_torch_file,
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)


pytestmark = [pytest.mark.unit, pytest.mark.lama]


def _patch_tiny_official_graph(monkeypatch, payload: bytes = b"lama") -> torch.Tensor:
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(lama_nn, "OFFICIAL_ONNX_SIZE_BYTES", len(payload))
    monkeypatch.setattr(lama_nn, "OFFICIAL_ONNX_SHA256", digest)
    return torch.tensor(list(payload), dtype=torch.uint8)


def _wrapped(graph: torch.Tensor) -> dict:
    return wrap_libreyolo_checkpoint(
        {"onnx_graph": graph},
        model_family="lama",
        size="b",
        task="restore",
        nc=1,
        names={0: "image"},
        imgsz=512,
        source_sha256=lama_nn.OFFICIAL_ONNX_SHA256,
    )


def test_opaque_graph_is_a_safe_persistent_cpu_buffer(monkeypatch):
    graph = _patch_tiny_official_graph(monkeypatch)
    runtime = OpaqueLaMaONNX(graph)

    assert runtime.state_dict()["onnx_graph"].dtype == torch.uint8
    assert torch.equal(runtime.state_dict()["onnx_graph"], graph)
    runtime.to(dtype=torch.float16)
    assert runtime.onnx_graph.device.type == "cpu"
    assert runtime.onnx_graph.dtype == torch.uint8


def test_missing_runtime_points_to_the_onnx_extra(monkeypatch):
    graph = _patch_tiny_official_graph(monkeypatch)
    runtime = OpaqueLaMaONNX(graph)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(ImportError, match=r'pip install "libreyolo\[onnx\]"'):
        runtime._get_session("cpu")


def _install_recording_onnxruntime(monkeypatch):
    calls = []

    class _SessionOptions:
        log_severity_level = 0

    class _RecordingSession:
        def __init__(self, serialized, *, sess_options, providers):
            calls.append(
                {
                    "serialized": serialized,
                    "sess_options": sess_options,
                    "providers": providers,
                    "session": self,
                }
            )

        @staticmethod
        def get_inputs():
            return [SimpleNamespace(name="image"), SimpleNamespace(name="mask")]

        @staticmethod
        def get_outputs():
            return [SimpleNamespace(name="output")]

    fake_ort = SimpleNamespace(
        __version__="1.18.0",
        SessionOptions=_SessionOptions,
        InferenceSession=_RecordingSession,
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    return calls


def test_cuda_sessions_use_device_id_and_cache_each_index(monkeypatch):
    graph = _patch_tiny_official_graph(monkeypatch)
    runtime = OpaqueLaMaONNX(graph)
    calls = _install_recording_onnxruntime(monkeypatch)

    cuda0 = runtime._get_session(torch.device("cuda:0"))
    cuda1 = runtime._get_session(torch.device("cuda:1"))

    assert runtime._get_session(torch.device("cuda:0")) is cuda0
    assert runtime._get_session(torch.device("cuda:1")) is cuda1
    assert cuda0 is not cuda1
    assert set(runtime._sessions) == {"cuda:0", "cuda:1"}
    assert [call["providers"] for call in calls] == [
        [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ],
        [
            ("CUDAExecutionProvider", {"device_id": 1}),
            "CPUExecutionProvider",
        ],
    ]


def test_implicit_cuda_device_uses_current_index_and_cpu_stays_cpu(monkeypatch):
    graph = _patch_tiny_official_graph(monkeypatch)
    runtime = OpaqueLaMaONNX(graph)
    calls = _install_recording_onnxruntime(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    implicit = runtime._get_session(torch.device("cuda"))
    assert runtime._get_session(torch.device("cuda:1")) is implicit
    cpu = runtime._get_session(torch.device("cpu"))
    assert runtime._get_session(torch.device("cpu")) is cpu

    assert set(runtime._sessions) == {"cuda:1", "cpu"}
    assert calls[0]["providers"] == [
        ("CUDAExecutionProvider", {"device_id": 1}),
        "CPUExecutionProvider",
    ]
    assert calls[1]["providers"] == ["CPUExecutionProvider"]


def test_forward_preserves_the_full_torch_device_for_session_lookup(monkeypatch):
    runtime = OpaqueLaMaONNX()
    seen = []

    def get_session(device):
        seen.append(device)
        return _FakeSession()

    monkeypatch.setattr(runtime, "_get_session", get_session)
    guided = torch.zeros(1, 4, 512, 512)
    runtime(guided)

    assert seen == [torch.device("cpu")]


def test_single_checkpoint_roundtrip_loads_with_weights_only(monkeypatch, tmp_path):
    graph = _patch_tiny_official_graph(monkeypatch)
    checkpoint = _wrapped(graph)
    path = tmp_path / "LibreLaMab-restore.pt"
    torch.save(checkpoint, path)

    safe = load_untrusted_torch_file(path, context="LibreLaMa unit test")
    assert validate_checkpoint_metadata(safe, strict=True) == []
    loaded = LibreLaMa(path, size="b", device="cpu")
    assert torch.equal(loaded.model.onnx_graph, graph)
    assert loaded.model.onnx_graph.device.type == "cpu"
    assert loaded.names == {0: "image"}


def test_raw_state_dict_construction_allocates_opaque_buffer(monkeypatch):
    graph = _patch_tiny_official_graph(monkeypatch)

    loaded = LibreLaMa(
        model_path={"onnx_graph": graph.clone()},
        size="b",
        device="cpu",
    )

    assert torch.equal(loaded.model.onnx_graph, graph)
    assert loaded.model.onnx_graph.device.type == "cpu"


def test_checkpoint_rejects_changed_embedded_bytes(monkeypatch, tmp_path):
    graph = _patch_tiny_official_graph(monkeypatch)
    graph[-1] ^= 1
    path = tmp_path / "LibreLaMab-restore.pt"
    torch.save(_wrapped(graph), path)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        LibreLaMa(path, size="b", device="cpu")


def test_preprocess_requires_aligned_canvas_and_binary_nearest_mask():
    image = Image.fromarray(np.full((3, 5, 3), [10, 20, 30], dtype=np.uint8))
    mask = np.zeros((3, 5), dtype=np.uint8)
    mask[1, 2] = 7

    guided, _, original_size, ratio, context = preprocess_image_and_mask(image, mask)
    assert tuple(guided.shape) == (1, 4, 512, 512)
    assert original_size == (5, 3)
    assert ratio == 1.0
    assert set(torch.unique(guided[:, 3]).tolist()) <= {0.0, 1.0}
    assert context.fill_mask.dtype == bool
    assert context.fill_mask.sum() == 1
    assert context.fill_mask[1, 2]
    # Input graph convention is BGR, while the public canvas remains RGB.
    assert guided[0, 0, 0, 0].item() == pytest.approx(30 / 255)
    assert guided[0, 2, 0, 0].item() == pytest.approx(10 / 255)

    with pytest.raises(ValueError, match="same original canvas"):
        preprocess_image_and_mask(image, np.zeros((2, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match="fixed 512x512"):
        preprocess_image_and_mask(image, mask, input_size=256)


def test_postprocess_keeps_original_canvas_and_unmasked_pixels_exact():
    original = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    fill_mask = np.zeros((4, 5), dtype=bool)
    fill_mask[1:3, 2:4] = True
    raw = torch.empty(1, 3, 512, 512)
    raw[:, 0] = 30  # B
    raw[:, 1] = 20  # G
    raw[:, 2] = 10  # R

    restored = postprocess(
        raw,
        (5, 4),
        original_rgb=original,
        fill_mask=fill_mask,
    )
    assert restored.shape == original.shape
    assert restored.dtype == np.uint8
    assert np.array_equal(restored[~fill_mask], original[~fill_mask])
    assert np.all(restored[fill_mask] == np.array([10, 20, 30], dtype=np.uint8))


class _FakeSession:
    def run(self, output_names, feeds):
        assert output_names == ["output"]
        assert set(feeds) == {"image", "mask"}
        output = feeds["image"].copy() * 255.0
        output[:, 0][feeds["mask"][:, 0] > 0] = 200.0
        return [output]


def test_public_predict_requires_mask_and_preserves_outside(monkeypatch):
    model = LibreLaMa(model_path=None, size="b", device="cpu")
    monkeypatch.setattr(model.model, "_get_session", lambda device_type: _FakeSession())
    image = np.random.default_rng(4).integers(0, 256, (7, 9, 3), dtype=np.uint8)
    mask = np.zeros((7, 9), dtype=np.uint8)
    mask[2:5, 3:7] = 255

    with pytest.raises(ValueError, match="requires prediction input option.*mask"):
        model.predict(image)
    result = model.predict(image, mask=mask, color_format="rgb")
    assert result.boxes is None
    assert result.restored.array.shape == image.shape
    assert np.array_equal(result.restored.array[mask == 0], image[mask == 0])
    # NumPy input defaults to BGR; the restored canvas is RGB.
    result = model.predict(image, mask=mask)
    assert np.array_equal(
        result.restored.array[mask == 0], image[..., ::-1][mask == 0]
    )
    assert not hasattr(model, "_pending_context")


def test_concurrent_predictions_keep_context_request_local(monkeypatch):
    model = LibreLaMa(model_path=None, size="b", device="cpu")
    monkeypatch.setattr(model.model, "_get_session", lambda device_type: _FakeSession())
    first = np.full((7, 9, 3), [11, 22, 33], dtype=np.uint8)
    second = np.full((5, 8, 3), [44, 55, 66], dtype=np.uint8)
    first_mask = np.zeros(first.shape[:2], dtype=np.uint8)
    second_mask = np.zeros(second.shape[:2], dtype=np.uint8)
    first_mask[1:4, 2:6] = 255
    second_mask[2:5, 1:3] = 255

    original_preprocess = model._preprocess_predict
    both_preprocessed = Barrier(2)

    def synchronized_preprocess(*args, **kwargs):
        prepared = original_preprocess(*args, **kwargs)
        both_preprocessed.wait(timeout=5)
        return prepared

    monkeypatch.setattr(model, "_preprocess_predict", synchronized_preprocess)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(model.predict, first, mask=first_mask, color_format="rgb"),
            executor.submit(model.predict, second, mask=second_mask, color_format="rgb"),
        ]
        results = [future.result(timeout=10) for future in futures]

    for result, source, mask in zip(
        results, (first, second), (first_mask, second_mask)
    ):
        assert result.restored.array.shape == source.shape
        assert np.array_equal(result.restored.array[mask == 0], source[mask == 0])
    assert not hasattr(model, "_pending_context")


def test_family_contract_and_inference_only_boundaries(monkeypatch):
    graph = _patch_tiny_official_graph(monkeypatch)
    assert LibreLaMa.can_load({"onnx_graph": graph})
    assert LibreLaMa.detect_size({"onnx_graph": graph}) == "b"
    assert LibreLaMa.detect_size_from_filename("LibreLaMab-restore.pt") == "b"
    assert LibreLaMa.detect_size_from_filename("LibreLaMab.pt") is None
    assert LibreLaMa.PREDICT_INPUT_KWARGS == ("mask",)
    assert LibreLaMa.REQUIRED_PREDICT_INPUT_KWARGS == ("mask",)

    model = LibreLaMa(model_path=None, size="b", device="cpu")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="unused.yaml")
    with pytest.raises(NotImplementedError, match="already executes"):
        model.export(format="onnx")
    with pytest.raises(NotImplementedError, match="already a QDQ-quantized"):
        model.quantize()


def test_converter_is_strict_atomic_and_single_file(monkeypatch, tmp_path):
    converter_path = (
        Path(__file__).resolve().parents[2] / "weights" / "convert_lama_weights.py"
    )
    weights_dir = str(converter_path.parent)
    if weights_dir not in sys.path:
        sys.path.insert(0, weights_dir)
    spec = importlib.util.spec_from_file_location(
        "convert_lama_weights", converter_path
    )
    assert spec is not None and spec.loader is not None
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    payload = b"tiny-lama-graph"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(converter, "OFFICIAL_SIZE_BYTES", len(payload))
    monkeypatch.setattr(converter, "OFFICIAL_SHA256", digest)
    monkeypatch.setattr(lama_nn, "OFFICIAL_ONNX_SIZE_BYTES", len(payload))
    monkeypatch.setattr(lama_nn, "OFFICIAL_ONNX_SHA256", digest)

    source = tmp_path / "inpainting_lama_2025jan.onnx"
    source.write_bytes(payload)
    destination = tmp_path / "LibreLaMab-restore.pt"
    checkpoint = converter.convert_weights(str(source), str(destination))

    assert destination.is_file()
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert checkpoint["source_sha256"] == digest
    assert checkpoint["dataset"] == "Places365-Challenge"
    assert set(checkpoint["model"]) == {"onnx_graph"}
    safe = load_untrusted_torch_file(destination, context="converted LibreLaMa")
    assert bytes(safe["model"]["onnx_graph"].tolist()) == payload


def test_converter_does_not_replace_destination_on_bad_hash(monkeypatch, tmp_path):
    converter_path = (
        Path(__file__).resolve().parents[2] / "weights" / "convert_lama_weights.py"
    )
    weights_dir = str(converter_path.parent)
    if weights_dir not in sys.path:
        sys.path.insert(0, weights_dir)
    spec = importlib.util.spec_from_file_location(
        "convert_lama_weights_bad_hash", converter_path
    )
    assert spec is not None and spec.loader is not None
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    source = tmp_path / "bad.onnx"
    source.write_bytes(b"bad")
    destination = tmp_path / "LibreLaMab-restore.pt"
    destination.write_bytes(b"keep")
    monkeypatch.setattr(converter, "OFFICIAL_SIZE_BYTES", 3)
    monkeypatch.setattr(converter, "OFFICIAL_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        converter.convert_weights(str(source), str(destination))
    assert destination.read_bytes() == b"keep"


def test_validation_dataset_requires_explicit_masks(tmp_path):
    root = tmp_path / "dataset"
    for directory in ("inputs/val", "targets/val", "masks/val"):
        (root / directory).mkdir(parents=True)
    image = np.full((5, 7, 3), 30, dtype=np.uint8)
    Image.fromarray(image).save(root / "inputs/val/sample.png")
    Image.fromarray(image).save(root / "targets/val/sample.png")
    Image.fromarray(np.zeros((5, 7), dtype=np.uint8)).save(
        root / "masks/val/sample.png"
    )
    config = {
        "val": str(root / "inputs/val"),
        "input_dir": "inputs",
        "target_dir": "targets",
    }

    with pytest.raises(ValueError, match="'mask_dir'"):
        LaMaValidationDataset(config, split="val")

    config["mask_dir"] = "masks"
    dataset = LaMaValidationDataset(config, split="val")
    source, target, mask, index = dataset[0]
    assert Path(source) == root / "inputs/val/sample.png"
    assert Path(target) == root / "targets/val/sample.png"
    assert Path(mask) == root / "masks/val/sample.png"
    assert index == 0

    (root / "masks/val/sample.png").unlink()
    with pytest.raises(FileNotFoundError, match="inpainting mask"):
        LaMaValidationDataset(config, split="val")


def test_family_validator_predicts_each_source_with_its_mask(monkeypatch, tmp_path):
    root = tmp_path / "dataset"
    for directory in ("inputs/val", "targets/val", "masks/val"):
        (root / directory).mkdir(parents=True)
    for index, color in enumerate((25, 175)):
        image = np.full((12, 13, 3), color, dtype=np.uint8)
        mask = np.zeros((12, 13), dtype=np.uint8)
        mask[2:8, 3:9] = 255
        filename = f"sample{index}.png"
        Image.fromarray(image).save(root / "inputs/val" / filename)
        Image.fromarray(image).save(root / "targets/val" / filename)
        Image.fromarray(mask).save(root / "masks/val" / filename)

    yaml_path = root / "lama.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "val": "inputs/val",
                "input_dir": "inputs",
                "target_dir": "targets",
                "mask_dir": "masks",
                "nc": 1,
                "names": {0: "image"},
            }
        ),
        encoding="utf-8",
    )
    calls = []
    model = LibreLaMa(model_path=None, size="b", device="cpu")

    def fake_predict(source, *, mask, **kwargs):
        calls.append((Path(source), Path(mask), kwargs))
        with Image.open(source) as image:
            restored = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return SimpleNamespace(restored=SimpleNamespace(array=restored))

    monkeypatch.setattr(model, "predict", fake_predict)
    metrics = model.val(
        data=str(yaml_path),
        batch=2,
        workers=0,
        device="cpu",
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )

    assert len(calls) == 2
    assert [call[0].name for call in calls] == ["sample0.png", "sample1.png"]
    assert [call[1].parent.name for call in calls] == ["val", "val"]
    assert all(call[1].parent.parent.name == "masks" for call in calls)
    assert all(call[2]["imgsz"] == 512 for call in calls)
    assert metrics["metrics/PSNR"] == pytest.approx(100.0)
    assert metrics["metrics/SSIM"] == pytest.approx(1.0)
    assert metrics["speed/images_seen"] == 2
