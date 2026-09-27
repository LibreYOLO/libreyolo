"""Unit tests for the LibreQuickSRNet super-resolution family."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo import LibreQuickSRNet
from libreyolo.models import LibreYOLO
from libreyolo.models.autoconvert import autoconvert_upstream_checkpoint
from libreyolo.models.nafnet import LibreNAFNet
from libreyolo.models.realesrgan import LibreRealESRGAN
from libreyolo.models.swinir import LibreSwinIR
from libreyolo.postprocess.quicksrnet import postprocess
from libreyolo.utils.serialization import (
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)

pytestmark = [pytest.mark.unit, pytest.mark.quicksrnet]


def _image(height: int, width: int, seed: int = 0) -> Image.Image:
    array = np.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    return Image.fromarray(array, mode="RGB")


def test_public_contract_and_download_url():
    assert LibreQuickSRNet.FAMILY == "quicksrnet"
    assert LibreQuickSRNet.SUPPORTED_TASKS == ("restore",)
    assert (
        LibreQuickSRNet.detect_size_from_filename("LibreQuickSRNetm2-restore.pt")
        == "m2"
    )
    assert LibreQuickSRNet.detect_size_from_filename("LibreQuickSRNetm2.pt") is None
    assert LibreQuickSRNet.get_download_url("LibreQuickSRNetm2-restore.pt") == (
        "https://huggingface.co/LibreYOLO/LibreQuickSRNetm2-restore/resolve/"
        "main/LibreQuickSRNetm2-restore.pt"
    )


def test_state_dict_detection_and_parameter_count():
    model = LibreQuickSRNet(size="m2", device="cpu")
    state = model.model.state_dict()
    assert len(state) == 14
    assert sum(parameter.numel() for parameter in model.model.parameters()) == 50_604
    assert LibreQuickSRNet.can_load(state)
    assert LibreQuickSRNet.detect_size(state) == "m2"
    assert LibreQuickSRNet.detect_checkpoint_task(state) == "restore"
    assert model.restore_scale == 2


def test_bidirectional_rejection_of_other_restore_families():
    quick = LibreQuickSRNet(size="m2", device="cpu").model.state_dict()
    other_models = (
        LibreNAFNet(size="s", device="cpu"),
        LibreRealESRGAN(size="x4t", device="cpu"),
        LibreSwinIR(size="s", device="cpu"),
    )
    for other in other_models:
        state = other.model.state_dict()
        assert not LibreQuickSRNet.can_load(state)
        assert not type(other).can_load(quick)


def test_forward_predict_and_postprocess():
    model = LibreQuickSRNet(size="m2", device="cpu")
    model.model.eval()
    with torch.inference_mode():
        output = model.model(torch.rand(1, 3, 11, 17))
    assert tuple(output.shape) == (1, 3, 22, 34)
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0

    result = model.predict(_image(13, 19, seed=1))
    assert result.restore_scale == 2
    assert result.restored.array.shape == (26, 38, 3)
    assert result.restored.array.dtype == np.uint8
    assert result.summary() == [{"name": "restored", "shape": [26, 38, 3], "scale": 2}]

    tensor = torch.zeros(1, 3, 20, 24)
    tensor[:, 1] = 0.5
    restored = postprocess(tensor, original_size=(7, 5), scale=2)
    assert restored.shape == (10, 14, 3)
    assert np.all(restored[..., 1] == 128)


def test_factory_roundtrip_and_plain_state_autoconvert(tmp_path):
    source = LibreQuickSRNet(size="m2", device="cpu")
    checkpoint = wrap_libreyolo_checkpoint(
        source.model.state_dict(),
        model_family="quicksrnet",
        size="m2",
        task="restore",
        nc=1,
        names={0: "image"},
        imgsz=64,
        scale=2,
        degradation="super-resolution",
        dataset="DIV2K",
    )
    path = tmp_path / "LibreQuickSRNetm2-restore.pt"
    torch.save(checkpoint, path)
    loaded = LibreYOLO(str(path), device="cpu")
    assert isinstance(loaded, LibreQuickSRNet)
    assert loaded.size == "m2"
    assert loaded.task == "restore"
    assert loaded.names == {0: "image"}

    raw = tmp_path / "quicksrnet_state_dict.pth"
    torch.save(source.model.state_dict(), raw)
    converted = autoconvert_upstream_checkpoint(str(raw))
    assert converted is not None
    assert converted.endswith("-LibreQuickSRNetm2-restore.pt")
    assert isinstance(LibreYOLO(converted, device="cpu"), LibreQuickSRNet)


def test_converter_is_lean_valid_and_atomic(tmp_path):
    converter_path = (
        Path(__file__).resolve().parents[2]
        / "weights"
        / ("convert_quicksrnet_weights.py")
    )
    weights_dir = str(converter_path.parent)
    if weights_dir not in sys.path:
        sys.path.insert(0, weights_dir)
    spec = importlib.util.spec_from_file_location(
        "convert_quicksrnet_weights", converter_path
    )
    assert spec is not None and spec.loader is not None
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    source_model = LibreQuickSRNet(size="m2", device="cpu")
    source = tmp_path / "source.pth.tar"
    torch.save(
        {
            "epoch": 100,
            "state_dict": source_model.model.state_dict(),
            "optimizer": {"state": {1: {"momentum": torch.ones(1)}}},
        },
        source,
    )
    destination = tmp_path / "LibreQuickSRNetm2-restore.pt"
    checkpoint = converter.convert_weights(
        str(source), str(destination), expected_sha256=None
    )
    assert destination.is_file()
    assert not (tmp_path / f".{destination.name}.tmp").exists()
    assert validate_checkpoint_metadata(checkpoint) == []
    assert set(checkpoint["model"]) == set(source_model.model.state_dict())
    assert "optimizer" not in checkpoint and "epoch" not in checkpoint
    assert (
        checkpoint["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    safely_loaded = torch.load(destination, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(safely_loaded) == []


def _write_restore_dataset(root: Path) -> Path:
    for branch in ("inputs", "targets"):
        (root / branch / "val").mkdir(parents=True, exist_ok=True)
    for index in range(2):
        _image(8, 10, seed=index).save(root / "inputs" / "val" / f"{index}.png")
        _image(16, 20, seed=10 + index).save(root / "targets" / "val" / f"{index}.png")
    config = {
        "path": str(root),
        "val": str(root / "inputs" / "val"),
        "input_dir": "inputs",
        "target_dir": "targets",
        "nc": 1,
        "names": {0: "image"},
    }
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump(config), encoding="utf-8")
    return data


def test_validation_and_training_contract(tmp_path):
    model = LibreQuickSRNet(size="m2", device="cpu")
    metrics = model.val(
        data=str(_write_restore_dataset(tmp_path)),
        split="val",
        batch=1,
        workers=0,
        device="cpu",
    )
    assert np.isfinite(metrics["metrics/PSNR"])
    assert np.isfinite(metrics["metrics/SSIM"])
    with pytest.raises(NotImplementedError, match="Training is not implemented"):
        model.train(data="unused.yaml")


def test_onnx_dynamic_export_roundtrip(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from libreyolo.backends.onnx import OnnxBackend

    torch.manual_seed(0)
    model = LibreQuickSRNet(size="m2", device="cpu")
    artifact = model.export(
        format="onnx",
        imgsz=32,
        dynamic=True,
        simplify=False,
        output_path=str(tmp_path / "quicksrnet.onnx"),
    )
    backend = OnnxBackend(str(artifact))
    image = _image(21, 29, seed=7)
    native = model.predict(image).restored.array
    exported = backend.predict(image).restored.array
    assert backend.restore_scale == 2
    assert exported.shape == native.shape == (42, 58, 3)
    assert int(np.abs(native.astype(int) - exported.astype(int)).max()) <= 1


def test_torchscript_export_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = LibreQuickSRNet(size="m2", device="cpu")
    artifact = model.export(
        format="torchscript",
        imgsz=24,
        dynamic=False,
        output_path=str(tmp_path / "quicksrnet.torchscript"),
    )
    backend = LibreYOLO(artifact, device="cpu")
    image = _image(24, 24, seed=8)
    native = model.predict(image).restored.array
    exported = backend.predict(image).restored.array
    assert backend.restore_scale == 2
    assert exported.shape == native.shape == (48, 48, 3)
    assert int(np.abs(native.astype(int) - exported.astype(int)).max()) <= 1
