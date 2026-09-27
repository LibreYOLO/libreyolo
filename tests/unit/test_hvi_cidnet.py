"""Unit coverage for the LibreHVI-CIDNet low-light family."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from safetensors.torch import save_file

from libreyolo.models.hvi_cidnet import LibreHVICIDNet
from libreyolo.models.hvi_cidnet import model as hvi_cidnet_model
from libreyolo.models.nafnet import LibreNAFNet
from libreyolo.models.quicksrnet import LibreQuickSRNet
from libreyolo.models.realesrgan import LibreRealESRGAN
from libreyolo.models.swinir import LibreSwinIR
from libreyolo.postprocess.hvi_cidnet import postprocess
from libreyolo.utils.serialization import validate_checkpoint_metadata


pytestmark = [pytest.mark.unit, pytest.mark.hvi_cidnet]


def _image(height: int = 17, width: int = 25, seed: int = 0) -> Image.Image:
    array = np.random.default_rng(seed).integers(
        0, 80, (height, width, 3), dtype=np.uint8
    )
    return Image.fromarray(array, mode="RGB")


def test_public_contract_state_dict_and_download_url():
    model = LibreHVICIDNet(size="t", device="cpu")
    state = model.model.state_dict()
    assert LibreHVICIDNet.FAMILY == "hvi_cidnet"
    assert LibreHVICIDNet.SUPPORTED_TASKS == ("restore",)
    assert LibreHVICIDNet.detect_size(state) == "t"
    assert LibreHVICIDNet.detect_checkpoint_task(state) == "restore"
    assert sum(value.numel() for value in state.values()) == 1_975_569
    assert LibreHVICIDNet.detect_size_from_filename("LibreHVICIDNett-restore.pt") == "t"
    assert LibreHVICIDNet.detect_size_from_filename("LibreHVICIDNett.pt") is None
    assert LibreHVICIDNet.get_download_url("LibreHVICIDNett-restore.pt") == (
        "https://huggingface.co/LibreYOLO/LibreHVICIDNett-restore/resolve/"
        "main/LibreHVICIDNett-restore.pt"
    )


def test_strict_source_checkpoint_load_and_forward_shape():
    source_path = os.environ.get("LIBREYOLO_HVI_CIDNET_CHECKPOINT")
    if not source_path:
        pytest.skip("LIBREYOLO_HVI_CIDNET_CHECKPOINT is not set")
    source = Path(source_path)
    if not source.is_file():
        pytest.skip("official HVI-CIDNet checkpoint is not staged")
    from safetensors.torch import load_file

    model = LibreHVICIDNet(size="t", device="cpu")
    model.model.load_state_dict(load_file(source), strict=True)
    with torch.inference_mode():
        output = model.model(torch.rand(1, 3, 16, 24))
    assert tuple(output.shape) == (1, 3, 16, 24)


def test_bidirectional_rejection_of_other_restore_families():
    ours = LibreHVICIDNet(size="t", device="cpu").model.state_dict()
    others = (
        LibreNAFNet(size="s", device="cpu"),
        LibreQuickSRNet(size="m2", device="cpu"),
        LibreRealESRGAN(size="x4t", device="cpu"),
        LibreSwinIR(size="s", device="cpu"),
    )
    for other in others:
        state = other.model.state_dict()
        assert not LibreHVICIDNet.can_load(state)
        assert not type(other).can_load(ours)


def test_predict_restores_original_canvas_and_controls_do_not_leak():
    torch.manual_seed(5)
    model = LibreHVICIDNet(size="t", device="cpu")
    image = _image()
    neutral = model.predict(image).restored.array
    dimmed = model.predict(image, intensity=0.5).restored.array
    neutral_again = model.predict(image).restored.array
    assert neutral.shape == dimmed.shape == (17, 25, 3)
    assert neutral.dtype == np.uint8
    assert np.array_equal(neutral, neutral_again)
    assert not np.array_equal(neutral, dimmed)
    with pytest.raises(ValueError, match="gamma must be positive"):
        model.predict(image, gamma=0)
    with pytest.raises(ValueError, match="saturation must be positive"):
        model.predict(image, saturation=-1)


@pytest.mark.parametrize(
    ("control", "value"),
    (
        ("gamma", float("nan")),
        ("saturation", float("inf")),
        ("intensity", float("-inf")),
    ),
)
def test_predict_rejects_non_finite_controls(control, value):
    model = LibreHVICIDNet(size="t", device="cpu")
    with pytest.raises(ValueError, match=rf"{control} must be positive"):
        model.predict(_image(), **{control: value})


def test_stream_controls_apply_lazily_without_leaking(monkeypatch):
    model = LibreHVICIDNet(size="t", device="cpu")
    image = _image(height=8, width=8)
    seen_gamma = []
    seen_output_scales = []
    preprocess_image = hvi_cidnet_model.preprocess_image

    def record_preprocess(*args, **kwargs):
        seen_gamma.append(kwargs["gamma"])
        return preprocess_image(*args, **kwargs)

    def record_forward(input_tensor, *, saturation_scale, intensity_scale):
        seen_output_scales.append((saturation_scale, intensity_scale))
        return input_tensor

    monkeypatch.setattr(hvi_cidnet_model, "preprocess_image", record_preprocess)
    monkeypatch.setattr(model.model, "forward", record_forward)

    results = model.predict(
        [image, image, image],
        gamma=0.6,
        saturation=0.7,
        intensity=0.8,
        stream=True,
    )
    assert seen_gamma == []
    assert seen_output_scales == []

    first = next(results)
    assert first.restored.array.shape == (8, 8, 3)
    assert seen_gamma == [0.6]
    assert seen_output_scales == [(0.7, 0.8)]

    # The stream is still suspended with one image left. A separate call must
    # nevertheless see neutral controls, both before and after explicit close.
    model.predict(image)
    assert seen_gamma[-1] == 1.0
    assert seen_output_scales[-1] == (1.0, 1.0)

    second = next(results)
    assert second.restored.array.shape == (8, 8, 3)
    assert seen_gamma[-1] == 0.6
    assert seen_output_scales[-1] == (0.7, 0.8)
    results.close()
    model.predict(image)
    assert seen_gamma[-1] == 1.0
    assert seen_output_scales[-1] == (1.0, 1.0)


def test_tiny_image_padding_and_postprocess_clamp():
    model = LibreHVICIDNet(size="t", device="cpu")
    result = model.predict(_image(height=1, width=1))
    assert result.restored.array.shape == (1, 1, 3)

    output = torch.tensor(
        [[[[2.0, -1.0]], [[0.5, 0.5]], [[0.0, 1.0]]]], dtype=torch.float32
    )
    restored = postprocess(output, original_size=(2, 1))
    assert restored.tolist() == [[[255, 128, 0], [0, 128, 255]]]


def test_converter_is_atomic_lean_and_valid(monkeypatch, tmp_path):
    converter_path = (
        Path(__file__).resolve().parents[2]
        / "weights"
        / "convert_hvi_cidnet_weights.py"
    )
    weights_dir = str(converter_path.parent)
    if weights_dir not in sys.path:
        sys.path.insert(0, weights_dir)
    spec = importlib.util.spec_from_file_location(
        "convert_hvi_cidnet_weights", converter_path
    )
    assert spec is not None and spec.loader is not None
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    source_model = LibreHVICIDNet(size="t", device="cpu")
    source = tmp_path / "model.safetensors"
    save_file(source_model.model.state_dict(), source)
    destination = tmp_path / "LibreHVICIDNett-restore.pt"
    monkeypatch.setattr(converter, "OFFICIAL_SIZE", source.stat().st_size)
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        converter.convert_weights(str(source), str(destination))
    monkeypatch.setattr(converter, "OFFICIAL_SHA256", converter.file_sha256(source))
    checkpoint = converter.convert_weights(str(source), str(destination))
    assert destination.is_file()
    assert not (tmp_path / f".{destination.name}.tmp").exists()
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert checkpoint["degradation"] == "low-light"
    assert checkpoint["dataset"] == "LOLv2-Synthetic"
    assert set(checkpoint["model"]) == set(source_model.model.state_dict())


def _write_restore_dataset(root: Path) -> Path:
    for branch in ("inputs", "targets"):
        (root / branch / "val").mkdir(parents=True, exist_ok=True)
    _image(8, 10, seed=21).save(root / "inputs" / "val" / "sample.png")
    _image(8, 10, seed=22).save(root / "targets" / "val" / "sample.png")
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "val": str(root / "inputs" / "val"),
                "input_dir": "inputs",
                "target_dir": "targets",
                "nc": 1,
                "names": {0: "image"},
            }
        ),
        encoding="utf-8",
    )
    return data


def test_validation_and_training_contract(tmp_path):
    model = LibreHVICIDNet(size="t", device="cpu")
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
