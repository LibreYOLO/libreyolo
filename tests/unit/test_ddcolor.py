"""Focused unit tests for the isolated LibreDDColor family."""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

from libreyolo.models.ddcolor import DDCOLOR_SIZE_CONFIGS, LibreDDColor
from libreyolo.models.ddcolor.utils import (
    DDCOLOR_ORIGINAL_L_KEY,
    preprocess_image,
)
from libreyolo.models.nafnet import LibreNAFNet
from libreyolo.models.quicksrnet import LibreQuickSRNet
from libreyolo.models.realesrgan import LibreRealESRGAN
from libreyolo.models.swinir import LibreSwinIR
from libreyolo.postprocess.ddcolor import postprocess


pytestmark = [pytest.mark.unit, pytest.mark.ddcolor]


def _image(height: int, width: int, seed: int = 0) -> Image.Image:
    array = np.random.default_rng(seed).integers(
        0,
        256,
        (height, width, 3),
        dtype=np.uint8,
    )
    return Image.fromarray(array, mode="RGB")


@pytest.fixture(scope="module")
def tiny_model() -> LibreDDColor:
    torch.manual_seed(0)
    model = LibreDDColor(size="t", device="cpu")
    model.model.eval()
    return model


def test_public_contract_and_two_size_configs():
    assert LibreDDColor.FAMILY == "ddcolor"
    assert LibreDDColor.SUPPORTED_TASKS == ("restore",)
    assert LibreDDColor.INPUT_SIZES == {"t": 512, "l": 512}
    assert DDCOLOR_SIZE_CONFIGS["t"]["depths"] == (3, 3, 9, 3)
    assert DDCOLOR_SIZE_CONFIGS["t"]["dims"] == (96, 192, 384, 768)
    assert DDCOLOR_SIZE_CONFIGS["l"]["depths"] == (3, 3, 27, 3)
    assert DDCOLOR_SIZE_CONFIGS["l"]["dims"] == (192, 384, 768, 1536)
    assert LibreDDColor.detect_size_from_filename("LibreDDColort-restore.pt") == "t"
    assert LibreDDColor.detect_size_from_filename("LibreDDColorl-restore.pt") == "l"
    assert LibreDDColor.detect_size_from_filename("LibreDDColort.pt") is None
    assert LibreDDColor.get_download_url("LibreDDColort-restore.pt") == (
        "https://huggingface.co/LibreYOLO/LibreDDColort-restore/resolve/"
        "main/LibreDDColort-restore.pt"
    )


def test_random_weight_signature_and_small_forward(tiny_model: LibreDDColor):
    state = tiny_model.model.state_dict()
    assert len(state) == 440
    assert (
        sum(parameter.numel() for parameter in tiny_model.model.parameters())
        == 55_006_640
    )
    assert LibreDDColor.can_load(state)
    assert LibreDDColor.detect_size(state) == "t"
    assert LibreDDColor.detect_nb_classes(state) == 1
    assert LibreDDColor.detect_checkpoint_task(state) == "restore"
    assert tuple(state["refine_net.0.0.weight_orig"].shape) == (2, 103, 1, 1)

    with torch.inference_mode():
        output = tiny_model.model(torch.rand(1, 3, 32, 32))
    assert tuple(output.shape) == (1, 2, 32, 32)
    assert torch.isfinite(output).all()


def test_concurrent_forwards_keep_encoder_features_request_local(
    tiny_model: LibreDDColor,
    monkeypatch,
):
    """Force both encoders to finish before either decoder consumes its skips."""

    network = tiny_model.model
    first = torch.zeros(1, 3, 32, 32)
    second = torch.ones(1, 3, 32, 32)
    with torch.inference_mode():
        expected_first = network(first).clone()
        expected_second = network(second).clone()

    both_encoded = Barrier(2)
    original_forward = network.encoder.forward

    def synchronized_encoder(image):
        features = original_forward(image)
        both_encoded.wait(timeout=10)
        return features

    monkeypatch.setattr(network.encoder, "forward", synchronized_encoder)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(network, first)
        second_future = executor.submit(network, second)
        actual_first = first_future.result(timeout=30)
        actual_second = second_future.result(timeout=30)

    torch.testing.assert_close(actual_first, expected_first, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_second, expected_second, rtol=0.0, atol=0.0)


def test_large_signature_is_recognized_without_allocating_large_model():
    state = {
        "encoder.arch.downsample_layers.0.0.weight": torch.empty(192, 3, 4, 4),
        "encoder.arch.norm3.weight": torch.empty(1536),
        "decoder.color_decoder.query_feat.weight": torch.empty(100, 256),
        "decoder.color_decoder.transformer_cross_attention_layers.8.multihead_attn.in_proj_weight": torch.empty(
            768, 256
        ),
        "refine_net.0.0.weight_orig": torch.empty(2, 103, 1, 1),
    }
    assert LibreDDColor.detect_size(state) == "l"
    assert LibreDDColor.can_load(state)


def test_bidirectional_rejection_of_restore_families(tiny_model: LibreDDColor):
    ddcolor_state = tiny_model.model.state_dict()
    other_models = (
        LibreNAFNet(size="s", device="cpu"),
        LibreQuickSRNet(size="m2", device="cpu"),
        LibreRealESRGAN(size="x4t", device="cpu"),
        LibreSwinIR(size="s", device="cpu"),
    )
    for other in other_models:
        other_state = other.model.state_dict()
        assert not LibreDDColor.can_load(other_state)
        assert not type(other).can_load(ddcolor_state)


def test_preprocess_matches_pinned_bgr_lab_pipeline():
    image = _image(7, 11, seed=3)
    tensor, returned, original_size, metadata = preprocess_image(image, input_size=512)
    assert returned.mode == "RGB"
    assert returned.size == image.size
    assert original_size == (11, 7)
    assert tuple(tensor.shape) == (1, 3, 512, 512)
    assert tensor.dtype == torch.float32

    image_rgb = np.asarray(image, dtype=np.uint8)
    image_bgr = np.ascontiguousarray(image_rgb[..., ::-1])
    image_float = (image_bgr / 255.0).astype(np.float32)
    expected_l = cv2.cvtColor(image_float, cv2.COLOR_BGR2Lab)[:, :, :1]
    resized = cv2.resize(image_float, (512, 512))
    resized_l = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    gray_lab = np.concatenate(
        (resized_l, np.zeros_like(resized_l), np.zeros_like(resized_l)),
        axis=-1,
    )
    expected_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)
    expected_tensor = torch.from_numpy(expected_rgb.transpose(2, 0, 1)).unsqueeze(0)

    assert np.array_equal(metadata[DDCOLOR_ORIGINAL_L_KEY], expected_l)
    assert torch.equal(tensor, expected_tensor)
    with pytest.raises(ValueError, match="fixed 512x512"):
        preprocess_image(image, input_size=256)


def test_postprocess_matches_nearest_lab_to_bgr_then_rgb():
    original_size = (5, 4)
    original_l = np.linspace(5.0, 95.0, 20, dtype=np.float32).reshape(4, 5, 1)
    output_ab = torch.tensor(
        [
            [
                [[-20.0, 5.0, 20.0], [30.0, -10.0, 15.0]],
                [[10.0, -15.0, 25.0], [-25.0, 20.0, -5.0]],
            ]
        ],
        dtype=torch.float32,
    )
    actual_rgb = postprocess(
        output_ab,
        original_size,
        original_l=original_l,
    )

    expected_ab = F.interpolate(output_ab, size=(4, 5))[0].numpy().transpose(1, 2, 0)
    expected_lab = np.concatenate((original_l, expected_ab), axis=-1)
    expected_bgr = cv2.cvtColor(expected_lab, cv2.COLOR_LAB2BGR)
    expected_bgr = (expected_bgr * 255.0).round().astype(np.uint8)
    expected_rgb = np.ascontiguousarray(expected_bgr[..., ::-1])
    assert actual_rgb.dtype == np.uint8
    assert actual_rgb.shape == (4, 5, 3)
    assert np.array_equal(actual_rgb, expected_rgb)


class _ZeroAB(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image.new_zeros((image.shape[0], 2, image.shape[2], image.shape[3]))


def test_predict_keeps_each_original_canvas_in_a_real_batch(monkeypatch):
    monkeypatch.setattr(LibreDDColor, "_init_model", lambda self: _ZeroAB())
    model = LibreDDColor(size="t", device="cpu")
    images = (_image(9, 13, seed=5), _image(6, 10, seed=6))
    results = model.predict(list(images), batch=2)
    assert len(results) == 2
    assert results[0].restored.array.shape == (9, 13, 3)
    assert results[1].restored.array.shape == (6, 10, 3)
    assert results[0].restored.array.dtype == np.uint8
    assert results[1].restored.array.dtype == np.uint8
    assert results[0].summary() == [
        {"name": "restored", "shape": [9, 13, 3], "scale": 1}
    ]


def _write_restore_dataset(root: Path) -> Path:
    for branch in ("inputs", "targets"):
        (root / branch / "val").mkdir(parents=True, exist_ok=True)
    for index in range(2):
        _image(8 + index, 10 + index, seed=index).save(
            root / "inputs" / "val" / f"{index}.png"
        )
        _image(8 + index, 10 + index, seed=10 + index).save(
            root / "targets" / "val" / f"{index}.png"
        )
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


def test_training_block_and_exact_predict_based_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(LibreDDColor, "_init_model", lambda self: _ZeroAB())
    model = LibreDDColor(size="t", device="cpu")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="unused.yaml")
    metrics = model.val(
        data=str(_write_restore_dataset(tmp_path)),
        batch=2,
        workers=0,
        device="cpu",
        verbose=False,
    )
    assert np.isfinite(metrics["metrics/PSNR"])
    assert np.isfinite(metrics["metrics/SSIM"])
    assert metrics["speed/images_seen"] == 2


def _load_converter_module():
    path = (
        Path(__file__).resolve().parents[2] / "weights" / "convert_ddcolor_weights.py"
    )
    weights_dir = str(path.parent)
    if weights_dir not in sys.path:
        sys.path.insert(0, weights_dir)
    spec = importlib.util.spec_from_file_location("convert_ddcolor_weights", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_pins_both_artifacts_and_rejects_other_bytes(tmp_path):
    converter = _load_converter_module()
    assert converter.OFFICIAL_CHECKPOINTS["t"]["sha256"] == (
        "8a1277bc90a1bfbb6d2d83933a9a6bc821931879ca93e26e4fcec12165d41fce"
    )
    assert converter.OFFICIAL_CHECKPOINTS["l"]["sha256"] == (
        "d81711971ec59200da26d5e8a1afae8dd3778d495ea8ad7a7dadc769f403f7e7"
    )
    assert "/cf9fd99c1d7472689ec7413441c1b799a51866a3/" in converter.official_url("t")
    assert "/060f67494e31883a4b13cb27f889f3154847ada4/" in converter.official_url("l")

    state = {"weight": torch.ones(1)}
    assert converter.extract_ddcolor_state_dict(state) == state
    assert converter.extract_ddcolor_state_dict({"params": state}) == state
    with pytest.raises(TypeError, match="only named tensors"):
        converter.extract_ddcolor_state_dict({"params": {"epoch": 1}})

    unknown = tmp_path / "not-an-official-checkpoint.bin"
    unknown.write_bytes(b"not DDColor")
    destination = tmp_path / "LibreDDColort-restore.pt"
    with pytest.raises(ValueError, match="source mismatch"):
        converter.convert_weights(str(unknown), str(destination), size="t")
    assert not destination.exists()
    assert not (tmp_path / f".{destination.name}.tmp").exists()
