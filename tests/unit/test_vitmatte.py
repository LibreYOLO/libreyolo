"""Focused offline coverage for the guided LibreViTMatte family."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from libreyolo.models.ben2 import LibreBEN2
from libreyolo.models.birefnet import LibreBiRefNet
from libreyolo.models.vitmatte import LibreViTMatte
from libreyolo.models.vitmatte.nn import constrain_alpha_to_trimap
from libreyolo.models.vitmatte.utils import (
    normalize_trimap,
    preprocess_guided_image,
    preprocess_numpy,
)
from libreyolo.models.vitmatte.validator import derive_trimap_from_matte
from libreyolo.postprocess.vitmatte import postprocess


pytestmark = [pytest.mark.unit, pytest.mark.vitmatte]


def _vitmatte_state() -> dict[str, torch.Tensor]:
    return {
        "backbone.embeddings.projection.weight": torch.empty(384, 4, 16, 16),
        "backbone.embeddings.position_embeddings": torch.empty(1, 197, 384),
        "backbone.encoder.layer.2.attention.rel_pos_h": torch.empty(63, 64),
        "backbone.encoder.layer.11.residual.conv3.weight": torch.empty(384, 192, 1, 1),
        "decoder.convstream.convs.0.conv.weight": torch.empty(48, 4, 3, 3),
        "decoder.matting_head.matting_convs.3.weight": torch.empty(1, 16, 1, 1),
    }


def _image(height: int = 5, width: int = 7) -> Image.Image:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[..., 0] = 255
    array[..., 1] = np.arange(width, dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_family_contract_detection_filename_and_notice():
    state = _vitmatte_state()
    assert LibreViTMatte.FAMILY == "vitmatte"
    assert LibreViTMatte.INPUT_SIZES == {"s": 512}
    assert LibreViTMatte.SUPPORTED_TASKS == ("matte",)
    assert LibreViTMatte.PREDICT_INPUT_KWARGS == ("trimap",)
    assert LibreViTMatte.REQUIRED_PREDICT_INPUT_KWARGS == ("trimap",)
    assert LibreViTMatte.can_load(state)
    assert LibreViTMatte.detect_size(state) == "s"
    assert LibreViTMatte.detect_nb_classes(state) == 1
    assert LibreViTMatte.detect_checkpoint_task(state) == "matte"
    assert LibreViTMatte.default_checkpoint_names(1) == {0: "matte"}
    assert LibreViTMatte.detect_size_from_filename("LibreViTMattes-matte.pt") == "s"
    assert LibreViTMatte.detect_size_from_filename("LibreViTMattes.pt") is None
    assert LibreViTMatte.get_download_url("LibreViTMattes-matte.pt") == (
        "https://huggingface.co/LibreYOLO/LibreViTMattes-matte/resolve/"
        "main/LibreViTMattes-matte.pt"
    )
    notice = LibreViTMatte.get_download_notice("LibreViTMattes-matte.pt", "unused")
    assert "NON-COMMERCIAL" in notice
    assert "Adobe Deep Image Matting" in notice


def test_checkpoint_detection_does_not_collide_with_automatic_matte_families():
    vitmatte = _vitmatte_state()
    birefnet = {
        "bb.patch_embed.proj.weight": torch.empty(192, 3, 4, 4),
        "squeeze_module.0.conv_in.weight": torch.empty(1),
        "decoder.ipt_blk5.conv1.weight": torch.empty(1),
        "decoder.gdt_convs_attn_4.0.weight": torch.empty(1),
    }
    ben2 = {
        "backbone.patch_embed.proj.weight": torch.empty(128, 3, 4, 4),
        "multifieldcrossatt.attention.4.out_proj.weight": torch.empty(128, 128),
        "dec_blk4.sal_conv.weight": torch.empty(1, 128, 1, 1),
        "insmask_head.6.weight": torch.empty(128, 384, 3, 3),
    }
    assert not LibreViTMatte.can_load(birefnet)
    assert not LibreViTMatte.can_load(ben2)
    assert not LibreBiRefNet.can_load(vitmatte)
    assert not LibreBEN2.can_load(vitmatte)


@pytest.mark.parametrize(
    ("array", "expected_middle"),
    [
        (np.asarray([[0, 128, 255]], dtype=np.uint8), 128.0 / 255.0),
        (np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32), 0.5),
    ],
)
def test_trimap_accepts_only_the_two_three_level_encodings(array, expected_middle):
    normalized = normalize_trimap(array)
    assert tuple(normalized.shape) == (1, 1, 3)
    assert normalized.dtype == torch.float32
    assert normalized[0, 0].tolist() == pytest.approx([0.0, expected_middle, 1.0])


@pytest.mark.parametrize(
    "trimap",
    [
        np.asarray([[0, 127, 255]], dtype=np.uint8),
        np.asarray([[0.0, 0.25, 1.0]], dtype=np.float32),
        np.asarray([[0.0, np.nan, 1.0]], dtype=np.float32),
    ],
)
def test_trimap_rejects_invalid_levels(trimap):
    with pytest.raises(ValueError, match="trimap="):
        normalize_trimap(trimap)


def test_trimap_rejects_non_grayscale_rgb():
    trimap = np.zeros((3, 4, 3), dtype=np.uint8)
    trimap[..., 1] = 128
    with pytest.raises(ValueError, match="grayscale"):
        normalize_trimap(trimap)


def test_guided_preprocess_resizes_nearest_and_pads_bottom_right():
    guide = Image.fromarray(
        np.asarray([[0, 255], [128, 0]], dtype=np.uint8),
        mode="L",
    )
    tensor, original, original_size, ratio = preprocess_guided_image(_image(), guide)
    assert tuple(tensor.shape) == (1, 4, 32, 32)
    assert original.size == original_size == (7, 5)
    assert ratio == 1.0
    assert sorted(torch.unique(tensor[0, 3, :5, :7]).tolist()) == pytest.approx(
        [0.0, 128.0 / 255.0, 1.0]
    )
    assert torch.count_nonzero(tensor[:, :, 5:, :]) == 0
    assert torch.count_nonzero(tensor[:, :, :, 7:]) == 0


def test_numpy_preprocess_requires_combined_four_channels():
    with pytest.raises(ValueError, match="HxWx4"):
        preprocess_numpy(np.zeros((5, 7, 3), dtype=np.uint8))
    combined = np.zeros((5, 7, 4), dtype=np.uint8)
    combined[..., 3] = 128
    output, ratio = preprocess_numpy(combined)
    assert output.shape == (4, 32, 32)
    assert ratio == 1.0


def test_postprocess_crops_probabilities_without_a_second_sigmoid():
    output = torch.tensor([[[[0.0, 0.25, 0.75, 1.0], [0.1, 0.2, 0.3, 0.4]]]])
    matte = postprocess(output, original_size=(3, 1))["matte"]
    assert matte.shape == (1, 3)
    assert matte.tolist() == [[0.0, 0.25, 0.75]]


def test_known_trimap_regions_are_forced_to_exact_zero_and_one():
    alpha = torch.full((1, 1, 1, 3), 0.37)
    pixels = torch.zeros(1, 4, 1, 3)
    pixels[:, 3] = torch.tensor([0.0, 0.5, 1.0])
    constrained = constrain_alpha_to_trimap(alpha, pixels)
    assert constrained.flatten().tolist() == pytest.approx([0.0, 0.37, 1.0])


def test_validation_trimap_derivation_has_fixed_symmetric_unknown_band():
    matte = np.zeros((11, 11), dtype=np.float32)
    matte[2:9, 2:9] = 1.0
    trimap = derive_trimap_from_matte(matte, radius=1)
    assert trimap.dtype == np.uint8
    assert set(np.unique(trimap)) == {0, 128, 255}
    assert trimap[0, 0] == 0
    assert trimap[2, 2] == 128
    assert trimap[3, 3] == 255
    with pytest.raises(ValueError, match="trimap_radius"):
        derive_trimap_from_matte(matte, radius=-1)


class _GuideEcho(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values[:, 3:4] + self.anchor * 0


def test_public_predict_requires_and_forwards_one_trimap(monkeypatch):
    monkeypatch.setattr(LibreViTMatte, "_init_model", lambda self: _GuideEcho())
    model = LibreViTMatte(size="s", device="cpu")
    guide_array = np.full((5, 7), 128, dtype=np.uint8)
    guide_array[:, :2] = 0
    guide_array[:, -2:] = 255
    guide = Image.fromarray(guide_array, mode="L")

    with pytest.raises(ValueError, match="requires prediction input.*trimap"):
        model.predict(_image())

    result = model.predict(_image(), trimap=guide)
    assert result.matte is not None
    assert result.matte.array.shape == (5, 7)
    np.testing.assert_allclose(result.matte.array, guide_array / 255.0)

    with pytest.raises(ValueError, match="only one non-streamed"):
        model.predict([_image(), _image()], trimap=guide)
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="unused.yaml")


def test_public_val_honors_matching_trimap_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(LibreViTMatte, "_init_model", lambda self: _GuideEcho())
    model = LibreViTMatte(size="s", device="cpu")
    predict = model.predict
    seen_devices = []

    def record_predict(*args, **kwargs):
        seen_devices.append(kwargs.get("device"))
        return predict(*args, **kwargs)

    monkeypatch.setattr(model, "predict", record_predict)
    image_dir = tmp_path / "images"
    matte_dir = tmp_path / "mattes"
    trimap_dir = tmp_path / "trimaps"
    image_dir.mkdir()
    matte_dir.mkdir()
    trimap_dir.mkdir()
    _image(height=16, width=20).save(image_dir / "sample.png")
    matte = np.zeros((16, 20), dtype=np.uint8)
    matte[4:12, 5:15] = 255
    Image.fromarray(matte, mode="L").save(matte_dir / "sample.png")
    Image.fromarray(matte, mode="L").save(trimap_dir / "sample.png")

    metrics = model.val(
        data=str(tmp_path),
        trimap_dir=str(trimap_dir),
        device="cpu",
        verbose=False,
    )
    assert seen_devices == ["cpu"]
    assert metrics["metrics/MAE"] == pytest.approx(0.0)
    assert metrics["metrics/Smeasure"] == pytest.approx(1.0)


def test_converter_is_checksum_pinned_and_rejects_other_files(tmp_path):
    converter_path = (
        Path(__file__).resolve().parents[2] / "weights" / "convert_vitmatte_weights.py"
    )
    weights_dir = str(converter_path.parent)
    if weights_dir not in sys.path:
        sys.path.insert(0, weights_dir)
    spec = importlib.util.spec_from_file_location(
        "convert_vitmatte_weights", converter_path
    )
    assert spec is not None and spec.loader is not None
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    wrong = tmp_path / "model.safetensors"
    wrong.write_bytes(b"not the audited checkpoint")
    with pytest.raises(ValueError, match="size mismatch"):
        converter.verify_source_checkpoint(wrong)
    assert converter.OFFICIAL_SIZE == 103_294_572
    assert converter.OFFICIAL_SHA256 == (
        "bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb"
    )
