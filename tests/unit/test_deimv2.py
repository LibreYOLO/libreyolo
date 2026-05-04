"""Unit tests for the native DEIMv2 family."""

from __future__ import annotations

import numpy as np
import torch
import pytest


pytestmark = pytest.mark.unit

DEIMV2_SIZE_CASES = [
    ("atto", 320, 100),
    ("femto", 416, 150),
    ("pico", 640, 200),
    ("n", 640, 300),
    ("s", 640, 300),
    ("m", 640, 300),
    ("l", 640, 300),
    ("x", 640, 300),
]


def test_deimv2_is_registered_and_detects_filenames():
    from libreyolo import LibreDEIMv2
    from libreyolo.models.base.model import BaseModel
    from libreyolo.training.config import DEIMv2Config

    assert any(cls.__name__ == "LibreDEIMv2" for cls in BaseModel._registry)
    assert LibreDEIMv2.FAMILY == "deimv2"
    assert LibreDEIMv2.TRAIN_CONFIG is DEIMv2Config
    assert LibreDEIMv2.detect_size_from_filename("LibreDEIMv2atto.pt") == "atto"
    assert (
        LibreDEIMv2.detect_size_from_filename("deimv2_hgnetv2_pico_coco.pth") == "pico"
    )
    assert LibreDEIMv2.detect_size_from_filename("deimv2_dinov3_s_coco.pth") == "s"
    assert LibreDEIMv2.detect_size_from_filename("deimv2_hgnetv2_s_coco.pth") is None


@pytest.mark.parametrize(("size", "input_size", "queries"), DEIMV2_SIZE_CASES)
def test_deimv2_forward_shapes(size, input_size, queries):
    from libreyolo import LibreDEIMv2

    model = LibreDEIMv2(None, size=size, device="cpu")
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, input_size, input_size))

    assert out["pred_logits"].shape == (1, queries, 80)
    assert out["pred_boxes"].shape == (1, queries, 4)


def test_deimv2_factory_detects_upstream_style_checkpoint(tmp_path):
    from libreyolo import LibreDEIMv2, LibreYOLO

    src = LibreDEIMv2(None, size="atto", device="cpu")
    ckpt = tmp_path / "deimv2_hgnetv2_atto_coco.pth"
    torch.save({"model": src.model.state_dict()}, ckpt)

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "deimv2"
    assert loaded.size == "atto"
    assert loaded.input_size == 320


def test_deimv2_factory_loads_safetensors_checkpoint(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import save_model

    from libreyolo import LibreDEIMv2, LibreYOLO

    src = LibreDEIMv2(None, size="atto", device="cpu")
    ckpt = tmp_path / "model.safetensors"
    save_model(src.model, ckpt)

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "deimv2"
    assert loaded.size == "atto"
    assert loaded.input_size == 320


def test_deimv2_dino_sizes_use_imagenet_preprocessing():
    from libreyolo import LibreDEIMv2

    atto = LibreDEIMv2(None, size="atto", device="cpu")
    dino = LibreDEIMv2(None, size="s", device="cpu")

    assert atto.size not in {"s", "m", "l", "x"}
    assert dino.size in {"s", "m", "l", "x"}
    assert atto.model.uses_imagenet_norm is False
    assert dino.model.uses_imagenet_norm is True

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    atto_chw, _ = atto._get_preprocess_numpy()(img, input_size=2)
    dino_chw, _ = dino._get_preprocess_numpy()(img, input_size=2)
    np.testing.assert_allclose(atto_chw, 0.0)
    assert dino_chw.mean() < -1.0


def test_deimv2_val_preprocessor_matches_upstream_pil_resize():
    """DEIMv2 validation should match upstream PIL RGB resize semantics."""
    from PIL import Image

    from libreyolo.validation.preprocessors import (
        DEIMValPreprocessor,
        DEIMv2ValPreprocessor,
        DFINEValPreprocessor,
    )

    img_bgr = np.arange(4 * 7 * 3, dtype=np.uint8).reshape(4, 7, 3)
    preproc = DEIMv2ValPreprocessor(img_size=(3, 5))

    out, targets = preproc(img_bgr, np.zeros((0, 5), dtype=np.float32), (3, 5))

    expected_rgb = img_bgr[:, :, ::-1]
    expected = np.array(
        Image.fromarray(expected_rgb).resize((5, 3), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ).transpose(2, 0, 1)

    np.testing.assert_array_equal(out, expected)
    assert targets.shape == (preproc.max_labels, 5)
    assert not issubclass(DFINEValPreprocessor, DEIMv2ValPreprocessor)
    assert not issubclass(DEIMValPreprocessor, DEIMv2ValPreprocessor)
