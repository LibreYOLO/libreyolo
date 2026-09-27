"""Unit tests for the native TinyFormer family."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = [pytest.mark.unit, pytest.mark.tinyformer]


def test_tinyformer_is_registered_and_detects_filenames():
    from libreyolo import LibreTinyFormer
    from libreyolo.models.base.model import BaseModel
    from libreyolo.training.config import TinyFormerConfig

    assert any(cls.__name__ == "LibreTinyFormer" for cls in BaseModel._registry)
    assert LibreTinyFormer.FAMILY == "tinyformer"
    assert LibreTinyFormer.TRAIN_CONFIG is TinyFormerConfig
    assert LibreTinyFormer.SUPPORTED_TASKS == ("detect",)
    assert LibreTinyFormer.WEIGHT_VARIANTS == ("visdrone", "obj2coco")
    assert LibreTinyFormer.detect_size_from_filename("LibreTinyFormers.pt") == "s"
    # "xl" must win over the single-char "x" and "l" codes.
    assert LibreTinyFormer.detect_size_from_filename("LibreTinyFormerxl.pt") == "xl"
    assert (
        LibreTinyFormer.detect_size_from_filename("LibreTinyFormers-visdrone.pt")
        == "s"
    )
    assert LibreTinyFormer.detect_size_from_filename("TinyFormer-XL-pbm.pth") == "xl"
    assert LibreTinyFormer.detect_size_from_filename("TinyFormer-S-pbm.pth") == "s"
    assert LibreTinyFormer.detect_size_from_filename("LibreDEIMv2s.pt") is None


@pytest.mark.parametrize("size", ["s", "m"])
def test_tinyformer_forward_shapes(size):
    from libreyolo import LibreTinyFormer

    model = LibreTinyFormer(None, size=size, device="cpu")
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, 640, 640))

    assert out["pred_logits"].shape == (1, 300, 80)
    assert out["pred_boxes"].shape == (1, 300, 4)


@pytest.mark.parametrize(
    ("size", "expected"),
    [("s", "s"), ("m", "m"), ("l", "l"), ("x", "x"), ("xl", "xl")],
)
def test_tinyformer_detect_size_from_state_dict(size, expected):
    from libreyolo import LibreTinyFormer
    from libreyolo.models.tinyformer.nn import LibreTinyFormerModel

    sd = LibreTinyFormerModel(config=size, nb_classes=80).state_dict()
    assert LibreTinyFormer.detect_size(sd) == expected
    assert LibreTinyFormer.detect_nb_classes(sd) == 80
    assert LibreTinyFormer.can_load(sd) is True


def test_tinyformer_factory_loads_v1_metadata_checkpoint(tmp_path):
    from libreyolo import LibreTinyFormer, LibreYOLO

    src = LibreTinyFormer(None, size="s", device="cpu")
    ckpt = tmp_path / "LibreTinyFormers.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            src.model.state_dict(),
            model_family="tinyformer",
            size="s",
            task="detect",
            nc=80,
            names={i: f"class_{i}" for i in range(80)},
            imgsz=640,
        ),
        ckpt,
    )

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "tinyformer"
    assert loaded.size == "s"
    assert loaded.input_size == 640


def test_tinyformer_factory_autoconverts_upstream_style_checkpoint(tmp_path):
    from libreyolo import LibreTinyFormer, LibreYOLO

    src = LibreTinyFormer(None, size="s", device="cpu")
    ckpt = tmp_path / "TinyFormer-S-pbm.pth"
    torch.save({"model": src.model.state_dict()}, ckpt)

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "tinyformer"
    assert loaded.size == "s"


def test_tinyformer_and_deimv2_can_load_reject_each_other():
    """Bidirectional routing: TinyFormer carries backbone.sda./proj_c1 markers
    that DEIMv2 must reject, and DEIMv2's backbone.sta. layout must not be
    claimed by TinyFormer."""
    from libreyolo.models.deimv2.model import LibreDEIMv2
    from libreyolo.models.tinyformer.model import LibreTinyFormer

    tinyformer_sd = {
        "backbone.sda.0.0.weight": torch.zeros(1),
        "backbone.proj_c1.0.weight": torch.zeros(1),
        "backbone.dinov3.cls_token": torch.zeros(1, 1, 384),
    }
    deimv2_sd = {
        "backbone.sta.stem.0.weight": torch.zeros(1),
        "backbone.dinov3.cls_token": torch.zeros(1, 1, 384),
    }

    assert LibreTinyFormer.can_load(tinyformer_sd) is True
    assert LibreDEIMv2.can_load(tinyformer_sd) is False
    assert LibreTinyFormer.can_load(deimv2_sd) is False
    assert LibreDEIMv2.can_load(deimv2_sd) is True


def test_tinyformer_preprocessing_is_always_imagenet_normalised():
    from libreyolo import LibreTinyFormer

    model = LibreTinyFormer(None, size="s", device="cpu")
    assert model.model.uses_imagenet_norm is True

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    chw, _ = model._get_preprocess_numpy()(img, input_size=2)
    assert chw.mean() < -1.0


def test_tinyformer_public_preprocessors_reject_unaligned_imgsz():
    from libreyolo import LibreTinyFormer

    model = LibreTinyFormer(None, size="s", device="cpu")
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive multiple of 32"):
        model._preprocess(image, input_size=513)
    with pytest.raises(ValueError, match="positive multiple of 32"):
        model._get_val_preprocessor(img_size=513)


def test_tinyformer_export_wrapper_returns_tuple():
    from libreyolo.models.tinyformer.nn import (
        LibreTinyFormerModel,
        TinyFormerExportWrapper,
    )

    model = LibreTinyFormerModel(config="s", nb_classes=80)
    wrapper = TinyFormerExportWrapper(model)
    with torch.no_grad():
        out = wrapper(torch.zeros(1, 3, 640, 640))

    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (1, 300, 80)
    assert out[1].shape == (1, 300, 4)


def test_tinyformer_is_nms_free_in_backends():
    from libreyolo.backends.base import _is_nms_free_family

    assert _is_nms_free_family("tinyformer") is True
