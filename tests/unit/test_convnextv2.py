"""Hermetic ConvNeXt V2 recognition, checkpoint, and prediction contracts."""

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreConvNeXt, LibreConvNeXtV2, LibreYOLO
from libreyolo.models.convnext.nn import ConvNeXt
from libreyolo.models.convnextv2.nn import ARCH_DEFS, GRN, ConvNeXtV2
from libreyolo.utils.serialization import validate_checkpoint_metadata

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("size", list(ARCH_DEFS))
def test_size_recognition_and_filename(size):
    with torch.device("meta"):
        model = ConvNeXtV2(size, num_classes=7)
    state = model.state_dict()
    assert LibreConvNeXtV2.can_load(state)
    assert LibreConvNeXtV2.detect_size(state) == size
    assert LibreConvNeXtV2.detect_nb_classes(state) == 7
    filename = f"LibreConvNeXtV2{size}-cls.pt"
    assert LibreConvNeXtV2.detect_size_from_filename(filename) == size
    assert LibreConvNeXtV2.get_download_url(filename).endswith(f"/{filename}")
    assert not LibreConvNeXt.can_load(state)
    assert LibreConvNeXt.detect_size_from_filename(filename) is None
    assert (
        LibreConvNeXtV2.detect_size_from_filename(f"LibreConvNeXtV2{size}.pt") is None
    )


def test_reject_v1_and_incomplete_depth():
    with torch.device("meta"):
        v1 = ConvNeXt().state_dict()
        v2 = ConvNeXtV2().state_dict()
    assert not LibreConvNeXtV2.can_load(v1)
    del v2["stages.2.2.grn.gamma"]
    assert not LibreConvNeXtV2.can_load(v2)


def test_grn_nonzero_parameters_and_gradient():
    grn = GRN(12)
    torch.nn.init.normal_(grn.gamma)
    torch.nn.init.normal_(grn.beta)
    x = torch.randn(2, 5, 7, 12, requires_grad=True)
    norm = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
    expected = (
        grn.gamma * (x * (norm / (norm.mean(-1, keepdim=True) + 1e-6))) + grn.beta + x
    )
    torch.testing.assert_close(grn(x), expected, rtol=0, atol=0)
    grn(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert grn.gamma.grad.abs().sum() > 0


def test_prediction_save_reload_and_rebuild(tmp_path):
    model = LibreConvNeXtV2(nb_classes=7, device="cpu")
    image = Image.fromarray(
        np.random.default_rng(3).integers(0, 256, (90, 140, 3), dtype=np.uint8)
    )
    before = model.predict(image, imgsz=64)[0].probs.data
    path = tmp_path / "arbitrary.pt"
    model.save(path)
    validate_checkpoint_metadata(torch.load(path, weights_only=True))
    restored = LibreYOLO(str(path), device="cpu")
    assert restored.family == "convnextv2"
    torch.testing.assert_close(
        restored.predict(image, imgsz=64)[0].probs.data, before, rtol=0, atol=0
    )
    head_dtype = restored.model.head.weight.dtype
    restored._rebuild_for_new_classes(3)
    assert restored.model.head.out_features == 3
    assert restored.model.head.weight.dtype == head_dtype


def test_raw_autoconversion(tmp_path):
    state = ConvNeXtV2(num_classes=7).state_dict()
    path = tmp_path / "foreign.pth"
    torch.save({"model": state}, path)
    model = LibreYOLO(str(path), device="cpu")
    assert (model.family, model.size, model.task, model.nb_classes) == (
        "convnextv2",
        "atto",
        "classify",
        7,
    )
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_preprocessing_matches_official_224():
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    image = Image.fromarray(
        np.random.default_rng(5).integers(0, 256, (259, 413, 3), dtype=np.uint8)
    )
    expected = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )(image)
    model = LibreConvNeXtV2(device="cpu")
    actual = model._preprocess(image)[0][0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_weight_terms_require_known_source(monkeypatch, tmp_path):
    from libreyolo.models.convnextv2 import utils

    monkeypatch.setattr(
        utils, "checkpoint_sha256", lambda _: utils.SOURCE_SHA256["atto"]
    )
    official = LibreConvNeXtV2.upstream_checkpoint_metadata(
        {}, source=tmp_path / "source.pt"
    )
    assert official["weight_license"] == "cc-by-nc-4.0"
    monkeypatch.setattr(utils, "checkpoint_sha256", lambda _: "0" * 64)
    custom = LibreConvNeXtV2.upstream_checkpoint_metadata(
        {}, source=tmp_path / "source.pt"
    )
    assert "weight_license" not in custom


def test_official_converter_rejects_custom_weights(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "weights"))
    from convert_convnextv2_weights import convert

    path = tmp_path / "custom.pt"
    torch.save({"model": ConvNeXtV2().state_dict()}, path)
    with pytest.raises(ValueError, match="not the pinned official"):
        convert(path, tmp_path / "converted.pt")
