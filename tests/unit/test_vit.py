"""LibreViT unit coverage for registry, inference, and preprocessing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.vit.model import LibreViT
from libreyolo.models.vit.nn import VisionTransformer
from libreyolo.models.vit.utils import build_eval_transform
from libreyolo.postprocess.vit import postprocess

pytestmark = [pytest.mark.unit, pytest.mark.vit]


def _signature_state(size: str, num_classes: int = 1000) -> dict:
    specs = {
        "ti": (192, 12),
        "s": (384, 12),
        "b": (768, 12),
        "l": (1024, 24),
    }
    embed_dim, depth = specs[size]
    state = {
        "cls_token": torch.empty(1, 1, embed_dim),
        "pos_embed": torch.empty(1, 197, embed_dim),
        "patch_embed.proj.weight": torch.empty(embed_dim, 3, 16, 16),
        "head.weight": torch.empty(num_classes, embed_dim),
    }
    state.update(
        {
            f"blocks.{index}.norm1.weight": torch.empty(embed_dim)
            for index in range(depth)
        }
    )
    return state


def test_registered_and_classify_only_contract():
    from libreyolo.models.base import BaseModel
    from libreyolo.validation.vit_validator import ViTClassifyValidator

    assert LibreViT in BaseModel._registry
    model = LibreViT(size="ti", nb_classes=7, device="cpu")
    assert model.family == "vit"
    assert model.task == "classify"
    assert model.input_size == 224
    assert model.crop_pct == 0.9
    assert model.interpolation == "bicubic"
    assert model.validator_class is ViTClassifyValidator
    # crop_pct now routes through self.config so val(crop_pct=...) is honored
    # (#878); with no override the family's native 0.9 is unchanged.
    _no_override = ViTClassifyValidator.__new__(ViTClassifyValidator)
    _no_override.config = SimpleNamespace(crop_pct=None)
    assert _no_override._dataset_transform_kwargs() == {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "interpolation": "bicubic",
        "crop_pct": 0.9,
    }


def test_canonical_multichar_filename_and_required_suffix():
    assert LibreViT.detect_size_from_filename("LibreViTti-cls.pt") == "ti"
    assert LibreViT.detect_size_from_filename("LibreViTs-cls.pt") == "s"
    assert LibreViT.detect_size_from_filename("LibreViTb-cls.pt") == "b"
    assert LibreViT.detect_size_from_filename("LibreViTl-cls.pt") == "l"
    assert LibreViT.detect_task_from_filename("LibreViTti-cls.pt") == "classify"
    assert LibreViT.detect_size_from_filename("LibreViTti.pt") is None
    assert LibreViT.detect_size_from_filename("LibreVitt-cls.pt") is None


@pytest.mark.parametrize("size", ["ti", "s", "b", "l"])
def test_exact_shipped_signatures_detect_size_and_classes(size):
    state = _signature_state(size, num_classes=37)
    assert LibreViT.can_load(state)
    assert LibreViT.detect_size(state) == size
    assert LibreViT.detect_nb_classes(state) == 37


def test_unsupported_patch_resolution_and_incomplete_depth_are_rejected():
    patch32 = _signature_state("b")
    patch32["patch_embed.proj.weight"] = torch.empty(768, 3, 32, 32)
    resolution384 = _signature_state("b")
    resolution384["pos_embed"] = torch.empty(1, 577, 768)
    missing_block = _signature_state("ti")
    missing_block.pop("blocks.4.norm1.weight")

    for state in (patch32, resolution384, missing_block):
        assert LibreViT.detect_size(state) is None
        assert not LibreViT.can_load(state)


def test_sibling_discriminators_reject_vit_bidirectionally():
    from libreyolo import (
        LibreCLIP,
        LibreConvNeXt,
        LibreDEIMv2,
        LibreDINOv2,
        LibreEfficientNetV2,
        LibreEoMT,
        LibreMobileNetV4,
        LibreRFDETR,
        LibreResNet,
    )

    vit_state = _signature_state("ti")
    siblings = (
        LibreDINOv2,
        LibreCLIP,
        LibreResNet,
        LibreConvNeXt,
        LibreMobileNetV4,
        LibreEfficientNetV2,
        LibreRFDETR,
        LibreDEIMv2,
        LibreEoMT,
    )
    for sibling in siblings:
        assert not sibling.can_load(vit_state), sibling.__name__

    sibling_signatures = (
        {
            "backbone.patch_embed.proj.weight": torch.empty(1),
            "linear.weight": torch.empty(1),
        },
        {
            "logit_scale": torch.empty(1),
            "text_projection": torch.empty(1),
            "visual.conv1.weight": torch.empty(1),
        },
        {
            "conv1.weight": torch.empty(1),
            "fc.weight": torch.empty(1),
            "layer1.0.conv1.weight": torch.empty(1),
        },
        {
            "stem.0.weight": torch.empty(1),
            "head.fc.weight": torch.empty(1),
            "stages.0.blocks.0.gamma": torch.empty(1),
        },
        {
            "conv_stem.weight": torch.empty(1),
            "conv_head.weight": torch.empty(1),
            "blocks.0.pw_exp.conv.weight": torch.empty(1),
        },
        {
            "conv_stem.weight": torch.empty(1),
            "conv_head.weight": torch.empty(1),
            "blocks.0.se.conv_reduce.weight": torch.empty(1),
        },
        {"transformer.decoder.weight": torch.empty(1)},
        {"decoder.swish_ffn.weight": torch.empty(1)},
        {
            "query.weight": torch.empty(1),
            "mask_head.fc1.weight": torch.empty(1),
            "mask_head.fc2.weight": torch.empty(1),
            "mask_head.fc3.weight": torch.empty(1),
            "class_predictor.weight": torch.empty(1),
            "embeddings.patch_embeddings.projection.weight": torch.empty(1),
        },
    )
    for state in sibling_signatures:
        assert not LibreViT.can_load(state)


def test_forward_reset_classifier_and_postprocess():
    model = VisionTransformer(size="ti", num_classes=7).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 224, 224))
    assert logits.shape == (1, 7)

    model.reset_classifier(3)
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 224, 224))
    assert logits.shape == (1, 3)
    probs = postprocess(logits)["probs"]
    assert probs.shape == (3,)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert int(probs.argmax()) == int(logits.argmax())


def test_eval_preprocessing_exactly_matches_timm_augreg():
    timm = pytest.importorskip("timm")
    from timm.data import create_transform, resolve_model_data_config

    reference = timm.create_model(
        "vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=False
    )
    config = resolve_model_data_config(reference)
    pixels = np.arange(277 * 301 * 3, dtype=np.uint32).reshape(277, 301, 3)
    image = Image.fromarray(pixels.astype(np.uint8))

    expected = create_transform(**config, is_training=False)(image)
    actual = build_eval_transform(224, crop_pct=0.9)(image)
    assert config["mean"] == (0.5, 0.5, 0.5)
    assert config["std"] == (0.5, 0.5, 0.5)
    assert config["crop_pct"] == 0.9
    assert torch.equal(actual, expected)


def test_training_is_explicitly_unsupported():
    model = LibreViT(size="ti", nb_classes=2, device="cpu")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="smoke10")
