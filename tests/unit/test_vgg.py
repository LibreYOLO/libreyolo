"""LibreVGG registry, architecture, preprocessing, and postprocess tests."""

from __future__ import annotations

import gc

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.vgg]


def test_registered_as_inference_only_classify_family():
    from libreyolo.models.base import BaseModel
    from libreyolo.models.vgg.model import LibreVGG

    assert any(family is LibreVGG for family in BaseModel._registry)
    model = LibreVGG(size="16", nb_classes=10, device="cpu")
    assert model.family == "vgg"
    assert model.task == "classify"
    assert model.input_size == 224
    assert model.crop_pct == 0.875
    assert model.interpolation == "bilinear"
    assert model.TRAIN_CONFIG is None
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="unused")


def test_canonical_filename_resolution_and_download_url():
    from libreyolo.models.vgg.model import LibreVGG

    for size in ("16", "19", "16bn", "19bn"):
        filename = f"LibreVGG{size}-cls.pt"
        assert LibreVGG.detect_size_from_filename(filename) == size
        assert LibreVGG.detect_task_from_filename(filename) == "classify"
        assert LibreVGG.get_download_url(filename) == (
            f"https://huggingface.co/LibreYOLO/LibreVGG{size}-cls/resolve/"
            f"main/{filename}"
        )
        assert LibreVGG.detect_size_from_filename(f"LibreVGG{size}.pt") is None

    # Regression: 16 is a prefix of 16bn, but the longest size must win.
    assert LibreVGG.detect_size_from_filename("LibreVGG16bn-cls.pt") == "16bn"


@pytest.mark.parametrize("size", ["16", "19", "16bn", "19bn"])
def test_detect_size_classes_and_forward(size):
    from libreyolo.models.vgg.model import LibreVGG
    from libreyolo.models.vgg.nn import VGG

    model = VGG(size=size, num_classes=10).eval()
    state_dict = model.state_dict()
    assert LibreVGG.can_load(state_dict)
    assert LibreVGG.detect_size(state_dict) == size
    assert LibreVGG.detect_nb_classes(state_dict) == 10
    malformed = state_dict.copy()
    malformed["classifier.6.weight"] = torch.empty(10, 512, device="meta")
    assert LibreVGG.can_load(malformed) is False
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 10)
    del output, state_dict, model
    gc.collect()


def test_rejects_alexnet_shape_signature():
    from libreyolo.models.vgg.model import LibreVGG

    alexnet_like = {
        "features.0.weight": torch.empty(64, 3, 11, 11),
        "classifier.1.weight": torch.empty(4096, 256 * 6 * 6),
        "classifier.6.weight": torch.empty(1000, 4096),
    }
    assert LibreVGG.can_load(alexnet_like) is False
    assert LibreVGG.detect_size(alexnet_like) is None


def test_preprocess_matches_official_weight_transform():
    from torchvision.models import VGG16_Weights

    from libreyolo.models.vgg.model import LibreVGG

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (317, 241, 3), dtype=np.uint8))
    expected = VGG16_Weights.IMAGENET1K_V1.transforms()(image)
    actual = LibreVGG(size="16", nb_classes=10, device="cpu").eval_transform(224)(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_postprocess_returns_normalized_probabilities():
    from libreyolo.postprocess.vgg import postprocess

    logits = torch.randn(1, 10)
    result = postprocess(logits)
    assert set(result) == {"probs"}
    assert result["probs"].shape == (10,)
    torch.testing.assert_close(result["probs"].sum(), torch.tensor(1.0))
    assert int(result["probs"].argmax()) == int(logits.argmax())
