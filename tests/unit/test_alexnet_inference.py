"""Unified AlexNet prediction and classification-validation acceptance tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from PIL import Image


pytestmark = [pytest.mark.unit, pytest.mark.alexnet]


@pytest.mark.external_data
@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("LIBREYOLO_ALEXNET_CKPT"),
    reason="Set LIBREYOLO_ALEXNET_CKPT to LibreAlexNetb-cls.pt.",
)
def test_unified_predict_real_image_golden():
    """The converted official checkpoint returns the frozen parkour golden."""
    from libreyolo import LibreYOLO
    from tests.fixture_images import PARKOUR_FIXTURE

    model = LibreYOLO(os.environ["LIBREYOLO_ALEXNET_CKPT"], device="cpu")
    result = model.predict(PARKOUR_FIXTURE)

    assert result.boxes is None
    assert result.probs is not None
    assert result.probs.data.shape == (1000,)
    torch.testing.assert_close(result.probs.data.sum(), torch.tensor(1.0))
    assert result.probs.top1 == 795
    assert result.names[result.probs.top1] == "ski"
    assert result.probs.top5 == [795, 908, 667, 701, 442]


def test_alexnet_val_uses_classification_pipeline(tmp_path):
    """The public val API returns classification metrics for an ImageFolder."""
    from libreyolo import LibreAlexNet

    for split in ("train", "val"):
        for class_name, color in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
            directory = tmp_path / split / class_name
            directory.mkdir(parents=True)
            Image.new("RGB", (256, 256), color=color).save(directory / "image.png")

    model = LibreAlexNet(size="b", nb_classes=2, device="cpu")
    metrics = model.val(
        data=str(Path(tmp_path)),
        batch=2,
        workers=0,
        device="cpu",
        verbose=False,
    )

    assert set(metrics) >= {
        "metrics/accuracy_top1",
        "metrics/accuracy_top5",
        "fitness",
    }
    assert 0.0 <= metrics["metrics/accuracy_top1"] <= 1.0
    assert metrics["metrics/accuracy_top5"] == 1.0
