"""Published ConvNeXt V2 classifiers and supervised fine-tuning smoke.

Run with -m e2e. Uses CUDA when available, otherwise CPU. These tests are
not promoted into the detection-only general nightly catalog.
"""

from pathlib import Path

import pytest
import torch

from libreyolo import LibreYOLO
from libreyolo.models.convnextv2.nn import ARCH_DEFS
from tests.e2e.conftest import require_test_weights

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.convnextv2,
    pytest.mark.external_data,
    pytest.mark.network,
]


@pytest.mark.parametrize("size", list(ARCH_DEFS))
def test_published_classifier(size, sample_image):
    weight = require_test_weights(f"LibreConvNeXtV2{size}-cls.pt")
    model = LibreYOLO(weight, device="cuda" if torch.cuda.is_available() else "cpu")
    result = model.predict(sample_image)[0]
    assert model.family == "convnextv2" and model.size == size
    assert result.probs.data.shape == (1000,)
    assert torch.isfinite(result.probs.data).all()
    assert len(result.probs.top5) == 5
    assert model._weight_metadata["weight_license"] == "cc-by-nc-4.0"


@pytest.mark.network
def test_finetune_smoke10(tmp_path):
    weight = require_test_weights("LibreConvNeXtV2atto-cls.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreYOLO(weight, device=device)
    original = model.model.stages[0][0].grn.gamma.detach().cpu().clone()
    result = model.train(
        data="smoke10",
        epochs=3,
        batch=10,
        imgsz=224,
        workers=0,
        device=device,
        amp=False,
        ema=False,
        project=str(tmp_path),
        name="finetune",
        exist_ok=True,
    )
    assert model.nb_classes == 10
    assert not torch.equal(original, model.model.stages[0][0].grn.gamma.detach().cpu())
    best = Path(result["best_checkpoint"])
    restored = LibreYOLO(str(best), device=device)
    assert restored._weight_metadata["weight_commercial_use"] is False
    metrics = restored.val(data="smoke10", workers=0, device=device, imgsz=224)
    assert 0 <= metrics["metrics/accuracy_top1"] <= 1
