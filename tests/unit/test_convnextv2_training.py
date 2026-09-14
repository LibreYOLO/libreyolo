"""CPU classification training, resume and weight-provenance lifecycle."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreConvNeXtV2, LibreYOLO
from libreyolo.models.convnextv2.trainer import ConvNeXtV2Trainer

pytestmark = pytest.mark.unit


def make_data(root):
    for split in ("train", "val"):
        for c in range(3):
            directory = root / split / f"c{c}"
            directory.mkdir(parents=True)
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, :, c] = 220
            for i in range(2):
                Image.fromarray(image).save(directory / f"{i}.png")


def test_train_and_resume(tmp_path):
    make_data(tmp_path / "data")
    torch.manual_seed(3)
    model = LibreConvNeXtV2(device="cpu")
    model._weight_metadata = {
        "weight_license": "cc-by-nc-4.0",
        "weight_commercial_use": False,
    }
    initial = model.model.stages[0][0].grn.gamma.detach().clone()
    result = model.train(
        data=str(tmp_path / "data"),
        epochs=1,
        batch=3,
        imgsz=64,
        workers=0,
        device="cpu",
        amp=False,
        ema=False,
        warmup_epochs=0,
        project=str(tmp_path / "runs"),
        name="train",
        exist_ok=True,
    )
    assert model.nb_classes == 3
    assert not torch.equal(initial, model.model.stages[0][0].grn.gamma)
    best = Path(result["best_checkpoint"])
    last = best.with_name("last.pt")
    checkpoint = torch.load(best, weights_only=True)
    assert checkpoint["model_family"] == "convnextv2"
    assert checkpoint["best_metric_key"] == "metrics/accuracy_top1"
    assert checkpoint["weight_license"] == "cc-by-nc-4.0"
    restored = LibreYOLO(str(last), device="cpu")
    assert restored._weight_metadata["weight_commercial_use"] is False
    restored.train(
        data=str(tmp_path / "data"),
        resume=True,
        epochs=2,
        workers=0,
        device="cpu",
        amp=False,
        project=str(tmp_path / "runs"),
        name="resumed",
        exist_ok=True,
    )
    assert ConvNeXtV2Trainer.best_metric_key == "metrics/accuracy_top1"


def test_save_and_ddp_preserve_terms_scratch_clears_them(tmp_path):
    model = LibreConvNeXtV2(device="cpu")
    model._weight_metadata = {
        "weight_license": "cc-by-nc-4.0",
        "weight_commercial_use": False,
    }
    model.save(tmp_path / "saved.pt")
    restored = LibreYOLO(str(tmp_path / "saved.pt"), device="cpu")
    assert restored._weight_metadata == model._weight_metadata
    from libreyolo.training.ddp_spawn import _bootstrap_checkpoint

    bootstrap = _bootstrap_checkpoint(restored)
    assert bootstrap["weight_license"] == "cc-by-nc-4.0"
    restored._reset_for_scratch(seed=3)
    assert "weight_license" not in restored._save_extra_metadata()
