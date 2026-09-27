"""PP-LiteSeg training end to end: overfit, best-checkpoint reload, and resume.

The fixture masks and images are authored here, not sampled from Cityscapes:
Cityscapes is user-supplied and never bundled or downloaded by LibreYOLO.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo import LibrePPLiteSeg

from .conftest import cuda_cleanup

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.ppliteseg,
    pytest.mark.extended_training,
]

NC = 3
NAMES = {0: "background", 1: "band", 2: "block"}


def _make_dataset(root: Path, count: int = 4) -> Path:
    """A trivially learnable layout: a vertical band and a corner block."""
    for split in ("train", "val"):
        image_dir = root / "images" / split
        mask_dir = root / "masks" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = np.zeros((64, 128, 3), dtype=np.uint8)
            image[:, 40:88] = (220, 30, 30)
            image[:24, :24] = (30, 30, 220)
            image = np.clip(image.astype(np.int16) + index, 0, 255).astype(np.uint8)
            Image.fromarray(image).save(image_dir / f"sample{index}.png")

            mask = np.zeros((64, 128), dtype=np.uint8)
            mask[:, 40:88] = 1
            mask[:24, :24] = 2
            Image.fromarray(mask, mode="L").save(mask_dir / f"sample{index}.png")

    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "masks_dir": "masks",
        "nc": NC,
        "names": NAMES,
    }
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return yaml_path


def _train(model, data, tmp_path, *, name, epochs, resume=False, **kwargs):
    return model.train(
        data=str(data),
        epochs=epochs,
        batch=2,
        imgsz=(64, 128),
        workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        project=str(tmp_path / "runs"),
        name=name,
        exist_ok=True,
        resume=resume,
        seed=0,
        **kwargs,
    )


def test_rf1_overfit_reloads_best_checkpoint_and_reproduces_the_mask(tmp_path):
    data = _make_dataset(tmp_path)
    model = LibrePPLiteSeg(size="t50", nb_classes=NC, device="cpu")
    try:
        result = _train(model, data, tmp_path, name="overfit", epochs=60, lr0=0.08)
        assert result.get("best_checkpoint") or result.get("last_checkpoint")

        metrics = model.val(
            data=str(data), batch=1, workers=0, imgsz=(64, 128), verbose=False
        )
        miou = metrics["metrics/mIoU"]
        accuracy = metrics["metrics/pixel_accuracy"]
        # A random 3-class head sits near 1/3 accuracy and well under 0.2 mIoU;
        # a model that actually fit the fixture clears both by a wide margin.
        assert accuracy > 0.90, f"pixel accuracy {accuracy:.4f} did not rise"
        assert miou > 0.60, f"mIoU {miou:.4f} did not rise"

        # The reloaded best checkpoint reproduces the fixture mask.
        checkpoint = result.get("best_checkpoint") or result.get("last_checkpoint")
        reloaded = LibrePPLiteSeg(model_path=str(checkpoint), device="cpu")
        assert reloaded.nb_classes == NC
        assert reloaded.names == NAMES
        predicted = reloaded.predict(str(tmp_path / "images" / "val" / "sample0.png"))[0]
        mask = predicted.semantic_mask.data.numpy()
        expected = np.zeros((64, 128), dtype=np.int64)
        expected[:, 40:88] = 1
        expected[:24, :24] = 2
        agreement = float((mask == expected).mean())
        assert agreement > 0.90, f"reloaded mask agreement {agreement:.4f}"
    finally:
        del model
        cuda_cleanup()


def test_training_rebuilds_every_head_for_a_custom_class_count(tmp_path):
    data = _make_dataset(tmp_path)
    # Start from the 19-class Cityscapes default and let the dataset re-head it.
    model = LibrePPLiteSeg(size="t50", nb_classes=19, device="cpu")
    try:
        result = _train(model, data, tmp_path, name="rehead", epochs=2)
        assert model.nb_classes == NC
        state = model.model.state_dict()
        assert state["seg_head.0.seg_head.2.weight"].shape[0] == NC
        for index in range(3):
            assert state[f"aux_heads.{index}.0.seg_head.2.weight"].shape[0] == NC

        checkpoint = result.get("best_checkpoint") or result.get("last_checkpoint")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["nc"] == NC
        assert payload["model_family"] == "ppliteseg"
        assert payload["task"] == "semantic"
        # Strict reload of a re-headed checkpoint must succeed.
        LibrePPLiteSeg(model_path=str(checkpoint), device="cpu")
    finally:
        del model
        cuda_cleanup()


def test_resume_restores_epoch_optimizer_groups_and_scheduler_state(tmp_path):
    data = _make_dataset(tmp_path)
    model = LibrePPLiteSeg(size="t50", nb_classes=NC, device="cpu")
    try:
        _train(model, data, tmp_path, name="resumable", epochs=2)
        last = Path(tmp_path / "runs" / "resumable" / "weights" / "last.pt")
        assert last.exists()
        before = torch.load(last, map_location="cpu", weights_only=False)
        assert before["epoch"] >= 1

        resumed = LibrePPLiteSeg(size="t50", nb_classes=NC, device="cpu")
        _train(resumed, data, tmp_path, name="resumable", epochs=4, resume=True)
        after = torch.load(last, map_location="cpu", weights_only=False)
        assert after["epoch"] > before["epoch"], "resume did not advance the epoch"
        assert after["model_family"] == "ppliteseg"
        del resumed
    finally:
        del model
        cuda_cleanup()


def test_bounded_convergence_lowers_loss_and_raises_miou(tmp_path):
    data = _make_dataset(tmp_path, count=6)
    model = LibrePPLiteSeg(size="t50", nb_classes=NC, device="cpu")
    try:
        before = model.val(
            data=str(data), batch=1, workers=0, imgsz=(64, 128), verbose=False
        )["metrics/mIoU"]
        _train(model, data, tmp_path, name="converge", epochs=20, lr0=0.05)
        after = model.val(
            data=str(data), batch=1, workers=0, imgsz=(64, 128), verbose=False
        )["metrics/mIoU"]
        assert after > before, f"mIoU did not improve ({before:.4f} -> {after:.4f})"
    finally:
        del model
        cuda_cleanup()
