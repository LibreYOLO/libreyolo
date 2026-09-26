"""CPU contracts for GTR oriented-box training (recipe, targets, loss, LoRA)."""

import math
import os
import random

import numpy as np
import pytest
import torch

from libreyolo import LibreGTR, LibreYOLO
from libreyolo.models.gtr.obb_nn import LibreGTROBBModel
from libreyolo.models.gtr.obb_trainer import (
    GTROBBConfig,
    GTROBBTrainer,
    GTROBBTrainTransform,
    rows_to_targets,
)
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


def test_recipe_follows_upstream_dota_finetune_configs():
    s = GTROBBConfig(size="s")
    assert (s.epochs, s.flat_epochs, s.imgsz) == (20, 20, 1024)
    assert s.backbone_lr_mult == pytest.approx(0.36)
    assert s.weight_decay == pytest.approx(1e-4)
    assert (s.flip_prob, s.degrees, s.no_aug_epochs) == (0.75, 180.0, 0)
    x = GTROBBConfig(size="x")
    assert (x.epochs, x.flat_epochs) == (30, 30)
    assert x.backbone_lr_mult == pytest.approx(0.032)
    assert x.weight_decay == pytest.approx(1.25e-4)
    assert GTROBBConfig(size="s", epochs=5).flat_epochs == 5
    with pytest.raises(ValueError, match="sizes 's' and 'x'"):
        GTROBBConfig(size="m")
    with pytest.raises(ValueError, match="Mosaic"):
        GTROBBConfig(size="s", mosaic_prob=0.5)


def _row(cx, cy, w, h, angle, cls=1):
    """Dataset row: un-rotated proxy xyxy, class, angle in [-pi/2, pi/2)."""
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cls, angle]


def _valid(rows):
    return rows[rows[:, 3] > 0]


def test_transform_fits_canvas_and_emits_long_edge_targets():
    transform = GTROBBTrainTransform(imgsz=256, flip_prob=0, degrees=0)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    # A 40x10 box at 30 degrees in a 200x100 image; the canvas scale is 1.28.
    targets = np.array([_row(100, 50, 40, 10, math.radians(30))], np.float32)
    chw, rows = transform(image, targets, (256, 256))
    assert chw.shape == (3, 256, 256) and rows.shape == (1000, 6)
    (row,) = _valid(rows)
    assert row[0] == 1
    np.testing.assert_allclose(row[1:3], [100 * 1.28 / 256, 50 * 1.28 / 256], atol=2e-3)
    np.testing.assert_allclose(row[3:5], [40 * 1.28 / 256, 10 * 1.28 / 256], atol=3e-3)
    assert row[5] == pytest.approx(30 / 180, abs=2e-3)
    # The padded bottom half is zero before normalization.
    assert np.allclose(chw[:, 200:, :], chw[:, 255, 0][:, None, None])


def test_flip_and_rotation_move_boxes_with_the_image(monkeypatch):
    targets = np.array([_row(64, 96, 60, 20, math.radians(30))], np.float32)
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    flip = GTROBBTrainTransform(imgsz=256, flip_prob=1.0, degrees=0)
    monkeypatch.setattr(random, "choice", lambda seq: "horizontal")
    (row,) = _valid(flip(image, targets, (256, 256))[1])
    assert row[1] == pytest.approx((256 - 64) / 256, abs=2e-3)
    assert row[2] == pytest.approx(96 / 256, abs=2e-3)
    assert row[5] == pytest.approx(150 / 180, abs=2e-3)  # mirrored angle

    rotate = GTROBBTrainTransform(
        imgsz=256, flip_prob=0.0, degrees=180, square_like_labels=(1,)
    )
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "choice", lambda seq: 90)
    (row,) = _valid(rotate(image, targets, (256, 256))[1])
    # 90 degrees counter-clockwise on screen about the canvas center.
    c = 255 / 2
    assert row[1] == pytest.approx((c + (96 - c)) / 256, abs=3e-3)
    assert row[2] == pytest.approx((c - (64 - c)) / 256, abs=3e-3)
    np.testing.assert_allclose(row[3:5], [60 / 256, 20 / 256], atol=3e-3)
    assert row[5] == pytest.approx(120 / 180, abs=3e-3)


def test_rotation_drops_boxes_whose_center_leaves_the_canvas(monkeypatch):
    transform = GTROBBTrainTransform(imgsz=256, flip_prob=0.0, degrees=180)
    targets = np.array([_row(5, 5, 8, 4, 0.0), _row(128, 128, 30, 10, 0.0)], np.float32)
    image = np.zeros((256, 256, 3), np.uint8)
    # Draws: flip check (skipped, flip_prob=0), rotation applied (< 0.5), then
    # 45 degrees: the corner box's center leaves the canvas and is dropped.
    draws = iter([0.9, 0.0, 0.625])
    monkeypatch.setattr(random, "random", lambda: next(draws))
    rows = _valid(transform(image, targets, (256, 256))[1])
    assert len(rows) == 1
    np.testing.assert_allclose(rows[0, 1:3], [0.5, 0.5], atol=3e-3)
    assert rows[0, 5] == pytest.approx(0.75, abs=3e-3)  # -45 deg as long edge
    transform.disable_strong_augs()
    rows = _valid(transform(image, targets, (256, 256))[1])
    np.testing.assert_allclose(
        rows[:, 1:3], [[5 / 256, 5 / 256], [0.5, 0.5]], atol=2e-3
    )


def test_rows_to_targets_drops_padding():
    rows = torch.zeros(2, 4, 6)
    rows[0, 0] = torch.tensor([3, 0.5, 0.5, 0.2, 0.1, 0.25])
    targets = rows_to_targets(rows, torch.device("cpu"))
    assert targets[0]["labels"].tolist() == [3]
    torch.testing.assert_close(targets[0]["boxes"], rows[0, :1, 1:])
    assert targets[1]["boxes"].shape == (0, 5)


def _trainer(num_classes=3):
    trainer = object.__new__(GTROBBTrainer)
    trainer.config = GTROBBConfig(size="s", num_classes=num_classes)
    trainer.device = torch.device("cpu")
    trainer.model = LibreGTROBBModel("s", num_classes, (256, 256)).train()
    trainer.criterion = trainer.build_criterion(distributed_normalize=False)
    return trainer


def test_training_step_is_finite_and_reaches_every_head():
    trainer = _trainer()
    rows = torch.zeros(2, 5, 6)
    rows[0, 0] = torch.tensor([1, 0.4, 0.5, 0.3, 0.1, 0.2])
    rows[0, 1] = torch.tensor([2, 0.7, 0.3, 0.1, 0.05, 0.9])
    out = trainer.on_forward(torch.randn(2, 3, 256, 256), rows)
    assert torch.isfinite(out["total_loss"])
    assert {"loss_mal", "loss_bbox", "loss_kld"} <= set(out)
    assert any(k.endswith("_dn_0") for k in out) and any("_enc_" in k for k in out)
    out["total_loss"].backward()
    for param in (
        trainer.model.decoder.dec_score_head[0].weight,
        trainer.model.backbone.backbone._model.blocks[0].attn.q_proj.weight,
    ):
        assert param.grad is not None and param.grad.abs().sum() > 0
    parts = trainer.get_loss_components(out)
    assert set(parts) == {"mal", "bbox", "kld"} and all(v > 0 for v in parts.values())


def test_lora_adapts_obb_decoder_and_backbone_and_reloads(tmp_path):
    pytest.importorskip("peft")
    from libreyolo.training.lora import apply_lora_to_gtr, module_has_lora

    model = LibreGTROBBModel("s", 15)
    apply_lora_to_gtr(model)
    adapted = {n for n, _ in model.named_modules() if n.endswith(".lora_A.default")}
    assert len(adapted) == 12 * 3 + 4 * 5
    assert any(".decoder.decoder.layers." in f".{n}" for n in adapted)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.02)
    ckpt = wrap_libreyolo_checkpoint(
        model.state_dict(),
        model_family="gtr",
        size="s",
        nc=15,
        task="obb",
        imgsz=1024,
    )
    path = tmp_path / "best.pt"
    torch.save(ckpt, path)
    loaded = LibreYOLO(str(path), device="cpu")
    assert isinstance(loaded, LibreGTR) and loaded.task == "obb"
    assert module_has_lora(loaded.model)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            value, loaded.model.state_dict()[key], rtol=0, atol=0
        )


def _obb_dataset(root):
    import cv2
    import yaml

    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(2):
            cv2.imwrite(
                str(root / "images" / split / f"{i}.jpg"),
                np.full((64, 96, 3), 40 * (i + 1), np.uint8),
            )
            (root / "labels" / split / f"{i}.txt").write_text(
                f"{i} 0.3 0.3 0.6 0.3 0.6 0.6 0.3 0.6\n"
            )
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": ["a", "b"],
            }
        )
    )
    return data


def test_obb_train_dispatches_to_obb_trainer_and_rebuilds_heads(tmp_path, monkeypatch):
    torch.manual_seed(0)
    ckpt = wrap_libreyolo_checkpoint(
        LibreGTROBBModel("s", 15).state_dict(),
        model_family="gtr",
        size="s",
        nc=15,
        task="obb",
        imgsz=1024,
    )
    path = tmp_path / "LibreGTRs-obb.pt"
    torch.save(ckpt, path)
    model = LibreYOLO(str(path), device="cpu")
    captured = {}
    monkeypatch.setattr(
        GTROBBTrainer, "train", lambda t: captured.update(trainer=t) or {}
    )
    model.train(data=str(_obb_dataset(tmp_path / "ds")), epochs=1, device="cpu")
    trainer = captured["trainer"]
    assert isinstance(trainer, GTROBBTrainer)
    assert trainer.config.imgsz == 1024 and trainer.config.flat_epochs == 1
    assert model.nb_classes == 2 and model.names == {0: "a", 1: "b"}
    assert model.model.decoder.dec_score_head[0].weight.shape[0] == 2


@pytest.mark.external_data
def test_criterion_matches_pinned_upstream(monkeypatch):
    """Set GTR_UPSTREAM to a GTR checkout at the pinned revision."""
    if not os.environ.get("GTR_UPSTREAM"):
        pytest.skip("Set GTR_UPSTREAM")
    import importlib
    import pathlib
    import subprocess
    import sys
    import types

    upstream = pathlib.Path(os.environ["GTR_UPSTREAM"])
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == "782e737efe2e6437ac537fbdcee089673d3376c1"
    root = upstream / "engine"

    def module(name, path=None):
        m = types.ModuleType(name)
        if path is not None:
            m.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, m)
        return m

    module("gtr_obb_ref", root)
    module("gtr_obb_ref.gtr", root / "gtr")
    module("gtr_obb_ref.gtr.obb", root / "gtr/obb")
    module("gtr_obb_ref.core", root / "core").register = lambda *a, **k: lambda cls: cls
    module("gtr_obb_ref.misc", root / "misc")
    dist = module("gtr_obb_ref.misc.dist_utils")
    dist.get_world_size = lambda: 1
    dist.is_dist_available_and_initialized = lambda: False
    module("gtr_obb_ref.gtr.obb.decoder").NUM_DIST = 6
    ref_criterion = importlib.import_module("gtr_obb_ref.gtr.obb.criterion")
    ref_matcher = importlib.import_module("gtr_obb_ref.gtr.obb.matcher")

    torch.manual_seed(0)
    model = LibreGTROBBModel("s", 15, (256, 256)).train()
    targets = []
    for n in (3, 0, 5):
        boxes = torch.rand(n, 5) * torch.tensor([0.8, 0.8, 0.2, 0.1, 0.999])
        boxes[:, :4] += torch.tensor([0.1, 0.1, 0.1, 0.03])
        targets.append({"labels": torch.randint(0, 15, (n,)), "boxes": boxes})
    outputs = model(torch.randn(3, 3, 256, 256), targets=targets)

    def clone(o):
        if isinstance(o, dict):
            return {k: clone(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clone(v) for v in o]
        return o

    trainer = _trainer(15)
    matcher = ref_matcher.OBBHungarianMatcher(
        weight_dict={"cost_class": 2, "cost_chamfer": 5, "cost_kld": 2},
        alpha=0.25,
        gamma=2.0,
    )
    reference = ref_criterion.OBBGTRCriterion(
        matcher=matcher,
        weight_dict={"loss_mal": 1, "loss_bbox": 5, "loss_kld": 5, "loss_fgl": 0.15},
        losses=["mal", "boxes"],
        gamma=1.5,
        alpha=0.75,
        num_classes=15,
        reg_max=32,
        group_detr=3,
    ).train()
    ours = trainer.criterion.train()(clone(outputs), targets)
    theirs = reference(clone(outputs), targets)
    assert set(ours) == set(theirs) and len(ours) > 20
    for key in ours:
        torch.testing.assert_close(ours[key], theirs[key], rtol=0, atol=1e-6)
