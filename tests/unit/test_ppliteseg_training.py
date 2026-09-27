"""Optimizer, schedule, and forward-pass tests for the PP-LiteSeg trainer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from libreyolo import LibrePPLiteSeg
from libreyolo.models.ppliteseg.trainer import PPLiteSegTrainer
from libreyolo.training.config import PPLiteSegConfig
from libreyolo.training.scheduler import PolyLRScheduler

pytestmark = [pytest.mark.unit, pytest.mark.ppliteseg]

BACKBONE_PREFIX = "encoder.backbone."


def _trainer(model: LibrePPLiteSeg, **overrides) -> PPLiteSegTrainer:
    """A trainer with just enough state wired for optimizer/schedule checks."""
    trainer = PPLiteSegTrainer.__new__(PPLiteSegTrainer)
    trainer.model = model.model
    trainer.wrapper_model = model
    trainer.device = torch.device("cpu")
    trainer.num_classes = model.nb_classes
    overrides.setdefault("num_classes", model.nb_classes)
    trainer.config = PPLiteSegConfig(size=model.size, **overrides)
    return trainer


def test_config_matches_the_source_recipe():
    config = PPLiteSegConfig()
    assert config.optimizer == "sgd"
    assert config.lr0 == 0.01
    assert config.momentum == 0.9
    assert config.weight_decay == 5e-4
    assert config.head_lr_mult == 10.0
    assert config.scheduler == "poly"
    assert config.warmup_epochs == 10
    assert config.epochs == 800
    assert config.ema is True and config.ema_decay == 0.9999
    assert config.amp is False, "the released recipe runs full precision"
    assert config.edge_kernel == 5


def test_optimizer_splits_backbone_from_the_rest_and_zeroes_bn_bias_decay():
    model = LibrePPLiteSeg(size="t50", device="cpu")
    trainer = _trainer(model)
    optimizer = trainer._setup_optimizer()
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == 0.9

    multipliers = {group["lr_mult"] for group in optimizer.param_groups}
    assert multipliers == {1.0, 10.0}
    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(0.01 * group["lr_mult"])
        # BN/bias groups carry ndim <= 1 params and no weight decay.
        if group["weight_decay"] == 0.0:
            assert all(param.ndim <= 1 for param in group["params"])
        else:
            assert group["weight_decay"] == pytest.approx(5e-4)
            assert all(param.ndim > 1 for param in group["params"])

    # Every trainable parameter lands in exactly one group.
    grouped = sum(len(group["params"]) for group in optimizer.param_groups)
    assert grouped == sum(1 for p in model.model.parameters() if p.requires_grad)


def test_backbone_group_holds_only_backbone_parameters():
    model = LibrePPLiteSeg(size="t50", device="cpu")
    trainer = _trainer(model)
    optimizer = trainer._setup_optimizer()
    backbone_ids = {
        id(param)
        for name, param in model.model.named_parameters()
        if name.startswith(BACKBONE_PREFIX)
    }
    for group in optimizer.param_groups:
        ids = {id(param) for param in group["params"]}
        if group["lr_mult"] == 1.0:
            assert ids <= backbone_ids
        else:
            assert not (ids & backbone_ids)


def test_head_multiplier_survives_warmup_and_decay():
    model = LibrePPLiteSeg(size="t50", device="cpu")
    trainer = _trainer(model)
    head_group = {"lr_mult": 10.0}
    backbone_group = {"lr_mult": 1.0}
    for base_lr in (0.0, 0.005, 0.01):
        assert trainer._scale_lr(base_lr, head_group) == pytest.approx(base_lr * 10.0)
        assert trainer._scale_lr(base_lr, backbone_group) == pytest.approx(base_lr)
    # A group without the key falls back to 1x rather than dropping to zero.
    assert trainer._scale_lr(0.01, {}) == pytest.approx(0.01)


def test_uniform_lr_collapses_to_two_groups():
    model = LibrePPLiteSeg(size="t50", device="cpu")
    optimizer = _trainer(model, head_lr_mult=1.0)._setup_optimizer()
    assert {group["lr_mult"] for group in optimizer.param_groups} == {1.0}
    assert len(optimizer.param_groups) == 2  # decay / no-decay only


def test_scheduler_is_poly_with_warmup():
    model = LibrePPLiteSeg(size="t50", device="cpu")
    trainer = _trainer(model, epochs=100, warmup_epochs=10)
    scheduler = trainer.create_scheduler(iters_per_epoch=10)
    assert isinstance(scheduler, PolyLRScheduler)
    assert scheduler.update_lr(0) == pytest.approx(0.0)
    assert scheduler.update_lr(100) == pytest.approx(0.01)  # end of warmup
    mid = scheduler.update_lr(550)
    assert 0.0 < mid < 0.01
    assert scheduler.update_lr(550) == pytest.approx(0.01 * (1 - 0.5) ** 0.9)
    assert scheduler.update_lr(1000) == pytest.approx(0.0, abs=1e-12)
    # The schedule is monotonically non-increasing after warmup.
    values = [scheduler.update_lr(i) for i in range(100, 1001, 50)]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def test_scheduler_rejects_a_non_poly_request():
    model = LibrePPLiteSeg(size="t50", device="cpu")
    trainer = _trainer(model, scheduler="cosine")
    with pytest.raises(ValueError, match="polynomial schedule"):
        trainer.create_scheduler(iters_per_epoch=10)


def test_on_forward_returns_total_loss_and_named_components():
    torch.manual_seed(0)
    model = LibrePPLiteSeg(size="t50", nb_classes=4, device="cpu")
    trainer = _trainer(model, num_classes=4)
    model.model.train()
    imgs = torch.rand(2, 3, 64, 128)
    targets = torch.randint(0, 4, (2, 64, 128))
    outputs = trainer.on_forward(imgs, targets)
    assert "total_loss" in outputs
    assert set(outputs) == {"main", "aux0", "aux1", "aux2", "loss", "total_loss"}
    assert torch.isfinite(outputs["total_loss"])
    components = trainer.get_loss_components(outputs)
    assert "total_loss" not in components
    assert all(isinstance(value, float) for value in components.values())


def test_criterion_tracks_the_resolved_class_count():
    model = LibrePPLiteSeg(size="t50", nb_classes=4, device="cpu")
    trainer = _trainer(model, num_classes=4)
    assert trainer.criterion.num_classes == 4
    trainer.num_classes = 9
    assert trainer.criterion.num_classes == 9


def test_trainer_identity_and_metric_key():
    model = LibrePPLiteSeg(size="b75", device="cpu")
    trainer = _trainer(model)
    assert trainer.get_model_family() == "ppliteseg"
    assert trainer.get_model_tag() == "LibrePPLiteSeg-b75"
    assert PPLiteSegTrainer.best_metric_key == "metrics/mIoU"
    assert PPLiteSegTrainer._config_class() is PPLiteSegConfig
    with pytest.raises(NotImplementedError, match="semantic-only"):
        trainer.create_transforms()


def test_train_rejects_an_off_stride_imgsz():
    model = LibrePPLiteSeg(size="t50", device="cpu")
    with pytest.raises(ValueError, match="divisible"):
        model.train(data="unused.yaml", imgsz=(500, 1000))


def test_optimizer_step_moves_every_group():
    torch.manual_seed(0)
    model = LibrePPLiteSeg(size="t50", nb_classes=3, device="cpu")
    trainer = _trainer(model, num_classes=3)
    optimizer = trainer._setup_optimizer()
    before = [
        [param.detach().clone() for param in group["params"][:2]]
        for group in optimizer.param_groups
    ]
    model.model.train()
    outputs = trainer.on_forward(torch.rand(2, 3, 64, 128), torch.randint(0, 3, (2, 64, 128)))
    optimizer.zero_grad()
    outputs["total_loss"].backward()
    optimizer.step()
    for group, originals in zip(optimizer.param_groups, before):
        moved = any(
            not torch.equal(param, original)
            for param, original in zip(group["params"][:2], originals)
        )
        assert moved, "every parameter group must actually take a step"


class _RecordingTrainer:
    """Stand-in for PPLiteSegTrainer that records the resume handshake."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        _RecordingTrainer.last = self

    def setup(self):
        self.calls.append("setup")

    def resume(self, path):
        self.calls.append(("resume", path))

    def train(self):
        self.calls.append("train")
        return {}


def test_resume_true_continues_the_loaded_run(tmp_path):
    import numpy as np
    from PIL import Image

    for split in ("train", "val"):
        for subdir in ("images", "masks"):
            (tmp_path / subdir / split).mkdir(parents=True)
        for index in range(2):
            pixels = np.random.default_rng(index).integers(
                0, 256, (64, 128, 3), dtype=np.uint8
            )
            mask = np.zeros((64, 128), dtype=np.uint8)
            mask[:, 64:] = 1
            Image.fromarray(pixels).save(tmp_path / "images" / split / f"{index}.png")
            Image.fromarray(mask).save(tmp_path / "masks" / split / f"{index}.png")
    data = tmp_path / "data.yaml"
    data.write_text(
        "train: images/train\nval: images/val\nmasks_dir: masks\nnames: [left, right]\n"
    )
    common = dict(
        data=str(data),
        batch=2,
        imgsz=(64, 128),
        device="cpu",
        workers=0,
        loggers=[],
        warmup_epochs=0,
    )
    model = LibrePPLiteSeg(size="t50", nb_classes=2, device="cpu")
    model.input_size = (64, 128)
    first = model.train(epochs=1, project=str(tmp_path / "runs"), **common)
    last = Path(first["last_checkpoint"])

    resumed = LibrePPLiteSeg(str(last), size="t50", device="cpu")
    resumed.input_size = (64, 128)
    results = resumed.train(epochs=2, resume=True, **common)

    # Only the second epoch runs, and it writes into the loaded run directory
    # instead of a fresh runs/train/ppliteseg_exp2.
    assert len(results["epoch_losses"]) == 1
    assert Path(results["last_checkpoint"]) == last
    assert [p.name for p in (tmp_path / "runs").iterdir()] == ["ppliteseg_exp"]


def test_resume_path_resumes_that_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("libreyolo.models.ppliteseg.trainer.PPLiteSegTrainer", _RecordingTrainer)
    checkpoint = tmp_path / "elsewhere.pt"
    model = LibrePPLiteSeg(size="t50", device="cpu")

    model.train(data="unused.yaml", resume=str(checkpoint))

    trainer = _RecordingTrainer.last
    assert trainer.calls == ["setup", ("resume", str(checkpoint)), "train"]
    assert trainer.kwargs["project"] == "runs/train"
    assert trainer.kwargs["name"] == "ppliteseg_exp"


def test_resume_true_requires_a_checkpoint(monkeypatch):
    monkeypatch.setattr("libreyolo.models.ppliteseg.trainer.PPLiteSegTrainer", _RecordingTrainer)
    with pytest.raises(ValueError, match="requires a checkpoint"):
        LibrePPLiteSeg(size="t50", device="cpu").train(data="unused.yaml", resume=True)
