"""YOLOv9-E2E trainer smoke tests — wiring only, no data."""

from __future__ import annotations

import pytest
import torch

from libreyolo import LibreYOLO9E2E
from libreyolo.training.callbacks import TrainEpochEvent

pytestmark = pytest.mark.unit


def _build_trainer(wrapper, **overrides):
    from libreyolo.models.yolo9_e2e.trainer import YOLO9E2ETrainer

    kwargs = dict(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="t",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=640,
        device="cpu",
        amp=False,
        ema=False,
        no_aug_epochs=0,
        warmup_epochs=0,
        eval_interval=-1,
    )
    kwargs.update(overrides)
    return YOLO9E2ETrainer(**kwargs)


def test_trainer_metadata():
    """Family tag, model tag, and config class must reflect yolo9_e2e."""
    from libreyolo.models.yolo9_e2e.config import YOLO9E2EConfig

    wrapper = LibreYOLO9E2E(None, size="t", device="cpu")
    trainer = _build_trainer(wrapper)
    assert trainer.get_model_family() == "yolo9_e2e"
    assert trainer.get_model_tag() == "YOLOv9-E2E-t"
    assert trainer._config_class() is YOLO9E2EConfig


def test_trainer_train_emits_epoch_callback(monkeypatch, tmp_path):
    """Smoke the real family trainer through BaseTrainer.train()."""
    wrapper = LibreYOLO9E2E(None, size="t", device="cpu")
    received = []
    trainer = _build_trainer(
        wrapper,
        callbacks=received.append,
        epochs=1,
        patience=0,
    )

    def setup():
        trainer.save_dir = tmp_path
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        trainer._is_setup = True

    monkeypatch.setattr(trainer, "setup", setup)
    monkeypatch.setattr(
        trainer,
        "_train_epoch",
        lambda epoch: (
            1.0,
            None,
            {"box": torch.tensor(0.1), "cls": 0.2},
            {"group0": 0.01},
        ),
    )
    monkeypatch.setattr(trainer, "_save_checkpoint", lambda *args, **kwargs: None)

    results = trainer.train()

    assert results["final_loss"] == pytest.approx(1.0)
    assert len(received) == 1
    assert isinstance(received[0], TrainEpochEvent)
    assert received[0].model_family == "yolo9_e2e"
    assert received[0].train_loss == pytest.approx(1.0)


def test_trainer_forward_returns_dual_branch_loss():
    """on_forward dispatches to the head's training-mode forward, which sums
    the one-to-many + one-to-one losses and returns a dual-branch loss dict."""
    wrapper = LibreYOLO9E2E(None, size="t", device="cpu")
    wrapper.model.train()
    trainer = _build_trainer(wrapper)

    imgs = torch.zeros(2, 3, 640, 640)
    targets = torch.zeros(2, 30, 5)
    targets[0, 0] = torch.tensor([3.0, 320.0, 240.0, 100.0, 80.0])
    targets[0, 1] = torch.tensor([17.0, 200.0, 200.0, 60.0, 40.0])
    targets[1, 0] = torch.tensor([1.0, 400.0, 320.0, 120.0, 100.0])

    out = trainer.on_forward(imgs, targets)
    assert "total_loss" in out
    assert torch.isfinite(out["total_loss"]), "total_loss must be finite"
    assert out["total_loss"].item() > 0
    for key in ("box_loss", "dfl_loss", "cls_loss"):
        assert torch.isfinite(out[key])


def test_trainer_backward_propagates_gradients():
    """A backward pass must produce non-zero gradients on backbone params."""
    wrapper = LibreYOLO9E2E(None, size="t", device="cpu")
    wrapper.model.train()
    trainer = _build_trainer(wrapper)

    imgs = torch.zeros(2, 3, 640, 640)
    targets = torch.zeros(2, 30, 5)
    targets[0, 0] = torch.tensor([3.0, 320.0, 240.0, 100.0, 80.0])
    targets[1, 0] = torch.tensor([1.0, 400.0, 320.0, 120.0, 100.0])

    out = trainer.on_forward(imgs, targets)
    out["total_loss"].backward()

    nonzero_grads = sum(
        1
        for p in wrapper.model.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert nonzero_grads > 0, "expected at least one parameter with nonzero grad"


def test_trainer_handles_empty_targets():
    """A batch where one image has zero GT boxes still yields a finite loss."""
    wrapper = LibreYOLO9E2E(None, size="t", device="cpu")
    wrapper.model.train()
    trainer = _build_trainer(wrapper)

    imgs = torch.zeros(2, 3, 640, 640)
    targets = torch.zeros(2, 30, 5)
    targets[0, 0] = torch.tensor([3.0, 320.0, 240.0, 100.0, 80.0])
    # Image 1: all padding (no boxes)

    out = trainer.on_forward(imgs, targets)
    assert torch.isfinite(out["total_loss"])


def test_trainer_one_to_one_branch_blocks_backbone_gradients():
    """The one-to-one towers read detached features, so their outputs must
    not send gradients back into the shared neck; the one-to-many branch
    still does, and the one-to-one towers still train."""
    wrapper = LibreYOLO9E2E(None, size="t", device="cpu")
    wrapper.model.train()
    head = wrapper.model.head

    def make_feats():
        return [
            torch.randn(2, tower[0].conv.in_channels, size, size, requires_grad=True)
            for tower, size in zip(head.anchor_convs, (80, 40, 20))
        ]

    feats = make_feats()
    branches = head(feats)
    sum(t.sum() for t in branches["one_to_one"]).backward()
    assert all(f.grad is None for f in feats)
    assert head.one_to_one_class_convs[0][-1].weight.grad is not None

    head.zero_grad(set_to_none=True)
    feats = make_feats()
    branches = head(feats)
    sum(t.sum() for t in branches["one_to_many"]).backward()
    assert all(f.grad is not None and f.grad.abs().sum() > 0 for f in feats)

    # The summed training loss still reaches the features (one-to-many part).
    feats = make_feats()
    targets = torch.zeros(2, 5, 5)
    targets[0, 0] = torch.tensor([3.0, 0.3, 0.2, 0.6, 0.5])
    head_out = head(feats, targets=targets, img_size=[640, 640])
    head_out["total_loss"].backward()
    assert all(f.grad is not None for f in feats)
