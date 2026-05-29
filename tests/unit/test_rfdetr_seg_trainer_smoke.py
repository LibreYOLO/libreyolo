"""BaseTrainer device-sync smoke tests — wiring only, no data."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit


class _MinimalTrainer:
    """Minimal concrete BaseTrainer that satisfies all abstract methods."""

    def get_model_family(self):
        return "test"

    def get_model_tag(self):
        return "test-s"

    def create_transforms(self):
        return None, None

    def create_scheduler(self, _iters_per_epoch):
        sched = MagicMock()
        sched.update_lr = MagicMock(return_value=1e-4)
        return sched

    def get_loss_components(self, _outputs):
        return {}


def _make_trainer(wrapper_device="meta"):
    from libreyolo.training.trainer import BaseTrainer

    # Dynamically create a concrete subclass.
    Trainer = type("_T", (_MinimalTrainer, BaseTrainer), {})

    model = nn.Linear(4, 4)
    wrapper = SimpleNamespace(device=torch.device(wrapper_device), task="detect")

    trainer = Trainer(
        model=model,
        wrapper_model=wrapper,
        size="s",
        num_classes=2,
        data=None,
        epochs=1,
        batch=2,
        imgsz=64,
        device="cpu",
        amp=False,
        ema=False,
        no_aug_epochs=0,
        warmup_epochs=0,
        eval_interval=-1,
    )
    return wrapper, trainer


def test_setup_syncs_wrapper_device_to_trainer_device():
    """wrapper_model.device must equal trainer.device after setup()."""
    wrapper, trainer = _make_trainer(wrapper_device="meta")
    assert wrapper.device != trainer.device

    fake_loader = MagicMock()
    fake_loader.__len__ = MagicMock(return_value=2)

    with tempfile.TemporaryDirectory() as tmp:
        with (
            patch.object(trainer, "_setup_data", side_effect=lambda: setattr(trainer, "train_loader", fake_loader)),
            patch.object(trainer, "_setup_optimizer", return_value=torch.optim.SGD(trainer.model.parameters(), lr=1e-4)),
            patch("libreyolo.training.trainer.barrier"),
            patch.object(trainer, "_get_save_dir", return_value=Path(tmp)),
        ):
            trainer.setup()

    assert wrapper.device == trainer.device, (
        f"wrapper.device={wrapper.device!r} not synced to trainer.device={trainer.device!r}"
    )
