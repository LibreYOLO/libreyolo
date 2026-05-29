"""RF-DETR seg trainer smoke tests — wiring only, no data."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

pytestmark = pytest.mark.unit


def _make_trainer(task="segment"):
    from libreyolo import LibreRFDETR
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    wrapper = LibreRFDETR(None, size="s", task=task, device="cpu")
    trainer = RFDETRTrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="s",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=560,
        device="cpu",
        amp=False,
        ema=False,
        no_aug_epochs=0,
        warmup_epochs=0,
        eval_interval=-1,
    )
    return wrapper, trainer


def _fake_train_loader(n=2):
    loader = MagicMock()
    loader.__len__ = MagicMock(return_value=n)
    return loader


def _fake_scheduler():
    sched = MagicMock()
    sched.update_lr = MagicMock(return_value=1e-4)
    return sched


def _fake_optimizer(model):
    dummy = torch.nn.Linear(2, 2)
    return torch.optim.AdamW(dummy.parameters(), lr=1e-4)


def test_setup_syncs_wrapper_device_to_trainer_device():
    """wrapper_model.device must equal trainer.device after setup()."""
    wrapper, trainer = _make_trainer(task="segment")

    # Poison the wrapper device so we can detect that setup() corrects it.
    wrapper.device = torch.device("meta")
    assert wrapper.device != trainer.device

    with tempfile.TemporaryDirectory() as tmp:
        with (
            patch.object(trainer, "_setup_data", side_effect=lambda: setattr(trainer, "train_loader", _fake_train_loader())),
            patch.object(trainer, "_setup_optimizer", side_effect=lambda: _fake_optimizer(trainer.model)),
            patch.object(trainer, "create_scheduler", return_value=_fake_scheduler()),
            patch.object(trainer, "on_setup"),
            patch("libreyolo.training.trainer.barrier"),
            patch.object(trainer, "_get_save_dir", return_value=Path(tmp)),
        ):
            trainer.setup()

    assert wrapper.device == trainer.device, (
        f"wrapper.device={wrapper.device!r} was not synced to trainer.device={trainer.device!r}"
    )
