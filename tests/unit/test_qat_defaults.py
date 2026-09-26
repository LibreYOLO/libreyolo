"""Tests for quantization-aware training safeguards."""

import logging
from types import SimpleNamespace

import pytest

from libreyolo.training.qat_defaults import apply_qat_training_guards
from libreyolo.training.trainer import BaseTrainer


pytestmark = pytest.mark.unit


class _StopSetup(Exception):
    pass


class _ProbeTrainer(BaseTrainer):
    """Concrete trainer that stops immediately after the QAT guard block."""

    def get_model_family(self):
        raise _StopSetup

    def get_model_tag(self):
        raise NotImplementedError

    def create_transforms(self):
        raise NotImplementedError

    def create_scheduler(self, iters_per_epoch):
        raise NotImplementedError

    def get_loss_components(self, outputs):
        raise NotImplementedError


def test_qat_guards_disable_ema_sync_bn_and_checkpoint_averaging():
    config = SimpleNamespace(ema=True, sync_bn=True, average_best=5)

    changed = apply_qat_training_guards(config)

    assert changed == ("ema", "sync_bn", "average_best")
    assert config.ema is False
    assert config.sync_bn is False
    assert config.average_best == 0


def test_qat_guards_leave_disabled_options_unchanged():
    config = SimpleNamespace(ema=False, sync_bn=False, average_best=0)

    assert apply_qat_training_guards(config) == ()
    assert config.ema is False
    assert config.sync_bn is False
    assert config.average_best == 0


def test_trainer_logs_qat_guard_changes(caplog):
    config = SimpleNamespace(ema=True, sync_bn=True, average_best=3, imgsz=(32, 64))
    trainer = _ProbeTrainer.__new__(_ProbeTrainer)
    trainer._is_setup = False
    trainer.config = config
    trainer.wrapper_model = SimpleNamespace(
        _quant_manifest={"recipe": "int8", "state": "prepared"}
    )

    with caplog.at_level(logging.WARNING, logger="libreyolo.training.trainer"):
        with pytest.raises(_StopSetup):
            trainer.setup()

    assert config.ema is False
    assert config.sync_bn is False
    assert config.average_best == 0
    assert "ema=False, sync_bn=False, average_best=0" in caplog.text
    assert "QAT recipe 'int8'" in caplog.text


def test_float_trainer_does_not_apply_qat_guards():
    config = SimpleNamespace(ema=True, sync_bn=True, average_best=3, imgsz=(32, 64))
    trainer = _ProbeTrainer.__new__(_ProbeTrainer)
    trainer._is_setup = False
    trainer.config = config
    trainer.wrapper_model = SimpleNamespace(_quant_manifest=None)

    with pytest.raises(_StopSetup):
        trainer.setup()

    assert config.ema is True
    assert config.sync_bn is True
    assert config.average_best == 3


def test_qat_trainer_runs_off_mps():
    """MPS lacks fake-quantize kernels, so QAT trains on CPU there."""
    import torch

    trainer = _ProbeTrainer(
        torch.nn.Linear(1, 1),
        wrapper_model=SimpleNamespace(_quant_manifest={"recipe": "int8"}),
        device="mps",
    )

    assert trainer.device == torch.device("cpu")
