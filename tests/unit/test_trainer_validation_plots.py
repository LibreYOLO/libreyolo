from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


class _Trainer:
    def get_model_family(self):
        return "dummy"

    def get_model_tag(self):
        return "dummy"

    def create_transforms(self):
        return None

    def create_scheduler(self):
        return None

    def get_loss_components(self, outputs):
        return {}


def _make_trainer(config):
    from libreyolo.training.trainer import BaseTrainer

    trainer_cls = type("_T", (_Trainer, BaseTrainer), {})
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = config
    return trainer


def test_save_plots_forces_validation_on_final_epoch():
    trainer = _make_trainer(
        SimpleNamespace(eval_interval=2, epochs=5, save_plots=True)
    )

    assert trainer._should_validate_epoch(1) is True
    assert trainer._should_validate_epoch(2) is False
    assert trainer._should_validate_epoch(4) is True


def test_final_epoch_validation_not_forced_when_plots_disabled():
    trainer = _make_trainer(
        SimpleNamespace(eval_interval=2, epochs=5, save_plots=False)
    )

    assert trainer._should_validate_epoch(4) is False
