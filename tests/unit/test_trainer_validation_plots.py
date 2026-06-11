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


def test_trainer_validation_routes_point_task(monkeypatch):
    import torch

    class _DummyPointValidator:
        def __init__(self, model, config):
            self._primary_threshold = 0.02
            self.model = model
            self.config = config

        def run(self):
            return {
                "metrics/precision": 0.85,
                "metrics/recall": 0.75,
                "metrics/f1": 0.80,
                "metrics/mAP@0.02": 0.70,
                "metrics/mAP@[0.01:0.10]": 0.65,
                "fitness": 0.65,
            }

    monkeypatch.setattr("libreyolo.validation.PointValidator", _DummyPointValidator)

    trainer = _make_trainer(
        SimpleNamespace(
            data="data.yaml",
            batch=4,
            imgsz=128,
            amp=False,
            workers=0,
            save_plots=False,
        )
    )
    trainer.device = torch.device("cpu")
    trainer.is_distributed = False
    trainer.ema_model = None
    trainer.model = SimpleNamespace(state_dict=lambda: {})
    
    trainer.wrapper_model = SimpleNamespace(
        task="point",
        model=trainer.model,
    )
    
    trainer._is_final_epoch = lambda epoch: False
    trainer.save_dir = SimpleNamespace()
    trainer._scalar_mapping = lambda x: x

    result = trainer._run_validation(0)

    assert result is not None
    assert result["best_metric_key"] == "fitness"
    assert result["best_metric"] == pytest.approx(0.65)
    assert result["mAP50"] == pytest.approx(0.70)
