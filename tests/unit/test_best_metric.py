"""Tests for the user-selectable best-checkpoint / early-stopping metric."""

from __future__ import annotations

import pytest

from libreyolo.training.best_metric import (
    BEST_METRIC_ALIASES,
    SUPPORTED_BEST_METRIC_TASKS,
    resolve_best_metric,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# resolve_best_metric
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task,alias,expected",
    [
        ("detect", "map50-95", "metrics/mAP50-95"),
        ("detect", "map50", "metrics/mAP50"),
        ("detect", "map75", "metrics/mAP75"),
        ("detect", "f1", "metrics/best_conf_f1"),
        ("classify", "top1", "metrics/accuracy_top1"),
        ("classify", "top5", "metrics/accuracy_top5"),
        ("classify", "f1", "metrics/f1"),
        ("classify", "precision", "metrics/precision"),
        ("classify", "recall", "metrics/recall"),
    ],
)
def test_resolve_best_metric_table(task, alias, expected):
    assert resolve_best_metric(alias, task) == expected


def test_resolve_best_metric_is_case_and_whitespace_insensitive():
    assert resolve_best_metric("  F1 ", "detect") == "metrics/best_conf_f1"
    assert resolve_best_metric("MAP50-95", "Detect") == "metrics/mAP50-95"


def test_resolve_best_metric_unknown_alias_lists_valid_values():
    with pytest.raises(ValueError) as excinfo:
        resolve_best_metric("map95", "detect")
    message = str(excinfo.value)
    assert "map95" in message
    assert "detect" in message
    assert "f1, map50, map50-95, map75" in message


def test_resolve_best_metric_unsupported_task_lists_supported_tasks():
    with pytest.raises(ValueError) as excinfo:
        resolve_best_metric("f1", "pose")
    message = str(excinfo.value)
    assert "pose" in message
    assert "classify, detect" in message


def test_alias_table_matches_supported_tasks():
    assert tuple(sorted(BEST_METRIC_ALIASES)) == SUPPORTED_BEST_METRIC_TASKS
    assert SUPPORTED_BEST_METRIC_TASKS == ("classify", "detect")


# ---------------------------------------------------------------------------
# TrainConfig.best_metric
# ---------------------------------------------------------------------------


def test_train_config_best_metric_defaults_to_none():
    from libreyolo.training.config import TrainConfig

    assert TrainConfig().best_metric is None


@pytest.mark.parametrize(
    "raw,expected",
    [(" F1 ", "f1"), ("MAP50", "map50"), ("", None), ("   ", None), (None, None)],
)
def test_train_config_best_metric_is_normalized(raw, expected):
    from libreyolo.training.config import TrainConfig

    assert TrainConfig(best_metric=raw).best_metric == expected


def test_train_config_from_kwargs_accepts_best_metric_without_warning(recwarn):
    from libreyolo.training.config import TrainConfig

    config = TrainConfig.from_kwargs(best_metric="f1")
    assert config.best_metric == "f1"
    assert not [w for w in recwarn if "Unknown training config keys" in str(w.message)]


# ---------------------------------------------------------------------------
# BaseTrainer resolution
# ---------------------------------------------------------------------------


def _dummy_trainer(**kwargs):
    from types import SimpleNamespace

    from torch import nn

    from libreyolo.training.trainer import BaseTrainer

    class DummyTrainer(BaseTrainer):
        def get_model_family(self) -> str:
            return "dummy"

        def get_model_tag(self) -> str:
            return "dummy"

        def create_transforms(self):
            raise NotImplementedError

        def create_scheduler(self, iters_per_epoch: int):
            raise NotImplementedError

        def get_loss_components(self, outputs):
            return {}

    task = kwargs.pop("task", "detect")
    return DummyTrainer(
        model=nn.Linear(1, 1),
        wrapper_model=SimpleNamespace(task=task),
        data=None,
        device="cpu",
        ema=False,
        **kwargs,
    )


def test_trainer_without_best_metric_keeps_class_default():
    trainer = _dummy_trainer()
    assert trainer.best_metric_key == "metrics/mAP50-95"
    assert trainer._best_metric_explicit is False
    assert trainer._branch_best_key("metrics/accuracy_top1") == "metrics/accuracy_top1"


def test_trainer_resolves_detect_alias_into_best_metric_key():
    trainer = _dummy_trainer(best_metric="f1", task="detect")
    assert trainer.best_metric_key == "metrics/best_conf_f1"
    assert trainer._best_metric_explicit is True
    assert trainer._branch_best_key("metrics/mAP50-95") == "metrics/best_conf_f1"


def test_trainer_resolves_classify_alias_into_best_metric_key():
    trainer = _dummy_trainer(best_metric="F1", task="classify")
    assert trainer.best_metric_key == "metrics/f1"
    assert trainer._branch_best_key("metrics/accuracy_top1") == "metrics/f1"


def test_trainer_rejects_best_metric_for_unsupported_task():
    with pytest.raises(ValueError, match="not supported for task 'pose'"):
        _dummy_trainer(best_metric="f1", task="pose")


def test_trainer_rejects_unknown_best_metric_alias():
    with pytest.raises(ValueError, match="Unknown best_metric 'map95'"):
        _dummy_trainer(best_metric="map95", task="detect")


# ---------------------------------------------------------------------------
# FOMO always tracks grid F1
# ---------------------------------------------------------------------------


def test_fomo_trainer_rejects_best_metric():
    from types import SimpleNamespace

    from torch import nn

    from libreyolo.models.fomo.trainer import FOMOTrainer

    with pytest.raises(NotImplementedError, match="grid_F1"):
        FOMOTrainer(
            model=nn.Linear(1, 1),
            wrapper_model=SimpleNamespace(task="detect"),
            data=None,
            device="cpu",
            ema=False,
            best_metric="f1",
        )


# ---------------------------------------------------------------------------
# V-JEPA2 has no working per-epoch validation path, so it cannot honor a
# selected metric.
# ---------------------------------------------------------------------------


def test_vjepa2_trainer_rejects_best_metric():
    from types import SimpleNamespace

    from torch import nn

    from libreyolo.models.vjepa2.trainer import VJEPA2Trainer

    with pytest.raises(NotImplementedError, match="V-JEPA2"):
        VJEPA2Trainer(
            model=nn.Linear(1, 1),
            wrapper_model=SimpleNamespace(task="classify"),
            data=None,
            device="cpu",
            ema=False,
            best_metric="f1",
        )


# ---------------------------------------------------------------------------
# Python API: covered families expose best_metric explicitly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module,cls",
    [
        ("libreyolo.models.yolo9.model", "LibreYOLO9"),
        ("libreyolo.models.rfdetr.model", "LibreRFDETR"),
        ("libreyolo.models.resnet.model", "LibreResNet"),
        ("libreyolo.models.convnext.model", "LibreConvNeXt"),
        ("libreyolo.models.mobilenetv4.model", "LibreMobileNetV4"),
        ("libreyolo.models.efficientnetv2.model", "LibreEfficientNetV2"),
    ],
)
def test_train_signature_exposes_best_metric(module, cls):
    import importlib
    import inspect

    train = getattr(importlib.import_module(module), cls).train
    parameters = inspect.signature(train).parameters
    assert "best_metric" in parameters
    assert parameters["best_metric"].default is None
