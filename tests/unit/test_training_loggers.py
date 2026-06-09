"""Unit tests for built-in experiment loggers (TensorBoard, MLflow, W&B)."""

from __future__ import annotations

import pickle
from types import MappingProxyType, SimpleNamespace

import pytest
from torch import nn

from libreyolo.training.callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from libreyolo.training.loggers import (
    MLflowLogger,
    TensorBoardLogger,
    WandbLogger,
    resolve_loggers,
)
from libreyolo.training.loggers import base as loggers_base
from libreyolo.training.loggers import (
    mlflow_logger as mlflow_module,
    tensorboard_logger as tensorboard_module,
    wandb_logger as wandb_module,
)
from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit


class LoggerTrainer(BaseTrainer):
    """Minimal trainer that fakes two epochs without touching data/optimizers."""

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

    def setup(self):
        import torch

        self.save_dir = self._test_save_dir
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self._is_setup = True

    def _train_epoch(self, epoch: int):
        metric = 0.7 if epoch == 0 else 0.5
        return (
            1.5 + epoch,
            {
                "mAP50": metric,
                "mAP50_95": metric,
                "best_metric": metric,
                "best_metric_key": "metrics/mAP50-95",
                "metrics": {"metrics/mAP50": metric, "metrics/mAP50-95": metric},
            },
            {"box": 0.2},
            {"group0": 0.01},
        )

    def _save_checkpoint(self, epoch, loss, val_metrics=None, is_best=None):
        pass


# ---------------------------------------------------------------------------
# Event fixtures
# ---------------------------------------------------------------------------


def _start_event(save_dir: str = "/tmp/libreyolo") -> TrainStartEvent:
    return TrainStartEvent(
        start_epoch=1,
        total_epochs=2,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir=save_dir,
        config={"epochs": 2, "lr0": 0.01, "batch": 16},
    )


def _epoch_event() -> TrainEpochEvent:
    return TrainEpochEvent(
        epoch=1,
        total_epochs=2,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir="/tmp/libreyolo",
        train_loss=1.5,
        train_loss_items={"box": 0.2, "cls": 0.3},
        lr={"group0": 0.01},
        val_metrics={"metrics/mAP50": 0.6, "metrics/mAP50-95": 0.4},
        validated=True,
        is_best=True,
        current_metric=0.4,
        current_metric_name="metrics/mAP50-95",
        best_metric=0.4,
        best_metric_name="metrics/mAP50-95",
        best_epoch=1,
        epoch_seconds=2.5,
    )


def _end_event(save_dir: str = "/tmp/libreyolo") -> TrainEndEvent:
    return TrainEndEvent(
        total_epochs=2,
        completed_epochs=2,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir=save_dir,
        final_loss=1.0,
        best_metric=0.4,
        best_epoch=1,
        total_seconds=5.0,
        results={"final_loss": 1.0},
    )


def _exception_event() -> TrainExceptionEvent:
    exc = RuntimeError("boom")
    return TrainExceptionEvent(
        epoch=1,
        total_epochs=2,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir="/tmp/libreyolo",
        exception=exc,
        exception_type="RuntimeError",
        exception_message="boom",
        elapsed_seconds=1.0,
    )


# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------


class FakeMlflow:
    def __init__(self):
        self.calls = []

    def set_tracking_uri(self, uri):
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name):
        self.calls.append(("set_experiment", name))

    def start_run(self, run_name=None):
        self.calls.append(("start_run", run_name))

    def log_params(self, params):
        self.calls.append(("log_params", params))

    def log_metrics(self, metrics, step=None):
        self.calls.append(("log_metrics", metrics, step))

    def log_artifact(self, path):
        self.calls.append(("log_artifact", path))

    def end_run(self, status=None):
        self.calls.append(("end_run", status))


class FakeWandbRun:
    def __init__(self):
        self.logged = []
        self.summary = {}
        self.artifacts = []
        self.finished = None

    def log(self, metrics, step=None):
        self.logged.append((metrics, step))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self, exit_code=0):
        self.finished = exit_code


class FakeWandb:
    def __init__(self):
        self.init_kwargs = None
        self.run = FakeWandbRun()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def Artifact(self, name, type):
        return SimpleNamespace(name=name, type=type, files=[], add_file=lambda p: None)


class FakeSummaryWriter:
    instances = []

    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.scalars = []
        self.texts = []
        self.closed = False
        FakeSummaryWriter.instances.append(self)

    def add_scalar(self, tag, value, global_step=None):
        self.scalars.append((tag, value, global_step))

    def add_text(self, tag, text):
        self.texts.append((tag, text))

    def flush(self):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setattr(mlflow_module, "_import_mlflow", lambda: fake)
    return fake


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = FakeWandb()
    monkeypatch.setattr(wandb_module, "_import_wandb", lambda: fake)
    return fake


@pytest.fixture
def fake_tensorboard(monkeypatch):
    FakeSummaryWriter.instances = []
    monkeypatch.setattr(
        tensorboard_module, "_import_summary_writer", lambda: FakeSummaryWriter
    )
    return FakeSummaryWriter


# ---------------------------------------------------------------------------
# TrainStartEvent.config
# ---------------------------------------------------------------------------


def test_train_start_event_exposes_read_only_config():
    event = _start_event()
    assert event.config["lr0"] == 0.01
    assert isinstance(event.config, MappingProxyType)
    with pytest.raises(TypeError):
        event.config["lr0"] = 0.5


def test_trainer_start_event_carries_resolved_config(tmp_path):
    received = []

    class StartCallback:
        def on_train_start(self, event):
            received.append(event)

    trainer = LoggerTrainer(
        model=nn.Linear(1, 1),
        data=None,
        device="cpu",
        ema=False,
        epochs=2,
        callbacks=StartCallback(),
    )
    trainer._test_save_dir = tmp_path

    trainer.train()

    assert len(received) == 1
    config = received[0].config
    assert config["epochs"] == 2
    assert config["device"] == "cpu"
    # Defaults the caller never passed are present too.
    assert "lr0" in config and "batch" in config


# ---------------------------------------------------------------------------
# resolve_loggers
# ---------------------------------------------------------------------------


def test_resolve_loggers_none_and_unknown():
    assert resolve_loggers(None) == []
    with pytest.raises(ValueError, match="Unknown logger"):
        resolve_loggers("clearml")


def test_resolve_loggers_strings_and_instances(fake_mlflow, fake_tensorboard):
    instance = MLflowLogger()
    resolved = resolve_loggers(["tensorboard", instance, "MLflow"])
    assert isinstance(resolved[0], TensorBoardLogger)
    assert resolved[1] is instance
    assert isinstance(resolved[2], MLflowLogger)


def test_missing_backend_raises_at_construction(monkeypatch):
    def boom():
        raise ImportError("MLflowLogger requires the 'mlflow' package.")

    monkeypatch.setattr(mlflow_module, "_import_mlflow", boom)
    with pytest.raises(ImportError, match="mlflow"):
        MLflowLogger()


def test_loggers_kwarg_resolves_on_trainer(fake_mlflow, tmp_path):
    trainer = LoggerTrainer(
        model=nn.Linear(1, 1),
        data=None,
        device="cpu",
        ema=False,
        epochs=2,
        loggers="mlflow",
    )
    trainer._test_save_dir = tmp_path

    trainer.train()

    names = [c[0] for c in fake_mlflow.calls]
    assert names[0] == "start_run"
    assert names.count("log_metrics") == 2
    assert names[-1] == "end_run"


# ---------------------------------------------------------------------------
# Metric schema
# ---------------------------------------------------------------------------


def test_epoch_metrics_canonical_names():
    metrics = loggers_base.epoch_metrics(_epoch_event())
    assert metrics == {
        "train/loss": 1.5,
        "train/loss/box": 0.2,
        "train/loss/cls": 0.3,
        "lr/group0": 0.01,
        "val/mAP50": 0.6,
        "val/mAP50-95": 0.4,
        "time/epoch_seconds": 2.5,
    }


# ---------------------------------------------------------------------------
# MLflowLogger
# ---------------------------------------------------------------------------


def test_mlflow_logger_full_lifecycle(fake_mlflow, tmp_path):
    (tmp_path / "results.csv").write_text("epoch\n1\n")
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "best.pt").write_bytes(b"00")

    logger = MLflowLogger(
        tracking_uri="http://localhost:5000",
        experiment_name="exp",
        log_checkpoints=True,
    )
    logger.on_train_start(_start_event(save_dir=str(tmp_path)))
    logger.on_train_epoch_end(_epoch_event())
    logger.on_train_end(_end_event(save_dir=str(tmp_path)))

    names = [c[0] for c in fake_mlflow.calls]
    assert ("set_tracking_uri", "http://localhost:5000") in fake_mlflow.calls
    assert ("set_experiment", "exp") in fake_mlflow.calls
    assert names.index("log_params") < names.index("log_metrics")
    start = next(c for c in fake_mlflow.calls if c[0] == "start_run")
    assert start[1] == "yolo9s-detect"
    params = next(c for c in fake_mlflow.calls if c[0] == "log_params")[1]
    assert params["lr0"] == "0.01"
    metrics_call = next(c for c in fake_mlflow.calls if c[0] == "log_metrics")
    assert metrics_call[1]["val/mAP50"] == 0.6
    assert metrics_call[2] == 1
    artifacts = [c[1] for c in fake_mlflow.calls if c[0] == "log_artifact"]
    assert any(a.endswith("results.csv") for a in artifacts)
    assert any(a.endswith("best.pt") for a in artifacts)
    assert fake_mlflow.calls[-1] == ("end_run", "FINISHED")


def test_mlflow_logger_marks_failed_run(fake_mlflow):
    logger = MLflowLogger()
    logger.on_train_start(_start_event())
    logger.on_train_exception(_exception_event())
    assert fake_mlflow.calls[-1] == ("end_run", "FAILED")


def test_mlflow_logger_sanitizes_metric_keys():
    assert mlflow_module._sanitize_key("val/precision(B)") == "val/precision_B_"
    assert mlflow_module._sanitize_key("val/mAP50-95") == "val/mAP50-95"


# ---------------------------------------------------------------------------
# WandbLogger
# ---------------------------------------------------------------------------


def test_wandb_logger_full_lifecycle(fake_wandb, tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "best.pt").write_bytes(b"00")

    logger = WandbLogger(project="proj", entity="team", log_checkpoints=True)
    logger.on_train_start(_start_event(save_dir=str(tmp_path)))
    logger.on_train_epoch_end(_epoch_event())
    logger.on_train_end(_end_event(save_dir=str(tmp_path)))

    assert fake_wandb.init_kwargs["project"] == "proj"
    assert fake_wandb.init_kwargs["entity"] == "team"
    assert fake_wandb.init_kwargs["config"]["lr0"] == 0.01
    assert fake_wandb.run.logged[0][0]["train/loss"] == 1.5
    assert fake_wandb.run.logged[0][1] == 1
    assert fake_wandb.run.summary["best_metric"] == 0.4
    assert len(fake_wandb.run.artifacts) == 1
    assert fake_wandb.run.finished == 0


def test_wandb_logger_failed_run_exit_code(fake_wandb):
    logger = WandbLogger()
    logger.on_train_start(_start_event())
    logger.on_train_exception(_exception_event())
    assert fake_wandb.run.finished == 1


# ---------------------------------------------------------------------------
# TensorBoardLogger
# ---------------------------------------------------------------------------


def test_tensorboard_logger_full_lifecycle(fake_tensorboard, tmp_path):
    logger = TensorBoardLogger()
    logger.on_train_start(_start_event(save_dir=str(tmp_path)))
    logger.on_train_epoch_end(_epoch_event())
    logger.on_train_end(_end_event(save_dir=str(tmp_path)))

    writer = fake_tensorboard.instances[0]
    assert writer.log_dir.endswith("tensorboard")
    assert writer.texts[0][0] == "train_config"
    tags = {tag for tag, _, _ in writer.scalars}
    assert "train/loss" in tags
    assert "val/mAP50" in tags
    assert all(step == 1 for _, _, step in writer.scalars)
    assert writer.closed is True


def test_tensorboard_logger_closes_on_exception(fake_tensorboard):
    logger = TensorBoardLogger()
    logger.on_train_start(_start_event())
    logger.on_train_exception(_exception_event())
    assert fake_tensorboard.instances[0].closed is True


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_backend_failure_disables_logger_but_training_survives(
    fake_mlflow, tmp_path, monkeypatch
):
    def explode(metrics, step=None):
        raise ConnectionError("tracking server down")

    monkeypatch.setattr(fake_mlflow, "log_metrics", explode)

    trainer = LoggerTrainer(
        model=nn.Linear(1, 1),
        data=None,
        device="cpu",
        ema=False,
        epochs=2,
        loggers="mlflow",
    )
    trainer._test_save_dir = tmp_path

    results = trainer.train()

    assert results["final_loss"] == pytest.approx(2.5)
    # Disabled after the first epoch's failure: no further backend calls.
    names = [c[0] for c in fake_mlflow.calls]
    assert "end_run" not in names


def test_logger_instances_are_picklable(fake_mlflow, fake_wandb, fake_tensorboard):
    for instance in (MLflowLogger(), WandbLogger(), TensorBoardLogger()):
        assert pickle.loads(pickle.dumps(instance)) is not None


def test_logger_names_are_picklable_through_ddp_filter():
    from libreyolo.training.ddp_spawn import _filter_picklable

    assert _filter_picklable({"loggers": "mlflow"}) == {"loggers": "mlflow"}


def test_ddp_filter_hint_mentions_loggers():
    from libreyolo.training.ddp_spawn import _filter_picklable

    with pytest.raises(RuntimeError, match="loggers='mlflow'"):
        _filter_picklable({"callbacks": lambda e: None})
