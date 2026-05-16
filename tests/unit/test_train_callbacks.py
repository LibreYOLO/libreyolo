"""Unit tests for public training callbacks."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

from libreyolo.training.callbacks import (
    TrainCallbackList,
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit


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


def _event() -> TrainEpochEvent:
    return TrainEpochEvent(
        epoch=1,
        total_epochs=1,
        model_family="dummy",
        model_size="s",
        task="detect",
        save_dir="/tmp/libreyolo",
        train_loss=1.0,
        train_loss_items={"box": 0.1},
        lr={"group0": 0.01},
        val_metrics={},
        validated=False,
        is_best=False,
        current_metric=None,
        current_metric_name=None,
        best_metric=None,
        best_metric_name=None,
        best_epoch=None,
        epoch_seconds=0.5,
    )


def _start_event() -> TrainStartEvent:
    return TrainStartEvent(
        start_epoch=1,
        total_epochs=1,
        model_family="dummy",
        model_size="s",
        task="detect",
        save_dir="/tmp/libreyolo",
    )


def _end_event() -> TrainEndEvent:
    return TrainEndEvent(
        total_epochs=1,
        completed_epochs=1,
        model_family="dummy",
        model_size="s",
        task="detect",
        save_dir="/tmp/libreyolo",
        final_loss=1.0,
        best_metric=None,
        best_epoch=None,
        total_seconds=0.5,
        results={"final_loss": 1.0},
    )


def _exception_event() -> TrainExceptionEvent:
    exc = RuntimeError("boom")
    return TrainExceptionEvent(
        epoch=1,
        total_epochs=1,
        model_family="dummy",
        model_size="s",
        task="detect",
        save_dir="/tmp/libreyolo",
        exception=exc,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        elapsed_seconds=0.5,
    )


def test_train_callback_list_dispatches_functions_and_object_methods():
    event = _event()
    received = []

    def callback_fn(received_event):
        received.append(("fn", received_event))

    class ObjectCallback:
        def __call__(self, received_event):
            received.append(("call", received_event))

        def on_train_epoch_end(self, received_event):
            received.append(("method", received_event))

    callbacks = TrainCallbackList([callback_fn, ObjectCallback()])

    callbacks.on_train_epoch_end(event)

    assert received == [("fn", event), ("method", event)]


def test_train_callback_list_dispatches_lifecycle_methods_only_to_objects():
    epoch_event = _event()
    start_event = _start_event()
    end_event = _end_event()
    exception_event = _exception_event()
    received = []

    def epoch_fn(received_event):
        received.append(("fn", type(received_event).__name__))

    class ObjectCallback:
        def on_train_start(self, received_event):
            received.append(("start", type(received_event).__name__))

        def on_train_epoch_end(self, received_event):
            received.append(("epoch", type(received_event).__name__))

        def on_train_end(self, received_event):
            received.append(("end", type(received_event).__name__))

        def on_train_exception(self, received_event):
            received.append(("exception", type(received_event).__name__))

    callbacks = TrainCallbackList([epoch_fn, ObjectCallback()])

    callbacks.on_train_start(start_event)
    callbacks.on_train_epoch_end(epoch_event)
    callbacks.on_train_end(end_event)
    callbacks.on_train_exception(exception_event)

    assert received == [
        ("start", "TrainStartEvent"),
        ("fn", "TrainEpochEvent"),
        ("epoch", "TrainEpochEvent"),
        ("end", "TrainEndEvent"),
        ("exception", "TrainExceptionEvent"),
    ]


def test_object_callbacks_may_implement_only_one_lifecycle_method():
    received = []

    class StartOnlyCallback:
        def on_train_start(self, event):
            received.append(type(event).__name__)

    callbacks = TrainCallbackList(StartOnlyCallback())

    callbacks.on_train_start(_start_event())
    callbacks.on_train_epoch_end(_event())
    callbacks.on_train_end(_end_event())
    callbacks.on_train_exception(_exception_event())

    assert received == ["TrainStartEvent"]


def test_train_epoch_event_mappings_are_read_only():
    event = _event()

    with pytest.raises(TypeError):
        event.train_loss_items["box"] = 0.2
    with pytest.raises(TypeError):
        event.lr["group0"] = 0.2

    end_event = _end_event()
    with pytest.raises(TypeError):
        end_event.results["final_loss"] = 2.0


def test_callbacks_are_not_treated_as_training_config_keys():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = DummyTrainer(
            model=nn.Linear(1, 1),
            data=None,
            device="cpu",
            ema=False,
            callbacks=[],
        )

    assert len(trainer.callbacks) == 0
    assert not any("Unknown training config keys" in str(w.message) for w in caught)


class CallbackTrainer(DummyTrainer):
    def setup(self):
        self.save_dir = self._test_save_dir
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.tensorboard_writer = None
        self._is_setup = True

    def _train_epoch(self, epoch: int):
        metric = 0.7 if epoch == 0 else 0.5
        map50 = 0.6 if epoch == 0 else 0.4
        return (
            1.5 + epoch,
            {
                "mAP50": map50,
                "mAP50_95": metric,
                "best_metric": metric,
                "best_metric_key": "metrics/mAP50-95",
                "metrics": {
                    "metrics/mAP50": map50,
                    "metrics/mAP50-95": metric,
                    "ignored/list": [1, 2],
                },
            },
            {"box": torch.tensor(0.2), "cls": 0.3, "dfl": 0.4},
            {"group0": 0.01},
        )

    def _save_checkpoint(
        self, epoch: int, loss: float, val_metrics=None, is_best=None
    ):
        self.saved_checkpoints.append(
            {
                "epoch": epoch,
                "loss": loss,
                "val_metrics": val_metrics,
                "is_best": is_best,
            }
        )


def test_train_emits_epoch_callback_after_best_update_and_checkpoint(tmp_path):
    received = []
    lifecycle = []

    class LifecycleCallback:
        def on_train_start(self, event):
            lifecycle.append(("start", event))

        def on_train_epoch_end(self, event):
            lifecycle.append(("epoch", event))

        def on_train_end(self, event):
            lifecycle.append(("end", event))

    trainer = CallbackTrainer(
        model=nn.Linear(1, 1),
        data=None,
        device="cpu",
        ema=False,
        epochs=2,
        save_period=10,
        callbacks=[received.append, LifecycleCallback()],
    )
    trainer._test_save_dir = tmp_path
    trainer.saved_checkpoints = []

    results = trainer.train()

    assert results["final_loss"] == pytest.approx(2.5)
    assert results["epoch_losses"] == [pytest.approx(1.5), pytest.approx(2.5)]
    assert results["epoch_lrs"] == [
        {"group0": pytest.approx(0.01)},
        {"group0": pytest.approx(0.01)},
    ]
    assert results["epoch_loss_items"] == [
        {
            "box": pytest.approx(0.2),
            "cls": pytest.approx(0.3),
            "dfl": pytest.approx(0.4),
        },
        {
            "box": pytest.approx(0.2),
            "cls": pytest.approx(0.3),
            "dfl": pytest.approx(0.4),
        },
    ]
    assert results["val_metrics"] == [
        {
            "metrics/mAP50": pytest.approx(0.6),
            "metrics/mAP50-95": pytest.approx(0.7),
        },
        {
            "metrics/mAP50": pytest.approx(0.4),
            "metrics/mAP50-95": pytest.approx(0.5),
        },
    ]
    assert results["epoch_metrics"][0]["train_loss"] == pytest.approx(1.5)
    assert results["epoch_metrics"][0]["is_best"] is True
    assert results["epoch_metrics"][0]["current_metric"] == pytest.approx(0.7)
    assert results["epoch_metrics"][0]["best_metric"] == pytest.approx(0.7)
    assert results["epoch_metrics"][1]["current_metric"] == pytest.approx(0.5)
    assert results["epoch_metrics"][1]["best_metric"] == pytest.approx(0.7)
    assert len(received) == 2
    assert [name for name, _ in lifecycle] == ["start", "epoch", "epoch", "end"]
    assert lifecycle[0][1].start_epoch == 1
    assert lifecycle[0][1].total_epochs == 2
    assert lifecycle[-1][1].completed_epochs == 2
    assert lifecycle[-1][1].results["final_loss"] == pytest.approx(2.5)
    assert trainer.saved_checkpoints[0]["is_best"] is True
    assert trainer.saved_checkpoints[1]["is_best"] is False

    event = received[0]
    assert event.epoch == 1
    assert event.total_epochs == 2
    assert event.model_family == "dummy"
    assert event.task == "detect"
    assert event.train_loss == pytest.approx(1.5)
    assert dict(event.train_loss_items) == {
        "box": pytest.approx(0.2),
        "cls": pytest.approx(0.3),
        "dfl": pytest.approx(0.4),
    }
    assert dict(event.lr) == {"group0": 0.01}
    assert event.validated is True
    assert dict(event.val_metrics) == {
        "metrics/mAP50": 0.6,
        "metrics/mAP50-95": 0.7,
    }
    assert event.is_best is True
    assert event.current_metric == pytest.approx(0.7)
    assert event.current_metric_name == "metrics/mAP50-95"
    assert event.best_metric == pytest.approx(0.7)
    assert event.best_metric_name == "metrics/mAP50-95"
    assert event.best_epoch == 1

    event = received[1]
    assert event.is_best is False
    assert event.current_metric == pytest.approx(0.5)
    assert event.current_metric_name == "metrics/mAP50-95"
    assert event.best_metric == pytest.approx(0.7)
    assert event.best_metric_name == "metrics/mAP50-95"
    assert event.best_epoch == 1


class FailingTrainer(CallbackTrainer):
    def _train_epoch(self, epoch: int):
        raise RuntimeError("training failed")


def test_train_exception_callback_fires_and_original_error_reraises(tmp_path):
    received = []

    class LifecycleCallback:
        def on_train_start(self, event):
            received.append(("start", event))

        def on_train_end(self, event):
            received.append(("end", event))

        def on_train_exception(self, event):
            received.append(("exception", event))

    trainer = FailingTrainer(
        model=nn.Linear(1, 1),
        data=None,
        device="cpu",
        ema=False,
        epochs=1,
        callbacks=LifecycleCallback(),
    )
    trainer._test_save_dir = tmp_path
    trainer.saved_checkpoints = []

    with pytest.raises(RuntimeError, match="training failed"):
        trainer.train()

    assert [name for name, _ in received] == ["start", "exception"]
    event = received[-1][1]
    assert event.epoch == 1
    assert event.exception_type == "RuntimeError"
    assert event.exception_message == "training failed"
    assert isinstance(event.exception, RuntimeError)


def test_legacy_two_value_epoch_result_still_normalizes_lr():
    trainer = DummyTrainer(
        model=nn.Linear(1, 1),
        data=None,
        device="cpu",
        ema=False,
    )
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.25)

    loss, val_metrics, loss_items, lr = trainer._normalize_epoch_result((2.0, None))

    assert loss == pytest.approx(2.0)
    assert val_metrics is None
    assert loss_items == {}
    assert lr == {"group0": 0.25}


def test_first_zero_validation_metric_counts_as_best():
    trainer = DummyTrainer(
        model=nn.Linear(1, 1),
        data=None,
        device="cpu",
        ema=False,
    )

    is_best = trainer._update_best_state(
        0,
        {
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "best_metric": 0.0,
            "best_metric_key": "metrics/mAP50-95",
        },
    )

    assert is_best is True
    assert trainer.best_mAP50 == pytest.approx(0.0)
    assert trainer.best_mAP50_95 == pytest.approx(0.0)
    assert trainer.best_epoch == 1


def test_yolo9_loss_components_match_epoch_event_names():
    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    components = YOLO9Trainer.get_loss_components(
        None,
        {
            "box": torch.tensor(0.2),
            "cls": torch.tensor(0.3),
            "dfl": torch.tensor(0.4),
        },
    )

    assert components == {
        "box": pytest.approx(0.2),
        "cls": pytest.approx(0.3),
        "dfl": pytest.approx(0.4),
    }


def test_rtdetr_loss_components_sum_main_aux_and_denoising_terms():
    from libreyolo.models.rtdetr.trainer import RTDETRTrainer

    outputs = {
        "total_loss": torch.tensor(99.0),
        "loss_vfl": torch.tensor(1.0),
        "loss_vfl_aux_0": torch.tensor(2.0),
        "loss_vfl_dn_0": torch.tensor(0.5),
        "loss_bbox": torch.tensor(3.0),
        "loss_bbox_aux_0": torch.tensor(4.0),
        "loss_giou": torch.tensor(5.0),
        "loss_giou_dn_0": torch.tensor(6.0),
    }

    components = RTDETRTrainer.get_loss_components(None, outputs)

    assert components == {
        "vfl": pytest.approx(3.5),
        "bbox": pytest.approx(7.0),
        "giou": pytest.approx(11.0),
    }
