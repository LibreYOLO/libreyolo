"""Unit tests for public training callbacks."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

from libreyolo.training.callbacks import TrainCallbackList, TrainEpochEvent
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
        best_metric=None,
        best_metric_name=None,
        best_epoch=None,
        epoch_seconds=0.5,
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


def test_train_epoch_event_mappings_are_read_only():
    event = _event()

    with pytest.raises(TypeError):
        event.train_loss_items["box"] = 0.2
    with pytest.raises(TypeError):
        event.lr["group0"] = 0.2


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
        return (
            1.5,
            {
                "mAP50": 0.6,
                "mAP50_95": 0.7,
                "best_metric": 0.7,
                "best_metric_key": "metrics/mAP50-95",
                "metrics": {
                    "metrics/mAP50": 0.6,
                    "metrics/mAP50-95": 0.7,
                    "ignored/list": [1, 2],
                },
            },
            {"box": torch.tensor(0.2), "cls": 0.3},
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
    trainer = CallbackTrainer(
        model=nn.Linear(1, 1),
        data=None,
        device="cpu",
        ema=False,
        epochs=1,
        save_period=1,
        callbacks=received.append,
    )
    trainer._test_save_dir = tmp_path
    trainer.saved_checkpoints = []

    results = trainer.train()

    assert results["final_loss"] == pytest.approx(1.5)
    assert len(received) == 1
    assert trainer.saved_checkpoints[0]["is_best"] is True

    event = received[0]
    assert event.epoch == 1
    assert event.total_epochs == 1
    assert event.model_family == "dummy"
    assert event.task == "detect"
    assert event.train_loss == pytest.approx(1.5)
    assert dict(event.train_loss_items) == {
        "box": pytest.approx(0.2),
        "cls": pytest.approx(0.3),
    }
    assert dict(event.lr) == {"group0": 0.01}
    assert event.validated is True
    assert dict(event.val_metrics) == {
        "metrics/mAP50": 0.6,
        "metrics/mAP50-95": 0.7,
    }
    assert event.is_best is True
    assert event.best_metric == pytest.approx(0.7)
    assert event.best_metric_name == "metrics/mAP50-95"
    assert event.best_epoch == 1


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
