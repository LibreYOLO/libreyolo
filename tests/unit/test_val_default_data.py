"""val() without data= reuses the dataset the checkpoint was trained on."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    class _Validator:
        def __init__(self, model, config, **kwargs):
            seen["data"] = config.data

        def __call__(self):
            return {}

    monkeypatch.setattr("libreyolo.validation.DetectionValidator", _Validator)
    return seen


def _model():
    from libreyolo import LibreYOLO9

    return LibreYOLO9(None, size="t", device="cpu")


def test_val_reuses_the_training_dataset(captured):
    """The ecosystem's model.val() needs no arguments after training; this
    raised 'Specify one of: data ...'."""
    model = _model()
    model._loaded_checkpoint_train_config = {"data": "/runs/data/coco8.yaml"}

    model.val(workers=0)

    assert captured["data"] == "/runs/data/coco8.yaml"


def test_explicit_data_wins(captured):
    model = _model()
    model._loaded_checkpoint_train_config = {"data": "/runs/data/coco8.yaml"}

    model.val(data="other.yaml", workers=0)

    assert captured["data"] == "other.yaml"


def test_released_weights_need_data(captured):
    model = _model()
    model._loaded_checkpoint_train_config = {}

    with pytest.raises(ValueError, match=r"val\(\) needs data="):
        model.val(workers=0)
    assert captured == {}


def test_val_after_training_without_a_checkpoint_reload(captured):
    """Families that keep the trained weights in memory without reloading a
    checkpoint (e.g. Dome-DETR) had no dataset in the cached config, so
    train() then val() raised 'carries no training dataset'."""
    from types import SimpleNamespace

    from libreyolo.training.trainer import BaseTrainer

    model = _model()  # built from scratch: no checkpoint, empty cache
    trainer = SimpleNamespace(
        wrapper_model=model, config=SimpleNamespace(data="/runs/data/new.yaml")
    )
    BaseTrainer._record_trained_dataset(trainer)  # what a finished train() does

    model.val(workers=0)

    assert captured["data"] == "/runs/data/new.yaml"


def test_failed_training_keeps_the_checkpoint_dataset(captured, tmp_path):
    """Setup failing on dataset B left the cache pointing at B while the
    weights were still the checkpoint's, trained on A."""
    from torch import nn

    from libreyolo.training.trainer import BaseTrainer

    class _SetupFails(BaseTrainer):
        def get_model_family(self):
            return "yolo9"

        def get_model_tag(self):
            return "tiny"

        def create_transforms(self):
            raise NotImplementedError

        def create_scheduler(self, iters_per_epoch):
            raise NotImplementedError

        def get_loss_components(self, outputs):
            return {}

        def _setup_data(self):
            raise FileNotFoundError("dataset B has no images")

    model = _model()
    model._loaded_checkpoint_train_config = {"data": "/runs/data/a.yaml"}
    trainer = _SetupFails(
        nn.Linear(1, 1), wrapper_model=model, data="/runs/data/b.yaml",
        device="cpu", ema=False, project=str(tmp_path),
    )
    with pytest.raises(FileNotFoundError):
        trainer.train()

    model.val(workers=0)

    assert captured["data"] == "/runs/data/a.yaml"
