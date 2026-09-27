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
