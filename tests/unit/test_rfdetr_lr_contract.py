from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

rfdetr_model = pytest.importorskip("libreyolo.models.rfdetr.model")
rfdetr_trainer = pytest.importorskip("libreyolo.models.rfdetr.trainer")


def test_rfdetr_effective_lr_is_absolute_under_accumulation():
    trainer = rfdetr_trainer.RFDETRTrainer.__new__(
        rfdetr_trainer.RFDETRTrainer
    )
    trainer.config = rfdetr_trainer.RFDETRConfig(
        data=None,
        batch=4,
        lr0=0.001,
        nbs=64,
    )
    trainer.world_size = 4

    assert trainer._accum_steps == 16
    assert trainer.effective_lr == pytest.approx(0.001)


def test_rfdetr_train_prefers_canonical_batch_and_lr0(monkeypatch, tmp_path):
    captured = {}

    class _DummyTrainer:
        def __init__(self, model, wrapper_model=None, **kwargs):
            captured["model"] = model
            captured["wrapper_model"] = wrapper_model
            captured["kwargs"] = kwargs

        def train(self):
            return {"save_dir": str(tmp_path / "exp")}

    monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", _DummyTrainer)

    wrapper = rfdetr_model.LibreRFDETR.__new__(rfdetr_model.LibreRFDETR)
    wrapper.model = object()
    wrapper.size = "n"
    wrapper.nb_classes = 2
    wrapper.input_size = 560

    result = wrapper.train(
        data="data.yaml",
        batch=2,
        lr0=0.001,
        output_dir=str(tmp_path / "canonical"),
    )

    assert result["output_dir"] == str(tmp_path / "exp")
    assert captured["kwargs"]["batch"] == 2
    assert captured["kwargs"]["lr0"] == pytest.approx(0.001)


def test_rfdetr_train_accepts_legacy_aliases(monkeypatch, tmp_path):
    captured = {}

    class _DummyTrainer:
        def __init__(self, model, wrapper_model=None, **kwargs):
            captured["kwargs"] = kwargs

        def train(self):
            return {"save_dir": str(tmp_path / "exp")}

    monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", _DummyTrainer)

    wrapper = rfdetr_model.LibreRFDETR.__new__(rfdetr_model.LibreRFDETR)
    wrapper.model = object()
    wrapper.size = "n"
    wrapper.nb_classes = 2
    wrapper.input_size = 560

    wrapper.train(
        data="data.yaml",
        batch_size=3,
        lr=0.002,
        output_dir=str(tmp_path / "aliases"),
    )

    assert captured["kwargs"]["batch"] == 3
    assert captured["kwargs"]["lr0"] == pytest.approx(0.002)


def test_rfdetr_train_rejects_conflicting_lr_aliases(tmp_path):
    wrapper = rfdetr_model.LibreRFDETR.__new__(rfdetr_model.LibreRFDETR)
    wrapper.model = object()
    wrapper.size = "n"
    wrapper.nb_classes = 2
    wrapper.input_size = 560

    with pytest.raises(ValueError, match="Conflicting RF-DETR LR values"):
        wrapper.train(
            data="data.yaml",
            lr=0.001,
            lr0=0.002,
            output_dir=str(tmp_path / "conflict"),
        )
