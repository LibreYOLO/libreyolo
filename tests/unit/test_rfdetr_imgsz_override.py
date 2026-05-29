"""Unit tests for RF-DETR imgsz handling in train() and create_transforms()."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

rfdetr_model = pytest.importorskip("libreyolo.models.rfdetr.model")
rfdetr_trainer = pytest.importorskip("libreyolo.models.rfdetr.trainer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wrapper(input_size: int = 504) -> rfdetr_model.LibreRFDETR:
    wrapper = rfdetr_model.LibreRFDETR.__new__(rfdetr_model.LibreRFDETR)
    wrapper.model = object()
    wrapper.size = "l"
    wrapper.nb_classes = 80
    wrapper.input_size = input_size
    return wrapper


def _make_trainer(imgsz: int, *, multi_scale: bool = False) -> rfdetr_trainer.RFDETRTrainer:
    """Construct a RFDETRTrainer stub without calling __init__."""
    trainer = rfdetr_trainer.RFDETRTrainer.__new__(rfdetr_trainer.RFDETRTrainer)
    trainer.config = rfdetr_trainer.RFDETRConfig(
        data=None,
        imgsz=imgsz,
        multi_scale=multi_scale,
        do_random_resize_via_padding=False,
    )

    class _FakeModel:
        patch_size = 6
        num_windows = 4  # block_size = 24

    class _FakeWrapper:
        task = "detect"

    trainer.model = _FakeModel()
    trainer.wrapper_model = _FakeWrapper()
    return trainer


# ---------------------------------------------------------------------------
# train() kwarg assembly
# ---------------------------------------------------------------------------

def test_explicit_imgsz_is_not_overridden(monkeypatch, tmp_path):
    """imgsz=624 supplied to train() must not be overwritten by model.input_size=504."""
    captured = {}

    class _DummyTrainer:
        def __init__(self, model, wrapper_model=None, **kwargs):
            captured["kwargs"] = kwargs

        def train(self):
            return {"save_dir": str(tmp_path / "exp")}

    monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", _DummyTrainer)

    wrapper = _make_wrapper(input_size=504)
    wrapper.train(data="data.yaml", imgsz=624, output_dir=str(tmp_path / "run"))

    assert captured["kwargs"]["imgsz"] == 624


def test_default_imgsz_applied_when_not_supplied(monkeypatch, tmp_path):
    """When imgsz is not passed, model.input_size is used as the fallback."""
    captured = {}

    class _DummyTrainer:
        def __init__(self, model, wrapper_model=None, **kwargs):
            captured["kwargs"] = kwargs

        def train(self):
            return {"save_dir": str(tmp_path / "exp")}

    monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", _DummyTrainer)

    wrapper = _make_wrapper(input_size=504)
    wrapper.train(data="data.yaml", output_dir=str(tmp_path / "run"))

    assert captured["kwargs"]["imgsz"] == 504


def test_imgsz_none_falls_back_to_model_default(monkeypatch, tmp_path):
    """imgsz=None must not be forwarded; model.input_size is used instead."""
    captured = {}

    class _DummyTrainer:
        def __init__(self, model, wrapper_model=None, **kwargs):
            captured["kwargs"] = kwargs

        def train(self):
            return {"save_dir": str(tmp_path / "exp")}

    monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", _DummyTrainer)

    wrapper = _make_wrapper(input_size=504)
    wrapper.train(data="data.yaml", imgsz=None, output_dir=str(tmp_path / "run"))

    assert captured["kwargs"]["imgsz"] == 504


# ---------------------------------------------------------------------------
# create_transforms() divisibility validation
# ---------------------------------------------------------------------------

def test_imgsz_not_divisible_by_block_size_raises():
    """imgsz=500 with block_size=24 (6*4) and multi_scale=False must raise ValueError."""
    trainer = _make_trainer(imgsz=500, multi_scale=False)
    with pytest.raises(ValueError, match="not divisible by 24"):
        trainer.create_transforms()


def test_imgsz_not_divisible_raises_with_multi_scale_too():
    """Validation always uses literal imgsz, so divisibility is required even with multi_scale=True."""
    trainer = _make_trainer(imgsz=500, multi_scale=True)
    with pytest.raises(ValueError, match="not divisible by 24"):
        trainer.create_transforms()
