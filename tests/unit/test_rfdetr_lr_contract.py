from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

rfdetr_model = pytest.importorskip("libreyolo.models.rfdetr.model")
rfdetr_trainer = pytest.importorskip("libreyolo.models.rfdetr.trainer")


def _make_wrapper():
    wrapper = rfdetr_model.LibreRFDETR.__new__(rfdetr_model.LibreRFDETR)
    wrapper.model = object()
    wrapper.size = "n"
    wrapper.nb_classes = 2
    wrapper.input_size = 560
    return wrapper


def _install_dummy_trainer(monkeypatch, result):
    captured = {}

    class _DummyTrainer:
        def __init__(self, model, wrapper_model=None, **kwargs):
            captured["kwargs"] = kwargs

        def setup(self):
            captured["setup"] = True

        def resume(self, checkpoint_path):
            captured["resume"] = checkpoint_path

        def train(self):
            return result

    monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", _DummyTrainer)
    return captured


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
    captured = _install_dummy_trainer(monkeypatch, {"save_dir": str(tmp_path / "exp")})

    result = _make_wrapper().train(
        data="data.yaml",
        batch=2,
        lr0=0.001,
        output_dir=str(tmp_path / "canonical"),
    )

    assert result["output_dir"] == str(tmp_path / "exp")
    assert captured["kwargs"]["batch"] == 2
    assert captured["kwargs"]["lr0"] == pytest.approx(0.001)
    assert captured["kwargs"]["project"] == str(tmp_path)
    assert captured["kwargs"]["name"] == "canonical"
    assert captured["kwargs"]["exist_ok"] is True


def test_rfdetr_train_accepts_legacy_aliases(monkeypatch, tmp_path):
    captured = _install_dummy_trainer(monkeypatch, {"save_dir": str(tmp_path / "exp")})

    _make_wrapper().train(
        data="data.yaml",
        batch_size=3,
        lr=0.002,
        output_dir=str(tmp_path / "aliases"),
    )

    assert captured["kwargs"]["batch"] == 3
    assert captured["kwargs"]["lr0"] == pytest.approx(0.002)


def test_rfdetr_train_honors_explicit_run_kwargs(monkeypatch, tmp_path):
    captured = _install_dummy_trainer(monkeypatch, {})

    result = _make_wrapper().train(
        data="data.yaml",
        output_dir=str(tmp_path / "ignored"),
        project=str(tmp_path / "project"),
        name="custom",
        exist_ok=False,
    )

    assert captured["kwargs"]["project"] == str(tmp_path / "project")
    assert captured["kwargs"]["name"] == "custom"
    assert captured["kwargs"]["exist_ok"] is False
    assert result["output_dir"] == str(tmp_path / "project" / "custom")


@pytest.mark.parametrize("resume_arg", [True, "explicit"])
def test_rfdetr_train_resolves_resume_paths(monkeypatch, tmp_path, resume_arg):
    captured = _install_dummy_trainer(monkeypatch, {"save_dir": str(tmp_path / "exp")})
    checkpoint_path = tmp_path / "resume.pt"
    resume = checkpoint_path if resume_arg == "explicit" else True
    expected = (
        tmp_path / "project" / "custom" / "weights" / "last.pt"
        if resume is True
        else checkpoint_path
    )

    _make_wrapper().train(
        data="data.yaml",
        output_dir=str(tmp_path / "ignored"),
        project=str(tmp_path / "project"),
        name="custom",
        resume=resume,
    )

    assert captured["setup"] is True
    assert captured["resume"] == str(expected)


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
