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
    assert captured["kwargs"]["exist_ok"] is False


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
    # resume=True continues in the same run_dir that weights/last.pt was read
    # from, so exist_ok must be forced True regardless of the new exist_ok=False
    # default -- otherwise _get_save_dir() would increment away from it and
    # split resumed state from newly written artifacts.
    assert captured["kwargs"]["exist_ok"] is (resume is True)


def test_rfdetr_resume_true_continues_the_loaded_run(monkeypatch, tmp_path):
    """Run dirs increment (rfdetr_exp, rfdetr_exp2, ...), so resume=True on a
    loaded rfdetr_exp2 checkpoint must not fall back to the first run."""
    import torch

    captured = _install_dummy_trainer(monkeypatch, {"save_dir": "unused"})
    checkpoint = tmp_path / "runs" / "train" / "rfdetr_exp2" / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"epoch": 0}, checkpoint)
    wrapper = _make_wrapper()
    wrapper.model_path = str(checkpoint)

    wrapper.train(data="data.yaml", resume=True)

    assert captured["resume"] == str(checkpoint)
    assert captured["kwargs"]["project"] == str(tmp_path / "runs" / "train")
    assert captured["kwargs"]["name"] == "rfdetr_exp2"
    assert captured["kwargs"]["exist_ok"] is True


def test_rfdetr_resume_exist_ok_false_starts_a_new_run(monkeypatch, tmp_path):
    """As for the other families, an explicit exist_ok=False resumes the loaded
    run's checkpoint into a new numbered run beside it."""
    import torch

    captured = _install_dummy_trainer(monkeypatch, {"save_dir": "unused"})
    checkpoint = tmp_path / "runs" / "train" / "rfdetr_exp2" / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"epoch": 0}, checkpoint)
    wrapper = _make_wrapper()
    wrapper.model_path = str(checkpoint)

    wrapper.train(data="data.yaml", resume=True, exist_ok=False)

    assert captured["resume"] == str(checkpoint)
    assert captured["kwargs"]["name"] == "rfdetr_exp2"
    assert captured["kwargs"]["exist_ok"] is False


def test_rfdetr_resume_path_continues_that_run(monkeypatch, tmp_path):
    """resume='<run>/weights/last.pt' wrote into a new runs/train/rfdetr_exp*
    instead of the run the checkpoint came from, unlike every other family."""
    import torch

    captured = _install_dummy_trainer(monkeypatch, {"save_dir": "unused"})
    checkpoint = tmp_path / "runs" / "rfdetr_exp3" / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"epoch": 0}, checkpoint)

    _make_wrapper().train(data="data.yaml", resume=str(checkpoint))
    kwargs = captured["kwargs"]
    assert captured["resume"] == str(checkpoint)
    assert (kwargs["project"], kwargs["name"]) == (str(tmp_path / "runs"), "rfdetr_exp3")
    assert kwargs["exist_ok"] is True

    _make_wrapper().train(data="data.yaml", resume=str(checkpoint), exist_ok=False)
    assert captured["kwargs"]["name"] == "rfdetr_exp3"
    assert captured["kwargs"]["exist_ok"] is False

    _make_wrapper().train(
        data="data.yaml", resume=str(checkpoint), project=str(tmp_path / "out"), name="n"
    )
    assert (captured["kwargs"]["project"], captured["kwargs"]["name"]) == (
        str(tmp_path / "out"),
        "n",
    )


def _save_rfdetr_run(path, **saved):
    import torch

    config = rfdetr_trainer.RFDETRConfig(**saved).to_dict()
    path.parent.mkdir(parents=True)
    torch.save({"epoch": 0, "config": config}, path)
    return path


def test_rfdetr_resume_restores_saved_settings(monkeypatch, tmp_path):
    """RF-DETR resumed with its signature defaults (100 epochs, batch 4,
    lr 1e-4) instead of the run's settings."""
    captured = _install_dummy_trainer(monkeypatch, {"save_dir": "unused"})
    last = _save_rfdetr_run(
        tmp_path / "rf" / "weights" / "last.pt",
        epochs=7, batch=3, lr0=0.0123, workers=0, ema=False, weight_decay=0.05,
    )

    _make_wrapper().train(data="data.yaml", resume=str(last))

    kwargs = captured["kwargs"]
    assert captured["resume"] == str(last)
    assert (kwargs["epochs"], kwargs["batch"], kwargs["workers"]) == (7, 3, 0)
    assert kwargs["lr0"] == pytest.approx(0.0123)
    assert (kwargs["ema"], kwargs["weight_decay"]) == (False, 0.05)


def test_rfdetr_resume_explicit_arguments_and_aliases_win(monkeypatch, tmp_path):
    captured = _install_dummy_trainer(monkeypatch, {"save_dir": "unused"})
    last = _save_rfdetr_run(
        tmp_path / "rf" / "weights" / "last.pt", epochs=7, batch=3, lr0=0.0123, workers=0
    )

    _make_wrapper().train(
        data="data.yaml", resume=str(last), epochs=9, batch_size=5, lr=0.002, num_workers=2
    )

    kwargs = captured["kwargs"]
    assert (kwargs["epochs"], kwargs["batch"], kwargs["workers"]) == (9, 5, 2)
    assert kwargs["lr0"] == pytest.approx(0.002)


def test_rfdetr_resume_without_saved_settings_keeps_the_defaults(monkeypatch, tmp_path):
    import torch

    captured = _install_dummy_trainer(monkeypatch, {"save_dir": "unused"})
    last = tmp_path / "rf" / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    torch.save({"epoch": 0}, last)

    _make_wrapper().train(data="data.yaml", resume=str(last))

    kwargs = captured["kwargs"]
    assert (kwargs["epochs"], kwargs["batch"]) == (100, 4)
    assert kwargs["lr0"] == pytest.approx(1e-4)


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


def test_rfdetr_resume_without_data_uses_the_saved_dataset(monkeypatch, tmp_path):
    """train(resume=True) with no data= crashed with a TypeError; the other
    families restore the checkpoint's dataset."""
    captured = _install_dummy_trainer(monkeypatch, {"save_dir": "unused"})
    data = str(tmp_path / "saved.yaml")
    last = _save_rfdetr_run(tmp_path / "rf" / "weights" / "last.pt", epochs=7, data=data)
    wrapper = _make_wrapper()
    wrapper.model_path = str(last)

    wrapper.train(resume=True)

    assert captured["resume"] == str(last)
    assert captured["kwargs"]["data"] == data


def test_rfdetr_train_without_data_says_so(monkeypatch, tmp_path):
    _install_dummy_trainer(monkeypatch, {"save_dir": "unused"})

    with pytest.raises(ValueError, match="needs data="):
        _make_wrapper().train(output_dir=str(tmp_path / "run"))

