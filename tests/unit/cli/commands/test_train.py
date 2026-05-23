"""Behavior tests for the train command.

These verify observable CLI behavior (dry-run config resolution).
Real training is covered in e2e/test_rf1_training.py.
"""

import json

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands.train import train_cmd
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer()
    app.command("train", cls=KeyValueCommand)(train_cmd)
    return app


def test_train_dry_run_uses_rtdetr_defaults():
    """Dry-run shows correct family-specific defaults for RT-DETR."""
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=rtdetr-r18",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rtdetr"
    assert data["resolved_config"]["epochs"] == 72
    assert data["resolved_config"]["batch"] == 4
    assert data["resolved_config"]["optimizer"] == "adamw"
    assert data["resolved_config"]["lr0"] == 0.0001
    assert data["resolved_config"]["scheduler"] == "constant"


def test_train_dry_run_uses_rtdetr_defaults_for_weight_filename():
    """Dry-run detects family defaults from supported weight filenames."""
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreRTDETRr18.pt",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rtdetr"
    assert data["resolved_config"]["epochs"] == 72
    assert data["resolved_config"]["batch"] == 4
    assert data["resolved_config"]["optimizer"] == "adamw"
    assert data["resolved_config"]["lr0"] == 0.0001
    assert data["resolved_config"]["scheduler"] == "constant"


def test_train_dry_run_uses_rfdetr_defaults():
    """Dry-run shows native RF-DETR defaults instead of generic YOLO defaults."""
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=rfdetr-m",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    cfg = data["resolved_config"]
    assert cfg["epochs"] == 100
    assert cfg["batch"] == 4
    assert cfg["lr0"] == 0.0001
    assert cfg["workers"] == 0
    assert cfg["weight_decay"] == 0.0001
    assert cfg["eval_interval"] == 1
    assert cfg["warmup_epochs"] == 0
    assert cfg["lr_drop"] == 100
    assert cfg["ema_decay"] == 0.993
    from libreyolo.models.rfdetr.config import RFDETRConfig

    assert RFDETRConfig().ema_tau == 100
    assert "optimizer" not in cfg
    assert "scheduler" not in cfg


def test_train_dry_run_rfdetr_user_override_wins():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=LibreRFDETRm.pt",
            "epochs=3",
            "batch=2",
            "lr0=0.001",
            "lr_drop=7",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    cfg = data["resolved_config"]
    assert cfg["epochs"] == 3
    assert cfg["batch"] == 2
    assert cfg["lr0"] == 0.001
    assert cfg["lr_drop"] == 7


def test_train_rfdetr_actual_call_uses_reported_defaults(monkeypatch, tmp_path):
    """RF-DETR train should receive the same defaults shown by dry-run."""
    app = _make_app()
    captured = {}

    class _RFDETRLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def train(self, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_exp")}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _RFDETRLike(),
    )

    result = runner.invoke(
        app,
        [
            "data=dummy.yaml",
            "model=LibreRFDETRm.pt",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["data"] == "dummy.yaml"
    kwargs = captured["kwargs"]
    assert kwargs["epochs"] == 100
    assert kwargs["batch_size"] == 4
    assert kwargs["lr"] == 0.0001
    assert kwargs["num_workers"] == 0
    assert kwargs["weight_decay"] == 0.0001
    assert kwargs["eval_interval"] == 1
    assert kwargs["warmup_epochs"] == 0
    assert kwargs["scheduler"] == "step"
    assert kwargs["lr_drop"] == 100
    assert kwargs["use_ema"] is True
    assert kwargs["ema_decay"] == 0.993
    assert kwargs["early_stopping"] is False

    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["epochs_completed"] == 100


def test_train_rfdetr_scheduler_override_reaches_trainer(monkeypatch, tmp_path):
    app = _make_app()
    captured = {}

    class _RFDETRLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def train(self, data, **kwargs):
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_exp")}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _RFDETRLike(),
    )

    result = runner.invoke(
        app,
        [
            "data=dummy.yaml",
            "model=LibreRFDETRm.pt",
            "scheduler=cosine",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["scheduler"] == "cosine"
    assert "ignores these parameters" not in result.output


def test_train_rfdetr_lr_drop_override_reaches_trainer(monkeypatch, tmp_path):
    app = _make_app()
    captured = {}

    class _RFDETRLike:
        FAMILY = "rfdetr"
        device = "cpu"

        def train(self, data, **kwargs):
            captured["kwargs"] = kwargs
            return {"output_dir": str(tmp_path / "rfdetr_exp")}

    monkeypatch.setattr(
        "libreyolo.cli.commands.train.load_model_or_exit",
        lambda out, model, model_path, device: _RFDETRLike(),
    )

    result = runner.invoke(
        app,
        [
            "data=dummy.yaml",
            "model=LibreRFDETRm.pt",
            "lr_drop=12",
            f"project={tmp_path}",
            "exist_ok=true",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["lr_drop"] == 12
    assert "ignores these parameters" not in result.output
