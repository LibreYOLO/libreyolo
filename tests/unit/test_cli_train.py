"""Tests for the train command."""

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


class DummyRFDETRModel:
    FAMILY = "rfdetr"
    device = "cpu"

    def __init__(self):
        self.received = None

    def train(self, **kwargs):
        self.received = kwargs
        return {
            "save_dir": "/tmp/runs/train/exp",
            "best_checkpoint": "/tmp/runs/train/exp/checkpoint_best_total.pth",
            "last_checkpoint": None,
        }


def test_train_cli_translates_rfdetr_kwargs_and_outputs(monkeypatch):
    dummy = DummyRFDETRModel()

    monkeypatch.setattr("libreyolo.cli.commands.train.detect_family_from_name", lambda model: "rfdetr")
    monkeypatch.setattr("libreyolo.cli.commands.train.apply_family_defaults", lambda params, family, mode: params)
    monkeypatch.setattr("libreyolo.cli.config.resolve_model_name", lambda model: model)
    monkeypatch.setattr("libreyolo.cli.config.is_user_provided", lambda name: name in {
        "batch",
        "lr0",
        "workers",
        "patience",
        "save_period",
        "warmup_epochs",
        "ema",
        "val",
    })
    monkeypatch.setattr(
        "libreyolo.utils.general.increment_path",
        lambda path, exist_ok=False, mkdir=False: path,
    )
    monkeypatch.setattr("libreyolo.LibreYOLO", lambda *args, **kwargs: dummy)

    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=rfdetr-s",
            "batch=32",
            "lr0=0.001",
            "workers=6",
            "patience=12",
            "save_period=3",
            "warmup_epochs=4",
            "ema=false",
            "val=false",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert (
        data["best_weights"]
        == "/tmp/runs/train/exp/checkpoint_best_total.pth"
    )
    assert data["last_weights"] is None

    assert dummy.received["batch_size"] == 32
    assert dummy.received["lr"] == 0.001
    assert dummy.received["num_workers"] == 6
    assert dummy.received["warmup_epochs"] == 4
    assert dummy.received["use_ema"] is False
    assert dummy.received["checkpoint_interval"] == 3
    assert dummy.received["early_stopping"] is True
    assert dummy.received["early_stopping_patience"] == 12
    assert dummy.received["output_dir"].endswith("runs/train/exp")
    assert "batch" not in dummy.received
    assert "lr0" not in dummy.received
    assert "workers" not in dummy.received
    assert "ema" not in dummy.received
    assert "eval_interval" not in dummy.received


def test_train_dry_run_uses_rtdetr_defaults():
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
    assert data["resolved_config"]["scheduler"] == "linear"


class DummyYOLOXModel:
    FAMILY = "yolox"
    device = "cpu"

    def __init__(self):
        self.received = None

    def train(self, **kwargs):
        self.received = kwargs
        return {"save_dir": "/tmp/runs/train/exp"}


def test_train_cli_passes_allow_download_scripts(monkeypatch):
    dummy = DummyYOLOXModel()

    monkeypatch.setattr("libreyolo.cli.commands.train.detect_family_from_name", lambda model: "yolox")
    monkeypatch.setattr("libreyolo.cli.commands.train.apply_family_defaults", lambda params, family, mode: params)
    monkeypatch.setattr("libreyolo.cli.config.resolve_model_name", lambda model: model)
    monkeypatch.setattr(
        "libreyolo.utils.general.increment_path",
        lambda path, exist_ok=False, mkdir=False: path,
    )
    monkeypatch.setattr("libreyolo.LibreYOLO", lambda *args, **kwargs: dummy)

    app = _make_app()
    result = runner.invoke(
        app,
        [
            "data=coco8.yaml",
            "model=yolox-s",
            "--allow-download-scripts",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert dummy.received["allow_download_scripts"] is True
