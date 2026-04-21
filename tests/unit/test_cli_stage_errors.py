"""Tests for stage-aware CLI runtime error messages."""

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands import export, train, val
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer(add_completion=False, no_args_is_help=True)
    app.command("train", cls=KeyValueCommand)(train.train_cmd)
    app.command("val", cls=KeyValueCommand)(val.val_cmd)
    app.command("export", cls=KeyValueCommand)(export.export_cmd)
    return app


class _FailingModel:
    FAMILY = "yolox"
    size = "s"
    device = "cpu"

    def train(self, **kwargs):
        raise RuntimeError("disk full")

    def val(self, **kwargs):
        raise RuntimeError("disk full")

    def export(self, **kwargs):
        raise RuntimeError("disk full")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("libreyolo.cli.config.resolve_model_name", lambda model: model)
    monkeypatch.setattr("libreyolo.LibreYOLO", lambda *args, **kwargs: _FailingModel())
    monkeypatch.setattr(
        "libreyolo.utils.general.increment_path",
        lambda path, exist_ok=False, mkdir=False: Path(path),
    )
    return _make_app()


def test_train_runtime_error_includes_stage_context(app):
    result = runner.invoke(
        app,
        ["train", "data=coco8.yaml", "model=yolox-s", "--json"],
    )

    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["error"] == "io_error"
    assert data["message"] == "Training failed: disk full"


def test_val_runtime_error_includes_stage_context(app):
    result = runner.invoke(
        app,
        ["val", "data=coco8.yaml", "model=yolox-s", "--json"],
    )

    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["error"] == "io_error"
    assert data["message"] == "Validation failed: disk full"


def test_export_runtime_error_includes_stage_context(app):
    result = runner.invoke(
        app,
        ["export", "model=yolox-s", "format=onnx", "--json"],
    )

    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["error"] == "io_error"
    assert data["message"] == "Export failed: disk full"
