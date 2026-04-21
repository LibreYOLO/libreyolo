"""Tests for shared CLI model reference validation."""

import json

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands import export, predict, special, train
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer(add_completion=False, no_args_is_help=True)
    app.command("predict", cls=KeyValueCommand)(predict.predict_cmd)
    app.command("train", cls=KeyValueCommand)(train.train_cmd)
    app.command("export", cls=KeyValueCommand)(export.export_cmd)
    app.command("info", cls=KeyValueCommand)(special.info_cmd)
    return app


def test_predict_unknown_model_uses_model_not_found_error():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "predict",
            "source=libreyolo/assets/parkour.jpg",
            "model=definitely-not-a-model",
            "--json",
        ],
    )

    assert result.exit_code == 4
    data = json.loads(result.stdout)
    assert data["error"] == "model_not_found"


def test_train_dry_run_rejects_unknown_model():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "train",
            "data=coco8.yaml",
            "model=definitely-not-a-model",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 4
    data = json.loads(result.stdout)
    assert data["error"] == "model_not_found"


def test_info_accepts_known_weight_filename(monkeypatch):
    app = _make_app()

    class _DummyParameter:
        def numel(self) -> int:
            return 42

    class _DummyTorchModel:
        def parameters(self):
            return [_DummyParameter()]

    class _DummyModel:
        FAMILY = "yolox"
        size = "s"
        nb_classes = 80
        device = "cpu"
        names = {}
        model = _DummyTorchModel()
        INPUT_SIZES = {"s": 640}

    monkeypatch.setattr("libreyolo.LibreYOLO", lambda *args, **kwargs: _DummyModel())

    result = runner.invoke(app, ["info", "model=LibreYOLOXs.pt", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model"] == "LibreYOLOXs.pt"
    assert data["model_family"] == "yolox"
