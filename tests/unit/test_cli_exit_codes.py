"""Command-level tests for CLI exit-code contracts."""

import json

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands import export, predict, special
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer(add_completion=False, no_args_is_help=True)
    app.command("predict", cls=KeyValueCommand)(predict.predict_cmd)
    app.command("export", cls=KeyValueCommand)(export.export_cmd)
    app.command("info", cls=KeyValueCommand)(special.info_cmd)
    return app


def test_predict_missing_source_exits_with_data_error_code():
    app = _make_app()
    result = runner.invoke(
        app,
        ["predict", "source=does-not-exist.jpg", "model=yolox-s", "--json"],
    )

    assert result.exit_code == 3
    data = json.loads(result.stdout)
    assert data["error"] == "source_not_found"


def test_export_precision_conflict_exits_with_usage_error_code():
    app = _make_app()
    result = runner.invoke(
        app,
        [
            "export",
            "model=yolox-s",
            "format=onnx",
            "half=true",
            "int8=true",
            "--json",
        ],
    )

    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert data["error"] == "config_conflict"


def test_info_unknown_model_exits_with_model_error_code():
    app = _make_app()
    result = runner.invoke(app, ["info", "model=definitely-not-a-model", "--json"])

    assert result.exit_code == 4
    data = json.loads(result.stdout)
    assert data["error"] == "model_not_found"
