"""Tests for command-specific --help-json schemas."""

import json

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands import export, predict, train, val
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer(add_completion=False, no_args_is_help=True)
    app.command("predict", cls=KeyValueCommand)(predict.predict_cmd)
    app.command("train", cls=KeyValueCommand)(train.train_cmd)
    app.command("val", cls=KeyValueCommand)(val.val_cmd)
    app.command("export", cls=KeyValueCommand)(export.export_cmd)
    return app


def _flags_for(command: str) -> list[str]:
    app = _make_app()
    result = runner.invoke(app, [command, "--help-json"])
    assert result.exit_code == 0
    return json.loads(result.stdout)["flags"]


def test_predict_help_json_only_lists_predict_flags():
    flags = _flags_for("predict")
    assert "--json" in flags
    assert "--quiet" in flags
    assert "--verbose" in flags
    assert "--dry-run" not in flags
    assert "--yes" not in flags


def test_train_help_json_lists_dry_run_but_not_yes():
    flags = _flags_for("train")
    assert "--dry-run" in flags
    assert "--json" in flags
    assert "--quiet" in flags
    assert "--yes" not in flags


def test_val_help_json_only_lists_val_flags():
    flags = _flags_for("val")
    assert "--json" in flags
    assert "--quiet" in flags
    assert "--dry-run" not in flags
    assert "--yes" not in flags


def test_export_help_json_only_lists_export_flags():
    flags = _flags_for("export")
    assert "--json" in flags
    assert "--quiet" in flags
    assert "--dry-run" not in flags
    assert "--yes" not in flags
