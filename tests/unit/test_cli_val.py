"""Tests for the val command."""

import json

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands.val import val_cmd
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer()
    app.command("val", cls=KeyValueCommand)(val_cmd)
    return app


class DummyModel:
    FAMILY = "yolox"
    size = "s"
    device = "cpu"

    def __init__(self):
        self.received = None

    def val(self, **kwargs):
        self.received = kwargs
        return {
            "metrics/mAP50": 0.5,
            "metrics/mAP50-95": 0.3,
            "metrics/precision": 0.7,
            "metrics/recall": 0.6,
        }


def test_val_cli_uses_public_argument_names(monkeypatch):
    dummy = DummyModel()

    monkeypatch.setattr(
        "libreyolo.cli.commands.val.resolve_model_or_exit", lambda out, model: model
    )
    monkeypatch.setattr("libreyolo.LibreYOLO", lambda *args, **kwargs: dummy)
    monkeypatch.setattr(
        "libreyolo.utils.general.increment_path",
        lambda path, exist_ok=False, mkdir=False: path,
    )

    app = _make_app()
    result = runner.invoke(
        app,
        [
            "model=yolox-s",
            "data=coco8.yaml",
            "batch=32",
            "conf=0.2",
            "iou=0.5",
            "workers=7",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["schema_version"] == 1

    assert dummy.received["batch"] == 32
    assert dummy.received["conf"] == 0.2
    assert dummy.received["iou"] == 0.5
    assert dummy.received["workers"] == 7
    assert "batch_size" not in dummy.received
    assert "conf_thres" not in dummy.received
    assert "iou_thres" not in dummy.received
    assert "num_workers" not in dummy.received


def test_val_cli_passes_allow_download_scripts(monkeypatch):
    dummy = DummyModel()

    monkeypatch.setattr(
        "libreyolo.cli.commands.val.resolve_model_or_exit", lambda out, model: model
    )
    monkeypatch.setattr("libreyolo.LibreYOLO", lambda *args, **kwargs: dummy)
    monkeypatch.setattr(
        "libreyolo.utils.general.increment_path",
        lambda path, exist_ok=False, mkdir=False: path,
    )

    app = _make_app()
    result = runner.invoke(
        app,
        [
            "model=yolox-s",
            "data=coco8.yaml",
            "--allow-download-scripts",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert dummy.received["allow_download_scripts"] is True
