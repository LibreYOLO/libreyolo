"""The val command's single_cls flag reaches val() and is gated like train."""

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


def _fake_model(family, task, captured):
    class _Model:
        FAMILY = family
        size = "t"
        device = "cpu"

        def val(self, **kwargs):
            captured["kwargs"] = kwargs
            return {"metrics/mAP50-95": 0.5, "metrics/mAP50": 0.6}

    model = _Model()
    model.task = task
    return model


@pytest.mark.parametrize(
    "grammar", [["single_cls=true"], ["--single-cls"]], ids=["key=value", "--flag"]
)
def test_val_single_cls_reaches_val(monkeypatch, tmp_path, grammar):
    captured = {}
    monkeypatch.setattr(
        "libreyolo.cli.commands.val.load_model_or_exit",
        lambda *a, **k: _fake_model("yolo9", "detect", captured),
    )
    result = runner.invoke(
        _make_app(),
        ["data=coco8.yaml", "model=LibreYOLO9t.pt", f"project={tmp_path}", *grammar, "--json"],
    )
    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["single_cls"] is True


def test_val_single_cls_is_not_sent_by_default(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "libreyolo.cli.commands.val.load_model_or_exit",
        lambda *a, **k: _fake_model("yolo9", "detect", captured),
    )
    result = runner.invoke(
        _make_app(),
        ["data=coco8.yaml", "model=LibreYOLO9t.pt", f"project={tmp_path}", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert "single_cls" not in captured["kwargs"]


def test_val_single_cls_rejected_outside_detection(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "libreyolo.cli.commands.val.load_model_or_exit",
        lambda *a, **k: _fake_model("dfine", "segment", captured),
    )
    result = runner.invoke(
        _make_app(),
        ["data=coco8.yaml", "model=LibreDFINEn-seg.pt", f"project={tmp_path}", "single_cls=true", "--json"],
    )
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"] == "config_unsupported"
    assert "kwargs" not in captured


def test_val_help_json_lists_single_cls():
    result = runner.invoke(_make_app(), ["--help-json"])
    assert result.exit_code == 0, result.output
    names = {p["name"] for p in json.loads(result.stdout)["parameters"]}
    assert "single_cls" in names
