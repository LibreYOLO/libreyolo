"""Both CLI grammars and JSON/error output for optional 3D detection."""

import json

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands.wilddet3d import wilddet3d_cmd
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit


def app():
    application = typer.Typer()
    application.command("wilddet3d", cls=KeyValueCommand)(wilddet3d_cmd)
    return application


@pytest.mark.parametrize("key_value", [False, True])
def test_cli_grammars(tmp_path, monkeypatch, key_value):
    from libreyolo.models.wilddet3d import LibreWildDet3D
    from libreyolo.utils.results import Boxes, Boxes3D, Results

    calibration = tmp_path / "k.npy"
    k = np.eye(3, dtype=np.float32)
    np.save(calibration, k)
    calls = []

    def init(self, path, **kwargs):
        calls.append((path, kwargs))

    def predict(self, source, **kwargs):
        print("upstream progress")  # Must not leak into JSON stdout.
        calls.append((source, kwargs))
        return Results(
            Boxes(np.empty((0, 4)), np.empty(0), np.empty(0)),
            (100, 100),
            path=source,
            boxes3d=Boxes3D(np.empty((0, 14)), (100, 100), k),
        )

    monkeypatch.setattr(LibreWildDet3D, "__init__", init)
    monkeypatch.setattr(LibreWildDet3D, "predict", predict)
    pairs = {
        "source": "image.jpg",
        "model": "weights.pt",
        "intrinsics": str(calibration),
        "text": '["car"]',
        "conf": "0.42",
    }
    args = (
        [f"{key}={value}" for key, value in pairs.items()]
        if key_value
        else [token for key, value in pairs.items() for token in (f"--{key}", value)]
    )
    args += ["json=true", "save=false"] if key_value else ["--json"]
    result = CliRunner().invoke(app(), args)
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert document["task"] == "detect3d"
    assert document["schema_version"] == 1
    assert document["results"][0]["detections"] == []
    assert calls[0][1]["conf"] == 0.42
    assert calls[1][1]["text"] == ["car"]
    assert calls[1][1]["save"] is False


def test_help_json():
    result = CliRunner().invoke(app(), ["--help-json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert "intrinsics" in str(data)
    assert "prompt_mode" in str(data)


def test_malformed_json_error(tmp_path):
    result = CliRunner().invoke(
        app(),
        ["source=x.jpg", "model=x.pt", "intrinsics=x.npy", "text=[invalid", "--json"],
    )
    assert result.exit_code != 0
    assert json.loads(result.stdout)["error"] == "config_type_error"


def test_quiet_suppresses_upstream_output(tmp_path, monkeypatch):
    import sys

    from libreyolo.models.wilddet3d import LibreWildDet3D

    k = tmp_path / "k.npy"
    np.save(k, np.eye(3))

    def init(self, *args, **kwargs):
        print("upstream stdout")
        print("upstream stderr", file=sys.stderr)

    monkeypatch.setattr(LibreWildDet3D, "__init__", init)
    monkeypatch.setattr(LibreWildDet3D, "predict", lambda *a, **k: [])
    result = CliRunner().invoke(
        app(),
        ["source=x", "model=x", f"intrinsics={k}", 'text=["car"]', "--quiet", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert json.loads(result.stdout)["results"] == []


def test_inventory_advertises_real_command():
    from libreyolo.cli.commands.special import models_cmd

    application = typer.Typer()
    application.command("models", cls=KeyValueCommand)(models_cmd)
    result = CliRunner().invoke(application, ["--json"])
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    # The command must not advertise a nonexistent generic factory alias.
    rows = document["families"]
    row = next(item for item in rows if item["name"] == "wilddet3d")
    assert row["cli_names"] == ["wilddet3d"]
