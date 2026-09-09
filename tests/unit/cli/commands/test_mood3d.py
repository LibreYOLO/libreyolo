"""Both CLI grammars and JSON behavior for 3D-MOOD."""

import json

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands.mood3d import mood3d_cmd
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit


def app():
    application = typer.Typer()
    application.command("3dmood", cls=KeyValueCommand)(mood3d_cmd)
    return application


@pytest.mark.parametrize("key_value", [False, True])
def test_cli_grammars(tmp_path, monkeypatch, key_value):
    from libreyolo.models.mood3d import Libre3DMOOD
    from libreyolo.utils.results import Boxes, Boxes3D, DepthMap, Results

    calibration = tmp_path / "k.npy"
    k = np.eye(3, dtype=np.float32)
    np.save(calibration, k)
    calls = []

    def init(self, path, **kwargs):
        calls.append((path, kwargs))
        self.size = kwargs["size"] or "t"

    def predict(self, source, **kwargs):
        calls.append((source, kwargs))
        return Results(
            Boxes(np.empty((0, 4)), np.empty(0), np.empty(0)),
            (10, 10),
            path=source,
            boxes3d=Boxes3D(np.empty((0, 14)), (10, 10), k),
            depth_map=DepthMap(np.ones((10, 10)), (10, 10)),
        )

    monkeypatch.setattr(Libre3DMOOD, "__init__", init)
    monkeypatch.setattr(Libre3DMOOD, "predict", predict)
    monkeypatch.setattr(Libre3DMOOD, "__enter__", lambda self: self)
    monkeypatch.setattr(Libre3DMOOD, "__exit__", lambda self, *args: None)
    pairs = {
        "source": "image.jpg",
        "model": "weights.pt",
        "size": "b",
        "intrinsics": str(calibration),
        "text": '["chair","table"]',
        "conf": "0.2",
        "iou": "0.4",
        "max_det": "20",
        "runtime_path": "/runtime",
        "runtime_python": "/python",
    }
    args = (
        [f"{key}={value}" for key, value in pairs.items()]
        if key_value
        else [
            token
            for key, value in pairs.items()
            for token in (f"--{key.replace(chr(95), chr(45))}", value)
        ]
    )
    args += ["json=true"] if key_value else ["--json"]
    result = CliRunner().invoke(app(), args)
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert document["task"] == "detect3d"
    assert document["results"][0]["depth"] == {"min": 1.0, "max": 1.0, "mean": 1.0}
    assert calls[0][1]["size"] == "b"
    assert calls[0][1]["conf"] == 0.2
    assert calls[0][1]["max_det"] == 20
    assert calls[1][1]["text"] == ["chair", "table"]


def test_help_json():
    result = CliRunner().invoke(app(), ["--help-json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert "intrinsics" in str(data)
    assert "max_det" in str(data)


def test_malformed_json_error():
    result = CliRunner().invoke(
        app(),
        ["source=x.jpg", "intrinsics=x.npy", "text=[invalid", "--json"],
    )
    assert result.exit_code != 0
    assert json.loads(result.stdout)["error"] == "config_type_error"


def test_inventory_advertises_dedicated_command():
    from libreyolo.cli.commands.special import models_cmd

    application = typer.Typer()
    application.command("models", cls=KeyValueCommand)(models_cmd)
    result = CliRunner().invoke(application, ["--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)["families"]
    row = next(item for item in rows if item["name"] == "3dmood")
    assert row["cli_names"] == ["3dmood"]
    assert row["tasks"] == ["detect3d"]
