"""Both FCOS3D CLI grammars and machine output."""

import json

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands.fcos3d import fcos3d_cmd
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit


def app():
    application = typer.Typer()
    application.command("fcos3d", cls=KeyValueCommand)(fcos3d_cmd)
    return application


@pytest.mark.parametrize("key_value", [False, True])
def test_grammars(tmp_path, monkeypatch, key_value):
    from libreyolo import LibreFCOS3D
    from libreyolo.utils.results import Boxes, Boxes3D, Results

    calls = []
    k = np.eye(3, dtype=np.float32)
    np.save(tmp_path / "k.npy", k)
    monkeypatch.setattr(LibreFCOS3D, "__init__", lambda self, path, **kwargs: None)

    def predict(self, source, **kwargs):
        calls.append(kwargs)
        print("loading progress")
        return Results(
            Boxes(np.empty((0, 4)), np.empty(0), np.empty(0)),
            (32, 32),
            boxes3d=Boxes3D(np.empty((0, 14)), (32, 32), k),
        )

    monkeypatch.setattr(LibreFCOS3D, "predict", predict)
    pairs = {
        "source": "image.jpg",
        "model": "weights.pth",
        "intrinsics": str(tmp_path / "k.npy"),
        "conf": "0.42",
    }
    args = (
        [f"{key}={val}" for key, val in pairs.items()]
        if key_value
        else [token for key, val in pairs.items() for token in (f"--{key}", val)]
    )
    args += (
        ["json=true", "save=false", "quiet=true"]
        if key_value
        else ["--json", "--quiet"]
    )
    result = CliRunner().invoke(app(), args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["task"] == "detect3d"
    assert calls[0]["conf"] == 0.42
    assert calls[0]["save"] is False
    assert result.stderr == ""


def test_help_and_error():
    result = CliRunner().invoke(app(), ["--help-json"])
    assert result.exit_code == 0
    assert "intrinsics" in json.loads(result.stdout).__str__()
    result = CliRunner().invoke(
        app(), ["source=x.jpg", "model=x.pth", "intrinsics=missing.npy", "--json"]
    )
    assert result.exit_code != 0
    assert "io_error" in result.stdout
