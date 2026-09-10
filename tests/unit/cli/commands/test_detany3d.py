"""DetAny3D CLI grammars, prompt forwarding and clean machine output."""

import json

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands.detany3d import detany3d_cmd
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit


def app():
    result = typer.Typer()
    result.command("detany3d", cls=KeyValueCommand)(detany3d_cmd)
    return result


@pytest.mark.parametrize("key_value", [False, True])
def test_grammars_and_json(monkeypatch, key_value):
    from libreyolo import LibreDetAny3D
    from libreyolo.utils.results import Boxes, Boxes3D, Results

    calls = []

    def init(self, path, **kwargs):
        calls.append(("init", path, kwargs))
        self._backend = None

    def predict(self, source, **kwargs):
        calls.append(("predict", source, kwargs))
        print("runtime progress")
        return Results(
            Boxes(np.empty((0, 4)), np.empty(0), np.empty(0)),
            (100, 100),
            boxes3d=Boxes3D(np.empty((0, 14)), (100, 100), np.eye(3)),
        )

    monkeypatch.setattr(LibreDetAny3D, "__init__", init)
    monkeypatch.setattr(LibreDetAny3D, "predict", predict)
    pairs = {
        "source": "image.jpg",
        "model": "model.pth",
        "runtime_path": "runtime",
        "text": '["car","person"]',
        "conf": "0.4",
    }
    args = (
        [f"{k}={v}" for k, v in pairs.items()]
        if key_value
        else [
            part for k, v in pairs.items() for part in (f"--{k.replace('_', '-')}", v)
        ]
    )
    args += (
        ["json=true", "quiet=true", "save=false"]
        if key_value
        else ["--json", "--quiet"]
    )
    result = CliRunner().invoke(app(), args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["task"] == "detect3d"
    assert result.stderr == ""
    assert calls[0][2]["conf"] == 0.4
    assert calls[1][2]["text"] == ["car", "person"]
    assert calls[1][2]["save"] is False


def test_help_and_bad_prompt():
    result = CliRunner().invoke(app(), ["--help-json"])
    assert result.exit_code == 0
    assert "runtime_python" in str(json.loads(result.stdout))
    result = CliRunner().invoke(app(), ["source=x", "model=x", "points=[bad", "--json"])
    assert result.exit_code != 0
    assert "config_type_error" in result.stdout
