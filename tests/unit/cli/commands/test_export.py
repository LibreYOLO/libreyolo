from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.parsing import KeyValueCommand


pytestmark = pytest.mark.unit
runner = CliRunner()


def _build_app() -> typer.Typer:
    from libreyolo.cli.commands import export

    app = typer.Typer(add_completion=False)
    app.command("export", cls=KeyValueCommand)(export.export_cmd)
    return app


def _parse_json_output(output: str) -> dict:
    for line in output.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"No JSON found in output:\n{output}")


class _LoadedModel:
    FAMILY = "yolo9"
    size = "t"
    INPUT_SIZES = {"t": 128}

    def __init__(self, output_path, captured):
        self.output_path = output_path
        self.captured = captured

    def _get_input_size(self):
        return 128

    def export(self, format, **kwargs):
        self.captured["format"] = format
        self.captured["kwargs"] = kwargs
        self.output_path.mkdir()
        return str(self.output_path)


def test_export_cli_allows_coreml_embedded_nms(monkeypatch, tmp_path):
    from libreyolo.cli.commands import export

    captured = {}
    monkeypatch.setattr(export, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        export,
        "load_model_or_exit",
        lambda out, model, model_path, device: _LoadedModel(
            tmp_path / "model.mlpackage", captured
        ),
    )

    result = runner.invoke(
        _build_app(),
        [
            "model=dummy.pt",
            "format=coreml",
            "nms=true",
            "conf=0.2",
            "iou=0.4",
            "max_det=12",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = _parse_json_output(result.output)
    assert data["format"] == "coreml"
    assert captured["format"] == "coreml"
    assert captured["kwargs"]["nms"] is True
    assert captured["kwargs"]["conf"] == 0.2
    assert captured["kwargs"]["iou"] == 0.4
    assert "max_det" not in captured["kwargs"]
