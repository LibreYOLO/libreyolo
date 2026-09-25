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


def test_export_cli_rejects_coreml_max_det(monkeypatch):
    from libreyolo.cli.commands import export

    monkeypatch.setattr(export, "resolve_model_or_exit", lambda out, model: model)

    result = runner.invoke(
        _build_app(),
        [
            "model=dummy.pt",
            "format=coreml",
            "nms=true",
            "max_det=12",
            "--json",
        ],
    )

    assert result.exit_code == 2
    data = _parse_json_output(result.output)
    assert data["error"] == "config_unsupported"
    assert "max_det is only supported for ONNX" in data["message"]


def test_export_cli_passes_rknn_target_and_verify(monkeypatch, tmp_path):
    from libreyolo.cli.commands import export

    captured = {}
    monkeypatch.setattr(export, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        export,
        "load_model_or_exit",
        lambda out, model, model_path, device: _LoadedModel(
            tmp_path / "model.rknn", captured
        ),
    )

    result = runner.invoke(
        _build_app(),
        [
            "model=dummy.pt",
            "format=rknn",
            "name=rk3588",
            "verify=true",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = _parse_json_output(result.output)
    assert captured["format"] == "rknn"
    assert captured["kwargs"]["name"] == "rk3588"
    assert captured["kwargs"]["verify"] is True
    assert data["target"] == "rk3588"
    assert data["verified"] is True


@pytest.mark.parametrize("option", ["name=rk3588", "verify=true"])
def test_export_cli_rejects_rknn_only_options_for_other_formats(option):
    result = runner.invoke(
        _build_app(),
        ["model=dummy.pt", "format=onnx", option, "--json"],
    )

    assert result.exit_code == 2
    data = _parse_json_output(result.output)
    assert data["error"] == "config_unsupported"


def test_export_cli_forwards_paddle_static_defaults(monkeypatch, tmp_path):
    from libreyolo.cli.commands import export

    captured = {}
    monkeypatch.setattr(export, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        export,
        "load_model_or_exit",
        lambda out, model, model_path, device: _LoadedModel(
            tmp_path / "model_paddle", captured
        ),
    )

    result = runner.invoke(
        _build_app(),
        ["model=dummy.pt", "format=paddle", "--json"],
    )

    assert result.exit_code == 0, result.output
    data = _parse_json_output(result.output)
    assert data["format"] == "paddle"
    assert captured["format"] == "paddle"
    assert captured["kwargs"]["dynamic"] is False
    assert captured["kwargs"]["batch"] == 1
    assert captured["kwargs"]["simplify"] is True


@pytest.mark.parametrize("channels", [2, 3])
@pytest.mark.parametrize(
    "arguments",
    [["model=dummy.pt", "format=onnx"], ["--model", "dummy.pt", "--format", "onnx"]],
)
def test_export_cli_reports_histogram_channel_count(
    monkeypatch, tmp_path, channels, arguments
):
    from libreyolo.cli.commands import export

    loaded = _LoadedModel(tmp_path / "model.onnx", {})
    loaded.input_profile = {"format": "event_histogram"} if channels == 2 else None
    monkeypatch.setattr(export, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(export, "load_model_or_exit", lambda *args, **kwargs: loaded)
    result = runner.invoke(_build_app(), [*arguments, "--json"])
    assert result.exit_code == 0, result.output
    assert _parse_json_output(result.output)["input_shape"] == [1, channels, 128, 128]


@pytest.mark.parametrize(
    ("native", "exported", "expected"),
    [
        ((192, 320), (192, 320), [192, 320]),  # rectangular export (#899)
        ((192, 320), (320, 320), [320, 320]),  # square fallback for the format
    ],
)
def test_export_cli_reports_the_exported_canvas(
    monkeypatch, tmp_path, native, exported, expected
):
    from libreyolo.cli.commands import export

    class _RectModel(_LoadedModel):
        def _get_input_size(self):
            return native

        def export(self, format, **kwargs):
            self._last_export_imgsz = exported
            return super().export(format, **kwargs)

    loaded = _RectModel(tmp_path / "model.onnx", {})
    monkeypatch.setattr(export, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(export, "load_model_or_exit", lambda *args, **kwargs: loaded)
    result = runner.invoke(_build_app(), ["model=dummy.pt", "format=onnx", "--json"])

    assert result.exit_code == 0, result.output
    assert _parse_json_output(result.output)["input_shape"] == [1, 3, *expected]
