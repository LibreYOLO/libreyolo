"""End-to-end tests for the LibreYOLO CLI.

Tests the full CLI pipeline using CliRunner (Typer/Click test client).
Requires GPU and model weights.
"""

import json

import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.e2e

runner = CliRunner()


# =========================================================================
# Helpers
# =========================================================================


def _build_app() -> typer.Typer:
    """Create a fresh Typer app with all commands.

    We build a new app per test group to avoid duplicate command
    registration on the shared module-level app.
    """
    from libreyolo.cli.commands import export, predict, special, train, val
    from libreyolo.utils.logging import setup_logging

    setup_logging(quiet=True)

    app = typer.Typer(add_completion=False, no_args_is_help=True)

    for cmd_name in ("version", "checks", "models", "formats", "cfg", "info"):
        app.command(cmd_name, cls=KeyValueCommand)(
            getattr(special, f"{cmd_name}_cmd")
        )
    app.command("predict", cls=KeyValueCommand)(predict.predict_cmd)
    app.command("train", cls=KeyValueCommand)(train.train_cmd)
    app.command("val", cls=KeyValueCommand)(val.val_cmd)
    app.command("export", cls=KeyValueCommand)(export.export_cmd)

    return app


def _parse_json_output(output: str) -> dict:
    """Extract JSON from CLI output, skipping library print() noise.

    The library still has bare print() calls (e.g. 'Auto-detected size: s')
    that go to stdout. We find the JSON line and parse only that.
    """
    for line in output.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"No JSON found in output:\n{output}")


# =========================================================================
# Special commands (no GPU needed)
# =========================================================================


class TestSpecialCommands:
    """Test special commands: version, checks, models, formats, cfg."""

    @pytest.fixture(scope="class")
    def app(self):
        return _build_app()

    def test_version(self, app):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "libreyolo" in result.output

    def test_version_json(self, app):
        result = runner.invoke(app, ["version", "--json"])
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert "version" in data
        assert "python" in data
        assert "torch" in data
        assert data["schema_version"] == 1

    def test_checks(self, app):
        result = runner.invoke(app, ["checks"])
        assert result.exit_code == 0
        assert "Python" in result.output
        assert "Torch" in result.output

    def test_checks_json(self, app):
        result = runner.invoke(app, ["checks", "--json"])
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert "python" in data
        assert "gpu" in data
        assert "packages" in data

    def test_models(self, app):
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
        assert "yolox" in result.output
        assert "yolo9" in result.output

    def test_models_json(self, app):
        result = runner.invoke(app, ["models", "--json"])
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        families = {f["name"] for f in data["families"]}
        assert "yolox" in families
        assert "yolo9" in families

    def test_formats(self, app):
        result = runner.invoke(app, ["formats"])
        assert result.exit_code == 0
        assert "onnx" in result.output

    def test_formats_json(self, app):
        result = runner.invoke(app, ["formats", "--json"])
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        format_names = {f["name"] for f in data["formats"]}
        assert "onnx" in format_names

    def test_cfg(self, app):
        result = runner.invoke(app, ["cfg"])
        assert result.exit_code == 0
        assert "epochs" in result.output

    def test_cfg_json(self, app):
        result = runner.invoke(app, ["cfg", "--json"])
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert "train_defaults" in data
        assert "val_defaults" in data
        assert "family_overrides" in data


# =========================================================================
# Predict command
# =========================================================================


class TestPredict:
    """Test predict command with real inference."""

    @pytest.fixture(scope="class")
    def app(self):
        return _build_app()

    def test_predict_basic(self, app):
        result = runner.invoke(
            app, ["predict", "source=libreyolo/assets/parkour.jpg", "model=yolox-s"]
        )
        assert result.exit_code == 0
        assert "person" in result.output

    def test_predict_json(self, app):
        result = runner.invoke(
            app,
            [
                "predict",
                "source=libreyolo/assets/parkour.jpg",
                "model=yolox-s",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["schema_version"] == 1
        assert data["model_family"] == "yolox"
        assert len(data["results"]) == 1
        assert len(data["results"][0]["detections"]) > 0

    def test_predict_with_conf(self, app):
        result = runner.invoke(
            app,
            [
                "predict",
                "source=libreyolo/assets/parkour.jpg",
                "model=yolox-s",
                "conf=0.9",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        for det in data["results"][0]["detections"]:
            assert det["confidence"] >= 0.9

    def test_predict_key_value_syntax(self, app):
        """key=value syntax works for predict."""
        result = runner.invoke(
            app,
            [
                "predict",
                "source=libreyolo/assets/parkour.jpg",
                "model=yolox-s",
                "conf=0.5",
            ],
        )
        assert result.exit_code == 0
        assert "person" in result.output

    def test_predict_standard_syntax(self, app):
        """--key value syntax works for predict."""
        result = runner.invoke(
            app,
            [
                "predict",
                "--source",
                "libreyolo/assets/parkour.jpg",
                "--model",
                "yolox-s",
                "--conf",
                "0.5",
            ],
        )
        assert result.exit_code == 0
        assert "person" in result.output

    def test_predict_missing_source(self, app):
        result = runner.invoke(
            app, ["predict", "source=nonexistent.jpg", "model=yolox-s"]
        )
        assert result.exit_code != 0

    def test_predict_missing_source_json(self, app):
        result = runner.invoke(
            app, ["predict", "source=nonexistent.jpg", "model=yolox-s", "--json"]
        )
        assert result.exit_code != 0
        data = _parse_json_output(result.output)
        assert data["error"] == "source_not_found"


# =========================================================================
# Train command (dry-run only — real training is in test_rf1_training.py)
# =========================================================================


class TestTrainDryRun:
    """Test train command with --dry-run to verify config resolution."""

    @pytest.fixture(scope="class")
    def app(self):
        return _build_app()

    def test_yolox_defaults(self, app):
        result = runner.invoke(
            app,
            ["train", "data=coco8.yaml", "model=yolox-s", "--dry-run", "--json"],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["model_family"] == "yolox"
        cfg = data["resolved_config"]
        assert cfg["momentum"] == 0.9  # YOLOX family default
        assert cfg["scheduler"] == "yoloxwarmcos"

    def test_yolo9_defaults(self, app):
        result = runner.invoke(
            app,
            ["train", "data=coco8.yaml", "model=yolo9-t", "--dry-run", "--json"],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["model_family"] == "yolo9"
        cfg = data["resolved_config"]
        assert cfg["scheduler"] == "linear"  # YOLO9 family default

    def test_user_override_wins(self, app):
        result = runner.invoke(
            app,
            [
                "train",
                "data=coco8.yaml",
                "model=yolox-s",
                "momentum=0.5",
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["resolved_config"]["momentum"] == 0.5  # user override

    def test_help_json(self, app):
        result = runner.invoke(app, ["train", "--help-json"])
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["command"] == "train"
        param_names = {p["name"] for p in data["parameters"]}
        assert "data" in param_names
        assert "model" in param_names
        assert "epochs" in param_names

    def test_missing_data_arg(self, app):
        result = runner.invoke(app, ["train", "model=yolox-s", "--dry-run"])
        assert result.exit_code != 0


# =========================================================================
# Export command
# =========================================================================


class TestExport:
    """Test export command."""

    @pytest.fixture(scope="class")
    def app(self):
        return _build_app()

    def test_export_onnx(self, app):
        result = runner.invoke(
            app,
            [
                "export",
                "model=yolox-s",
                "format=onnx",
                "device=cpu",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["format"] == "onnx"
        assert data["model_family"] == "yolox"
        assert data["output_path"].endswith(".onnx")

    def test_export_half_int8_conflict(self, app):
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
        assert result.exit_code != 0
        data = _parse_json_output(result.output)
        assert data["error"] == "config_conflict"
