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
        app.command(cmd_name, cls=KeyValueCommand)(getattr(special, f"{cmd_name}_cmd"))
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
    """Test special commands: version, checks, models, formats, cfg, info."""

    @pytest.fixture(scope="class")
    def app(self):
        return _build_app()

    def test_no_args_shows_help(self, app):
        result = runner.invoke(app, [])
        # Typer with multiple commands exits 2 and prints usage when no subcommand given
        assert result.exit_code == 2
        assert "LibreYOLO" in result.output

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

    @pytest.mark.onnx
    def test_formats(self, app):
        result = runner.invoke(app, ["formats"])
        assert result.exit_code == 0
        assert "onnx" in result.output

    @pytest.mark.onnx
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

    @pytest.mark.yolox
    def test_info(self, app):
        result = runner.invoke(app, ["info", "model=yolox-s"])
        assert result.exit_code == 0
        assert "yolox" in result.output
        assert "Parameters" in result.output

    @pytest.mark.yolox
    def test_info_json(self, app):
        result = runner.invoke(app, ["info", "model=yolox-s", "--json"])
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["model_family"] == "yolox"
        assert data["parameters"] > 0


# =========================================================================
# Predict command
# =========================================================================


@pytest.mark.yolox
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

    def test_predict_mixed_syntax(self, app):
        """Mixed key=value and --key value syntax works for predict."""
        result = runner.invoke(
            app,
            [
                "predict",
                "source=libreyolo/assets/parkour.jpg",
                "--model",
                "yolox-s",
                "conf=0.5",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["model_family"] == "yolox"

    def test_predict_bare_bool_flags(self, app):
        """Bare boolean flags (half, save) work for predict."""
        result = runner.invoke(
            app,
            [
                "predict",
                "source=libreyolo/assets/parkour.jpg",
                "model=yolox-s",
                "half",
                "save",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert len(data["results"]) == 1

    def test_predict_save(self, app):
        """Save flag produces output_path in JSON."""
        result = runner.invoke(
            app,
            [
                "predict",
                "source=libreyolo/assets/parkour.jpg",
                "model=yolox-s",
                "save",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert "output_path" in data

    def test_predict_local_weights(self, app):
        """Local weights path works for predict."""
        result = runner.invoke(
            app,
            [
                "predict",
                "source=libreyolo/assets/parkour.jpg",
                "model=weights/LibreYOLOXs.pt",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["model_family"] == "yolox"
        assert data["image_size"] == [640, 640]
        assert len(data["results"][0]["detections"]) > 0

    def test_predict_reports_effective_imgsz(self, app):
        """JSON output reports the requested inference size."""
        result = runner.invoke(
            app,
            [
                "predict",
                "source=libreyolo/assets/parkour.jpg",
                "model=weights/LibreYOLOXs.pt",
                "imgsz=320",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["image_size"] == [320, 320]

    def test_predict_missing_source(self, app):
        result = runner.invoke(
            app, ["predict", "source=nonexistent.jpg", "model=yolox-s"]
        )
        assert result.exit_code == 3

    def test_predict_missing_source_json(self, app):
        result = runner.invoke(
            app, ["predict", "source=nonexistent.jpg", "model=yolox-s", "--json"]
        )
        assert result.exit_code == 3
        data = _parse_json_output(result.output)
        assert data["error"] == "source_not_found"

    def test_predict_missing_required_source(self, app):
        """Missing source entirely exits with usage error."""
        result = runner.invoke(app, ["predict", "model=yolox-s"])
        assert result.exit_code == 2

    def test_predict_help(self, app):
        result = runner.invoke(app, ["predict", "--help"])
        assert result.exit_code == 0
        assert "source" in result.output
        assert "model" in result.output

    def test_detect_task_prefix(self, app, monkeypatch):
        """Optional 'detect' task prefix is stripped by the entrypoint."""
        import sys

        # Simulate what entrypoint() does: strip 'detect' from sys.argv
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "libreyolo",
                "detect",
                "predict",
                "source=libreyolo/assets/parkour.jpg",
                "model=yolox-s",
                "--json",
            ],
        )
        from libreyolo.cli import _strip_task_prefix

        argv = _strip_task_prefix(sys.argv)
        # After stripping, argv should not contain 'detect'
        assert "detect" not in argv
        result = runner.invoke(
            app,
            argv[1:],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["model_family"] == "yolox"


# =========================================================================
# Val command
# =========================================================================


class TestVal:
    """Test val command with real validation."""

    @pytest.fixture(scope="class")
    def app(self):
        return _build_app()

    @pytest.mark.yolox
    def test_val_json(self, app):
        """Full validation pipeline produces metrics."""
        result = runner.invoke(
            app,
            [
                "val",
                "data=coco8.yaml",
                "model=yolox-s",
                "device=cpu",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["schema_version"] == 1
        assert data["model_family"] == "yolox"
        assert data["split"] == "val"
        assert "metrics" in data
        assert "mAP50" in data["metrics"]
        assert "mAP50_95" in data["metrics"]
        assert isinstance(data["metrics"]["mAP50"], float)

    @pytest.mark.yolox
    def test_val_missing_data(self, app):
        """Missing required data arg errors cleanly."""
        result = runner.invoke(app, ["val", "model=yolox-s"])
        assert result.exit_code == 2

    @pytest.mark.yolox
    def test_val_with_overrides(self, app):
        """Validation with batch and conf overrides produces metrics."""
        result = runner.invoke(
            app,
            [
                "val",
                "model=yolox-s",
                "data=coco8.yaml",
                "batch=8",
                "conf=0.01",
                "device=cpu",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert "metrics" in data
        assert "mAP50" in data["metrics"]

    @pytest.mark.yolo9
    def test_val_yolo9(self, app):
        """YOLOv9 validation through CLI."""
        result = runner.invoke(
            app,
            [
                "val",
                "model=yolo9-t",
                "data=coco8.yaml",
                "device=cpu",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["model_family"] == "yolo9"
        assert "metrics" in data

    @pytest.mark.yolox
    def test_val_local_weights(self, app):
        """Local weights path works for val."""
        result = runner.invoke(
            app,
            [
                "val",
                "model=weights/LibreYOLOXs.pt",
                "data=coco8.yaml",
                "device=cpu",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["model_family"] == "yolox"
        assert "metrics" in data


# =========================================================================
# Predict (multi-family)
# =========================================================================


@pytest.mark.yolo9
class TestPredictMultiFamily:
    """Test predict across model families to verify factory routing."""

    @pytest.fixture(scope="class")
    def app(self):
        return _build_app()

    def test_predict_yolo9(self, app):
        """YOLO9 model loads and produces detections through the same CLI."""
        result = runner.invoke(
            app,
            [
                "predict",
                "source=libreyolo/assets/parkour.jpg",
                "model=yolo9-t",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["model_family"] == "yolo9"
        assert len(data["results"]) == 1
        assert len(data["results"][0]["detections"]) > 0


# =========================================================================
# Train command (dry-run only — real training is in test_rf1_training.py)
# =========================================================================


class TestTrainDryRun:
    """Test train command with --dry-run to verify config resolution."""

    @pytest.fixture(scope="class")
    def app(self):
        return _build_app()

    @pytest.mark.yolox
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

    @pytest.mark.yolo9
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

    @pytest.mark.rtdetr
    def test_rtdetr_weight_filename_uses_family_defaults(self, app):
        result = runner.invoke(
            app,
            [
                "train",
                "data=coco8.yaml",
                "model=LibreRTDETRr18.pt",
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["model_family"] == "rtdetr"
        cfg = data["resolved_config"]
        assert cfg["epochs"] == 72
        assert cfg["batch"] == 4
        assert cfg["optimizer"] == "adamw"
        assert cfg["scheduler"] == "linear"

    @pytest.mark.yolox
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

    @pytest.mark.yolox
    def test_missing_data_arg(self, app):
        result = runner.invoke(app, ["train", "model=yolox-s", "--dry-run"])
        assert result.exit_code == 2

    def test_train_help(self, app):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "data" in result.output
        assert "epochs" in result.output


# =========================================================================
# Export command
# =========================================================================


@pytest.mark.onnx
@pytest.mark.yolox
class TestExport:
    """Test export command."""

    @pytest.fixture(scope="class")
    def app(self):
        return _build_app()

    def test_export_onnx(self, app):
        pytest.importorskip("onnx")

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

    def test_export_onnx_with_options(self, app):
        pytest.importorskip("onnx")
        result = runner.invoke(
            app,
            [
                "export",
                "model=yolox-s",
                "format=onnx",
                "dynamic",
                "device=cpu",
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        assert data["format"] == "onnx"
        assert data["dynamic"] is True

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
        assert result.exit_code == 2
        data = _parse_json_output(result.output)
        assert data["error"] == "config_conflict"
