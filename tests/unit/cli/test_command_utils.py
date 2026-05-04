"""Tests for libreyolo/cli/command_utils.py.

Covers exit-code contracts, stage-aware error wrapping, model reference
validation, and --help-json callback wiring.
"""

import json
from pathlib import Path

import pytest
import torch
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands import export, predict, special, train, val
from libreyolo.cli.command_utils import (
    get_loaded_model_family,
    get_loaded_model_input_size,
)
from libreyolo.cli.parsing import KeyValueCommand
from libreyolo.utils.results import Boxes, Results

pytestmark = pytest.mark.unit

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(cmds) -> typer.Typer:
    app = typer.Typer(add_completion=False, no_args_is_help=True)
    for name, cmd in cmds:
        app.command(name, cls=KeyValueCommand)(cmd)
    return app


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


def test_predict_missing_source_exits_with_data_error_code():
    app = _make_app(
        [
            ("predict", predict.predict_cmd),
            ("export", export.export_cmd),
            ("info", special.info_cmd),
        ]
    )
    result = runner.invoke(
        app,
        ["predict", "source=does-not-exist.jpg", "model=yolox-s", "--json"],
    )

    assert result.exit_code == 3
    data = json.loads(result.stdout)
    assert data["error"] == "source_not_found"


def test_export_precision_conflict_exits_with_usage_error_code():
    app = _make_app(
        [
            ("predict", predict.predict_cmd),
            ("export", export.export_cmd),
            ("info", special.info_cmd),
        ]
    )
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
    data = json.loads(result.stdout)
    assert data["error"] == "config_conflict"


def test_info_unknown_model_exits_with_model_error_code():
    app = _make_app(
        [
            ("predict", predict.predict_cmd),
            ("export", export.export_cmd),
            ("info", special.info_cmd),
        ]
    )
    result = runner.invoke(app, ["info", "model=definitely-not-a-model", "--json"])

    assert result.exit_code == 4
    data = json.loads(result.stdout)
    assert data["error"] == "model_not_found"


# ---------------------------------------------------------------------------
# Stage-aware runtime errors
# ---------------------------------------------------------------------------


class _FailingModel:
    FAMILY = "yolox"
    size = "s"
    device = "cpu"

    def train(self, **kwargs):
        raise RuntimeError("disk full")

    def val(self, **kwargs):
        raise RuntimeError("disk full")

    def export(self, **kwargs):
        raise RuntimeError("disk full")


@pytest.fixture
def failing_app(monkeypatch):
    monkeypatch.setattr(
        "libreyolo.cli.commands.train.resolve_model_or_exit", lambda out, model: model
    )
    monkeypatch.setattr(
        "libreyolo.cli.commands.val.resolve_model_or_exit", lambda out, model: model
    )
    monkeypatch.setattr(
        "libreyolo.cli.commands.export.resolve_model_or_exit", lambda out, model: model
    )
    monkeypatch.setattr("libreyolo.LibreYOLO", lambda *args, **kwargs: _FailingModel())
    monkeypatch.setattr(
        "libreyolo.utils.general.increment_path",
        lambda path, exist_ok=False, mkdir=False: Path(path),
    )
    return _make_app(
        [
            ("train", train.train_cmd),
            ("val", val.val_cmd),
            ("export", export.export_cmd),
        ]
    )


def test_train_runtime_error_includes_stage_context(failing_app):
    result = runner.invoke(
        failing_app,
        ["train", "data=coco8.yaml", "model=yolox-s", "--json"],
    )

    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["error"] == "io_error"
    assert data["message"] == "Training failed: disk full"


def test_val_runtime_error_includes_stage_context(failing_app):
    result = runner.invoke(
        failing_app,
        ["val", "data=coco8.yaml", "model=yolox-s", "--json"],
    )

    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["error"] == "io_error"
    assert data["message"] == "Validation failed: disk full"


def test_export_runtime_error_includes_stage_context(failing_app):
    result = runner.invoke(
        failing_app,
        ["export", "model=yolox-s", "format=onnx", "--json"],
    )

    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["error"] == "io_error"
    assert data["message"] == "Export failed: disk full"


def test_export_cli_leaves_opset_auto_by_default(monkeypatch, tmp_path):
    captured = {}

    class _ExportModel:
        FAMILY = "deimv2"
        size = "atto"
        INPUT_SIZES = {"atto": 320}

        def export(self, **kwargs):
            captured.update(kwargs)
            out = tmp_path / "model.onnx"
            out.write_bytes(b"onnx")
            return str(out)

    monkeypatch.setattr(
        "libreyolo.cli.commands.export.resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        "libreyolo.cli.commands.export.load_model_or_exit",
        lambda *args, **kwargs: _ExportModel(),
    )
    app = _make_app([("export", export.export_cmd), ("info", special.info_cmd)])

    result = runner.invoke(
        app,
        ["export", "model=deimv2-atto", "format=onnx", "--json"],
    )

    assert result.exit_code == 0
    assert captured["opset"] is None


# ---------------------------------------------------------------------------
# Model reference validation
# ---------------------------------------------------------------------------


def test_predict_unknown_model_uses_model_not_found_error():
    app = _make_app(
        [
            ("predict", predict.predict_cmd),
            ("train", train.train_cmd),
            ("export", export.export_cmd),
            ("info", special.info_cmd),
        ]
    )
    result = runner.invoke(
        app,
        [
            "predict",
            "source=libreyolo/assets/parkour.jpg",
            "model=definitely-not-a-model",
            "--json",
        ],
    )

    assert result.exit_code == 4
    data = json.loads(result.stdout)
    assert data["error"] == "model_not_found"


def test_train_dry_run_rejects_unknown_model():
    app = _make_app(
        [
            ("predict", predict.predict_cmd),
            ("train", train.train_cmd),
            ("export", export.export_cmd),
            ("info", special.info_cmd),
        ]
    )
    result = runner.invoke(
        app,
        [
            "train",
            "data=coco8.yaml",
            "model=definitely-not-a-model",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 4
    data = json.loads(result.stdout)
    assert data["error"] == "model_not_found"


def test_info_accepts_known_weight_filename(monkeypatch):
    app = _make_app(
        [
            ("predict", predict.predict_cmd),
            ("train", train.train_cmd),
            ("export", export.export_cmd),
            ("info", special.info_cmd),
        ]
    )

    class _DummyParameter:
        def numel(self) -> int:
            return 42

    class _DummyTorchModel:
        def parameters(self):
            return [_DummyParameter()]

    class _DummyModel:
        FAMILY = "yolox"
        size = "s"
        nb_classes = 80
        device = "cpu"
        names = {}
        model = _DummyTorchModel()
        INPUT_SIZES = {"s": 640}

    monkeypatch.setattr("libreyolo.LibreYOLO", lambda *args, **kwargs: _DummyModel())

    result = runner.invoke(app, ["info", "model=LibreYOLOXs.pt", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model"] == "LibreYOLOXs.pt"
    assert data["model_family"] == "yolox"


def test_loaded_model_metadata_supports_wrappers_and_backends():
    class _Wrapper:
        FAMILY = "yolox"
        INPUT_SIZES = {"s": 640}
        size = "s"

    class _Backend:
        model_family = "yolox"
        imgsz = 320

    assert get_loaded_model_family(_Wrapper()) == "yolox"
    assert get_loaded_model_input_size(_Wrapper()) == 640
    assert get_loaded_model_family(_Backend()) == "yolox"
    assert get_loaded_model_input_size(_Backend()) == 320
    assert get_loaded_model_input_size(_Wrapper(), imgsz=416) == 416


def test_predict_json_supports_exported_backend_metadata(monkeypatch):
    app = _make_app([("predict", predict.predict_cmd), ("info", special.info_cmd)])

    class _BackendLike:
        model_family = "yolox"
        imgsz = 640
        device = "cpu"

        def __call__(self, source, **kwargs):
            boxes = Boxes(
                torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                torch.tensor([0.9]),
                torch.tensor([0]),
            )
            return Results(
                boxes=boxes,
                orig_shape=(10, 20),
                path=source,
                names={0: "person"},
            )

    monkeypatch.setattr(
        "libreyolo.cli.commands.predict.resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        "libreyolo.cli.commands.predict.load_model_or_exit",
        lambda out, model, model_path, device: _BackendLike(),
    )

    result = runner.invoke(
        app,
        [
            "predict",
            "source=libreyolo/assets/parkour.jpg",
            "model=model.onnx",
            "imgsz=320",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "yolox"
    assert data["image_size"] == [320, 320]
    assert data["results"][0]["detections"][0]["class"] == "person"


def test_predict_exported_backend_does_not_receive_native_only_kwargs(monkeypatch):
    app = _make_app([("predict", predict.predict_cmd), ("info", special.info_cmd)])

    class _StrictBackendLike:
        model_family = "yolox"
        imgsz = 640
        device = "cpu"

        def __call__(
            self,
            source,
            *,
            conf=0.25,
            iou=0.45,
            imgsz=None,
            classes=None,
            max_det=300,
            save=False,
            batch=1,
            output_path=None,
            color_format="auto",
        ):
            assert conf == 0.25
            assert iou == 0.45
            assert imgsz is None
            assert classes is None
            assert max_det == 300
            assert save is False
            assert batch == 1
            assert output_path is None
            assert color_format == "auto"
            boxes = Boxes(
                torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                torch.tensor([0.9]),
                torch.tensor([0]),
            )
            return Results(
                boxes=boxes,
                orig_shape=(10, 20),
                path=source,
                names={0: "person"},
            )

    monkeypatch.setattr(
        "libreyolo.cli.commands.predict.resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        "libreyolo.cli.commands.predict.load_model_or_exit",
        lambda out, model, model_path, device: _StrictBackendLike(),
    )

    result = runner.invoke(
        app,
        [
            "predict",
            "source=libreyolo/assets/parkour.jpg",
            "model=model.onnx",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "yolox"
    assert data["results"][0]["detections"][0]["class"] == "person"


@pytest.mark.parametrize(
    ("option", "name"),
    [
        ("tiling=true", "tiling"),
        ("overlap_ratio=0.3", "overlap_ratio"),
        ("output_file_format=png", "output_file_format"),
    ],
)
def test_predict_exported_backend_rejects_requested_native_only_kwargs(
    monkeypatch, option, name
):
    app = _make_app([("predict", predict.predict_cmd), ("info", special.info_cmd)])

    class _StrictBackendLike:
        model_family = "yolox"
        imgsz = 640
        device = "cpu"

        def __call__(
            self,
            source,
            *,
            conf=0.25,
            iou=0.45,
            imgsz=None,
            classes=None,
            max_det=300,
            save=False,
            batch=1,
            output_path=None,
            color_format="auto",
        ):
            raise AssertionError("should fail before backend inference")

    monkeypatch.setattr(
        "libreyolo.cli.commands.predict.resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        "libreyolo.cli.commands.predict.load_model_or_exit",
        lambda out, model, model_path, device: _StrictBackendLike(),
    )

    result = runner.invoke(
        app,
        [
            "predict",
            "source=libreyolo/assets/parkour.jpg",
            "model=model.onnx",
            option,
            "--json",
        ],
    )

    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert data["error"] == "config_unsupported"
    assert name in data["message"]


# ---------------------------------------------------------------------------
# --help-json schema
# ---------------------------------------------------------------------------


def _flags_for(command: str) -> list[str]:
    app = _make_app(
        [
            ("predict", predict.predict_cmd),
            ("train", train.train_cmd),
            ("val", val.val_cmd),
            ("export", export.export_cmd),
        ]
    )
    result = runner.invoke(app, [command, "--help-json"])
    assert result.exit_code == 0
    return json.loads(result.stdout)["flags"]


def test_predict_help_json_only_lists_predict_flags():
    flags = _flags_for("predict")
    assert "--json" in flags
    assert "--quiet" in flags
    assert "--verbose" in flags
    assert "--dry-run" not in flags
    assert "--yes" not in flags


def test_train_help_json_lists_dry_run_but_not_yes():
    flags = _flags_for("train")
    assert "--dry-run" in flags
    assert "--json" in flags
    assert "--quiet" in flags
    assert "--yes" not in flags


def test_val_help_json_only_lists_val_flags():
    flags = _flags_for("val")
    assert "--json" in flags
    assert "--quiet" in flags
    assert "--dry-run" not in flags
    assert "--yes" not in flags


def test_export_help_json_only_lists_export_flags():
    flags = _flags_for("export")
    assert "--json" in flags
    assert "--quiet" in flags
    assert "--dry-run" not in flags
    assert "--yes" not in flags
