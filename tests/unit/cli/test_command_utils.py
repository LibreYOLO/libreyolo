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
from libreyolo.utils.results import Boxes, Masks, Results
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

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


def test_metadata_command_inspects_checkpoint_without_loading_model(tmp_path):
    app = _make_app([("metadata", special.metadata_cmd), ("info", special.info_cmd)])
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            {"layer.weight": torch.ones(1)},
            model_family="yolo9",
            size="t",
            task="detect",
            nc=1,
            names={0: "object"},
            imgsz=640,
        ),
        checkpoint_path,
    )

    result = runner.invoke(
        app,
        ["metadata", f"path={checkpoint_path}", "--json"],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["valid"] is True
    assert data["metadata"]["schema_version"] == "1.0"
    assert data["metadata"]["task"] == "detect"


def test_metadata_command_shows_training_fields(tmp_path):
    app = _make_app([("metadata", special.metadata_cmd), ("info", special.info_cmd)])
    checkpoint_path = tmp_path / "last.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            {"layer.weight": torch.ones(1)},
            model_family="yolo9",
            size="t",
            task="detect",
            nc=1,
            names={0: "object"},
            imgsz=640,
            epoch=3,
            best_metric_value=0.5,
            is_ema_weights=False,
            train_model={"layer.weight": torch.ones(1)},
        ),
        checkpoint_path,
    )

    result = runner.invoke(
        app,
        ["metadata", f"path={checkpoint_path}", "--json"],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["metadata"]["epoch"] == 3
    assert data["metadata"]["best_metric_value"] == 0.5
    assert data["metadata"]["is_ema_weights"] is False
    assert data["metadata"]["train_model"] == {"type": "dict", "keys": 1}


def test_metadata_command_exits_nonzero_for_invalid_checkpoint(tmp_path):
    app = _make_app([("metadata", special.metadata_cmd), ("info", special.info_cmd)])
    checkpoint_path = tmp_path / "bad.pt"
    torch.save({"model": {"layer.weight": torch.ones(1)}}, checkpoint_path)

    result = runner.invoke(
        app,
        ["metadata", f"path={checkpoint_path}", "--json"],
    )

    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["valid"] is False
    assert "missing required key: schema_version" in data["errors"]


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


def test_val_json_reports_segmentation_metric_groups(monkeypatch):
    app = _make_app([("val", val.val_cmd), ("info", special.info_cmd)])

    class _SegModel:
        FAMILY = "rfdetr"
        size = "n"
        device = "cpu"

        def val(self, **kwargs):
            assert "use_coco_eval" not in kwargs
            return {
                "metrics/mAP50": 0.7,
                "metrics/mAP50-95": 0.6,
                "metrics/mAP50(B)": 0.55,
                "metrics/mAP50-95(B)": 0.45,
                "metrics/precision(M)": 0.82,
                "metrics/recall(M)": 0.52,
                "metrics/mAP50(M)": 0.72,
                "metrics/mAP50-95(M)": 0.62,
            }

    monkeypatch.setattr(
        "libreyolo.cli.commands.val.resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        "libreyolo.cli.commands.val.load_model_or_exit",
        lambda out, model, model_path, device: _SegModel(),
    )
    monkeypatch.setattr(
        "libreyolo.utils.general.increment_path",
        lambda path, exist_ok=False, mkdir=False: Path(path),
    )

    result = runner.invoke(
        app,
        [
            "val",
            "data=fire-smoke.yaml",
            "model=LibreRFDETRn-seg.pt",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["metrics"]["mAP50_95"] == 0.6
    assert data["metrics"]["precision"] == 0.82
    assert data["metrics"]["recall"] == 0.52
    assert data["box_metrics"] == {
        "mAP50": 0.55,
        "mAP50_95": 0.45,
    }
    assert data["mask_metrics"] == {
        "mAP50": 0.72,
        "mAP50_95": 0.62,
        "precision": 0.82,
        "recall": 0.52,
    }


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


def test_predict_json_reports_segmentation_masks(monkeypatch):
    app = _make_app([("predict", predict.predict_cmd), ("info", special.info_cmd)])

    class _SegBackendLike:
        model_family = "rfdetr"
        imgsz = 312
        device = "cpu"

        def __call__(self, source, **kwargs):
            boxes = Boxes(
                torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                torch.tensor([0.9]),
                torch.tensor([0]),
            )
            masks = Masks(torch.ones(1, 10, 20, dtype=torch.bool), (10, 20))
            return Results(
                boxes=boxes,
                masks=masks,
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
        lambda out, model, model_path, device: _SegBackendLike(),
    )

    result = runner.invoke(
        app,
        [
            "predict",
            "source=libreyolo/assets/parkour.jpg",
            "model=rfdetr-seg.onnx",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_family"] == "rfdetr"
    assert data["results"][0]["masks"] == {
        "count": 1,
        "shape": [1, 10, 20],
    }


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
