"""Behavior tests for the predict command."""

import json

import pytest
import torch
import typer
from PIL import Image, UnidentifiedImageError
from typer.testing import CliRunner

from libreyolo.cli.commands import predict as predict_module
from libreyolo.cli.commands.predict import predict_cmd
from libreyolo.cli.parsing import KeyValueCommand
from libreyolo.utils.results import Points, Probs, RestoredImage, Results

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer()
    app.command("predict", cls=KeyValueCommand)(predict_cmd)
    return app


class _FakeClassifyModel:
    FAMILY = "yolo9"
    task = "classify"
    size = "t"
    device = "cpu"

    def _get_input_size(self) -> int:
        return 224

    def __call__(self, source, **kwargs):
        return Results(
            boxes=None,
            orig_shape=(10, 12),
            path=str(source),
            names={0: "cat", 1: "dog"},
            probs=Probs(torch.tensor([0.2, 0.8])),
        )


class _FakePointModel:
    FAMILY = "librefomo"
    task = "point"
    size = "s"
    device = "cpu"

    def _get_input_size(self) -> int:
        return 96

    def __call__(self, source, **kwargs):
        return Results(
            boxes=None,
            orig_shape=(10, 12),
            path=str(source),
            names={0: "person"},
            points=Points(torch.tensor([[6.0, 5.0, 0.0, 0.9]])),
        )


class _FakeRestoreModel:
    FAMILY = "nafnet"
    task = "restore"
    size = "s"
    device = "cpu"

    def _get_input_size(self) -> int:
        return 256

    def __call__(self, source, **kwargs):
        del kwargs
        return Results(
            boxes=None,
            orig_shape=(10, 12),
            path=str(source),
            names={0: "image"},
            restored=RestoredImage(torch.zeros((10, 12, 3), dtype=torch.uint8)),
        )


class _FakeGuidedRestoreModel(_FakeRestoreModel):
    def __init__(self):
        self.calls = []

    def __call__(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return super().__call__(source, **kwargs)


class _FakeStreamModel:
    FAMILY = "yolo9"
    task = "classify"
    size = "t"
    device = "cpu"

    def __init__(self):
        self.calls = []

    def _get_input_size(self) -> int:
        return 224

    def __call__(self, source, **kwargs):
        self.calls.append((source, kwargs))

        def generate():
            for frame_idx in range(2):
                yield Results(
                    boxes=None,
                    orig_shape=(10, 12),
                    path=str(source),
                    names={0: "cat", 1: "dog"},
                    probs=Probs(torch.tensor([0.2, 0.8])),
                    frame_idx=frame_idx,
                )

        return generate()


def test_predict_formats_classification_probs(monkeypatch, tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    fake_model = _FakeClassifyModel()

    monkeypatch.setattr(
        predict_module,
        "resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=fake-cls.pt",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    item = data["results"][0]
    assert item["detections"] == []
    assert item["classification"]["name"] == "dog"
    assert item["classification"]["class"] == 1
    assert item["top5"][0]["name"] == "dog"


def test_predict_formats_point_results(monkeypatch, tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    fake_model = _FakePointModel()

    monkeypatch.setattr(
        predict_module,
        "resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=fake-point.pt",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    det = data["results"][0]["detections"][0]
    assert det["class"] == "person"
    assert det["class_id"] == 0
    assert det["confidence"] == 0.9
    assert det["point_xy"] == [6.0, 5.0]


def test_predict_formats_restore_results(monkeypatch, tmp_path):
    source = tmp_path / "image.jpg"
    Image.new("RGB", (12, 10)).save(source)
    fake_model = _FakeRestoreModel()

    monkeypatch.setattr(
        predict_module,
        "resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    json_result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=fake-restore.pt",
            "--json",
        ],
    )

    assert json_result.exit_code == 0
    data = json.loads(json_result.stdout)
    item = data["results"][0]
    assert item["detections"] == []
    assert item["restored"] == {"shape": [10, 12, 3], "dtype": "uint8"}

    human_result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=fake-restore.pt",
        ],
    )

    assert human_result.exit_code == 0
    assert "restored" in human_result.stdout


@pytest.mark.parametrize("guide_name", ["mask", "trimap"])
def test_predict_forwards_guided_image_option(monkeypatch, tmp_path, guide_name):
    source = tmp_path / "image.jpg"
    guide = tmp_path / f"{guide_name}.png"
    Image.new("RGB", (12, 10)).save(source)
    Image.new("L", (12, 10)).save(guide)
    fake_model = _FakeGuidedRestoreModel()

    monkeypatch.setattr(
        predict_module,
        "resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    result = runner.invoke(
        _make_app(),
        [
            f"source={source}",
            "model=fake-guided.pt",
            f"{guide_name}={guide}",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert fake_model.calls[0][1][guide_name] == str(guide)


def test_predict_webcam_source_auto_streams_as_ndjson(monkeypatch):
    fake_model = _FakeStreamModel()
    monkeypatch.setattr(
        predict_module,
        "resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    result = runner.invoke(
        _make_app(),
        [
            "source=0",
            "model=fake-stream.pt",
            "stream_buffer=true",
            "vid_stride=2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["frame_index"] for row in rows] == [0, 1]
    assert all(row["results"][0]["predictions"][0]["name"] == "dog" for row in rows)
    source, kwargs = fake_model.calls[0]
    assert source == 0
    assert kwargs["stream"] is True
    assert kwargs["stream_buffer"] is True
    assert kwargs["vid_stride"] == 2


def test_predict_rtsp_source_bypasses_local_path_validation(monkeypatch):
    fake_model = _FakeStreamModel()
    monkeypatch.setattr(
        predict_module,
        "resolve_model_or_exit",
        lambda out, model: model,
    )
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: fake_model,
    )

    source = "rtsp://127.0.0.1:8554/camera"
    result = runner.invoke(
        _make_app(),
        [f"source={source}", "model=fake-stream.pt", "--json"],
    )

    assert result.exit_code == 0
    assert fake_model.calls[0][0] == source


class _FailingModel:
    FAMILY = "yolo9"
    task = "detect"
    size = "t"
    device = "cpu"

    def __init__(self, exc, *, lazy=False):
        self.exc = exc
        self.lazy = lazy

    def _get_input_size(self) -> int:
        return 640

    def __call__(self, source, **kwargs):
        if not self.lazy:
            raise self.exc

        def generate():
            raise self.exc
            yield  # pragma: no cover

        return generate()


@pytest.mark.parametrize(
    ("source", "extra", "exc", "lazy", "code", "exit_code"),
    [
        (
            "image",
            ["imgsz=150"],
            RuntimeError("Sizes of tensors must match except in dimension 1."),
            False,
            "invalid_imgsz",
            2,
        ),
        (
            "image",
            [],
            UnidentifiedImageError("cannot identify image file"),
            False,
            "data_invalid",
            3,
        ),
        (
            "https://example.com/missing.jpg",
            [],
            ValueError("Failed to load image from URL 'x': 404 Client Error"),
            False,
            "source_not_found",
            3,
        ),
        ("7", [], ConnectionError("Cannot open live stream: 7"), True, "source_not_found", 3),
        ("image", [], RuntimeError("boom"), False, "io_error", 1),
    ],
    ids=["imgsz", "corrupt", "url-404", "camera", "other"],
)
def test_predict_model_failures_emit_json_errors(
    monkeypatch, tmp_path, source, extra, exc, lazy, code, exit_code
):
    if source == "image":
        source = tmp_path / "image.jpg"
        Image.new("RGB", (12, 10)).save(source)
    monkeypatch.setattr(predict_module, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        predict_module,
        "load_model_or_exit",
        lambda *args, **kwargs: _FailingModel(exc, lazy=lazy),
    )

    result = runner.invoke(
        _make_app(),
        [f"source={source}", "model=LibreYOLO9t.pt", *extra, "--json"],
    )

    assert result.exit_code == exit_code, result.output
    assert json.loads(result.stdout)["error"] == code
