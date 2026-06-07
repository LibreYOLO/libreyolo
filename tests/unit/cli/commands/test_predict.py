"""Behavior tests for the predict command."""

import json

import pytest
import torch
import typer
from PIL import Image
from typer.testing import CliRunner

from libreyolo.cli.commands import predict as predict_module
from libreyolo.cli.commands.predict import predict_cmd
from libreyolo.cli.parsing import KeyValueCommand
from libreyolo.utils.results import Points, Probs, Results

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
