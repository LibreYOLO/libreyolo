"""Behavior tests for the `compare` (face-verification) command."""

import json

import pytest
import typer
from PIL import Image
from typer.testing import CliRunner

from libreyolo.cli.commands import special as special_module
from libreyolo.cli.commands.special import compare_cmd
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer()
    app.command("compare", cls=KeyValueCommand)(compare_cmd)
    return app


class _FakeEmbedModel:
    task = "embed"
    family = "facerec"

    def __init__(self, sim: float):
        self._sim = sim
        self.face_detector = None

    def verify(self, a, b, *, threshold: float = 0.4, **_):
        return {
            "similarity": self._sim,
            "same_person": self._sim >= threshold,
            "threshold": threshold,
        }


def _patch(monkeypatch, fake):
    monkeypatch.setattr(special_module, "resolve_model_or_exit", lambda out, m: m)
    monkeypatch.setattr(
        special_module,
        "load_model_or_exit",
        lambda out, *, model, model_path, device, task=None: fake,
    )


def _imgs(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    Image.new("RGB", (20, 20)).save(a)
    Image.new("RGB", (20, 20)).save(b)
    det = tmp_path / "facedet.onnx"        # lazy OpenCVFaceDetector — never loaded here
    det.write_bytes(b"stub")
    return a, b, det


def test_compare_same_person(monkeypatch, tmp_path):
    a, b, det = _imgs(tmp_path)
    _patch(monkeypatch, _FakeEmbedModel(sim=0.82))
    result = runner.invoke(
        _make_app(),
        [f"model={a}", f"source={a}", f"source2={b}",
         "--face-detector", str(det), "--json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["same_person"] is True
    assert data["similarity"] == pytest.approx(0.82, abs=1e-4)


def test_compare_different_people(monkeypatch, tmp_path):
    a, b, det = _imgs(tmp_path)
    _patch(monkeypatch, _FakeEmbedModel(sim=0.05))
    result = runner.invoke(
        _make_app(),
        [f"model={a}", f"source={a}", f"source2={b}",
         "--face-detector", str(det), "--json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["same_person"] is False


def test_compare_requires_face_detector(monkeypatch, tmp_path):
    a, b, _ = _imgs(tmp_path)
    _patch(monkeypatch, _FakeEmbedModel(sim=0.82))
    result = runner.invoke(
        _make_app(),
        [f"model={a}", f"source={a}", f"source2={b}", "--json"],
    )
    assert result.exit_code != 0


def test_compare_missing_source(monkeypatch, tmp_path):
    a, _, det = _imgs(tmp_path)
    _patch(monkeypatch, _FakeEmbedModel(sim=0.82))
    result = runner.invoke(
        _make_app(),
        [f"model={a}", f"source={a}",
         f"source2={tmp_path / 'missing.jpg'}", "--face-detector", str(det), "--json"],
    )
    assert result.exit_code != 0
