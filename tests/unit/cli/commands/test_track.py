"""Behavior tests for the track command."""

import json
import sys

import pytest
import torch
import typer
from PIL import Image
from typer.testing import CliRunner

from libreyolo.cli.commands import track as track_module
from libreyolo.cli.commands.track import track_cmd
from libreyolo.cli.parsing import KeyValueCommand
from libreyolo.utils.results import Boxes, Results

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer()
    app.command("track", cls=KeyValueCommand)(track_cmd)
    return app


class _FakeTrackModel:
    FAMILY = "yolo9"
    task = "detect"
    size = "t"
    device = "cpu"

    def __init__(self):
        self.calls = []

    def _get_input_size(self) -> int:
        return 160

    def track(self, source, **kwargs):
        self.calls.append((source, kwargs))
        for frame_idx in range(2):
            result = Results(
                boxes=Boxes(
                    torch.tensor([[1.0, 2.0, 5.0, 6.0]]),
                    torch.tensor([0.9]),
                    torch.tensor([0.0]),
                ),
                orig_shape=(10, 12),
                path=f"{source}/{frame_idx:03d}.png",
                names={0: "person"},
                frame_idx=frame_idx,
            )
            result.track_id = torch.tensor([7])
            yield result


def _patch_loader(monkeypatch, model):
    monkeypatch.setattr(track_module, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(
        track_module, "load_model_or_exit", lambda *args, **kwargs: model
    )


def _frames_dir(tmp_path, n=2):
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(n):
        Image.new("RGB", (32, 32), (i * 40, 0, 0)).save(folder / f"{i:03d}.png")
    return folder


def test_track_is_registered_and_documents_its_options(monkeypatch, capsys):
    from libreyolo import cli

    monkeypatch.setattr(sys, "argv", ["libreyolo", "track", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.entrypoint()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--source", "--model", "--tracker", "--track-conf", "--json"):
        assert option in help_text


def test_track_help_json_lists_tracker_options():
    result = runner.invoke(_make_app(), ["--help-json"])
    assert result.exit_code == 0
    assert "track_conf" in result.stdout
    assert "tracker" in result.stdout


def test_track_emits_one_json_line_per_frame(monkeypatch, tmp_path):
    folder = _frames_dir(tmp_path)
    fake = _FakeTrackModel()
    _patch_loader(monkeypatch, fake)

    result = runner.invoke(
        _make_app(),
        [
            f"source={folder}",
            "model=fake.onnx",
            "tracker=botsort",
            "conf=0.2",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    lines = [json.loads(line) for line in result.stdout.splitlines() if line]
    assert [line["frame_index"] for line in lines] == [0, 1]
    detection = lines[0]["results"][0]["detections"][0]
    assert detection["track_id"] == 7
    assert detection["class"] == "person"
    assert lines[0]["tracker"] == "botsort"
    ((source, kwargs),) = fake.calls
    assert source == str(folder)
    assert kwargs["tracker"] == "botsort"
    assert kwargs["conf"] == 0.2


def test_track_rejects_unknown_tracker(monkeypatch, tmp_path):
    folder = _frames_dir(tmp_path)
    _patch_loader(monkeypatch, _FakeTrackModel())

    result = runner.invoke(_make_app(), [f"source={folder}", "tracker=sort", "--json"])

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"] == "config_type_error"


def test_track_missing_source_exits_with_data_error(monkeypatch, tmp_path):
    _patch_loader(monkeypatch, _FakeTrackModel())
    result = runner.invoke(
        _make_app(), [f"source={tmp_path / 'missing.mp4'}", "--json"]
    )
    assert result.exit_code == 3
    assert json.loads(result.stdout)["error"] == "source_not_found"


def test_track_cli_smoke_with_an_onnx_model(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from libreyolo import LibreYOLO9

    native = LibreYOLO9(model_path=None, size="t", device="cpu")
    model_path = native.export(
        format="onnx", imgsz=160, output_path=str(tmp_path / "tiny.onnx")
    )
    folder = _frames_dir(tmp_path, n=3)

    result = runner.invoke(
        _make_app(),
        [
            f"source={folder}",
            f"model={model_path}",
            "imgsz=160",
            "device=cpu",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    lines = [json.loads(line) for line in result.stdout.splitlines() if line]
    assert [line["frame_index"] for line in lines] == [0, 1, 2]
    assert all(line["model_family"] == "yolo9" for line in lines)
