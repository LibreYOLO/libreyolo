"""CLI behaviour for the optical flow task: pair sources, JSON summaries, val."""

from __future__ import annotations

import json

import pytest
import torch
import typer
from PIL import Image
from typer.testing import CliRunner

from libreyolo.cli.commands import predict as predict_module
from libreyolo.cli.commands.predict import _split_source_list, predict_cmd
from libreyolo.cli.commands.val import val_cmd
from libreyolo.cli.parsing import KeyValueCommand
from libreyolo.utils.results import FlowField, Results

pytestmark = pytest.mark.unit

runner = CliRunner()


def _predict_app() -> typer.Typer:
    app = typer.Typer()
    app.command("predict", cls=KeyValueCommand)(predict_cmd)
    return app


def _val_app() -> typer.Typer:
    app = typer.Typer()
    app.command("val", cls=KeyValueCommand)(val_cmd)
    return app


class _FakeFlowModel:
    FAMILY = "searaft"
    task = "flow"
    size = "m"
    device = "cpu"

    def __init__(self):
        self.calls = []

    def _get_input_size(self) -> int:
        return 32

    def __call__(self, source, **kwargs):
        self.calls.append((source, kwargs))
        frames = list(source)
        results = []
        for frame in frames[:-1]:
            field = torch.zeros(10, 12, 2)
            field[..., 0] = 3.0
            field[..., 1] = 4.0
            results.append(
                Results(
                    boxes=None,
                    orig_shape=(10, 12),
                    path=str(frame),
                    names={0: "flow"},
                    flow=FlowField(field, (10, 12)),
                )
            )
        return results


def _write_frames(tmp_path, n=2):
    paths = []
    for i in range(n):
        path = tmp_path / f"frame_{i}.png"
        Image.new("RGB", (12, 10), color=(10 * i, 10 * i, 10 * i)).save(path)
        paths.append(path)
    return paths


def test_split_source_list_splits_existing_files(tmp_path):
    a, b = _write_frames(tmp_path)
    assert _split_source_list(f"{a},{b}") == [str(a), str(b)]
    assert _split_source_list(f"{a}, {b}") == [str(a), str(b)]


def test_split_source_list_keeps_other_strings(tmp_path):
    (a,) = _write_frames(tmp_path, 1)
    assert _split_source_list(str(a)) == str(a)
    assert _split_source_list(f"{a},{tmp_path / 'missing.png'}") == f"{a},{tmp_path / 'missing.png'}"
    assert _split_source_list("rtsp://cam/1,2") == "rtsp://cam/1,2"
    assert _split_source_list(0) == 0
    comma_path = tmp_path / "a,b.png"
    Image.new("RGB", (4, 4)).save(comma_path)
    assert _split_source_list(str(comma_path)) == str(comma_path)


@pytest.mark.parametrize("grammar", ["key=value", "--flag"])
def test_predict_pair_source_reports_flow_summary(monkeypatch, tmp_path, grammar):
    a, b = _write_frames(tmp_path)
    fake_model = _FakeFlowModel()
    monkeypatch.setattr(predict_module, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(predict_module, "load_model_or_exit", lambda *args, **kwargs: fake_model)

    source = f"{a},{b}"
    args = (
        [f"source={source}", "model=LibreSEARAFTm-flow.pt", "--json"]
        if grammar == "key=value"
        else ["--source", source, "--model", "LibreSEARAFTm-flow.pt", "--json"]
    )
    result = runner.invoke(_predict_app(), args)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert fake_model.calls[0][0] == [str(a), str(b)]
    flow = payload["results"][0]["flow"]
    assert flow["shape"] == [10, 12, 2]
    assert flow["mean_magnitude"] == pytest.approx(5.0)
    assert flow["max_magnitude"] == pytest.approx(5.0)
    assert flow["has_confidence"] is False
    assert len(payload["results"]) == 1


@pytest.mark.parametrize("grammar", ["key=value", "--flag"])
def test_predict_flow_view_flags_reach_predict(monkeypatch, tmp_path, grammar):
    a, b = _write_frames(tmp_path)
    fake_model = _FakeFlowModel()
    monkeypatch.setattr(predict_module, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(predict_module, "load_model_or_exit", lambda *args, **kwargs: fake_model)
    source = f"{a},{b}"
    base = (
        [f"source={source}", "model=LibreSEARAFTm-flow.pt", f"output_path={tmp_path}"]
        if grammar == "key=value"
        else ["--source", source, "--model", "LibreSEARAFTm-flow.pt", "--output-path", str(tmp_path)]
    )
    flags = (
        ["save=True", "save_flow=True", "arrows=True", "warp=True"]
        if grammar == "key=value"
        else ["--save", "--save-flow", "--arrows", "--warp"]
    )
    result = runner.invoke(_predict_app(), base + flags + ["--json"])
    assert result.exit_code == 0, result.output
    kwargs = fake_model.calls[0][1]
    assert kwargs["save"] is True
    assert kwargs["save_flow"] is True and kwargs["arrows"] is True and kwargs["warp"] is True
    # Unset flags are not sent at all, so non-flow predict paths never see them.
    fake_model.calls.clear()
    result = runner.invoke(_predict_app(), base + ["--json"])
    assert result.exit_code == 0, result.output
    assert not {"save_flow", "arrows", "warp"} & set(fake_model.calls[0][1])


def test_predict_view_flags_need_save_or_show(monkeypatch, tmp_path):
    a, b = _write_frames(tmp_path)
    fake_model = _FakeFlowModel()
    monkeypatch.setattr(predict_module, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(predict_module, "load_model_or_exit", lambda *args, **kwargs: fake_model)
    result = runner.invoke(
        _predict_app(), [f"source={a},{b}", "model=LibreSEARAFTm-flow.pt", "--arrows", "--json"]
    )
    assert result.exit_code != 0
    assert "config_unsupported" in result.output
    assert fake_model.calls == []


def test_predict_flow_flags_rejected_for_other_tasks(monkeypatch, tmp_path):
    (a,) = _write_frames(tmp_path, 1)

    class _Detector(_FakeFlowModel):
        task = "detect"

    fake_model = _Detector()
    monkeypatch.setattr(predict_module, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(predict_module, "load_model_or_exit", lambda *args, **kwargs: fake_model)
    result = runner.invoke(
        _predict_app(), [f"source={a}", "model=LibreYOLO9t.pt", "--save", "--save-flow", "--json"]
    )
    assert result.exit_code != 0
    assert "config_unsupported" in result.output
    assert "--save-flow" in result.output
    assert fake_model.calls == []


def test_predict_human_summary_mentions_flow(monkeypatch, tmp_path):
    a, b = _write_frames(tmp_path)
    fake_model = _FakeFlowModel()
    monkeypatch.setattr(predict_module, "resolve_model_or_exit", lambda out, model: model)
    monkeypatch.setattr(predict_module, "load_model_or_exit", lambda *args, **kwargs: fake_model)

    result = runner.invoke(_predict_app(), [f"source={a},{b}", "model=LibreSEARAFTm-flow.pt"])

    assert result.exit_code == 0, result.output
    assert "flow mean=5px max=5px" in result.output


def test_val_reports_flow_metrics(monkeypatch, tmp_path):
    captured = {}

    class _Model:
        FAMILY = "searaft"
        task = "flow"
        size = "m"
        device = "cpu"

        def val(self, **kwargs):
            captured["kwargs"] = kwargs
            return {
                "metrics/epe": 1.23456,
                "metrics/px1": 0.5,
                "metrics/px3": 0.75,
                "metrics/fl_all": 0.125,
                "fitness": 0.5,
            }

    monkeypatch.setattr("libreyolo.cli.commands.val.load_model_or_exit", lambda *a, **k: _Model())
    result = runner.invoke(
        _val_app(),
        ["data=flow8.yaml", "model=LibreYOLO9t.pt", f"project={tmp_path}", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["metrics"] == {"epe": 1.2346, "px1": 0.5, "px3": 0.75, "fl_all": 0.125}
    assert captured["kwargs"]["data"] == "flow8.yaml"

    human = runner.invoke(
        _val_app(), ["data=flow8.yaml", "model=LibreYOLO9t.pt", f"project={tmp_path}"]
    )
    assert human.exit_code == 0, human.output
    assert "EPE: 1.2346 px" in human.output and "Fl-all: 0.1250" in human.output
