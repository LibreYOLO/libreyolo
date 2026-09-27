"""``val(project=..., name=...)`` set where validation outputs go.

Same names and meaning as the ecosystem's validation arguments: outputs land
in ``project/name``, with ``name`` incremented (``exp``, ``exp2``, ...) unless
``exist_ok=True``, as the CLI already did.
"""

from __future__ import annotations

import pytest

import libreyolo.validation as validation

pytestmark = pytest.mark.unit


class _RecordingValidator:
    configs = []

    def __init__(self, model, config):
        self.configs.append(config)

    def __call__(self):
        return {}


@pytest.fixture
def recorded(monkeypatch):
    _RecordingValidator.configs = []
    monkeypatch.setattr(validation, "DetectionValidator", _RecordingValidator)
    return _RecordingValidator.configs


def _model():
    from libreyolo import LibreYOLO9

    return LibreYOLO9(None, size="t", nb_classes=2, device="cpu")


def _backend():
    from libreyolo.backends.onnx import OnnxBackend

    backend = OnnxBackend.__new__(OnnxBackend)
    backend.imgsz = 64
    backend.model_family = "yolo9"
    backend.task = "detect"
    backend.device = "cpu"
    return backend


@pytest.mark.parametrize("make", [_model, _backend], ids=["pytorch", "backend"])
def test_project_and_name_set_the_output_directory(tmp_path, recorded, make):
    model = make()
    project = tmp_path / "evals"

    model.val(data="data.yaml", imgsz=64, project=str(project), name="run")
    (project / "run").mkdir(parents=True)
    model.val(data="data.yaml", imgsz=64, project=str(project), name="run")
    model.val(
        data="data.yaml", imgsz=64, project=str(project), name="run", exist_ok=True
    )

    assert [c.save_dir for c in recorded] == [
        str(project / "run"),
        str(project / "run2"),
        str(project / "run"),
    ]


def test_project_or_name_alone_uses_the_cli_defaults(tmp_path, recorded, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _model()

    model.val(data="data.yaml", imgsz=64, project="evals")
    model.val(data="data.yaml", imgsz=64, name="mine")

    assert [c.save_dir for c in recorded] == ["evals/exp", "runs/val/mine"]


def test_without_project_or_name_the_default_directory_is_unchanged(recorded):
    _model().val(data="data.yaml", imgsz=64)
    assert recorded[0].save_dir is None


def test_save_dir_and_project_name_cannot_both_be_given(tmp_path, recorded):
    with pytest.raises(ValueError, match="save_dir"):
        _model().val(
            data="data.yaml", imgsz=64, save_dir=str(tmp_path), name="run"
        )


def test_cli_val_uses_the_same_directory_rule(tmp_path, monkeypatch):
    import typer
    from typer.testing import CliRunner

    from libreyolo.cli.commands.val import val_cmd
    from libreyolo.cli.parsing import KeyValueCommand

    captured = {}

    class _Model:
        FAMILY = "yolo9"
        size = "t"
        device = "cpu"
        task = "detect"

        def val(self, **kwargs):
            captured.update(kwargs)
            return {"metrics/mAP50-95": 0.5, "metrics/mAP50": 0.6}

    monkeypatch.setattr(
        "libreyolo.cli.commands.val.load_model_or_exit", lambda *a, **k: _Model()
    )
    app = typer.Typer()
    app.command("val", cls=KeyValueCommand)(val_cmd)
    project = tmp_path / "evals"
    (project / "run").mkdir(parents=True)

    result = CliRunner().invoke(
        app,
        ["data=coco8.yaml", "model=LibreYOLO9t.pt", f"project={project}", "name=run", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured["save_dir"] == str(project / "run2")
