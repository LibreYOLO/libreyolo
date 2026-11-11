"""Ultralytics names typed into LibreYOLO get the LibreYOLO spelling back.

The hints are error text only: no Ultralytics name becomes an alias, and no
Ultralytics checkpoint is loaded or converted.
"""

from __future__ import annotations

import json
import re
import sys
import types

import pytest
import torch
import typer
from typer.testing import CliRunner

import libreyolo
from libreyolo.utils.predict_args import normalize_predict_kwargs
from libreyolo.utils.ultralytics_hints import (
    UltralyticsNameError,
    is_ultralytics_checkpoint,
    ultralytics_weight_hint,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# from libreyolo import YOLO
# ---------------------------------------------------------------------------


def test_from_import_yolo_names_libreyolo():
    with pytest.raises(ImportError, match=r'LibreYOLO\("LibreYOLO9s.pt"\)') as info:
        exec("from libreyolo import YOLO", {})
    # ImportError, not the AttributeError that "from x import y" swallows.
    assert not isinstance(info.value, AttributeError)
    assert info.value.name == "libreyolo"


def test_from_import_inside_a_function_names_libreyolo():
    def load():
        from libreyolo import YOLO  # noqa: F401

    with pytest.raises(ImportError, match="from libreyolo import LibreYOLO"):
        load()


@pytest.mark.parametrize(
    "name, expected",
    [
        ("YOLO", "LibreYOLO9s.pt"),
        ("RTDETR", "LibreRTDETRl.pt"),
        ("NAS", "LibreYOLONASs.pt"),
        ("YOLOWorld", "LibreOpenVocab"),
        ("YOLOE", "LibreOpenVocab"),
        ("SAM", "LibreSAM"),
    ],
)
def test_attribute_access_raises_attribute_error_with_hint(name, expected):
    with pytest.raises(UltralyticsNameError, match=expected) as info:
        getattr(libreyolo, name)
    assert isinstance(info.value, AttributeError)


@pytest.mark.parametrize("name", ["YOLO", "RTDETR", "YOLOWorld", "SAM"])
def test_hasattr_and_getattr_default_still_work(name):
    assert hasattr(libreyolo, name) is False
    assert getattr(libreyolo, name, "default") == "default"
    assert name not in libreyolo.__all__


def test_unknown_names_keep_the_plain_attribute_error():
    with pytest.raises(AttributeError) as info:
        libreyolo.DefinitelyNotAName  # noqa: B018
    assert not isinstance(info.value, UltralyticsNameError)
    with pytest.raises(ImportError, match="cannot import name"):
        exec("from libreyolo import DefinitelyNotAName", {})


# ---------------------------------------------------------------------------
# Weight names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, python, cli",
    [
        ("yolov8n.pt", 'LibreYOLO("LibreYOLO9t.pt")', "yolo9-t"),
        ("yolov5su.pt", 'LibreYOLO("LibreYOLO9s.pt")', "yolo9-s"),
        ("yolov5n6u.pt", 'LibreYOLO("LibreYOLO9t.pt")', "yolo9-t"),
        ("yolov3u.pt", 'LibreYOLO("LibreYOLO9s.pt")', "yolo9-s"),
        ("yolov3-tinyu.pt", 'LibreYOLO("LibreYOLO9t.pt")', "yolo9-t"),
        ("yolov9c.pt", 'LibreYOLO("LibreYOLO9c.pt")', "yolo9-c"),
        ("yolov9e.pt", 'LibreYOLO("LibreYOLO9c.pt")', "yolo9-c"),
        ("yolov10b.pt", 'LibreYOLO("LibreYOLO9c.pt")', "yolo9-c"),
        ("yolo11m.pt", 'LibreYOLO("LibreYOLO9m.pt")', "yolo9-m"),
        ("yolov11m.pt", 'LibreYOLO("LibreYOLO9m.pt")', "yolo9-m"),
        ("yolo12l.pt", 'LibreYOLO("LibreYOLO9c.pt")', "yolo9-c"),
        ("YOLO26x.pt", 'LibreYOLO("LibreYOLO9c.pt")', "yolo9-c"),
        ("yolov8n", 'LibreYOLO("LibreYOLO9t.pt")', "yolo9-t"),
        ("yolov8.pt", 'LibreYOLO("LibreYOLO9s.pt")', "yolo9-s"),
        ("yolov8-seg.pt", 'LibreYOLO("LibreDFINEs-seg.pt")', "dfine-s-seg"),
        ("weights/yolov8s.pt", 'LibreYOLO("LibreYOLO9s.pt")', "yolo9-s"),
        ("yolov8n-seg.pt", 'LibreYOLO("LibreDFINEn-seg.pt")', "dfine-n-seg"),
        ("yolov9e-seg.pt", 'LibreYOLO("LibreDFINEx-seg.pt")', "dfine-x-seg"),
        ("yolo26n-pose.pt", 'LibreYOLO("LibreGTRs-pose.pt")', "gtr-s-pose"),
        ("yolov8x-pose-p6.pt", 'LibreYOLO("LibreGTRx-pose.pt")', "gtr-x-pose"),
        ("yolo11m-obb.pt", 'LibreYOLO("LibreGTRs-obb.pt")', "gtr-s-obb"),
        ("yolo11l-obb.pt", 'LibreYOLO("LibreGTRx-obb.pt")', "gtr-x-obb"),
        (
            "yolo11n-cls.pt",
            'LibreYOLO("LibreMobileNetV4s-cls.pt")',
            "mobilenetv4-s-cls",
        ),
        (
            "yolov8x-cls.pt",
            'LibreYOLO("LibreMobileNetV4l-cls.pt")',
            "mobilenetv4-l-cls",
        ),
        ("yolo_nas_m.pt", 'LibreYOLO("LibreYOLONASm.pt")', "yolonas-m"),
    ],
)
def test_weight_name_mapping(name, python, cli):
    hint = ultralytics_weight_hint(name)
    assert hint is not None
    assert hint.python == python
    assert hint.cli == cli
    assert hint.ultralytics is True


@pytest.mark.parametrize(
    "name", ["yolov8s-worldv2.pt", "yolov8l-world.pt", "yoloe-11s-seg.pt"]
)
def test_open_vocabulary_names_point_at_libreopenvocab(name):
    hint = ultralytics_weight_hint(name)
    assert 'LibreOpenVocab("owlv2")' in hint.python
    assert hint.cli == ""


def test_model_yaml_means_untrained():
    hint = ultralytics_weight_hint("yolov8n.yaml")
    assert 'LibreYOLO9(None, size="t")' in hint.python
    assert hint.cli == "yolo9-t pretrained=False"


@pytest.mark.parametrize("name", ["yolov9-c.pt", "yolo9-t"])
def test_look_alikes_get_the_hint_without_the_ultralytics_label(name):
    hint = ultralytics_weight_hint(name)
    assert hint is not None
    assert hint.ultralytics is False


@pytest.mark.parametrize(
    "name",
    [
        "LibreYOLO9t.pt",
        "LibreYOLOXs.pt",
        "rtdetr-l.pt",  # already resolves to LibreRTDETRl.pt
        "yolox-s",
        "yolov7.pt",
        "yolo9_e2e-t",
        "best.pt",
        "foo.pt",
        "owner/repo",
    ],
)
def test_other_names_get_no_hint(name):
    assert ultralytics_weight_hint(name) is None


def _all_hints():
    names = [
        f"yolo{v}{s}{t}.pt"
        for v in ("v5", "v8", "v9", "v10", "11", "26")
        for s in "nsmblxtce"
        for t in ("", "-seg", "-pose", "-obb", "-cls")
    ]
    names.append("yolov8n.yaml")
    hints = {ultralytics_weight_hint(name) for name in names}
    return [h for h in hints if 'LibreYOLO("' in h.python]


def test_every_suggested_file_is_a_registered_libreyolo_weight():
    from libreyolo.models import BaseModel

    for hint in _all_hints():
        filename = re.search(r'LibreYOLO\("([^"]+)"\)', hint.python).group(1)
        cls = next(
            (c for c in BaseModel._registry if c.detect_size_from_filename(filename)),
            None,
        )
        assert cls is not None, filename
        assert cls.get_download_url(filename), filename


def test_every_suggested_cli_name_resolves():
    from libreyolo.cli.config import resolve_model_name

    for hint in _all_hints():
        cli_name = hint.cli.split()[0]
        assert resolve_model_name(cli_name) != cli_name, cli_name


def test_factory_names_the_libreyolo_model(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError) as info:
        libreyolo.LibreYOLO("yolov8n.pt")
    message = str(info.value)
    assert "Ultralytics model name" in message
    assert 'model = LibreYOLO("LibreYOLO9t.pt")' in message
    assert "unable to determine size" not in message


def test_factory_with_explicit_size_names_the_libreyolo_model(monkeypatch, tmp_path):
    from libreyolo import models

    def no_download(*args, **kwargs):
        raise RuntimeError("no URL")

    monkeypatch.setattr(models, "download_weights", no_download)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=r'LibreYOLO\("LibreYOLO9t.pt"\)'):
        models.LibreYOLO("yolov8n.pt", size="n")


def test_bare_unknown_name_gets_no_bogus_hub_hint(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="unable to determine size") as info:
        libreyolo.LibreYOLO("notamodel")
    assert "hf://" not in str(info.value)


# ---------------------------------------------------------------------------
# Ultralytics checkpoint files
# ---------------------------------------------------------------------------


@pytest.fixture
def ultralytics_checkpoint(tmp_path, monkeypatch):
    """A file pickled the way Ultralytics saves ``best.pt``, made without it."""

    class DetectionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 1)

    tasks = types.ModuleType("ultralytics.nn.tasks")
    DetectionModel.__module__ = tasks.__name__
    DetectionModel.__qualname__ = "DetectionModel"
    tasks.DetectionModel = DetectionModel
    for name in ("ultralytics", "ultralytics.nn"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, tasks.__name__, tasks)

    path = tmp_path / "best.pt"
    torch.save({"epoch": -1, "model": None, "ema": DetectionModel()}, path)
    for name in ("ultralytics.nn.tasks", "ultralytics.nn", "ultralytics"):
        monkeypatch.delitem(sys.modules, name)
    return path


def test_ultralytics_checkpoint_is_refused_with_a_libreyolo_start(
    ultralytics_checkpoint, monkeypatch
):
    from libreyolo.models import autoconvert

    def never(*args, **kwargs):
        raise AssertionError("an Ultralytics checkpoint reached the autoconverter")

    monkeypatch.setattr(autoconvert, "autoconvert_upstream_checkpoint", never)

    with pytest.raises(RuntimeError) as info:
        libreyolo.LibreYOLO(str(ultralytics_checkpoint))
    message = str(info.value)
    assert "is an Ultralytics checkpoint" in message
    assert 'model = LibreYOLO("LibreYOLO9s.pt")' in message
    assert "model.train(data=" in message
    # torch's own error advises weights_only=False; it must not be shown.
    assert info.value.__suppress_context__
    assert "weights_only" not in message


def test_checkpoint_detection_reads_the_pickle_without_unpickling(
    ultralytics_checkpoint, tmp_path
):
    assert is_ultralytics_checkpoint(str(ultralytics_checkpoint), RuntimeError("x"))

    plain = tmp_path / "plain.pt"
    torch.save({"conv.weight": torch.zeros(1)}, plain)
    assert not is_ultralytics_checkpoint(str(plain), RuntimeError("x"))
    assert not is_ultralytics_checkpoint(str(tmp_path / "missing.pt"), OSError("x"))


# ---------------------------------------------------------------------------
# predict() options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"project": "runs", "name": "exp"}, "output_path="),
        ({"tracker": "bytetrack.yaml"}, r"model\.track\("),
        ({"persist": True}, r"model\.track\("),
        ({"save_txt": True}, r"to_json\(\)"),
    ],
)
def test_predict_options_name_the_libreyolo_spelling(kwargs, expected):
    with pytest.raises(TypeError, match="Unsupported predict option") as info:
        normalize_predict_kwargs(kwargs)
    assert re.search(expected, str(info.value))
    # One hint line per distinct hint, even for project + name.
    assert str(info.value).count("output_path=") <= 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_app():
    from libreyolo.cli.commands import predict, special
    from libreyolo.cli.parsing import KeyValueCommand

    app = typer.Typer(add_completion=False, no_args_is_help=True)
    for name, cmd in (("predict", predict.predict_cmd), ("info", special.info_cmd)):
        app.command(name, cls=KeyValueCommand)(cmd)
    return app


@pytest.mark.parametrize(
    "model, suggestion",
    [
        ("yolov8n.pt", "model=yolo9-t"),
        ("yolo11s-seg.pt", "model=dfine-s-seg"),
        ("yolov8n.yaml", "model=yolo9-t pretrained=False"),
        ("yolov8s-world.pt", "LibreOpenVocab"),
    ],
)
def test_cli_names_the_libreyolo_model(model, suggestion):
    result = CliRunner().invoke(_cli_app(), ["info", f"model={model}", "--json"])

    assert result.exit_code == 4
    data = json.loads(result.stdout)
    assert data["error"] == "model_not_found"
    assert "Ultralytics" in data["message"]
    assert suggestion in data["suggestion"]
