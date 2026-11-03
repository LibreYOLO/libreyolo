"""Unit tests for the ExecuTorch exporter and artifact transaction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from libreyolo.export.executorch import (
    _capture_compatibility,
    _commit_artifact_pair,
)
from libreyolo.export.exporter import BaseExporter, ExecuTorchExporter

pytestmark = pytest.mark.unit


def _wrapper() -> MagicMock:
    wrapper = MagicMock()
    wrapper._get_model_name.return_value = "yolo9"
    wrapper._get_input_size.return_value = 64
    wrapper.task = "detect"
    wrapper.SUPPORTED_TASKS = ("detect",)
    wrapper.DEFAULT_TASK = "detect"
    wrapper.size = "t"
    wrapper.nb_classes = 2
    wrapper.names = {0: "a", 1: "b"}
    wrapper.device = torch.device("cpu")
    return wrapper


def test_executorch_constraints_reject_unsupported_requests(monkeypatch):
    exporter = ExecuTorchExporter(_wrapper())
    monkeypatch.setattr(
        "libreyolo.export.executorch.check_executorch_available", lambda: None
    )

    with pytest.raises(ValueError, match="xnnpack"):
        exporter._preflight(
            half=False, int8=False, data=None, delegate="vulkan"
        )
    with pytest.raises(ValueError, match="batch=1"):
        exporter._export(
            torch.nn.Identity(),
            torch.zeros(2, 3, 8, 8),
            output_path="unused.pte",
            metadata={},
            dynamic=False,
        )
    with pytest.raises(ValueError, match="dynamic=False"):
        exporter._export(
            torch.nn.Identity(),
            torch.zeros(1, 3, 8, 8),
            output_path="unused.pte",
            metadata={},
            dynamic=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch": 2, "dynamic": False}, "batch=1"),
        ({"batch": 1, "dynamic": True}, "dynamic=False"),
    ],
)
def test_executorch_shape_constraints_reject_before_base_export(
    monkeypatch, kwargs, message
):
    """Invalid requests must not reach BaseExporter's destructive LoRA merge."""
    base_call = MagicMock(side_effect=AssertionError("base export entered"))
    monkeypatch.setattr(BaseExporter, "__call__", base_call)

    with pytest.raises(ValueError, match=message):
        ExecuTorchExporter(_wrapper())(**kwargs)

    base_call.assert_not_called()


def test_executorch_metadata_is_fixed_fp32_contract():
    exporter = ExecuTorchExporter(_wrapper())
    metadata = exporter._build_metadata(
        "fp32", True, None, imgsz=(64, 64)
    )
    assert metadata["precision"] == "fp32"
    assert metadata["dynamic"] is False
    assert metadata["task"] == "detect"
    assert metadata["imgsz_h"] == metadata["imgsz_w"] == 64


def test_rtdetrv4_capture_compatibility_restores_module_flag():
    from libreyolo.models.dfine import ms_deform

    assert ms_deform._FORCE_MANUAL_GRID_SAMPLE_EXPORT is False
    with _capture_compatibility({"model_family": "rtdetrv4"}):
        assert ms_deform._FORCE_MANUAL_GRID_SAMPLE_EXPORT is True
    assert ms_deform._FORCE_MANUAL_GRID_SAMPLE_EXPORT is False


def test_artifact_pair_writes_program_and_sidecar(tmp_path):
    path = tmp_path / "model.pte"
    _commit_artifact_pair(b"program", {"task": "detect"}, path)

    assert path.read_bytes() == b"program"
    assert json.loads(Path(f"{path}.json").read_text(encoding="utf-8")) == {
        "task": "detect"
    }
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.backup"))


def test_artifact_pair_restores_preexisting_files_on_commit_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.pte"
    sidecar = Path(f"{path}.json")
    path.write_bytes(b"old-program")
    sidecar.write_text('{"old": true}', encoding="utf-8")

    real_replace = os.replace

    def fail_sidecar_commit(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.suffix == ".tmp" and destination_path == sidecar:
            raise OSError("simulated sidecar commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_sidecar_commit)
    with pytest.raises(OSError, match="simulated"):
        _commit_artifact_pair(b"new-program", {"old": False}, path)

    assert path.read_bytes() == b"old-program"
    assert sidecar.read_text(encoding="utf-8") == '{"old": true}'
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.backup"))


def test_flatc_from_an_unactivated_venv_is_handed_to_executorch(tmp_path, monkeypatch):
    # The executorch wheel installs flatc beside the venv's python, which is
    # not on PATH when the venv is not activated (IDE, Jupyter kernel).
    import libreyolo.export.executorch as executorch_export

    flatc = tmp_path / "flatc"
    flatc.write_text("")
    monkeypatch.delenv("FLATC_EXECUTABLE", raising=False)
    monkeypatch.setattr(executorch_export, "environment_script", lambda name: flatc)
    with executorch_export._environment_flatc():
        assert os.environ["FLATC_EXECUTABLE"] == str(flatc)
        # A concurrent export must still find it after another one finishes.
        with executorch_export._environment_flatc():
            pass
        assert os.environ["FLATC_EXECUTABLE"] == str(flatc)


def test_user_flatc_executable_is_left_alone(monkeypatch):
    import libreyolo.export.executorch as executorch_export

    monkeypatch.setenv("FLATC_EXECUTABLE", "/opt/flatc")
    monkeypatch.setattr(
        executorch_export,
        "environment_script",
        lambda name: pytest.fail("looked up flatc despite FLATC_EXECUTABLE"),
    )
    with executorch_export._environment_flatc():
        assert os.environ["FLATC_EXECUTABLE"] == "/opt/flatc"
    assert os.environ["FLATC_EXECUTABLE"] == "/opt/flatc"
