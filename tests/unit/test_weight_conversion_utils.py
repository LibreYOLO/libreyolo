"""Tests for shared weight-conversion script helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
if str(WEIGHTS_DIR) not in sys.path:
    sys.path.insert(0, str(WEIGHTS_DIR))

import _conversion_utils as conversion_utils  # noqa: E402


class DummyModel:
    def __init__(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict


def test_extract_state_dict_prefers_ema_module():
    checkpoint = {
        "ema": {"module": {"from_ema": 1}},
        "model": {"from_model": 2},
        "state_dict": {"from_state_dict": 3},
    }

    assert conversion_utils.extract_state_dict(checkpoint) == {"from_ema": 1}


def test_extract_state_dict_materializes_module_like_values():
    checkpoint = {"model": DummyModel({"layer.weight": 1})}

    assert conversion_utils.extract_state_dict(checkpoint, prefer_ema=False) == {
        "layer.weight": 1
    }


def test_strip_state_dict_prefix_only_changes_matching_keys():
    state_dict = {
        "model.model.backbone.conv.weight": 1,
        "head.cls.weight": 2,
    }

    stripped = conversion_utils.strip_state_dict_prefix(state_dict, "model.model.")

    assert stripped == {
        "backbone.conv.weight": 1,
        "head.cls.weight": 2,
    }


def test_wrap_libreyolo_checkpoint_uses_provided_names():
    checkpoint = conversion_utils.wrap_libreyolo_checkpoint(
        {"layer.weight": 1},
        model_family="dfine",
        size="n",
        task="detect",
        nc=2,
        names={0: "cat", 1: "dog"},
        imgsz=640,
    )

    assert checkpoint["libreyolo_version"]
    checkpoint = {k: v for k, v in checkpoint.items() if k != "libreyolo_version"}
    assert checkpoint == {
        "model": {"layer.weight": 1},
        "schema_version": "1.0",
        "model_family": "dfine",
        "size": "n",
        "task": "detect",
        "nc": 2,
        "names": {0: "cat", 1: "dog"},
        "imgsz": 640,
    }


def test_wrap_libreyolo_checkpoint_does_not_write_task_catalog_fields():
    checkpoint = conversion_utils.wrap_libreyolo_checkpoint(
        {"layer.weight": 1},
        model_family="ec",
        size="s",
        task="pose",
        nc=1,
        names={0: "person"},
        imgsz=640,
        supported_tasks=("detect", "pose", "segment"),
        default_task="detect",
    )

    assert checkpoint["task"] == "pose"
    assert "supported_tasks" not in checkpoint
    assert "default_task" not in checkpoint


def test_save_checkpoint_creates_parent_directory(tmp_path):
    output_path = tmp_path / "nested" / "checkpoint.pt"

    saved_path = conversion_utils.save_checkpoint(
        {"value": torch.tensor([1.0])},
        output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()
    loaded = torch.load(output_path, map_location="cpu", weights_only=False)
    assert torch.equal(loaded["value"], torch.tensor([1.0]))
