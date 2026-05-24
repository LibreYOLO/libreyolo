"""Tests for torch checkpoint loading helpers."""

import pytest

from libreyolo.utils import serialization

pytestmark = pytest.mark.unit


def test_untrusted_load_uses_weights_only(monkeypatch):
    calls = {}

    monkeypatch.setattr(serialization, "_supports_weights_only", lambda: True)

    def fake_load(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(serialization.torch, "load", fake_load)

    result = serialization.load_untrusted_torch_file("model.pt")

    assert result == {"ok": True}
    assert calls["kwargs"]["weights_only"] is True
    assert calls["kwargs"]["map_location"] == "cpu"


def test_trusted_load_uses_full_checkpoint_mode(monkeypatch):
    calls = {}

    monkeypatch.setattr(serialization, "_supports_weights_only", lambda: True)

    def fake_load(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(serialization.torch, "load", fake_load)

    result = serialization.load_trusted_torch_file("last.pt", map_location="cuda:0")

    assert result == {"ok": True}
    assert calls["kwargs"]["weights_only"] is False
    assert calls["kwargs"]["map_location"] == "cuda:0"


def test_untrusted_load_requires_modern_torch(monkeypatch):
    monkeypatch.setattr(serialization, "_supports_weights_only", lambda: False)

    with pytest.raises(RuntimeError, match="weights_only"):
        serialization.load_untrusted_torch_file("model.pt")


def test_wrap_libreyolo_checkpoint_emits_required_v1_metadata(monkeypatch):
    monkeypatch.setattr(serialization, "get_libreyolo_version", lambda: "1.2.3")

    state_dict = {"layer.weight": 1}
    checkpoint = serialization.wrap_libreyolo_checkpoint(
        state_dict,
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "cat", 1: "dog"},
        imgsz=640,
    )

    assert checkpoint == {
        "model": state_dict,
        "schema_version": serialization.SCHEMA_VERSION,
        "libreyolo_version": "1.2.3",
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "nc": 2,
        "names": {0: "cat", 1: "dog"},
        "imgsz": 640,
    }


def test_validate_checkpoint_metadata_requires_all_core_fields():
    checkpoint = {
        "model": {"layer.weight": object()},
        "schema_version": serialization.SCHEMA_VERSION,
        "libreyolo_version": "1.2.3",
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "nc": 1,
        "names": {0: "cat"},
    }

    errors = serialization.validate_checkpoint_metadata(checkpoint)

    assert "missing required key: imgsz" in errors
    with pytest.raises(serialization.CheckpointMetadataError, match="imgsz"):
        serialization.validate_checkpoint_metadata(checkpoint, strict=True)


def test_validate_checkpoint_metadata_accepts_string_name_keys_without_mutation():
    checkpoint = {
        "model": {"layer.weight": object()},
        "schema_version": serialization.SCHEMA_VERSION,
        "libreyolo_version": "1.2.3",
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "nc": 2,
        "names": {"0": "cat", "1": "dog"},
        "imgsz": 640,
    }

    assert serialization.validate_checkpoint_metadata(checkpoint) == []
    assert checkpoint["names"] == {"0": "cat", "1": "dog"}


def test_validate_checkpoint_metadata_pads_missing_name_indices():
    checkpoint = {
        "model": {"layer.weight": object()},
        "schema_version": serialization.SCHEMA_VERSION,
        "libreyolo_version": "1.2.3",
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "nc": 3,
        "names": {0: "cat", 2: "dog"},
        "imgsz": 640,
    }

    with pytest.warns(RuntimeWarning, match="padding"):
        assert serialization.validate_checkpoint_metadata(checkpoint, strict=True) == []
    with pytest.warns(RuntimeWarning, match="padding"):
        assert serialization.normalize_checkpoint_names(checkpoint["names"], 3) == {
            0: "cat",
            1: "class_1",
            2: "dog",
        }
    assert checkpoint["names"] == {0: "cat", 2: "dog"}


def test_validate_checkpoint_metadata_rejects_out_of_range_names():
    checkpoint = {
        "model": {"layer.weight": object()},
        "schema_version": serialization.SCHEMA_VERSION,
        "libreyolo_version": "1.2.3",
        "model_family": "yolo9",
        "size": "t",
        "task": "detect",
        "nc": 2,
        "names": {0: "cat", 2: "dog"},
        "imgsz": 640,
    }

    with pytest.raises(serialization.CheckpointMetadataError, match="out-of-range"):
        serialization.validate_checkpoint_metadata(checkpoint, strict=True)


def test_warn_on_metadata_schema_version_logs_legacy_metadata(caplog):
    import logging

    logger = logging.getLogger("test-schema")

    with caplog.at_level(logging.WARNING, logger="test-schema"):
        serialization.warn_on_metadata_schema_version(
            {"model_family": "yolo9"},
            artifact="test export",
            logger=logger,
        )

    assert "has no schema_version" in caplog.text


def test_wrap_checkpoint_does_not_fall_back_to_default_size_for_empty_task_map(
    monkeypatch,
):
    from libreyolo.models.base import BaseModel

    class DummyFamily:
        FAMILY = "dummy"
        INPUT_SIZES = {"s": 640}
        TASK_INPUT_SIZES = {"pose": {}}

    monkeypatch.setattr(BaseModel, "_registry", [DummyFamily])

    with pytest.raises(serialization.CheckpointMetadataError, match="imgsz"):
        serialization.wrap_libreyolo_checkpoint(
            {"layer.weight": object()},
            model_family="dummy",
            size="s",
            task="pose",
            nc=1,
            names={0: "person"},
        )
