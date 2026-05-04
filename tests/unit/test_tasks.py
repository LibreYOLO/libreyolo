"""Tests for task metadata helpers."""

import pytest

from libreyolo.tasks import TaskType, normalize_supported_tasks, normalize_task, resolve_task

pytestmark = pytest.mark.unit


def test_normalize_task_aliases():
    assert normalize_task("det") == "detect"
    assert normalize_task("seg") == "segment"
    assert normalize_task("cls") == "classify"


def test_task_type_literal_is_public():
    assert set(TaskType.__args__) == {"detect", "segment", "pose", "classify"}


def test_resolve_task_precedence():
    assert (
        resolve_task(
            explicit_task="detect",
            checkpoint_task="segment",
            filename_task="segment",
            supported_tasks=("detect", "segment"),
        )
        == "detect"
    )
    assert (
        resolve_task(
            checkpoint_task="segment",
            filename_task="detect",
            supported_tasks=("detect", "segment"),
        )
        == "segment"
    )


def test_resolve_task_rejects_unsupported_task():
    with pytest.raises(ValueError, match="not supported"):
        resolve_task(explicit_task="segment", supported_tasks=("detect",))


def test_normalize_supported_tasks_accepts_exported_json_string():
    assert normalize_supported_tasks('["detect", "segment"]') == ("detect", "segment")


def test_suffix_to_task_returns_none_for_unknown_suffix():
    from libreyolo.tasks import suffix_to_task

    assert suffix_to_task("-seg") == "segment"
    assert suffix_to_task("-unknown") is None
