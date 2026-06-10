"""Tests for task metadata helpers."""

import pytest

from libreyolo.tasks import (
    TaskType,
    normalize_supported_tasks,
    normalize_task,
    resolve_task,
    suffix_to_task,
    task_suffix_pattern,
    task_to_suffix,
)

pytestmark = pytest.mark.unit


def test_normalize_task_aliases():
    assert normalize_task("det") == "detect"
    assert normalize_task("seg") == "segment"
    assert normalize_task("cls") == "classify"
    assert normalize_task("obb") == "obb"
    assert normalize_task("point") == "point"


def test_normalize_semantic_task_aliases():
    assert normalize_task("semantic") == "semantic"
    assert normalize_task("semseg") == "semantic"
    assert normalize_task("sem") == "semantic"
    assert normalize_task("semantic-segmentation") == "semantic"
    assert normalize_task("semantic_segmentation") == "semantic"
    # Instance segmentation aliases must keep resolving to segment.
    assert normalize_task("segmentation") == "segment"
    assert normalize_task("seg") == "segment"


def test_point_task_has_no_aliases():
    for alias in ("points", "centroid", "centroids", "center", "centers"):
        with pytest.raises(ValueError, match="Unsupported task"):
            normalize_task(alias)


def test_task_type_literal_is_public():
    assert set(TaskType.__args__) == {
        "detect",
        "segment",
        "semantic",
        "pose",
        "classify",
        "gaze",
        "obb",
        "point",
    }


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


def test_resolve_task_accepts_obb_when_supported():
    assert resolve_task(explicit_task="obb", supported_tasks=("detect", "obb")) == "obb"


def test_normalize_supported_tasks_accepts_exported_json_string():
    assert normalize_supported_tasks('["detect", "point"]') == ("detect", "point")


def test_task_suffix_helpers():
    assert suffix_to_task("-seg") == "segment"
    assert suffix_to_task("-sem") == "semantic"
    assert suffix_to_task("-obb") == "obb"
    assert suffix_to_task("-point") == "point"
    assert task_to_suffix("obb") == "obb"
    assert task_to_suffix("point") == "point"
    assert task_to_suffix("semantic") == "sem"
    assert suffix_to_task("-unknown") is None


def test_semantic_and_segment_suffixes_do_not_collide():
    pattern = task_suffix_pattern(("segment", "semantic"))

    assert "-seg" in pattern
    assert "-sem" in pattern
    assert suffix_to_task("-sem") != suffix_to_task("-seg")


def test_task_suffix_pattern_includes_point_for_supported_families():
    pattern = task_suffix_pattern(("detect", "point"))

    assert "-point" in pattern
    assert "-seg" not in pattern
