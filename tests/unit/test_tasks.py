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


def test_normalize_depth_task_aliases():
    assert normalize_task("depth") == "depth"
    assert normalize_task("depth-estimation") == "depth"
    assert normalize_task("depth_estimation") == "depth"
    assert normalize_task("monodepth") == "depth"
    assert normalize_task("monocular-depth") == "depth"


def test_normalize_normal_task_aliases():
    for alias in (
        "normal",
        "normals",
        "surface-normal",
        "surface_normal",
        "surface-normals",
        "surface_normals",
    ):
        assert normalize_task(alias) == "normal"


def test_normalize_edge_task_aliases():
    for alias in (
        "edge",
        "edges",
        "boundary",
        "boundaries",
        "edge-detection",
        "edge_detection",
    ):
        assert normalize_task(alias) == "edge"


def test_normalize_restore_task_aliases():
    assert normalize_task("restore") == "restore"
    assert normalize_task("restoration") == "restore"
    assert normalize_task("deblur") == "restore"
    assert normalize_task("denoise") == "restore"


def test_normalize_matte_task_aliases():
    assert normalize_task("matte") == "matte"
    assert normalize_task("matting") == "matte"
    assert normalize_task("background-removal") == "matte"
    assert normalize_task("background_removal") == "matte"
    assert normalize_task("rembg") == "matte"
    assert normalize_task("dis") == "matte"


def test_task_type_literal_is_public():
    assert set(TaskType.__args__) == {
        "detect",
        "segment",
        "semantic",
        "panoptic",
        "pose",
        "classify",
        "gaze",
        "obb",
        "point",
        "depth",
        "edge",
        "normal",
        "restore",
        "matte",
        "ocr",
        "embed",
        "mesh",
        "detect3d",
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
    assert suffix_to_task("-depth") == "depth"
    assert suffix_to_task("-edge") == "edge"
    assert suffix_to_task("-normal") == "normal"
    assert suffix_to_task("-restore") == "restore"
    assert suffix_to_task("-matte") == "matte"
    assert task_to_suffix("obb") == "obb"
    assert task_to_suffix("point") == "point"
    assert task_to_suffix("depth") == "depth"
    assert task_to_suffix("edge") == "edge"
    assert task_to_suffix("normal") == "normal"
    assert task_to_suffix("restore") == "restore"
    assert task_to_suffix("matte") == "matte"
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
