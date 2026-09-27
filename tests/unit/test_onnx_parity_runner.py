from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest
import yaml

from vision_analysis_benchmark.onnx_parity import (
    COCO80_NAMES,
    CheckpointUnavailableError,
    DEFAULT_CASES,
    MINI500_ANNOTATION,
    MINI500_IMAGE_DIR,
    MINI500_POSE_ANNOTATION,
    ParityCase,
    _checkpoint_unavailable_reason,
    _ensure_case_weights_available,
    _load_onnx_for_parity,
    _preflight_case_weights,
    _safe_hf_dataset_path,
    compare_metrics,
    download_hf_dataset_snapshot,
    extract_metrics,
    filter_cases,
    metric_keys_for_task,
    pose_direct_paths,
    prepare_pose_data,
    write_summary,
    write_mini500_yaml,
    write_mini500_pose_yaml,
)


pytestmark = pytest.mark.unit


def _case_ids() -> set[str]:
    return {case.id for case in DEFAULT_CASES}


def test_default_cases_include_supported_onnx_scope():
    case_ids = _case_ids()

    assert "yolo9-t" in case_ids
    assert "rfdetr-n" in case_ids
    assert "yolox-n" in case_ids
    assert "yolo9_e2e-t" in case_ids
    assert "yolonas-s" in case_ids
    assert "dfine-n" in case_ids
    assert "deim-n" in case_ids
    assert "deimv2-atto" in case_ids
    assert "rtdetr-r18" in case_ids
    assert "rtdetrv2-r18" in case_ids
    assert "rtdetrv4-s" in case_ids
    assert "picodet-s" in case_ids
    assert "rtmdet-t" in case_ids
    assert "ec-s" in case_ids


def test_default_cases_include_new_segment_and_pose_paths():
    case_ids = _case_ids()

    assert "rfdetr-s-segment" in case_ids
    assert "ec-s-segment" in case_ids
    assert "yolo9-s-pose" in case_ids
    assert "rfdetr-x-pose" in case_ids
    assert "yolonas-n-pose" in case_ids
    assert "ec-s-pose" in case_ids


def test_metric_keys_are_task_specific():
    assert "metrics/mAP50-95" in metric_keys_for_task("detect")
    assert "metrics/mAP50-95(M)" in metric_keys_for_task("segment")
    assert "metrics/keypoints_mAP50-95" in metric_keys_for_task("pose")


def test_extract_metrics_ignores_non_task_and_non_numeric_values():
    metrics = {
        "metrics/mAP50-95": "0.40",
        "metrics/mAP50": 0.55,
        "metrics/precision": "bad",
        "metrics/recall": 0.72,
        "metrics/mAP50-95(M)": 0.1,
    }

    assert extract_metrics(metrics, "detect") == {
        "metrics/mAP50-95": 0.40,
        "metrics/mAP50": 0.55,
        "metrics/recall": 0.72,
    }


def test_compare_metrics_passes_within_absolute_tolerance():
    comparisons = compare_metrics(
        {
            "metrics/mAP50-95": 0.50000,
            "metrics/mAP50": 0.60000,
            "metrics/precision": 0.70000,
            "metrics/recall": 0.80000,
        },
        {
            "metrics/mAP50-95": 0.50004,
            "metrics/mAP50": 0.60003,
            "metrics/precision": 0.70002,
            "metrics/recall": 0.80001,
        },
        "detect",
        abs_tolerance=1e-4,
        rel_tolerance=0.0,
    )

    assert all(item.passed for item in comparisons)


def test_compare_metrics_fails_missing_and_drifted_values():
    comparisons = compare_metrics(
        {
            "metrics/keypoints_mAP50-95": 0.50,
            "metrics/keypoints_mAP50": 0.60,
            "metrics/keypoints_mAP75": 0.40,
        },
        {
            "metrics/keypoints_mAP50-95": 0.45,
            "metrics/keypoints_mAP50": 0.60,
        },
        "pose",
        abs_tolerance=1e-4,
        rel_tolerance=0.0,
    )

    by_key = {item.key: item for item in comparisons}
    assert not by_key["metrics/keypoints_mAP50-95"].passed
    assert by_key["metrics/keypoints_mAP75"].reason == "missing metric"


def test_checkpoint_unavailable_reason_is_weight_specific():
    assert _checkpoint_unavailable_reason(
        FileNotFoundError("Model weights file not found: weights/model.pt")
    )
    assert _checkpoint_unavailable_reason(
        RuntimeError("Failed to download weights from https://example.test: 404 Client Error")
    )
    assert _checkpoint_unavailable_reason(
        ValueError("Could not determine download URL for 'missing.pt'.")
    )
    assert _checkpoint_unavailable_reason(
        FileNotFoundError(
            "No weights are published at https://example.test/model.pt (HTTP 401)"
        )
    )
    assert _checkpoint_unavailable_reason(
        FileNotFoundError("annotations file not found")
    ) is None
    assert _checkpoint_unavailable_reason(RuntimeError("ONNX export failed")) is None


def test_ensure_case_weights_available_downloads_canonical_path(tmp_path):
    seen = {}
    weight_path = tmp_path / "weights" / "missing.pt"

    _ensure_case_weights_available(
        ParityCase(
            id="missing",
            family="family",
            size="s",
            task="detect",
            weights="missing.pt",
            onnx_claim="available",
        ),
        resolve_weights_path=lambda _name: str(weight_path),
        downloader=lambda path, size: seen.update({"path": path, "size": size}),
    )

    assert seen == {"path": str(weight_path), "size": "s"}


def test_ensure_case_weights_available_skips_existing_file(tmp_path):
    weight_path = tmp_path / "weights" / "exists.pt"
    weight_path.parent.mkdir()
    weight_path.write_bytes(b"weights")

    _ensure_case_weights_available(
        ParityCase(
            id="exists",
            family="family",
            size="s",
            task="detect",
            weights="exists.pt",
            onnx_claim="available",
        ),
        resolve_weights_path=lambda _name: str(weight_path),
        downloader=lambda _path, _size: pytest.fail("download should not run"),
    )


def test_preflight_case_weights_raises_unavailable_for_weight_download_failure(monkeypatch):
    import vision_analysis_benchmark.onnx_parity as parity

    case = ParityCase(
        id="missing",
        family="family",
        size="s",
        task="detect",
        weights="missing.pt",
        onnx_claim="available",
    )
    monkeypatch.setattr(
        parity,
        "_ensure_case_weights_available",
        lambda _case: (_ for _ in ()).throw(
            RuntimeError(
                "Failed to download weights from https://example.test: "
                "404 Client Error"
            )
        ),
    )

    with pytest.raises(CheckpointUnavailableError, match="Failed to download weights"):
        _preflight_case_weights(case)


def test_preflight_case_weights_does_not_hide_unrelated_file_errors(monkeypatch):
    import vision_analysis_benchmark.onnx_parity as parity

    case = ParityCase(
        id="broken-data",
        family="family",
        size="s",
        task="detect",
        weights="exists.pt",
        onnx_claim="available",
    )
    monkeypatch.setattr(
        parity,
        "_ensure_case_weights_available",
        lambda _case: (_ for _ in ()).throw(
            FileNotFoundError("annotations file not found")
        ),
    )

    with pytest.raises(FileNotFoundError, match="annotations file not found"):
        _preflight_case_weights(case)


def test_safe_hf_dataset_path_rejects_traversal(tmp_path):
    with pytest.raises(ValueError, match="Unsafe Hugging Face dataset path"):
        _safe_hf_dataset_path("../outside.jpg", tmp_path)


def test_safe_hf_dataset_path_returns_target_inside_dest(tmp_path):
    rel, target = _safe_hf_dataset_path("images/val2017/frame.jpg", tmp_path)

    assert str(rel) == "images/val2017/frame.jpg"
    assert target == tmp_path.resolve() / "images" / "val2017" / "frame.jpg"


def test_download_coco_person_keypoints_recovers_corrupt_zip(tmp_path, monkeypatch):
    import vision_analysis_benchmark.onnx_parity as parity

    zip_path = tmp_path / "annotations" / "annotations_trainval2017.zip"
    zip_path.parent.mkdir()
    zip_path.write_bytes(b"partial")

    def fake_download(_url, target) -> None:
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr(parity.COCO_PERSON_KEYPOINTS_MEMBER, "{}")

    monkeypatch.setattr(parity, "_download_atomic", fake_download)

    path = parity._download_coco_person_keypoints(tmp_path)

    assert path == tmp_path / parity.COCO_PERSON_KEYPOINTS_MEMBER
    assert path.read_text(encoding="utf-8") == "{}"


def test_load_onnx_for_parity_uses_exported_metadata_task(tmp_path):
    class FakeModel:
        task = "segment"

    seen = {}

    def fake_loader(path, **kwargs):
        seen.update({"path": path, "kwargs": kwargs})
        return FakeModel()

    args = type("Args", (), {"onnx_device": "cuda"})()
    case = ParityCase(
        id="seg",
        family="family",
        size="s",
        task="segment",
        weights="seg.pt",
        onnx_claim="complete",
    )

    model = _load_onnx_for_parity(tmp_path / "seg.onnx", case, args, loader=fake_loader)

    assert model.task == "segment"
    assert seen == {"path": str(tmp_path / "seg.onnx"), "kwargs": {"device": "cuda"}}


def test_load_onnx_for_parity_rejects_wrong_exported_task(tmp_path):
    class FakeModel:
        task = "detect"

    args = type("Args", (), {"onnx_device": "cpu"})()
    case = ParityCase(
        id="seg",
        family="family",
        size="s",
        task="segment",
        weights="seg.pt",
        onnx_claim="complete",
    )

    with pytest.raises(RuntimeError, match="task metadata resolved to 'detect'"):
        _load_onnx_for_parity(
            tmp_path / "seg.onnx",
            case,
            args,
            loader=lambda *_args, **_kwargs: FakeModel(),
        )


def test_write_summary_counts_unavailable_cases(tmp_path):
    records = [
        {
            "case": {
                "id": "missing-model",
                "family": "family",
                "size": "s",
                "task": "detect",
                "onnx_claim": "available",
                "notes": "",
            },
            "status": "unavailable",
            "error": "Model weights file not found: weights/missing.pt",
            "unavailable_reason": "Model weights file not found: weights/missing.pt",
        }
    ]

    write_summary(records, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    markdown = (tmp_path / "summary.md").read_text()

    assert summary["counts"] == {
        "total": 1,
        "passed": 0,
        "failed": 0,
        "error": 0,
        "unavailable": 1,
    }
    assert "unavailable `1`" in markdown
    assert "Model weights file not found: weights/missing.pt" in markdown


def test_filter_cases_by_family_task_and_claim():
    selected = filter_cases(
        DEFAULT_CASES,
        families=["ec"],
        tasks=["pose"],
        claims=["available"],
    )

    assert {case.id for case in selected} == {
        "ec-s-pose",
        "ec-m-pose",
        "ec-l-pose",
        "ec-x-pose",
    }


def test_filter_cases_can_return_empty_selection():
    assert filter_cases(DEFAULT_CASES, families=["does-not-exist"]) == []


def test_write_mini500_yaml_uses_explicit_annotation_mapping(tmp_path):
    root = tmp_path / "mini500"
    (root / Path(MINI500_ANNOTATION).parent).mkdir(parents=True)
    (root / MINI500_ANNOTATION).write_text(json.dumps({"images": []}))
    (root / MINI500_IMAGE_DIR).mkdir(parents=True)

    yaml_path = write_mini500_yaml(root)
    config = yaml.safe_load(yaml_path.read_text())

    assert config["path"] == str(root.resolve())
    assert config["val"] == MINI500_IMAGE_DIR
    assert config["annotations"]["val"] == MINI500_ANNOTATION
    assert config["names"] == COCO80_NAMES


def test_write_mini500_pose_yaml_filters_keypoints_to_mini500_images(tmp_path):
    root = tmp_path / "mini500"
    (root / Path(MINI500_ANNOTATION).parent).mkdir(parents=True)
    (root / MINI500_IMAGE_DIR).mkdir(parents=True)
    (root / MINI500_ANNOTATION).write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
                "annotations": [],
            }
        )
    )
    keypoints_path = tmp_path / "person_keypoints_val2017.json"
    keypoints_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.jpg"},
                    {"id": 2, "file_name": "b.jpg"},
                    {"id": 3, "file_name": "c.jpg"},
                ],
                "annotations": [
                    {"id": 10, "image_id": 1, "category_id": 1},
                    {"id": 11, "image_id": 3, "category_id": 1},
                ],
                "categories": [{"id": 1, "name": "person"}],
            }
        )
    )

    yaml_path = write_mini500_pose_yaml(root, full_keypoints_path=keypoints_path)
    config = yaml.safe_load(yaml_path.read_text())
    filtered = json.loads((root / MINI500_POSE_ANNOTATION).read_text())

    assert config["annotations"]["val"] == MINI500_POSE_ANNOTATION
    assert config["kpt_shape"] == [17, 3]
    assert config["names"] == {0: "person"}
    assert filtered["categories"] == [{"id": 0, "name": "person"}]
    assert {image["id"] for image in filtered["images"]} == {1, 2}
    assert [ann["id"] for ann in filtered["annotations"]] == [10]
    assert {ann["category_id"] for ann in filtered["annotations"]} == {0}


def test_prepare_pose_data_keeps_existing_pose_yaml(tmp_path):
    pose_yaml = tmp_path / "pose.yaml"
    pose_yaml.write_text(yaml.safe_dump({"kpt_shape": [17, 3]}))

    assert prepare_pose_data(pose_yaml) == pose_yaml


def test_prepare_pose_data_keeps_native_coco_keypoints_yaml(tmp_path):
    annotation_path = tmp_path / "annotations" / "person_keypoints_val2017.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        json.dumps(
            {
                "annotations": [],
                "categories": [
                    {
                        "id": 1,
                        "name": "person",
                        "keypoints": [f"kpt_{idx}" for idx in range(17)],
                    }
                ],
            }
        )
    )
    pose_yaml = tmp_path / "coco-pose.yaml"
    pose_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val2017",
                "annotations": {"val": "annotations/person_keypoints_val2017.json"},
            }
        )
    )

    assert prepare_pose_data(pose_yaml) == pose_yaml


def test_prepare_pose_data_inspects_arbitrary_keypoint_annotation_name(tmp_path):
    annotation_path = tmp_path / "annotations" / "pose_val.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "keypoints": [0] * 51,
                        "num_keypoints": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "person"}],
            }
        )
    )
    pose_yaml = tmp_path / "coco-pose.yaml"
    pose_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val2017",
                "annotations": {"val": "annotations/pose_val.json"},
            }
        )
    )

    assert prepare_pose_data(pose_yaml) == pose_yaml


def test_prepare_pose_data_regenerates_for_box_only_annotation(tmp_path, monkeypatch):
    import vision_analysis_benchmark.onnx_parity as parity

    annotation_path = tmp_path / "annotations" / "person_keypoints_val2017.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        json.dumps(
            {
                "annotations": [{"id": 1, "image_id": 1, "category_id": 0}],
                "categories": [{"id": 0, "name": "person"}],
            }
        )
    )
    (tmp_path / MINI500_ANNOTATION).write_text(
        json.dumps({"images": [{"id": 1, "file_name": "a.jpg"}], "annotations": []})
    )
    (tmp_path / MINI500_IMAGE_DIR).mkdir(parents=True)
    full_keypoints = tmp_path / "full_person_keypoints.json"
    full_keypoints.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg"}],
                "annotations": [
                    {
                        "id": 10,
                        "image_id": 1,
                        "category_id": 1,
                        "keypoints": [0] * 51,
                        "num_keypoints": 0,
                    }
                ],
                "categories": [
                    {
                        "id": 1,
                        "name": "person",
                        "keypoints": [f"kpt_{idx}" for idx in range(17)],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        parity,
        "_download_coco_person_keypoints",
        lambda _dataset_root: full_keypoints,
    )
    pose_yaml = tmp_path / "coco-pose.yaml"
    pose_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val2017",
                "annotations": {"val": "annotations/person_keypoints_val2017.json"},
            }
        )
    )

    assert prepare_pose_data(pose_yaml) == tmp_path / "coco-val2017-mini500-pose.yaml"


def test_prepare_pose_data_honors_selected_split(tmp_path, monkeypatch):
    import vision_analysis_benchmark.onnx_parity as parity

    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / "instances_val2017.json").write_text(
        json.dumps(
            {
                "annotations": [{"id": 1, "image_id": 1, "category_id": 0}],
                "categories": [{"id": 0, "name": "person"}],
            }
        )
    )
    (annotations_dir / "person_keypoints_test-dev2017.json").write_text(
        json.dumps(
            {
                "annotations": [],
                "categories": [
                    {
                        "id": 0,
                        "name": "person",
                        "keypoints": [f"kpt_{idx}" for idx in range(17)],
                    }
                ],
            }
        )
    )
    (tmp_path / MINI500_ANNOTATION).write_text(
        json.dumps({"images": [{"id": 1, "file_name": "a.jpg"}], "annotations": []})
    )
    (tmp_path / MINI500_IMAGE_DIR).mkdir(parents=True)
    full_keypoints = tmp_path / "full_person_keypoints.json"
    full_keypoints.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg"}],
                "annotations": [
                    {
                        "id": 10,
                        "image_id": 1,
                        "category_id": 1,
                        "keypoints": [0] * 51,
                        "num_keypoints": 0,
                    }
                ],
                "categories": [
                    {
                        "id": 1,
                        "name": "person",
                        "keypoints": [f"kpt_{idx}" for idx in range(17)],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        parity,
        "_download_coco_person_keypoints",
        lambda _dataset_root: full_keypoints,
    )
    pose_yaml = tmp_path / "coco-pose.yaml"
    pose_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val2017",
                "test": "images/test2017",
                "annotations": {
                    "val": "annotations/instances_val2017.json",
                    "test": "annotations/person_keypoints_test-dev2017.json",
                },
            }
        )
    )

    assert prepare_pose_data(pose_yaml, split="val") == (
        tmp_path / "coco-val2017-mini500-pose.yaml"
    )
    assert prepare_pose_data(pose_yaml, split="test") == pose_yaml


def test_pose_direct_paths_honor_selected_split(tmp_path):
    annotation_path = tmp_path / "annotations" / "person_keypoints_test-dev2017.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        json.dumps(
            {
                "annotations": [],
                "categories": [
                    {
                        "id": 0,
                        "name": "person",
                        "keypoints": [f"kpt_{idx}" for idx in range(17)],
                    }
                ],
            }
        )
    )
    pose_yaml = tmp_path / "coco-pose.yaml"
    pose_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val2017",
                "test": "images/test2017",
                "annotations": {
                    "val": "annotations/person_keypoints_val2017.json",
                    "test": "annotations/person_keypoints_test-dev2017.json",
                },
            }
        )
    )

    annotation_path, image_dir = pose_direct_paths(pose_yaml, split="test")

    assert annotation_path == tmp_path / "annotations" / "person_keypoints_test-dev2017.json"
    assert image_dir == tmp_path / "images" / "test2017"


def test_pose_direct_paths_ignores_box_only_coco_annotations(tmp_path):
    annotation_path = tmp_path / "annotations" / "instances_val2017.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 0,
                        "bbox": [0, 0, 10, 10],
                    }
                ],
                "categories": [{"id": 0, "name": "person"}],
            }
        )
    )
    pose_yaml = tmp_path / "pose.yaml"
    pose_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val2017",
                "annotations": {"val": "annotations/instances_val2017.json"},
                "kpt_shape": [17, 3],
            }
        )
    )

    assert pose_direct_paths(pose_yaml) == (None, None)


def test_download_hf_dataset_snapshot_redownloads_unknown_size_partial(
    tmp_path,
    monkeypatch,
):
    import vision_analysis_benchmark.onnx_parity as parity

    class FakeTreeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return [{"type": "file", "path": "images/val2017/a.jpg"}]

    def fake_get(_url, headers=None, timeout=60):
        assert headers == {"Accept": "application/json"}
        assert timeout == 60
        return FakeTreeResponse()

    seen = {}

    def fake_download(url, target, *, headers=None) -> None:
        seen.update({"url": url, "headers": headers})
        target.write_bytes(b"complete")

    target = tmp_path / "images" / "val2017" / "a.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"partial")
    monkeypatch.setattr(parity.requests, "get", fake_get)
    monkeypatch.setattr(parity, "_download_atomic", fake_download)

    download_hf_dataset_snapshot("LibreYOLO/coco-val2017-mini500", tmp_path)

    assert target.read_bytes() == b"complete"
    assert seen["headers"] == {}
