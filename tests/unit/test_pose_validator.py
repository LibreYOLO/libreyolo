from __future__ import annotations

import json
import logging

import pytest
import yaml
from PIL import Image

from libreyolo.validation import PoseValidator, ValidationConfig

pytestmark = pytest.mark.unit


class _DummyPoseModel:
    size = "s"

    def _get_model_name(self):
        return "dummy"


def test_pose_validator_accepts_yolo_pose_yaml(tmp_path):
    images_dir = tmp_path / "images" / "val"
    labels_dir = tmp_path / "labels" / "val"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    Image.new("RGB", (640, 480)).save(images_dir / "img0.jpg")
    (labels_dir / "img0.txt").write_text(
        "0 0.5 0.5 0.25 0.5 "
        "0.4 0.3 2 0.6 0.3 2 0.6 0.7 1 0.4 0.7 0\n"
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val",
                "nc": 1,
                "names": {0: "runway"},
                "kpt_shape": [4, 3],
                "keypoints": ["tl", "tr", "br", "bl"],
                "skeleton": [[1, 2], [3, 4]],
            }
        )
    )

    config = ValidationConfig(
        data=str(data_yaml),
        save_dir=str(tmp_path / "runs"),
        verbose=False,
    )
    validator = PoseValidator(_DummyPoseModel(), config=config)
    validator._setup_paths()

    gt_path = tmp_path / "runs" / "ground_truth_yolo_pose.json"
    assert validator._kpts_json == gt_path
    coco = json.loads(gt_path.read_text())
    assert coco["images"][0]["file_name"] == str(images_dir / "img0.jpg")
    assert coco["categories"][0]["name"] == "runway"
    assert coco["categories"][0]["keypoints"] == ["tl", "tr", "br", "bl"]
    assert coco["categories"][0]["skeleton"] == [[1, 2], [3, 4]]
    assert coco["annotations"][0]["num_keypoints"] == 3
    assert len(coco["annotations"][0]["keypoints"]) == 12
    assert validator._num_keypoints == 4
    assert validator._normalize_skeleton([[1, 2], [3, 4]]) == ((0, 1), (2, 3))
    assert validator._normalize_skeleton([[0, 1], [2, 3]]) == ((0, 1), (2, 3))
    assert validator._resolve_oks_sigmas() == [0.25] * 4


def test_pose_validator_rejects_missing_yolo_pose_labels(tmp_path):
    images_dir = tmp_path / "images" / "val"
    labels_dir = tmp_path / "labels" / "val"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.new("RGB", (640, 480)).save(images_dir / "img0.jpg")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val",
                "nc": 1,
                "names": {0: "person"},
                "kpt_shape": [17, 3],
            }
        )
    )
    config = ValidationConfig(
        data=str(data_yaml),
        save_dir=str(tmp_path / "runs"),
        verbose=False,
    )
    validator = PoseValidator(_DummyPoseModel(), config=config)

    with pytest.raises(FileNotFoundError, match="No YOLO pose labels"):
        validator._setup_paths()


def test_pose_validator_zero_predictions_returns_all_keypoint_metrics(tmp_path):
    config = ValidationConfig(
        save_dir=str(tmp_path),
        verbose=False,
        keypoints_json=str(tmp_path / "unused.json"),
        images_dir=str(tmp_path),
    )
    validator = PoseValidator(_DummyPoseModel(), config=config)
    validator.save_dir = tmp_path
    validator._predictions = []

    metrics = validator._evaluate_oks_ap()

    assert metrics == {
        "metrics/keypoints_mAP50-95": 0.0,
        "metrics/keypoints_mAP50": 0.0,
        "metrics/keypoints_mAP75": 0.0,
        "metrics/keypoints_mAP_M": 0.0,
        "metrics/keypoints_mAP_L": 0.0,
        "metrics/keypoints_AR50-95": 0.0,
        "metrics/keypoints_AR50": 0.0,
        "metrics/keypoints_AR75": 0.0,
        "metrics/keypoints_AR_M": 0.0,
        "metrics/keypoints_AR_L": 0.0,
    }
    assert (tmp_path / "predictions.json").exists()


def test_pose_validator_uses_coco_keypoints_annotation_from_yaml(tmp_path):
    images_dir = tmp_path / "images" / "val2017"
    annotations_dir = tmp_path / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    Image.new("RGB", (320, 240)).save(images_dir / "000000123456.jpg")

    keypoints_json = annotations_dir / "person_keypoints_val2017_mini500.json"
    keypoints_json.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 123456,
                        "file_name": "000000123456.jpg",
                        "width": 320,
                        "height": 240,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 123456,
                        "category_id": 0,
                        "bbox": [0, 0, 100, 100],
                        "area": 10000,
                        "iscrowd": 0,
                        "num_keypoints": 1,
                        "keypoints": [50, 60, 2] + [0, 0, 0] * 16,
                    }
                ],
                "categories": [
                    {
                        "id": 0,
                        "name": "person",
                        "keypoints": [f"kpt_{i}" for i in range(17)],
                        "skeleton": [],
                    }
                ],
            }
        )
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val2017",
                "annotations": {"val": "annotations/person_keypoints_val2017_mini500.json"},
                "nc": 1,
                "names": {0: "person"},
                "kpt_shape": [17, 3],
            }
        )
    )

    config = ValidationConfig(
        data=str(data_yaml),
        save_dir=str(tmp_path / "runs"),
        verbose=False,
    )
    validator = PoseValidator(_DummyPoseModel(), config=config)
    validator._setup_paths()
    validator._load_coco_gt()

    assert validator._kpts_json == keypoints_json
    assert validator._images_dir == images_dir
    assert validator._image_records[0]["id"] == 123456
    assert not (tmp_path / "runs" / "ground_truth_yolo_pose.json").exists()


def test_pose_validator_ignores_box_only_coco_annotation_for_pose(tmp_path):
    images_dir = tmp_path / "images" / "val2017"
    labels_dir = tmp_path / "labels" / "val2017"
    annotations_dir = tmp_path / "annotations"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    Image.new("RGB", (320, 240)).save(images_dir / "000000123456.jpg")
    (labels_dir / "000000123456.txt").write_text(
        "0 0.5 0.5 0.25 0.5 "
        "0.4 0.3 2 0.6 0.3 2 0.6 0.7 1 0.4 0.7 0\n"
    )
    instances_json = annotations_dir / "instances_val2017.json"
    instances_json.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 123456,
                        "file_name": "000000123456.jpg",
                        "width": 320,
                        "height": 240,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 123456,
                        "category_id": 0,
                        "bbox": [0, 0, 100, 100],
                        "area": 10000,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 0, "name": "person"}],
            }
        )
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val2017",
                "annotations": {"val": "annotations/instances_val2017.json"},
                "nc": 1,
                "names": {0: "person"},
                "kpt_shape": [4, 3],
            }
        )
    )

    config = ValidationConfig(
        data=str(data_yaml),
        save_dir=str(tmp_path / "runs"),
        verbose=False,
    )
    validator = PoseValidator(_DummyPoseModel(), config=config)
    validator._setup_paths()

    assert validator._kpts_json == tmp_path / "runs" / "ground_truth_yolo_pose.json"
    coco = json.loads(validator._kpts_json.read_text())
    assert coco["images"][0]["file_name"] == str(images_dir / "000000123456.jpg")
    assert "keypoints" in coco["annotations"][0]


def test_pose_validator_maps_contiguous_labels_to_coco_category_ids(tmp_path):
    config = ValidationConfig(
        save_dir=str(tmp_path),
        verbose=False,
        keypoints_json=str(tmp_path / "unused.json"),
        images_dir=str(tmp_path),
    )
    validator = PoseValidator(_DummyPoseModel(), config=config)
    validator._category_id = 1
    validator._category_ids = [1, 3]

    assert validator._prediction_category_id(0) == 1
    assert validator._prediction_category_id(1) == 3
    assert validator._prediction_category_id(3) == 3
    assert validator._prediction_category_id(7) == 1


def test_pose_validator_warns_about_skipped_gt_label_lines(tmp_path, caplog):
    """Ground-truth lines dropped from the metrics must name file and line (#873)."""
    images_dir = tmp_path / "images" / "val"
    labels_dir = tmp_path / "labels" / "val"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    Image.new("RGB", (640, 480)).save(images_dir / "img0.jpg")
    (labels_dir / "img0.txt").write_text(
        "0 0.5 0.5 0.25 0.5 0.4 0.3 2 0.6 0.3 2 0.6 0.7 1 0.4 0.7 0\n"
        "0 0.5 0.5 0.25 0.5 0.4 0.3 2\n"  # only one keypoint
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val",
                "nc": 1,
                "names": {0: "runway"},
                "kpt_shape": [4, 3],
            }
        )
    )

    config = ValidationConfig(
        data=str(data_yaml),
        save_dir=str(tmp_path / "runs"),
        verbose=False,
    )
    validator = PoseValidator(_DummyPoseModel(), config=config)
    with caplog.at_level(logging.WARNING):
        validator._setup_paths()

    assert "skipped 1 ground-truth label line" in caplog.text
    assert "img0.txt:2" in caplog.text
    coco = json.loads((tmp_path / "runs" / "ground_truth_yolo_pose.json").read_text())
    assert len(coco["annotations"]) == 1
