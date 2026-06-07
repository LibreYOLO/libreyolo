"""Tests for COCO keypoints to YOLO-pose conversion."""

from __future__ import annotations

import json

import pytest
from PIL import Image

pytestmark = pytest.mark.unit


def test_coco_keypoints_converter_writes_yolo_pose_labels(tmp_path):
    from libreyolo.data.coco_pose import convert_coco_keypoints_json_to_yolo_pose

    images_dir = tmp_path / "images" / "train"
    labels_dir = tmp_path / "labels" / "train"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (100, 50)).save(images_dir / "sample.jpg")
    Image.new("RGB", (100, 50)).save(images_dir / "empty.jpg")
    coco = {
        "images": [
            {"id": 1, "file_name": "sample.jpg", "width": 100, "height": 50},
            {"id": 2, "file_name": "empty.jpg", "width": 100, "height": 50},
            {"id": 3, "file_name": "not_selected.jpg", "width": 100, "height": 50},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 5, 40, 20],
                "keypoints": [20, 10, 2, 0, 0, 0, 150, 10, 2],
                "num_keypoints": 1,
            },
            {
                "id": 2,
                "image_id": 3,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "keypoints": [1, 1, 2, 2, 2, 2, 3, 3, 2],
                "num_keypoints": 3,
            },
        ],
    }
    json_path = tmp_path / "person_keypoints_train2017.json"
    json_path.write_text(json.dumps(coco), encoding="utf-8")

    summary = convert_coco_keypoints_json_to_yolo_pose(
        json_path,
        images_dir,
        labels_dir,
        num_keypoints=3,
    )

    assert summary == {"images": 2, "labels": 2, "annotations": 1, "skipped": 0}
    label = (labels_dir / "sample.txt").read_text(encoding="utf-8").strip().split()
    assert len(label) == 14
    assert [float(v) for v in label] == pytest.approx(
        [0.0, 0.3, 0.3, 0.4, 0.4, 0.2, 0.2, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert (labels_dir / "empty.txt").read_text(encoding="utf-8") == ""
