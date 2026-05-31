from __future__ import annotations

import json

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
    assert coco["annotations"][0]["num_keypoints"] == 3
    assert len(coco["annotations"][0]["keypoints"]) == 12
    assert validator._num_keypoints == 4
    assert validator._resolve_oks_sigmas() == [0.25] * 4
