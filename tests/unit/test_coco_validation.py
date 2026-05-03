"""
Unit tests for COCO evaluation in DetectionValidator.

Tests that COCO evaluator is properly integrated into the validation pipeline
using mock data (no GPU or real datasets needed).
"""

import pytest
import yaml
from PIL import Image


def _coco_metrics(base: float):
    return {
        "precision": base + 0.001,
        "recall": base + 0.002,
        "mAP": base,
        "mAP50": base + 0.01,
        "mAP75": base + 0.02,
        "mAP_small": base + 0.03,
        "mAP_medium": base + 0.04,
        "mAP_large": base + 0.05,
        "AR1": base + 0.06,
        "AR10": base + 0.07,
        "AR100": base + 0.08,
        "AR_small": base + 0.09,
        "AR_medium": base + 0.10,
        "AR_large": base + 0.11,
    }


class _DummyEvaluator:
    def __init__(self, metrics=None):
        self.metrics = metrics or _coco_metrics(0.1)
        self.updates = []
        self.saved_paths = []

    def update(self, pred, image_id):
        self.updates.append((pred, image_id))

    def compute(self, save_json=None):
        self.saved_paths.append(save_json)
        return dict(self.metrics)


def create_mock_yolo_dataset(tmp_path):
    """Create a minimal mock YOLO dataset for testing."""
    # Create directory structure
    images_dir = tmp_path / "images" / "val"
    labels_dir = tmp_path / "labels" / "val"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create 3 dummy images
    for i in range(3):
        img = Image.new("RGB", (640, 640), color=(i * 50, i * 50, i * 50))
        img.save(images_dir / f"img{i}.jpg")

        # Create corresponding label
        # Format: class cx cy w h (normalized)
        labels = [
            "0 0.5 0.5 0.3 0.3\n",  # Center object
            "1 0.25 0.25 0.2 0.2\n",  # Top-left object
        ]
        (labels_dir / f"img{i}.txt").write_text("".join(labels[: i + 1]))

    # Create data.yaml
    data_yaml = {
        "path": str(tmp_path),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["cat", "dog"],
    }

    yaml_path = tmp_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    return yaml_path


@pytest.mark.unit
def test_coco_evaluator_integration(tmp_path):
    """Test that COCOEvaluator integrates into DetectionValidator."""
    from libreyolo.data import create_yolo_coco_api
    from libreyolo.validation import COCOEvaluator

    yaml_path = create_mock_yolo_dataset(tmp_path)

    # YOLOCocoAPI creation
    coco_api = create_yolo_coco_api(str(yaml_path), split="val")
    assert len(coco_api.imgs) == 3

    # COCOEvaluator initialization
    evaluator = COCOEvaluator(coco_api, iou_type="bbox")

    # Update with dummy predictions
    dummy_pred = {
        "boxes": [[100, 100, 200, 200]],
        "scores": [0.9],
        "classes": [0],
    }
    evaluator.update(dummy_pred, image_id=0)
    assert len(evaluator.results) == 1

    # Compute metrics
    metrics = evaluator.compute()
    expected_keys = [
        "mAP",
        "mAP50",
        "mAP75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"


@pytest.mark.unit
def test_coco_evaluator_encodes_masks_json_safe():
    from libreyolo.validation.coco_evaluator import COCOEvaluator

    mask = [[0, 0, 0], [0, 1, 1], [0, 1, 0]]
    rle = COCOEvaluator._encode_mask(mask)

    assert rle["size"] == [3, 3]
    assert isinstance(rle["counts"], str)


@pytest.mark.unit
def test_coco_evaluator_uses_mask_area_for_segmentation(tmp_path):
    from libreyolo.data import create_yolo_coco_api
    from libreyolo.validation import COCOEvaluator

    yaml_path = create_mock_yolo_dataset(tmp_path)
    coco_api = create_yolo_coco_api(str(yaml_path), split="val")
    evaluator = COCOEvaluator(coco_api, iou_type="segm")

    mask = [[0, 0, 0], [0, 1, 1], [0, 1, 0]]
    evaluator.update(
        {
            "boxes": [[0, 0, 3, 3]],
            "scores": [0.9],
            "classes": [0],
            "masks": [mask],
        },
        image_id=0,
    )

    assert evaluator.results[0]["area"] == 3.0


@pytest.mark.unit
def test_segmentation_validator_updates_bbox_and_mask_evaluators():
    from types import SimpleNamespace

    from libreyolo.validation.detection_validator import SegmentationValidator

    validator = SegmentationValidator.__new__(SegmentationValidator)
    validator.bbox_evaluator = _DummyEvaluator()
    validator.mask_evaluator = _DummyEvaluator()

    pred = {"boxes": [], "scores": [], "classes": [], "masks": []}
    validator._update_metrics([pred], targets=None, img_info=[], img_ids=[123])

    assert validator.bbox_evaluator.updates == [(pred, 123)]
    assert validator.mask_evaluator.updates == [(pred, 123)]

    validator.config = SimpleNamespace(verbose=False, save_json=False)
    validator.save_dir = None
    validator.bbox_evaluator = _DummyEvaluator(_coco_metrics(0.2))
    validator.mask_evaluator = _DummyEvaluator(_coco_metrics(0.7))

    metrics = validator._compute_metrics()

    assert metrics["metrics/mAP50-95"] == pytest.approx(0.7)
    assert metrics["metrics/precision(B)"] == pytest.approx(0.201)
    assert metrics["metrics/recall(B)"] == pytest.approx(0.202)
    assert metrics["metrics/mAP50-95(B)"] == pytest.approx(0.2)
    assert metrics["metrics/precision(M)"] == pytest.approx(0.701)
    assert metrics["metrics/recall(M)"] == pytest.approx(0.702)
    assert metrics["metrics/mAP50-95(M)"] == pytest.approx(0.7)
