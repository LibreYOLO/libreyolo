"""
Unit tests for COCO evaluation in DetectionValidator.

Tests that COCO evaluator is properly integrated into the validation pipeline
using mock data (no GPU or real datasets needed).
"""

import pytest
import numpy as np
import yaml
from pathlib import Path
from PIL import Image


def create_mock_yolo_dataset(tmp_path):
    """Create a minimal mock YOLO dataset for testing."""
    # Create directory structure
    images_dir = tmp_path / "images" / "val"
    labels_dir = tmp_path / "labels" / "val"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create 3 dummy images
    for i in range(3):
        img = Image.new('RGB', (640, 640), color=(i*50, i*50, i*50))
        img.save(images_dir / f"img{i}.jpg")

        # Create corresponding label
        # Format: class cx cy w h (normalized)
        labels = [
            f"0 0.5 0.5 0.3 0.3\n",  # Center object
            f"1 0.25 0.25 0.2 0.2\n",  # Top-left object
        ]
        (labels_dir / f"img{i}.txt").write_text("".join(labels[:i+1]))

    # Create data.yaml
    data_yaml = {
        'path': str(tmp_path),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': ['cat', 'dog']
    }

    yaml_path = tmp_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    return yaml_path


@pytest.mark.unit
def test_coco_evaluator_integration(tmp_path):
    """Test that COCOEvaluator integrates into DetectionValidator."""
    from libreyolo.validation import DetectionValidator, ValidationConfig
    from libreyolo.data import create_yolo_coco_api
    from libreyolo.validation import COCOEvaluator

    yaml_path = create_mock_yolo_dataset(tmp_path)

    # YOLOCocoAPI creation
    coco_api = create_yolo_coco_api(str(yaml_path), split='val')
    assert len(coco_api.imgs) == 3

    # COCOEvaluator initialization
    evaluator = COCOEvaluator(coco_api, iou_type='bbox')

    # Update with dummy predictions
    dummy_pred = {
        'boxes': [[100, 100, 200, 200]],
        'scores': [0.9],
        'classes': [0],
    }
    evaluator.update(dummy_pred, image_id=0)
    assert len(evaluator.results) == 1

    # Compute metrics
    metrics = evaluator.compute()
    expected_keys = [
        'mAP', 'mAP50', 'mAP75', 'mAP_small', 'mAP_medium', 'mAP_large',
        'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large',
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"


@pytest.mark.unit
def test_validation_config_coco_flag():
    """Test that ValidationConfig has use_coco_eval flag."""
    from libreyolo.validation import ValidationConfig

    config = ValidationConfig(data="dummy.yaml")
    assert hasattr(config, 'use_coco_eval')
    assert config.use_coco_eval is True

    config = ValidationConfig(data="dummy.yaml", use_coco_eval=False)
    assert config.use_coco_eval is False
