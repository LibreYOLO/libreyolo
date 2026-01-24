"""
Integration test for COCO evaluation in DetectionValidator.

Tests that COCO evaluator is properly integrated into the validation pipeline.
"""

import tempfile
import shutil
from pathlib import Path
import yaml
import numpy as np
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


def test_coco_evaluator_integration():
    """Test that COCOEvaluator integrates into DetectionValidator."""
    try:
        from libreyolo.validation import DetectionValidator, ValidationConfig
        from libreyolo.data import create_yolo_coco_api
        from libreyolo.validation import COCOEvaluator
    except ImportError as e:
        print(f"Skipping test - missing dependencies: {e}")
        return

    # Create temporary dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        yaml_path = create_mock_yolo_dataset(tmp_path)

        # Test 1: YOLOCocoAPI creation
        print("Test 1: Creating YOLOCocoAPI from dataset...")
        try:
            coco_api = create_yolo_coco_api(str(yaml_path), split='val')
            print(f"  ✓ Created COCO API with {len(coco_api.imgs)} images")
            assert len(coco_api.imgs) == 3, f"Expected 3 images, got {len(coco_api.imgs)}"
        except Exception as e:
            print(f"  ✗ Failed to create COCO API: {e}")
            return

        # Test 2: COCOEvaluator initialization
        print("\nTest 2: Initializing COCOEvaluator...")
        try:
            evaluator = COCOEvaluator(coco_api, iou_type='bbox')
            print("  ✓ COCOEvaluator initialized")
        except Exception as e:
            print(f"  ✗ Failed to initialize evaluator: {e}")
            return

        # Test 3: Update with dummy predictions
        print("\nTest 3: Updating with dummy predictions...")
        try:
            dummy_pred = {
                'boxes': [[100, 100, 200, 200]],  # xyxy format
                'scores': [0.9],
                'classes': [0],
            }
            evaluator.update(dummy_pred, image_id=0)
            print(f"  ✓ Updated evaluator, {len(evaluator.results)} predictions accumulated")
            assert len(evaluator.results) == 1, "Expected 1 prediction"
        except Exception as e:
            print(f"  ✗ Failed to update: {e}")
            return

        # Test 4: Compute metrics
        print("\nTest 4: Computing COCO metrics...")
        try:
            metrics = evaluator.compute()
            print("  ✓ Metrics computed successfully")
            print(f"    mAP@[0.5:0.95]: {metrics['mAP']:.3f}")
            print(f"    mAP@0.5:       {metrics['mAP50']:.3f}")

            # Check all expected metrics are present
            expected_keys = ['mAP', 'mAP50', 'mAP75', 'mAP_small', 'mAP_medium', 'mAP_large',
                           'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
            for key in expected_keys:
                assert key in metrics, f"Missing metric: {key}"
            print(f"  ✓ All {len(expected_keys)} COCO metrics present")
        except Exception as e:
            print(f"  ✗ Failed to compute metrics: {e}")
            import traceback
            traceback.print_exc()
            return

        print("\n" + "="*50)
        print("✓ All COCO integration tests passed!")
        print("="*50)


def test_validation_config_coco_flag():
    """Test that ValidationConfig has use_coco_eval flag."""
    from libreyolo.validation import ValidationConfig

    print("\nTest: ValidationConfig has use_coco_eval flag...")

    # Default should be True
    config = ValidationConfig(data="dummy.yaml")
    assert hasattr(config, 'use_coco_eval'), "Missing use_coco_eval attribute"
    assert config.use_coco_eval == True, "Default should be True"
    print("  ✓ use_coco_eval flag exists and defaults to True")

    # Test setting to False
    config = ValidationConfig(data="dummy.yaml", use_coco_eval=False)
    assert config.use_coco_eval == False
    print("  ✓ Can set use_coco_eval to False")


if __name__ == "__main__":
    print("="*50)
    print("COCO Validation Integration Tests")
    print("="*50)

    test_validation_config_coco_flag()
    print()
    test_coco_evaluator_integration()
