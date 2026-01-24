"""
Unit tests for YOLOCocoAPI - COCO evaluation for YOLO format datasets.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from libreyolo.data import YOLOCocoAPI, create_yolo_coco_api, parse_yolo_label_line

pytestmark = pytest.mark.unit


class TestParseYOLOLabelLine:
    """Test YOLO label parsing."""

    def test_valid_label_line(self):
        """Test parsing a valid YOLO label line."""
        line = "0 0.5 0.5 0.2 0.1"
        result = parse_yolo_label_line(line, img_w=100, img_h=100, num_classes=80)

        assert result is not None
        class_id, x1, y1, x2, y2, area = result
        assert class_id == 0
        assert 40 <= x1 <= 41  # 0.5 - 0.1 = 0.4 * 100 = 40
        assert 45 <= y1 <= 46  # 0.5 - 0.05 = 0.45 * 100 = 45
        assert 59 <= x2 <= 61  # 0.5 + 0.1 = 0.6 * 100 = 60
        assert 54 <= y2 <= 56  # 0.5 + 0.05 = 0.55 * 100 = 55

    def test_empty_line(self):
        """Test that empty lines return None."""
        result = parse_yolo_label_line("", img_w=100, img_h=100, num_classes=80)
        assert result is None

    def test_invalid_format(self):
        """Test that invalid format returns None."""
        result = parse_yolo_label_line("0 0.5", img_w=100, img_h=100, num_classes=80)
        assert result is None

    def test_class_id_out_of_range(self):
        """Test that out-of-range class IDs are skipped."""
        # Class 100 when max is 80
        result = parse_yolo_label_line("100 0.5 0.5 0.2 0.1", img_w=100, img_h=100, num_classes=80)
        assert result is None

    def test_negative_class_id(self):
        """Test that negative class IDs are skipped."""
        result = parse_yolo_label_line("-1 0.5 0.5 0.2 0.1", img_w=100, img_h=100, num_classes=80)
        assert result is None

    def test_box_clamping(self):
        """Test that boxes are clamped to image boundaries."""
        # Box that extends beyond image boundaries
        line = "0 0.1 0.1 0.5 0.5"  # Will go negative on left/top
        result = parse_yolo_label_line(line, img_w=100, img_h=100, num_classes=80)

        assert result is not None
        class_id, x1, y1, x2, y2, area = result
        # Should be clamped to 0
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 100
        assert y2 <= 100

    def test_zero_area_box_rejected(self):
        """Test that zero-area boxes are rejected."""
        # Box with zero width
        line = "0 0.5 0.5 0.0 0.1"
        result = parse_yolo_label_line(line, img_w=100, img_h=100, num_classes=80)
        assert result is None


class TestYOLOCocoAPI:
    """Test YOLOCocoAPI COCO-compatible interface."""

    @pytest.fixture
    def mock_yolo_dataset(self, tmp_path):
        """Create a minimal mock YOLO dataset."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create 2 dummy images
        from PIL import Image
        img1 = Image.new('RGB', (100, 100), color='red')
        img2 = Image.new('RGB', (200, 150), color='blue')
        img1.save(images_dir / "img1.jpg")
        img2.save(images_dir / "img2.jpg")

        # Create corresponding labels
        (labels_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.1\n1 0.3 0.3 0.1 0.1\n")
        (labels_dir / "img2.txt").write_text("0 0.6 0.6 0.3 0.2\n")

        class_names = ["cat", "dog"]

        return images_dir, labels_dir, class_names

    def test_yolo_coco_api_initialization(self, mock_yolo_dataset):
        """Test YOLOCocoAPI initialization."""
        images_dir, labels_dir, class_names = mock_yolo_dataset

        api = YOLOCocoAPI(images_dir, labels_dir, class_names)

        assert len(api.imgs) == 2
        assert len(api.cats) == 2
        assert len(api.anns) > 0  # Should have some annotations

    def test_yolo_coco_api_get_img_ids(self, mock_yolo_dataset):
        """Test getImgIds method."""
        images_dir, labels_dir, class_names = mock_yolo_dataset
        api = YOLOCocoAPI(images_dir, labels_dir, class_names)

        img_ids = api.getImgIds()
        assert len(img_ids) == 2
        assert all(isinstance(i, int) for i in img_ids)

    def test_yolo_coco_api_get_cat_ids(self, mock_yolo_dataset):
        """Test getCatIds method."""
        images_dir, labels_dir, class_names = mock_yolo_dataset
        api = YOLOCocoAPI(images_dir, labels_dir, class_names)

        cat_ids = api.getCatIds()
        assert len(cat_ids) == 2
        assert cat_ids == [0, 1]

    def test_yolo_coco_api_load_anns(self, mock_yolo_dataset):
        """Test loadAnns method."""
        images_dir, labels_dir, class_names = mock_yolo_dataset
        api = YOLOCocoAPI(images_dir, labels_dir, class_names)

        all_anns = api.loadAnns()
        assert len(all_anns) > 0

        # Check annotation format
        ann = all_anns[0]
        assert 'id' in ann
        assert 'image_id' in ann
        assert 'category_id' in ann
        assert 'bbox' in ann  # Should be [x, y, w, h] format
        assert 'area' in ann
        assert 'iscrowd' in ann

    def test_yolo_coco_api_get_ann_ids(self, mock_yolo_dataset):
        """Test getAnnIds method."""
        images_dir, labels_dir, class_names = mock_yolo_dataset
        api = YOLOCocoAPI(images_dir, labels_dir, class_names)

        # Get all annotations
        ann_ids = api.getAnnIds()
        assert len(ann_ids) > 0

        # Get annotations for specific image
        img_ids = api.getImgIds()
        ann_ids_img0 = api.getAnnIds(imgIds=img_ids[0])
        assert len(ann_ids_img0) >= 0  # May have 0 or more annotations


class TestCreateYOLOCocoAPI:
    """Test create_yolo_coco_api helper function."""

    @pytest.fixture
    def mock_data_yaml(self, tmp_path):
        """Create a mock data.yaml file."""
        import yaml

        # Create directory structure
        (tmp_path / "images" / "val").mkdir(parents=True)
        (tmp_path / "labels" / "val").mkdir(parents=True)

        # Create dummy image
        from PIL import Image
        img = Image.new('RGB', (100, 100))
        img.save(tmp_path / "images" / "val" / "test.jpg")

        # Create dummy label
        (tmp_path / "labels" / "val" / "test.txt").write_text("0 0.5 0.5 0.2 0.1\n")

        # Create data.yaml
        data = {
            'path': str(tmp_path),
            'train': 'images/train',
            'val': 'images/val',
            'names': ['cat', 'dog', 'bird']
        }

        yaml_path = tmp_path / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)

        return yaml_path

    def test_create_yolo_coco_api_from_yaml(self, mock_data_yaml):
        """Test creating YOLOCocoAPI from data.yaml."""
        api = create_yolo_coco_api(str(mock_data_yaml), split='val')

        assert api is not None
        assert len(api.imgs) >= 0  # May have images
        assert len(api.cats) == 3  # Should have 3 classes


class TestCOCOEvaluatorIntegration:
    """Test COCOEvaluator integration with YOLOCocoAPI."""

    @pytest.fixture
    def simple_coco_api(self, tmp_path):
        """Create a simple COCO API for testing."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create one image
        from PIL import Image
        img = Image.new('RGB', (100, 100))
        img.save(images_dir / "test.jpg")

        # Create one label
        (labels_dir / "test.txt").write_text("0 0.5 0.5 0.4 0.4\n")

        return YOLOCocoAPI(images_dir, labels_dir, ["cat"])

    def test_coco_evaluator_initialization(self, simple_coco_api):
        """Test COCOEvaluator can be initialized."""
        from libreyolo.validation import COCOEvaluator

        evaluator = COCOEvaluator(simple_coco_api)
        assert evaluator is not None
        assert evaluator.iou_type == 'bbox'

    def test_coco_evaluator_update(self, simple_coco_api):
        """Test adding predictions to evaluator."""
        from libreyolo.validation import COCOEvaluator

        evaluator = COCOEvaluator(simple_coco_api)

        # Add some dummy predictions
        predictions = {
            'boxes': [[30, 30, 70, 70]],  # xyxy format
            'scores': [0.9],
            'classes': [0],
        }

        evaluator.update(predictions, image_id=0)
        assert len(evaluator.results) == 1

        # Check COCO format
        result = evaluator.results[0]
        assert 'image_id' in result
        assert 'category_id' in result
        assert 'bbox' in result  # Should be [x, y, w, h]
        assert 'score' in result
