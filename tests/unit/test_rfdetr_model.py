"""
Unit tests for RF-DETR model integration.

Tests model initialization, preprocessing, forward pass, and postprocessing
without requiring actual weights or expensive inference.
"""

import types
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

# Skip all tests if rfdetr is not installed
pytest.importorskip("rfdetr", reason="rfdetr package not installed")

from libreyolo.rfdetr import LIBREYOLORFDETR
from libreyolo.rfdetr.utils import box_cxcywh_to_xyxy, postprocess
from libreyolo.rfdetr.nn import RFDETRModel, RFDETR_CONFIGS

pytestmark = pytest.mark.unit


def build_rfdetr_stub(size="b", nb_classes=80):
    """Create a lightweight RF-DETR instance that skips weight loading."""
    inst = LIBREYOLORFDETR.__new__(LIBREYOLORFDETR)
    inst.size = size
    inst.nb_classes = nb_classes
    inst.device = torch.device("cpu")

    # Get resolution from config
    config_cls = RFDETR_CONFIGS[size]
    inst.resolution = config_cls().resolution
    inst._pretrain_weights = None

    # Create a minimal mock model
    class DummyRFDETRModel:
        def __init__(self):
            self.model = self

        def __call__(self, x):
            # Return DETR-like output
            batch_size = x.shape[0]
            num_queries = 300
            return {
                'pred_logits': torch.randn(batch_size, num_queries, nb_classes),
                'pred_boxes': torch.rand(batch_size, num_queries, 4),  # cxcywh in [0, 1]
            }

        def eval(self):
            return self

        def to(self, device):
            return self

    inst.model = DummyRFDETRModel()
    return inst


class TestRFDETRModelInit:
    """Test RF-DETR model initialization."""

    @pytest.mark.parametrize("size", ["n", "s", "b", "m", "l"])
    def test_valid_sizes(self, size):
        """Test that all valid sizes can be initialized."""
        model = build_rfdetr_stub(size=size)
        assert model.size == size
        assert model._get_valid_sizes() == ["n", "s", "b", "m", "l"]

    def test_invalid_size_raises(self):
        """Test that invalid size raises ValueError."""
        with pytest.raises(ValueError, match="Invalid size"):
            # This will fail in __init__ before we can stub
            LIBREYOLORFDETR(size="invalid", model_path={})

    def test_model_name(self):
        """Test that model name is correct."""
        model = build_rfdetr_stub()
        assert model._get_model_name() == "LIBREYOLORFDETR"

    @pytest.mark.parametrize("size,expected_resolution", [
        ("n", 384),
        ("s", 384),
        ("b", 384),
        ("m", 512),
        ("l", 512),
    ])
    def test_input_sizes(self, size, expected_resolution):
        """Test that input sizes match RF-DETR configs."""
        model = build_rfdetr_stub(size=size)
        assert model._get_input_size() == expected_resolution


class TestRFDETRPreprocessing:
    """Test RF-DETR preprocessing pipeline."""

    def test_preprocess_pil_image(self):
        """Test preprocessing of PIL image."""
        model = build_rfdetr_stub()
        img = Image.new("RGB", (640, 480), color="red")

        tensor, orig_img, orig_size = model._preprocess(img)

        # Check output types
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(orig_img, Image.Image)
        assert isinstance(orig_size, tuple)

        # Check shapes
        assert tensor.shape == (1, 3, model.resolution, model.resolution)
        assert orig_size == (480, 640)  # (height, width)

    def test_preprocess_numpy_array(self):
        """Test preprocessing of numpy array."""
        model = build_rfdetr_stub()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        tensor, orig_img, orig_size = model._preprocess(img)

        assert tensor.shape == (1, 3, model.resolution, model.resolution)
        assert orig_size == (480, 640)

    def test_preprocess_normalization(self):
        """Test that ImageNet normalization is applied."""
        model = build_rfdetr_stub()
        # Create a white image (255, 255, 255)
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))

        tensor, _, _ = model._preprocess(img)

        # After normalization, values should not be in [0, 1] range
        # ImageNet normalization shifts values
        assert tensor.min() < 0 or tensor.max() > 1


class TestRFDETRPostprocessing:
    """Test RF-DETR postprocessing utilities."""

    def test_box_cxcywh_to_xyxy(self):
        """Test box format conversion."""
        # Center (0.5, 0.5), size (0.2, 0.2)
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])

        xyxy = box_cxcywh_to_xyxy(boxes)

        expected = torch.tensor([[0.4, 0.4, 0.6, 0.6]])
        assert torch.allclose(xyxy, expected)

    def test_postprocess_output_format(self):
        """Test that postprocess returns correct format."""
        # Mock DETR output
        batch_size = 2
        num_queries = 300
        num_classes = 80

        outputs = {
            'pred_logits': torch.randn(batch_size, num_queries, num_classes),
            'pred_boxes': torch.rand(batch_size, num_queries, 4),
        }
        target_sizes = torch.tensor([[480, 640], [720, 1280]])  # (H, W)

        results = postprocess(outputs, target_sizes, num_select=100)

        assert len(results) == batch_size
        for result in results:
            assert 'scores' in result
            assert 'labels' in result
            assert 'boxes' in result
            assert result['scores'].shape[0] == 100
            assert result['labels'].shape[0] == 100
            assert result['boxes'].shape == (100, 4)

    def test_postprocess_box_scaling(self):
        """Test that boxes are scaled to image coordinates."""
        # Single image with known size
        outputs = {
            'pred_logits': torch.ones(1, 100, 80) * 10,  # High confidence
            'pred_boxes': torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]).repeat(1, 100, 1),
        }
        target_sizes = torch.tensor([[480, 640]])  # H=480, W=640

        results = postprocess(outputs, target_sizes, num_select=10)
        boxes = results[0]['boxes']

        # Boxes should be in absolute coordinates
        assert boxes[:, 0].max() <= 640  # x coordinates
        assert boxes[:, 1].max() <= 480  # y coordinates
        assert boxes[:, 2].max() <= 640  # x coordinates
        assert boxes[:, 3].max() <= 480  # y coordinates


class TestRFDETRForward:
    """Test RF-DETR forward pass."""

    def test_forward_output_format(self):
        """Test that forward pass returns DETR output format."""
        model = build_rfdetr_stub()
        input_tensor = torch.randn(1, 3, model.resolution, model.resolution)

        output = model._forward(input_tensor)

        assert 'pred_logits' in output
        assert 'pred_boxes' in output
        assert output['pred_logits'].dim() == 3  # (B, queries, classes)
        assert output['pred_boxes'].dim() == 3   # (B, queries, 4)


class TestRFDETRInference:
    """Test end-to-end inference pipeline."""

    def test_predict_output_schema(self):
        """Test that predict returns correct output schema."""
        model = build_rfdetr_stub()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = model._predict_single(
            img,
            save=False,
            conf_thres=0.5,
            iou_thres=0.45,  # Ignored for RF-DETR but accepted
        )

        # Check all required keys
        assert "boxes" in detections
        assert "scores" in detections
        assert "classes" in detections
        assert "num_detections" in detections

        # Check consistency
        assert detections["num_detections"] == len(detections["boxes"])
        assert len(detections["boxes"]) == len(detections["scores"])
        assert len(detections["scores"]) == len(detections["classes"])

    def test_confidence_threshold_filtering(self):
        """Test that confidence threshold filters detections."""
        model = build_rfdetr_stub()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # High threshold should give fewer detections
        high_thresh = model._predict_single(img, save=False, conf_thres=0.9)
        low_thresh = model._predict_single(img, save=False, conf_thres=0.1)

        assert high_thresh["num_detections"] <= low_thresh["num_detections"]


class TestRFDETRAvailableLayers:
    """Test layer access for XAI/visualization."""

    def test_get_available_layers(self):
        """Test that model exposes layers for inspection."""
        model = build_rfdetr_stub()
        layers = model._get_available_layers()

        assert isinstance(layers, dict)
        # RF-DETR may not have all layers in stub, but should return dict


class TestRFDETRConfigs:
    """Test RF-DETR configuration classes."""

    def test_all_configs_exist(self):
        """Test that all size configs are available."""
        assert 'n' in RFDETR_CONFIGS
        assert 's' in RFDETR_CONFIGS
        assert 'b' in RFDETR_CONFIGS
        assert 'm' in RFDETR_CONFIGS
        assert 'l' in RFDETR_CONFIGS

    def test_config_has_required_attributes(self):
        """Test that configs have required attributes."""
        for size, config_cls in RFDETR_CONFIGS.items():
            config = config_cls()
            assert hasattr(config, 'resolution')
            assert hasattr(config, 'hidden_dim')
            assert hasattr(config, 'num_queries')
            assert config.resolution > 0
            assert config.hidden_dim > 0
            assert config.num_queries > 0
