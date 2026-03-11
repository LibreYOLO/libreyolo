"""Unit tests for RTDETR model.

Tests for RTDETR model registry, weight loading, class detection, and configuration.
"""

import numpy as np
import pytest
import torch

from libreyolo.models.rtdetr.config import RTDETRConfig
from libreyolo.models.rtdetr.model import LibreYOLORTDETR
from libreyolo.validation.preprocessors import RTDETRValPreprocessor

pytestmark = pytest.mark.unit


class TestRTDETRRegistry:
    """Test RTDETR model registration."""

    def test_rtdetr_is_registered(self):
        """LibreYOLORTDETR should be in the BaseModel registry."""
        from libreyolo.models.base.model import BaseModel

        assert LibreYOLORTDETR in BaseModel._registry or any(
            cls.__name__ == "LibreYOLORTDETR" for cls in BaseModel._registry
        )


class TestRTDETRCanLoad:
    """Test RTDETR weight key detection."""

    def test_rtdetr_can_load_with_rtdetr_keys(self):
        """can_load() should return True for RTDETR-specific weight keys."""
        fake_weights = {
            "backbone.res_layers.0.blocks.0.conv1.weight": torch.zeros(64, 3, 3, 3),
            "encoder.input_proj.0.0.weight": torch.zeros(256, 512, 1, 1),
            "decoder.dec_score_head.0.weight": torch.zeros(80, 256),
            "decoder.dec_score_head.0.bias": torch.zeros(80),
        }
        assert LibreYOLORTDETR.can_load(fake_weights) is True

    def test_rtdetr_cannot_load_rfdetr_keys(self):
        """can_load() should return False for RF-DETR weight keys."""
        fake_rfdetr_weights = {
            "model.backbone.0.body.layer1.0.conv1.weight": torch.zeros(64, 64, 1, 1),
            "model.encoder_projection.weight": torch.zeros(256, 512),
        }
        assert LibreYOLORTDETR.can_load(fake_rfdetr_weights) is False


class TestRTDETRDetectNbClasses:
    """Test RTDETR class count detection."""

    def test_rtdetr_detect_nb_classes(self):
        """detect_nb_classes() should read from dec_score_head bias."""
        fake_weights = {
            "decoder.dec_score_head.0.bias": torch.zeros(80),
            "decoder.dec_score_head.1.bias": torch.zeros(80),
            "decoder.dec_score_head.2.bias": torch.zeros(80),
        }
        assert LibreYOLORTDETR.detect_nb_classes(fake_weights) == 80

    def test_rtdetr_detect_nb_classes_custom(self):
        """detect_nb_classes() should work with non-COCO class counts."""
        fake_weights = {
            "decoder.dec_score_head.0.bias": torch.zeros(10),
        }
        assert LibreYOLORTDETR.detect_nb_classes(fake_weights) == 10


class TestRTDETRDetectSizeFromFilename:
    """Test RTDETR size detection from filename."""

    def test_rtdetr_detect_size_from_filename(self):
        """detect_size_from_filename() should handle multi-char size codes."""
        assert LibreYOLORTDETR.detect_size_from_filename("rtdetr-r50.pt") == "r50"
        assert LibreYOLORTDETR.detect_size_from_filename("rtdetr-r50m.pt") == "r50m"
        assert LibreYOLORTDETR.detect_size_from_filename("rtdetr-r18.pt") == "r18"
        assert LibreYOLORTDETR.detect_size_from_filename("rtdetr-r101.pt") == "r101"


class TestRTDETRModelInstantiation:
    """Test RTDETR model creation."""

    def test_rtdetr_model_instantiation(self):
        """LibreYOLORTDETR should be instantiatable from scratch."""
        model = LibreYOLORTDETR(nb_classes=80, size="r18")
        assert model is not None
        assert model.nb_classes == 80
        assert model.size == "r18"


class TestRTDETRValPreprocessor:
    """Test RTDETR validation preprocessor."""

    def test_rtdetr_val_preprocessor(self):
        """RTDETRValPreprocessor should resize and normalize correctly."""
        preprocessor = RTDETRValPreprocessor(img_size=(640, 640))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Empty targets for testing
        targets = np.zeros((0, 5), dtype=np.float32)
        result, padded_targets = preprocessor(img, targets, (640, 640))
        assert result.shape == (3, 640, 640)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert padded_targets.shape == (120, 5)  # max_labels, 5


class TestRTDETRExportMetadata:
    """Test RTDETR export metadata."""

    def test_rtdetr_export_metadata(self):
        """RTDETR model family should be 'rtdetr'."""
        model = LibreYOLORTDETR(nc=80, size="r18")
        assert model.FAMILY == "rtdetr"


class TestRTDETRConfig:
    """Test RTDETR training configuration."""

    def test_rtdetr_config_defaults(self):
        """RTDETRConfig should have RTDETR-specific defaults."""
        config = RTDETRConfig(data="dummy.yaml")
        assert config.lr_backbone == 0.00001
        assert config.betas == (0.9, 0.999)
