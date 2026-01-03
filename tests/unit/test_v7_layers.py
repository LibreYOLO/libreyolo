"""Unit tests for YOLOv7 layers."""
import pytest
import torch

from libreyolo.v7.nn import (
    Conv, RepConv, SPPCSPC, MP, ImplicitA, ImplicitM,
    IDetect, Backbone7, Neck7, LibreYOLO7Model
)
from libreyolo.v7 import utils as v7_utils

pytestmark = pytest.mark.unit


class TestV7ConvLayers:
    """Test basic convolution layers."""

    def test_conv_forward(self):
        """Test Conv layer forward pass."""
        layer = Conv(3, 64, k=3, s=1)
        x = torch.randn(1, 3, 64, 64)
        out = layer(x)
        assert out.shape == (1, 64, 64, 64)

    def test_conv_stride(self):
        """Test Conv with stride 2 downsamples correctly."""
        layer = Conv(64, 128, k=3, s=2)
        x = torch.randn(1, 64, 64, 64)
        out = layer(x)
        assert out.shape == (1, 128, 32, 32)

    def test_repconv_forward(self):
        """Test RepConv layer forward pass."""
        layer = RepConv(64, 64, k=3, s=1)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)


class TestV7Pooling:
    """Test pooling layers."""

    def test_mp_forward(self):
        """Test Max Pooling layer forward pass."""
        layer = MP(k=2)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 16, 16)


class TestV7SPPCSPC:
    """Test SPPCSPC module."""

    def test_sppcspc_forward(self):
        """Test SPPCSPC forward pass."""
        layer = SPPCSPC(256, 128)
        x = torch.randn(1, 256, 16, 16)
        out = layer(x)
        assert out.shape == (1, 128, 16, 16)


class TestV7ImplicitLayers:
    """Test Implicit layers."""

    def test_implicit_a_forward(self):
        """Test ImplicitA layer forward pass."""
        layer = ImplicitA(64)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)

    def test_implicit_m_forward(self):
        """Test ImplicitM layer forward pass."""
        layer = ImplicitM(64)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)


class TestV7DetectionHead:
    """Test detection head components."""

    def test_idetect_forward(self):
        """Test IDetect head forward pass."""
        layer = IDetect(
            nc=80,
            anchors=[
                [12, 16, 19, 36, 40, 28],
                [36, 75, 76, 55, 72, 146],
                [142, 110, 192, 243, 459, 401],
            ],
            ch=(128, 256, 512)
        )
        x = [
            torch.randn(1, 128, 80, 80),
            torch.randn(1, 256, 40, 40),
            torch.randn(1, 512, 20, 20),
        ]
        out = layer(x)
        # In training mode, returns list of predictions per scale
        assert len(out) == 3
        # Each output has shape (batch, anchors, grid_h, grid_w, 5+nc)
        assert out[0].shape[0] == 1
        assert out[0].shape[-1] == 85  # 5 + 80


class TestV7FullModel:
    """Test full model architecture."""

    def test_backbone_forward(self):
        """Test Backbone7 forward pass."""
        backbone = Backbone7()
        x = torch.randn(1, 3, 640, 640)
        p3, p4, p5 = backbone(x)
        assert p3.shape[2] == 80  # 640 / 8
        assert p4.shape[2] == 40  # 640 / 16
        assert p5.shape[2] == 20  # 640 / 32

    def test_neck_forward(self):
        """Test Neck7 forward pass."""
        neck = Neck7()
        # Use correct input channel sizes for base config
        p3 = torch.randn(1, 512, 80, 80)  # B3 output
        p4 = torch.randn(1, 1024, 40, 40)  # B4 output
        p5 = torch.randn(1, 1024, 20, 20)  # B5 output
        n3, n4, n5 = neck(p3, p4, p5)
        assert n3.shape[2] == 80
        assert n4.shape[2] == 40
        assert n5.shape[2] == 20

    def test_full_model_forward(self):
        """Test full LibreYOLO7Model forward pass."""
        model = LibreYOLO7Model(config='base', nb_classes=80)
        model.eval()  # Set to eval mode to get dict output
        x = torch.randn(1, 3, 640, 640)
        out = model(x)
        # In eval mode, returns dict with 'predictions' key
        assert isinstance(out, dict)
        assert 'predictions' in out


class TestV7Utils:
    """Test utility functions."""

    def test_preprocess_image(self):
        """Test image preprocessing."""
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor, original_img, original_size = v7_utils.preprocess_image(img, input_size=640)
        assert tensor.shape == (1, 3, 640, 640)
        assert original_size == (100, 100)

    def test_nms(self):
        """Test non-maximum suppression.
        
        nms(boxes, scores, iou_threshold) expects separate boxes and scores tensors.
        """
        boxes = torch.tensor([
            [0, 0, 10, 10],
            [1, 1, 11, 11],  # Overlapping, should be suppressed
            [100, 100, 110, 110],  # Non-overlapping
        ], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8, 0.7])
        result = v7_utils.nms(boxes, scores, iou_threshold=0.5)
        assert len(result) == 2  # Should keep 2 boxes

    def test_xywh2xyxy(self):
        """Test xywh to xyxy conversion."""
        xywh = torch.tensor([[50.0, 50.0, 20.0, 20.0]])
        xyxy = v7_utils.xywh2xyxy(xywh)
        expected = torch.tensor([[40.0, 40.0, 60.0, 60.0]])
        assert torch.allclose(xyxy, expected)
