"""Unit tests for YOLOv7 layers."""
import pytest
import torch

from libreyolo.v7.nn import (
    Conv, RepConv, Bottleneck, BottleneckCSP, ELAN, ELAN_H,
    SPPCSPC, MP, SP, Concat, ImplicitA, ImplicitM,
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


class TestV7Bottlenecks:
    """Test bottleneck modules."""

    def test_bottleneck_forward(self):
        """Test Bottleneck forward pass."""
        layer = Bottleneck(64, 64)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)

    def test_bottleneck_csp_forward(self):
        """Test BottleneckCSP forward pass."""
        layer = BottleneckCSP(64, 64, n=1)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)


class TestV7ELANBlocks:
    """Test ELAN-based blocks."""

    def test_elan_forward(self):
        """Test ELAN forward pass."""
        layer = ELAN(64, 256, c_hidden=64, n=2)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 256, 32, 32)

    def test_elan_h_forward(self):
        """Test ELAN-H forward pass."""
        layer = ELAN_H(256, 512, c_hidden=128, n=2)
        x = torch.randn(1, 256, 16, 16)
        out = layer(x)
        assert out.shape == (1, 512, 16, 16)


class TestV7Pooling:
    """Test pooling layers."""

    def test_mp_forward(self):
        """Test Max Pooling layer forward pass."""
        layer = MP(k=2)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 16, 16)

    def test_sp_forward(self):
        """Test SP (Spatial Pyramid) forward pass."""
        layer = SP(k=3)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)


class TestV7SPPCSPC:
    """Test SPPCSPC module."""

    def test_sppcspc_forward(self):
        """Test SPPCSPC forward pass."""
        layer = SPPCSPC(256, 128)
        x = torch.randn(1, 256, 16, 16)
        out = layer(x)
        assert out.shape == (1, 128, 16, 16)


class TestV7Concat:
    """Test Concat layer."""

    def test_concat_forward(self):
        """Test Concat layer forward pass."""
        layer = Concat(dimension=1)
        x1 = torch.randn(1, 64, 32, 32)
        x2 = torch.randn(1, 128, 32, 32)
        out = layer([x1, x2])
        assert out.shape == (1, 192, 32, 32)


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
                [[12, 16], [19, 36], [40, 28]],
                [[36, 75], [76, 55], [72, 146]],
                [[142, 110], [192, 243], [459, 401]],
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
        backbone = Backbone7(config='tiny')
        x = torch.randn(1, 3, 640, 640)
        p3, p4, p5 = backbone(x)
        assert p3.shape[2] == 80  # 640 / 8
        assert p4.shape[2] == 40  # 640 / 16
        assert p5.shape[2] == 20  # 640 / 32

    def test_neck_forward(self):
        """Test Neck7 forward pass."""
        neck = Neck7(config='tiny')
        # Use correct input channel sizes for 'tiny' config
        p3 = torch.randn(1, 128, 80, 80)
        p4 = torch.randn(1, 256, 40, 40)
        p5 = torch.randn(1, 512, 20, 20)
        n3, n4, n5 = neck(p3, p4, p5)
        assert n3.shape[2] == 80
        assert n4.shape[2] == 40
        assert n5.shape[2] == 20

    def test_full_model_forward(self):
        """Test full LibreYOLO7Model forward pass."""
        model = LibreYOLO7Model(config='tiny', nb_classes=80)
        x = torch.randn(1, 3, 640, 640)
        out = model(x)
        # Output is list of 3 scale predictions
        assert len(out) == 3


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
        """Test non-maximum suppression."""
        boxes = torch.tensor([
            [0, 0, 10, 10, 0.9, 0],
            [1, 1, 11, 11, 0.8, 0],  # Overlapping, should be suppressed
            [100, 100, 110, 110, 0.7, 0],  # Non-overlapping
        ])
        result = v7_utils.nms(boxes, iou_thres=0.5)
        assert len(result) == 2  # Should keep 2 boxes

    def test_xywh2xyxy(self):
        """Test xywh to xyxy conversion."""
        xywh = torch.tensor([[50, 50, 20, 20]])
        xyxy = v7_utils.xywh2xyxy(xywh)
        expected = torch.tensor([[40, 40, 60, 60]])
        assert torch.allclose(xyxy, expected)
