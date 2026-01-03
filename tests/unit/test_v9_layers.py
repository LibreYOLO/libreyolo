"""Unit tests for YOLOv9 layers."""
import pytest
import torch

from libreyolo.v9.nn import (
    Conv, RepConvN, Bottleneck, RepNBottleneck, RepNCSP,
    ELAN, RepNCSPELAN, AConv, ADown, SPPELAN, Concat,
    DFL, DDetect, Backbone9, Neck9, LibreYOLO9Model
)
from libreyolo.v9 import utils as v9_utils

pytestmark = pytest.mark.unit


class TestV9ConvLayers:
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

    def test_repconvn_forward(self):
        """Test RepConvN layer forward pass."""
        layer = RepConvN(64, 64, k=3, s=1)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)


class TestV9Bottlenecks:
    """Test bottleneck modules."""

    def test_bottleneck_forward(self):
        """Test Bottleneck forward pass."""
        layer = Bottleneck(64, 64)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)

    def test_repn_bottleneck_forward(self):
        """Test RepNBottleneck forward pass."""
        layer = RepNBottleneck(64, 64)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)

    def test_repn_csp_forward(self):
        """Test RepNCSP forward pass."""
        layer = RepNCSP(64, 64, n=1)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)


class TestV9ELANBlocks:
    """Test ELAN-based blocks."""

    def test_elan_forward(self):
        """Test ELAN forward pass."""
        layer = ELAN(64, 128, n=2)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 128, 32, 32)

    def test_repncspelan_forward(self):
        """Test RepNCSPELAN forward pass."""
        layer = RepNCSPELAN(64, 128, c3=64, c4=64, n=1)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 128, 32, 32)


class TestV9Downsampling:
    """Test downsampling layers."""

    def test_aconv_forward(self):
        """Test AConv (Average Convolution) forward pass."""
        layer = AConv(64, 128)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 128, 16, 16)

    def test_adown_forward(self):
        """Test ADown forward pass."""
        layer = ADown(64, 128)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 128, 16, 16)


class TestV9SPPELAN:
    """Test SPP-ELAN module."""

    def test_sppelan_forward(self):
        """Test SPPELAN forward pass."""
        layer = SPPELAN(256, 256, c_hidden=128)
        x = torch.randn(1, 256, 16, 16)
        out = layer(x)
        assert out.shape == (1, 256, 16, 16)


class TestV9Concat:
    """Test Concat layer."""

    def test_concat_forward(self):
        """Test Concat layer forward pass."""
        layer = Concat(dimension=1)
        x1 = torch.randn(1, 64, 32, 32)
        x2 = torch.randn(1, 128, 32, 32)
        out = layer([x1, x2])
        assert out.shape == (1, 192, 32, 32)


class TestV9DetectionHead:
    """Test detection head components."""

    def test_dfl_forward(self):
        """Test DFL (Distribution Focal Loss) forward pass."""
        layer = DFL(c1=16)
        x = torch.randn(1, 16, 100)
        out = layer(x)
        assert out.shape == (1, 1, 100)

    def test_ddetect_forward(self):
        """Test DDetect head forward pass."""
        layer = DDetect(nc=80, ch=(64, 128, 256), reg_max=16)
        x = [
            torch.randn(1, 64, 80, 80),
            torch.randn(1, 128, 40, 40),
            torch.randn(1, 256, 20, 20),
        ]
        out = layer(x)
        # Output is (batch, 4*reg_max + nc, total_anchors) = (1, 144, 8400)
        assert out.shape[0] == 1
        assert out.shape[1] == 4 * 16 + 80  # 64 + 80 = 144


class TestV9FullModel:
    """Test full model architecture."""

    def test_backbone_forward(self):
        """Test Backbone9 forward pass."""
        backbone = Backbone9(config='t')
        x = torch.randn(1, 3, 640, 640)
        p3, p4, p5 = backbone(x)
        assert p3.shape[2] == 80  # 640 / 8
        assert p4.shape[2] == 40  # 640 / 16
        assert p5.shape[2] == 20  # 640 / 32

    def test_neck_forward(self):
        """Test Neck9 forward pass."""
        neck = Neck9(config='t')
        # Use correct input channel sizes for 't' config
        p3 = torch.randn(1, 64, 80, 80)
        p4 = torch.randn(1, 128, 40, 40)
        p5 = torch.randn(1, 256, 20, 20)
        n3, n4, n5 = neck(p3, p4, p5)
        assert n3.shape[2] == 80
        assert n4.shape[2] == 40
        assert n5.shape[2] == 20

    def test_full_model_forward(self):
        """Test full LibreYOLO9Model forward pass."""
        model = LibreYOLO9Model(config='t', nb_classes=80)
        x = torch.randn(1, 3, 640, 640)
        out = model(x)
        # Output should be (batch, 4*reg_max + nc, total_anchors)
        assert out.shape[0] == 1
        assert out.shape[1] == 4 * 16 + 80  # 144


class TestV9Utils:
    """Test utility functions."""

    def test_preprocess_image(self):
        """Test image preprocessing."""
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor, original_img, original_size = v9_utils.preprocess_image(img, input_size=640)
        assert tensor.shape == (1, 3, 640, 640)
        assert original_size == (100, 100)

    def test_make_anchors(self):
        """Test anchor generation."""
        feature_maps = [
            torch.randn(1, 64, 80, 80),
            torch.randn(1, 128, 40, 40),
            torch.randn(1, 256, 20, 20),
        ]
        anchors, strides = v9_utils.make_anchors(feature_maps, strides=[8, 16, 32])
        # Total anchors = 80*80 + 40*40 + 20*20 = 8400
        assert anchors.shape[1] == 8400
        assert strides.shape[1] == 8400
