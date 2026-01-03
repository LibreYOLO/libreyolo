"""Unit tests for YOLO-RD layers."""
import pytest
import torch

from libreyolo.rd.nn import (
    Conv, RepConvN, Bottleneck, RepNBottleneck, RepNCSP,
    RepNCSPELAN, ADown, SPPELAN, Concat,
    DFL, DDetect, PONO, DConv, RepNCSPELAND,
    BackboneRD, NeckRD, LibreYOLORDModel
)
from libreyolo.rd import utils as rd_utils

pytestmark = pytest.mark.unit


class TestRDBaseLayers:
    """Test basic layers inherited from v9."""

    def test_conv_forward(self):
        """Test Conv layer forward pass."""
        layer = Conv(3, 64, k=3, s=1)
        x = torch.randn(1, 3, 64, 64)
        out = layer(x)
        assert out.shape == (1, 64, 64, 64)

    def test_repncspelan_forward(self):
        """Test RepNCSPELAN forward pass.
        
        RepNCSPELAN(c1, c2, c3, c4, n) where:
        - c1: input channels
        - c2: intermediate channels 1
        - c3: intermediate channels 2
        - c4: output channels
        """
        # Input: 64, cv1 output: 64, cv2/cv3 output: 32, final output: 128
        layer = RepNCSPELAN(64, 64, 32, 128, n=1)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 128, 32, 32)


class TestRDSpecificLayers:
    """Test YOLO-RD specific layers (DConv, PONO)."""

    def test_pono_forward(self):
        """Test PONO (Position-wise Channel Normalization) forward pass.
        
        PONO takes eps parameter (not channels) and returns normalized tensor only.
        """
        layer = PONO(eps=1e-5)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)

    def test_dconv_forward(self):
        """Test DConv (Dynamic Convolution) forward pass.
        
        DConv maintains input channels (in_channels -> in_channels).
        """
        layer = DConv(in_channels=64, atoms=64)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)

    def test_dconv_alpha_blending(self):
        """Test DConv alpha blending is in valid range."""
        layer = DConv(in_channels=64, atoms=64, alpha=0.3)
        assert 0 <= layer.alpha <= 1
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)

    def test_repncspelan_d_forward(self):
        """Test RepNCSPELAND (with DConv) forward pass.
        
        RepNCSPELAND(c1, c2, c3, c4, n, atoms) where c4 is output channels.
        """
        layer = RepNCSPELAND(64, 64, 32, 128, n=1, atoms=64)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 128, 32, 32)


class TestRDDownsampling:
    """Test downsampling layers."""

    def test_adown_forward(self):
        """Test ADown forward pass."""
        layer = ADown(64, 128)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 128, 16, 16)


class TestRDSPPELAN:
    """Test SPP-ELAN module."""

    def test_sppelan_forward(self):
        """Test SPPELAN forward pass.
        
        SPPELAN(c1, c2, neck_channels=None, k) where:
        - c1: input channels
        - c2: output channels
        - neck_channels: intermediate channels (default: c2 // 2)
        - k: pool kernel size
        """
        layer = SPPELAN(256, 256, neck_channels=128, k=5)
        x = torch.randn(1, 256, 16, 16)
        out = layer(x)
        assert out.shape == (1, 256, 16, 16)


class TestRDDetectionHead:
    """Test detection head components."""

    def test_dfl_forward(self):
        """Test DFL (Distribution Focal Loss) forward pass.
        
        DFL expects input shape (batch, 4*reg_max, anchors).
        """
        reg_max = 16
        layer = DFL(c1=reg_max)
        # Input: (batch, 4*reg_max, anchors)
        x = torch.randn(1, 4 * reg_max, 100)
        out = layer(x)
        # Output: (batch, 4, anchors)
        assert out.shape == (1, 4, 100)

    def test_ddetect_forward(self):
        """Test DDetect head forward pass."""
        layer = DDetect(nc=80, ch=(64, 128, 256), reg_max=16, stride=(8, 16, 32))
        layer.eval()  # Set to eval mode to get tensor output
        x = [
            torch.randn(1, 64, 80, 80),
            torch.randn(1, 128, 40, 40),
            torch.randn(1, 256, 20, 20),
        ]
        out = layer(x)
        # Eval mode returns (decoded_output, raw_outputs) tuple
        decoded, raw = out
        # decoded: (batch, 4+nc, total_anchors)
        assert decoded.shape[0] == 1
        assert decoded.shape[1] == 4 + 80  # 84 (decoded boxes + class scores)


class TestRDFullModel:
    """Test full model architecture."""

    def test_backbone_forward(self):
        """Test BackboneRD forward pass."""
        backbone = BackboneRD(atoms=512)
        x = torch.randn(1, 3, 640, 640)
        p3, p4, p5 = backbone(x)
        assert p3.shape[2] == 80  # 640 / 8
        assert p4.shape[2] == 40  # 640 / 16
        assert p5.shape[2] == 20  # 640 / 32

    def test_backbone_has_dconv(self):
        """Test that BackboneRD contains RepNCSPELAND layer with DConv."""
        backbone = BackboneRD(atoms=512)
        # Check that elan2 is RepNCSPELAND (contains DConv)
        assert hasattr(backbone, 'elan2')
        assert isinstance(backbone.elan2, RepNCSPELAND)
        assert hasattr(backbone.elan2, 'dconv')
        assert isinstance(backbone.elan2.dconv, DConv)

    def test_neck_forward(self):
        """Test NeckRD forward pass."""
        neck = NeckRD()
        # Use correct input channel sizes from BackboneRD outputs
        p3 = torch.randn(1, 512, 80, 80)  # B3 from elan2
        p4 = torch.randn(1, 512, 40, 40)  # B4 from elan3
        p5 = torch.randn(1, 512, 20, 20)  # B5/SPP output
        n3, n4, n5 = neck(p3, p4, p5)
        assert n3.shape[2] == 80
        assert n4.shape[2] == 40
        assert n5.shape[2] == 20

    def test_full_model_forward(self):
        """Test full LibreYOLORDModel forward pass."""
        model = LibreYOLORDModel(config='c', nb_classes=80)
        model.eval()  # Set to eval mode to get dict output
        x = torch.randn(1, 3, 640, 640)
        out = model(x)
        # In eval mode, returns dict with 'predictions' key
        assert isinstance(out, dict)
        assert 'predictions' in out


class TestRDUtils:
    """Test utility functions (re-exported from v9)."""

    def test_preprocess_image(self):
        """Test image preprocessing."""
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor, original_img, original_size = rd_utils.preprocess_image(img, input_size=640)
        assert tensor.shape == (1, 3, 640, 640)
        assert original_size == (100, 100)

    def test_make_anchors(self):
        """Test anchor generation.
        
        make_anchors returns (anchor_points, stride_tensor) with shapes:
        - anchor_points: (total_anchors, 2)
        - stride_tensor: (total_anchors, 1)
        """
        feature_maps = [
            torch.randn(1, 64, 80, 80),
            torch.randn(1, 128, 40, 40),
            torch.randn(1, 256, 20, 20),
        ]
        anchors, strides = rd_utils.make_anchors(feature_maps, strides=[8, 16, 32])
        # Total anchors = 80*80 + 40*40 + 20*20 = 8400
        assert anchors.shape[0] == 8400
        assert anchors.shape[1] == 2
        assert strides.shape[0] == 8400
        assert strides.shape[1] == 1


class TestRDRegionalDiversity:
    """Test regional diversity features specific to YOLO-RD."""

    def test_dconv_position_aware(self):
        """Test that DConv produces position-aware features."""
        layer = DConv(in_channels=64, atoms=64)
        # Create input with known pattern
        x = torch.zeros(1, 64, 32, 32)
        # Add different values in different regions
        x[:, :, :16, :16] = 1.0  # Top-left
        x[:, :, 16:, 16:] = 2.0  # Bottom-right

        out = layer(x)
        assert out.shape == (1, 64, 32, 32)
        # The outputs in different regions should differ due to position-aware normalization
        top_left = out[:, :, :16, :16].mean()
        bottom_right = out[:, :, 16:, 16:].mean()
        # They should be different (not exactly equal)
        assert not torch.isclose(top_left, bottom_right, atol=1e-3)

    def test_pono_statistics(self):
        """Test PONO computes correct position-wise statistics."""
        layer = PONO(eps=1e-5)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)

        # Verify that the output is normalized (zero mean, unit variance per position)
        # After normalization, mean should be close to 0 and std close to 1
        normalized_mean = out.mean(dim=1, keepdim=True)
        normalized_std = out.std(dim=1, keepdim=True)

        assert torch.allclose(normalized_mean, torch.zeros_like(normalized_mean), atol=1e-5)
        assert torch.allclose(normalized_std, torch.ones_like(normalized_std), atol=1e-1)
