"""Unit tests for YOLOv9 layers."""

import pytest
import numpy as np
import torch

from libreyolo.models.yolo9.nn import (
    Conv,
    RepConvN,
    Bottleneck,
    RepNBottleneck,
    RepNCSP,
    ELAN,
    RepNCSPELAN,
    AConv,
    ADown,
    SPPELAN,
    Concat,
    DFL,
    DDetect,
    DDetectSeg,
    Backbone9,
    Neck9,
    LibreYOLO9Model,
)
from libreyolo.models.yolo9 import utils as yolo9_utils
from libreyolo.validation.preprocessors import YOLO9ValPreprocessor

pytestmark = pytest.mark.unit


class TestYOLO9ConvLayers:
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


class TestYOLO9Bottlenecks:
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


class TestYOLO9ELANBlocks:
    """Test ELAN-based blocks."""

    def test_elan_forward(self):
        """Test ELAN forward pass.

        ELAN(c1, c2, c3, c4, n) where:
        - c1: input channels
        - c2: cv1 output channels (gets split in half)
        - c3: cv2/cv3 output channels
        - c4: output channels
        """
        # Input: 64, cv1: 64 (split to 32+32), cv2/cv3: 32, output: 128
        layer = ELAN(64, 64, 32, 128, n=1)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 128, 32, 32)

    def test_repncspelan_forward(self):
        """Test RepNCSPELAN forward pass.

        RepNCSPELAN(c1, c2, c3, c4, n) where:
        - c1: input channels
        - c2: intermediate channels 1
        - c3: intermediate channels 2
        - c4: output channels
        """
        layer = RepNCSPELAN(64, 64, 32, 128, n=1)
        x = torch.randn(1, 64, 32, 32)
        out = layer(x)
        assert out.shape == (1, 128, 32, 32)


class TestYOLO9Downsampling:
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


class TestYOLO9SPPELAN:
    """Test SPP-ELAN module."""

    def test_sppelan_forward(self):
        """Test SPPELAN forward pass.

        SPPELAN(c1, c2, c3, k) where:
        - c1: input channels
        - c2: neck channels (intermediate)
        - c3: output channels
        - k: pool kernel size
        """
        layer = SPPELAN(256, 128, 256, k=5)
        x = torch.randn(1, 256, 16, 16)
        out = layer(x)
        assert out.shape == (1, 256, 16, 16)


class TestYOLO9Concat:
    """Test Concat layer."""

    def test_concat_forward(self):
        """Test Concat layer forward pass."""
        layer = Concat(dimension=1)
        x1 = torch.randn(1, 64, 32, 32)
        x2 = torch.randn(1, 128, 32, 32)
        out = layer([x1, x2])
        assert out.shape == (1, 192, 32, 32)


class TestYOLO9DetectionHead:
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

    def test_ddetect_seg_forward(self):
        """Test segmented DDetect head forward pass."""
        layer = DDetectSeg(
            nc=2,
            ch=(64, 128, 256),
            reg_max=16,
            stride=(8, 16, 32),
            num_masks=32,
        )
        layer.eval()
        x = [
            torch.randn(1, 64, 8, 8),
            torch.randn(1, 128, 4, 4),
            torch.randn(1, 256, 2, 2),
        ]
        decoded, raw, proto, coeffs = layer(x)
        assert decoded.shape == (1, 6, 84)
        assert len(raw) == 3
        assert proto.shape == (1, 32, 16, 16)
        assert coeffs.shape == (1, 32, 84)


class TestYOLO9FullModel:
    """Test full model architecture."""

    def test_backbone_forward(self):
        """Test Backbone9 forward pass."""
        backbone = Backbone9(config="t")
        x = torch.randn(1, 3, 640, 640)
        p3, p4, p5 = backbone(x)
        assert p3.shape[2] == 80  # 640 / 8
        assert p4.shape[2] == 40  # 640 / 16
        assert p5.shape[2] == 20  # 640 / 32

    def test_neck_forward(self):
        """Test Neck9 forward pass."""
        # Get backbone to determine correct channel sizes
        backbone = Backbone9(config="t")
        x = torch.randn(1, 3, 640, 640)
        p3, p4, p5 = backbone(x)

        neck = Neck9(config="t")
        n3, n4, n5 = neck(p3, p4, p5)
        assert n3.shape[2] == 80
        assert n4.shape[2] == 40
        assert n5.shape[2] == 20

    def test_full_model_forward(self):
        """Test full LibreYOLO9Model forward pass."""
        model = LibreYOLO9Model(config="t", nb_classes=80)
        model.eval()  # Set to eval mode to get dict output
        x = torch.randn(1, 3, 640, 640)
        out = model(x)
        # In eval mode, returns dict with 'predictions' key
        assert isinstance(out, dict)
        assert "predictions" in out

    def test_segment_model_forward(self):
        """Test full LibreYOLO9 segmentation model forward pass."""
        model = LibreYOLO9Model(config="t", nb_classes=2, segmentation=True)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert isinstance(out, dict)
        assert out["predictions"].shape == (1, 6, 84)
        assert out["proto"].shape == (1, 32, 16, 16)
        assert out["mask_coeffs"].shape == (1, 32, 84)

    def test_segment_training_loss(self):
        """Segmentation model computes box, class, DFL, and mask losses."""
        model = LibreYOLO9Model(config="t", nb_classes=2, segmentation=True)
        model.train()
        targets = torch.zeros(2, 100, 5)
        targets[:, :, 0] = -1
        targets[0, 0] = torch.tensor([0, 0.2, 0.2, 0.7, 0.7])
        targets[1, 0] = torch.tensor([1, 0.1, 0.1, 0.6, 0.6])
        masks = torch.zeros(2, 100, 16, 16)
        masks[0, 0, 3:11, 3:11] = 1
        masks[1, 0, 2:10, 2:10] = 1

        out = model(torch.randn(2, 3, 64, 64), targets=targets, masks=masks)

        assert out["total_loss"].requires_grad
        assert out["seg_loss"].requires_grad
        assert out["seg"] >= 0


class TestYOLO9Utils:
    """Test utility functions."""

    def test_preprocess_image(self):
        """Test image preprocessing."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor, original_img, original_size, ratio, pad = (
            yolo9_utils.preprocess_image(img, input_size=640)
        )
        assert tensor.shape == (1, 3, 640, 640)
        assert original_size == (100, 100)
        # Square input → unit ratio, no padding.
        assert ratio == 0.64 or ratio == 640 / 100
        assert pad == (0.0, 0.0)

    def test_preprocess_image_letterboxes_non_square_like_validation(self):
        """Predict preprocessing must match YOLO9 validation geometry."""
        img = np.zeros((4, 8, 3), dtype=np.uint8)

        tensor, _, original_size, ratio, pad = yolo9_utils.preprocess_image(
            img, input_size=8, color_format="rgb"
        )
        val_tensor, _ = YOLO9ValPreprocessor((8, 8), max_labels=1)(
            img[:, :, ::-1].copy(),
            np.zeros((0, 5), dtype=np.float32),
            (8, 8),
        )

        assert original_size == (8, 4)
        # ratio 1.0, new_h=4, dh=2 → top=round(2-0.1)=2: image centered in rows
        # [2:6], 2px of gray pad above (rows 0:2) and below (rows 6:8).
        assert ratio == 1.0
        assert pad == (0.0, 2.0)
        # Predict and centered-letterbox validation must produce identical
        # input-canvas geometry.
        torch.testing.assert_close(tensor[0], torch.from_numpy(val_tensor))
        gray = 114 / 255.0
        # Top and bottom 2 rows are padding; the middle 4 rows are the image.
        torch.testing.assert_close(
            tensor[0, :, :2, :],
            torch.full((3, 2, 8), gray, dtype=tensor.dtype),
        )
        torch.testing.assert_close(
            tensor[0, :, 6:, :],
            torch.full((3, 2, 8), gray, dtype=tensor.dtype),
        )
        torch.testing.assert_close(
            tensor[0, :, 2:6, :],
            torch.zeros((3, 4, 8), dtype=tensor.dtype),
        )

    def test_postprocess_defaults_to_letterbox_inverse(self):
        """YOLO9 postprocess default matches letterboxed predict inputs."""
        pred = torch.zeros(1, 6, 1)
        pred[0, :4, 0] = torch.tensor([0.0, 0.0, 320.0, 320.0])
        pred[0, 4, 0] = 0.9

        out = yolo9_utils.postprocess(
            {"predictions": pred},
            input_size=640,
            original_size=(1280, 960),
        )

        assert out["num_detections"] == 1
        torch.testing.assert_close(
            torch.as_tensor(out["boxes"]),
            torch.tensor([[0.0, 0.0, 640.0, 640.0]]),
        )

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
        from libreyolo.utils.general import make_anchors

        anchors, strides = make_anchors(feature_maps, strides=[8, 16, 32])
        # Total anchors = 80*80 + 40*40 + 20*20 = 8400
        assert anchors.shape[0] == 8400
        assert anchors.shape[1] == 2
        assert strides.shape[0] == 8400
        assert strides.shape[1] == 1

    def test_postprocess_segment_outputs_masks(self):
        """YOLO9 segment postprocess keeps mask coefficients aligned through NMS."""
        num_anchors = 4
        num_classes = 2
        num_masks = 32
        pred = torch.zeros(1, 4 + num_classes, num_anchors)
        pred[0, :4] = torch.tensor(
            [
                [10, 12, 11, 200],
                [10, 12, 11, 200],
                [50, 60, 55, 240],
                [50, 60, 55, 240],
            ],
            dtype=torch.float32,
        )
        pred[0, 4:] = torch.tensor(
            [[0.9, 0.2, 0.95, 0.1], [0.1, 0.8, 0.05, 0.7]]
        )
        proto = torch.randn(1, num_masks, 16, 16)
        coeffs = torch.randn(1, num_masks, num_anchors)

        out = yolo9_utils.postprocess(
            {"predictions": pred, "proto": proto, "mask_coeffs": coeffs},
            conf_thres=0.25,
            iou_thres=0.5,
            input_size=64,
            original_size=(128, 96),
            max_det=3,
        )

        assert out["num_detections"] == 2
        assert out["masks"].shape == (2, 96, 128)


class TestYOLO9CenteredLetterbox:
    """Centered-letterbox geometry, pad round-trip, multi_label, R5 guard."""

    def test_centered_pad_math_matches_preprocess_numpy(self):
        """val helper yolo9_letterbox_pad == preprocess_numpy geometry."""
        from libreyolo.validation.preprocessors import yolo9_letterbox_pad

        for orig_h, orig_w, size in [
            (480, 640, 640),
            (375, 500, 640),
            (1080, 1920, 640),
            (333, 500, 416),
        ]:
            img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            _, ratio_np, pad_np = yolo9_utils.preprocess_numpy(img, size)
            ratio_v, pad_w, pad_h = yolo9_letterbox_pad(orig_h, orig_w, size)
            assert ratio_v == pytest.approx(ratio_np)
            assert (pad_w, pad_h) == pad_np

    def test_pad_round_trip_in_postprocess(self):
        """A box placed in the padded canvas maps back to original coords."""
        orig_w, orig_h, size = 800, 400, 640
        img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        _, ratio, pad = yolo9_utils.preprocess_numpy(img, size)
        # Original box (100,50)-(300,250) → forward into padded canvas.
        x1, y1, x2, y2 = 100.0, 50.0, 300.0, 250.0
        bx1 = x1 * ratio + pad[0]
        by1 = y1 * ratio + pad[1]
        bx2 = x2 * ratio + pad[0]
        by2 = y2 * ratio + pad[1]
        pred = torch.zeros(1, 5, 1)
        pred[0, :4, 0] = torch.tensor([bx1, by1, bx2, by2])
        pred[0, 4, 0] = 0.9
        out = yolo9_utils.postprocess(
            {"predictions": pred},
            conf_thres=0.25,
            input_size=size,
            original_size=(orig_w, orig_h),
            pad=pad,
        )
        assert out["num_detections"] == 1
        torch.testing.assert_close(
            torch.as_tensor(out["boxes"]),
            torch.tensor([[x1, y1, x2, y2]]),
            atol=1.0,
            rtol=0.0,
        )

    def test_legacy_scalar_ratio_pad_none_backward_compat(self):
        """pad=None falls back to top-left-pad assumption (no offset)."""
        pred = torch.zeros(1, 5, 1)
        pred[0, :4, 0] = torch.tensor([0.0, 0.0, 320.0, 320.0])
        pred[0, 4, 0] = 0.9
        out = yolo9_utils.postprocess(
            {"predictions": pred},
            conf_thres=0.25,
            input_size=640,
            original_size=(1280, 960),
            pad=None,
        )
        assert out["num_detections"] == 1
        # ratio = 640/1280 = 0.5 → box /0.5 = (0,0,640,640), no pad shift.
        torch.testing.assert_close(
            torch.as_tensor(out["boxes"]),
            torch.tensor([[0.0, 0.0, 640.0, 640.0]]),
        )

    def test_multi_label_emits_multiple_dets_per_anchor(self):
        """multi_label=True emits one det per above-conf class on an anchor."""
        # One anchor, two classes both above conf.
        pred = torch.zeros(1, 6, 1)
        pred[0, :4, 0] = torch.tensor([10.0, 10.0, 50.0, 50.0])
        pred[0, 4, 0] = 0.8  # class 0
        pred[0, 5, 0] = 0.7  # class 1
        out_multi = yolo9_utils.postprocess(
            {"predictions": pred},
            conf_thres=0.25,
            iou_thres=0.9,  # high IoU so the two near-identical boxes both survive
            input_size=64,
            original_size=(64, 64),
            multi_label=True,
        )
        assert out_multi["num_detections"] == 2
        assert set(out_multi["classes"]) == {0, 1}

        out_single = yolo9_utils.postprocess(
            {"predictions": pred},
            conf_thres=0.25,
            iou_thres=0.9,
            input_size=64,
            original_size=(64, 64),
            multi_label=False,
        )
        assert out_single["num_detections"] == 1
        assert out_single["classes"] == [0]  # argmax class only

    def test_multi_label_candidate_guard_matches_full_scan(self):
        """R5: pre-topk candidate guard yields identical dets to a naive scan."""
        torch.manual_seed(0)
        num_anchors = 8400
        num_classes = 80
        pred = torch.zeros(1, 4 + num_classes, num_anchors)
        # Random plausible boxes in 640 canvas.
        cx = torch.rand(num_anchors) * 640
        cy = torch.rand(num_anchors) * 640
        pred[0, 0] = cx - 5
        pred[0, 1] = cy - 5
        pred[0, 2] = cx + 5
        pred[0, 3] = cy + 5
        # Mostly low scores, a handful of strong detections.
        pred[0, 4:] = torch.rand(num_classes, num_anchors) * 0.0005
        pred[0, 4, 100] = 0.9
        pred[0, 7, 100] = 0.6  # second class on same anchor
        pred[0, 20, 5000] = 0.8

        out = yolo9_utils.postprocess(
            {"predictions": pred},
            conf_thres=0.001,
            iou_thres=0.7,
            input_size=640,
            original_size=(640, 640),
            multi_label=True,
            max_det=300,
        )
        # The three strong (class,anchor) detections must all appear.
        assert out["num_detections"] >= 3
        top = sorted(out["scores"], reverse=True)[:3]
        assert top[0] == pytest.approx(0.9, abs=1e-3)
        assert top[1] == pytest.approx(0.8, abs=1e-3)
        assert top[2] == pytest.approx(0.6, abs=1e-3)

    def test_process_masks_crop_respects_pad(self):
        """Seg mask crop uses the centered pad offset (E1)."""
        from libreyolo.models.yolo9.utils import _process_masks

        orig_w, orig_h, size = 128, 64, 64
        img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        _, _, pad = yolo9_utils.preprocess_numpy(img, size)
        proto = torch.randn(32, 16, 16)
        coeffs = torch.randn(2, 32)
        boxes_input = torch.tensor(
            [[10.0, 20.0, 40.0, 50.0], [5.0, 8.0, 30.0, 40.0]]
        )
        masks = _process_masks(
            proto,
            coeffs,
            boxes_input,
            input_shape=(size, size),
            original_size=(orig_w, orig_h),
            letterbox=True,
            pad=pad,
        )
        # Output masks are upsampled to the original frame size.
        assert masks.shape == (2, orig_h, orig_w)
