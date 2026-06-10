"""Unit tests for YOLOv9 layers."""

import pytest
import numpy as np
import torch
from PIL import Image

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
    DDetectOBB,
    Backbone9,
    Neck9,
    LibreYOLO9Model,
)
from libreyolo.models.yolo9.loss import YOLO9OBBLoss
from libreyolo.models.yolo9 import utils as yolo9_utils
from libreyolo.postprocess import yolo9 as yolo9_postprocess_mod
from libreyolo.models.yolo9.trainer import YOLO9Trainer
from libreyolo.models.yolo9.transforms import YOLO9TrainTransform
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

    def test_ddetect_obb_forward(self):
        """Test oriented-box DDetect head forward pass."""
        layer = DDetectOBB(nc=2, ch=(64, 128, 256), reg_max=16, stride=(8, 16, 32))
        layer.eval()
        x = [
            torch.randn(1, 64, 8, 8),
            torch.randn(1, 128, 4, 4),
            torch.randn(1, 256, 2, 2),
        ]
        decoded, raw, angle_logits = layer(x)
        assert decoded.shape == (1, 7, 84)
        assert len(raw) == 3
        assert len(angle_logits) == 3
        assert angle_logits[0].shape == (1, 1, 8, 8)

    def test_ddetect_obb_export_forward_returns_prediction_tensor(self):
        """OBB export mode returns a single traceable prediction tensor."""
        layer = DDetectOBB(nc=2, ch=(64, 128, 256), reg_max=16, stride=(8, 16, 32))
        layer.eval()
        layer.export = True
        x = [
            torch.randn(1, 64, 8, 8),
            torch.randn(1, 128, 4, 4),
            torch.randn(1, 256, 2, 2),
        ]

        decoded = layer(x)

        assert isinstance(decoded, torch.Tensor)
        assert decoded.shape == (1, 7, 84)


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

    def test_obb_model_forward(self):
        """Test full LibreYOLO9 OBB model forward pass."""
        model = LibreYOLO9Model(config="t", nb_classes=2, obb=True)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert isinstance(out, dict)
        assert out["obb"] is True
        assert out["predictions"].shape == (1, 7, 84)

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

    def test_obb_training_loss(self):
        """OBB model computes box, class, DFL, and angle losses."""
        model = LibreYOLO9Model(config="t", nb_classes=2, obb=True)
        model.train()
        targets = torch.zeros(2, 100, 6)
        targets[:, :, 0] = -1
        targets[0, 0] = torch.tensor([0, 0.2, 0.2, 0.7, 0.7, 0.25])
        targets[1, 0] = torch.tensor([1, 0.1, 0.1, 0.6, 0.6, -0.25])

        out = model(torch.randn(2, 3, 64, 64), targets=targets)

        assert out["total_loss"].requires_grad
        assert out["angle_loss"].requires_grad
        assert out["angle"] >= 0


def test_yolo9_obb_transform_vertical_flip_updates_box_and_angle():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    targets = np.array([[2.0, 1.0, 10.0, 4.0, 0.0, 0.25]], dtype=np.float32)
    transform = YOLO9TrainTransform(
        max_labels=2,
        flip_prob=0.0,
        vertical_flip_prob=1.0,
        hsv_prob=0.0,
        output_label_dim=6,
    )

    _, labels = transform(image, targets, (10, 20))

    np.testing.assert_allclose(labels[0], [0.0, 0.1, 0.6, 0.5, 0.9, -0.25])
    assert labels[1, 0] == -1


def test_yolo9_obb_transform_horizontal_flip_updates_box_and_angle():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    targets = np.array([[2.0, 1.0, 10.0, 4.0, 0.0, 0.25]], dtype=np.float32)
    transform = YOLO9TrainTransform(
        max_labels=2,
        flip_prob=1.0,
        vertical_flip_prob=0.0,
        hsv_prob=0.0,
        output_label_dim=6,
    )

    _, labels = transform(image, targets, (10, 20))

    np.testing.assert_allclose(labels[0], [0.0, 0.5, 0.1, 0.9, 0.4, -0.25])
    assert labels[1, 0] == -1


def test_yolo9_obb_loss_default_angle_weight_is_one():
    loss = YOLO9OBBLoss(
        num_classes=2,
        reg_max=16,
        strides=[8, 16, 32],
        image_size=None,
        device=torch.device("cpu"),
    )

    assert loss.angle_weight == 1.0


class TestYOLO9Utils:
    """Test utility functions."""

    def test_preprocess_image(self):
        """Test image preprocessing."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor, original_img, original_size = yolo9_utils.preprocess_image(
            img, input_size=640
        )
        assert tensor.shape == (1, 3, 640, 640)
        assert original_size == (100, 100)

    def test_preprocess_image_letterboxes_non_square_like_validation(self):
        """Predict preprocessing must match YOLO9 validation geometry."""
        img = np.zeros((4, 8, 3), dtype=np.uint8)

        tensor, _, original_size = yolo9_utils.preprocess_image(
            img, input_size=8, color_format="rgb"
        )
        val_tensor, _ = YOLO9ValPreprocessor((8, 8), max_labels=1)(
            img[:, :, ::-1].copy(),
            np.zeros((0, 5), dtype=np.float32),
            (8, 8),
        )

        assert original_size == (8, 4)
        torch.testing.assert_close(tensor[0], torch.from_numpy(val_tensor))
        torch.testing.assert_close(
            tensor[0, :, 4:, :],
            torch.full((3, 4, 8), 114 / 255.0, dtype=tensor.dtype),
        )

    def test_preprocess_image_accepts_rectangular_input_size(self):
        img = np.zeros((4, 8, 3), dtype=np.uint8)

        tensor, _, original_size = yolo9_utils.preprocess_image(
            img, input_size=(8, 16), color_format="rgb"
        )

        assert original_size == (8, 4)
        assert tensor.shape == (1, 3, 8, 16)
        torch.testing.assert_close(
            tensor[0, :, :, :16],
            torch.zeros((3, 8, 16), dtype=tensor.dtype),
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

    def test_postprocess_accepts_rectangular_input_size(self):
        pred = torch.zeros(1, 6, 1)
        pred[0, :4, 0] = torch.tensor([0.0, 0.0, 320.0, 320.0])
        pred[0, 4, 0] = 0.9

        out = yolo9_utils.postprocess(
            {"predictions": pred},
            input_size=(320, 640),
            original_size=(1280, 960),
        )

        assert out["num_detections"] == 1
        torch.testing.assert_close(
            torch.as_tensor(out["boxes"]),
            torch.tensor([[0.0, 0.0, 960.0, 960.0]]),
        )

    def test_postprocess_detection_is_multilabel(self):
        """Detection postprocess emits one detection per class above conf on an
        anchor (multi-label), matching MultimediaTechLab/YOLO ``bbox_nms``."""
        pred = torch.zeros(1, 6, 1)
        pred[0, :4, 0] = torch.tensor([0.0, 0.0, 100.0, 100.0])
        pred[0, 4:, 0] = torch.tensor([0.9, 0.8])  # two classes over conf

        out = yolo9_utils.postprocess(
            {"predictions": pred}, conf_thres=0.25, iou_thres=0.5
        )

        assert out["num_detections"] == 2
        assert sorted(out["classes"]) == [0, 1]

    def test_postprocess_detection_caps_multilabel_candidates(self, monkeypatch):
        """Detection limits low-threshold multi-label expansion before NMS."""
        # Patch the postprocess module — that's where postprocess() resolves it.
        monkeypatch.setattr(yolo9_postprocess_mod, "_YOLO9_MAX_NMS_CANDIDATES", 3)
        pred = torch.zeros(1, 6, 4)
        pred[0, :4] = torch.tensor(
            [
                [0.0, 20.0, 40.0, 60.0],
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 30.0, 50.0, 70.0],
                [10.0, 10.0, 10.0, 10.0],
            ]
        )
        pred[0, 4:] = torch.tensor(
            [[0.1, 0.9, 0.7, 0.5], [0.8, 0.2, 0.6, 0.4]]
        )

        out = yolo9_utils.postprocess(
            {"predictions": pred}, conf_thres=0.01, iou_thres=0.5, max_det=3
        )

        assert out["num_detections"] == 3
        assert sorted(round(float(s), 1) for s in out["scores"]) == [
            0.7,
            0.8,
            0.9,
        ]

    def test_postprocess_segment_keeps_best_class(self):
        """Segment postprocess stays best-class (not multi-label) so each
        detection keeps a single mask-coefficient vector."""
        pred = torch.zeros(1, 6, 1)
        pred[0, :4, 0] = torch.tensor([0.0, 0.0, 100.0, 100.0])
        pred[0, 4:, 0] = torch.tensor([0.9, 0.8])  # two classes over conf
        proto = torch.randn(1, 32, 16, 16)
        coeffs = torch.randn(1, 32, 1)

        out = yolo9_utils.postprocess(
            {"predictions": pred, "proto": proto, "mask_coeffs": coeffs},
            conf_thres=0.25,
            iou_thres=0.5,
            original_size=(100, 100),
        )

        assert out["num_detections"] == 1
        assert out["classes"] == [0]

    def test_postprocess_obb_outputs_obb_payload(self):
        pred = torch.zeros(1, 7, 1)
        pred[0, :4, 0] = torch.tensor([10.0, 20.0, 50.0, 40.0])
        pred[0, 4, 0] = 0.25
        pred[0, 5:, 0] = torch.tensor([0.9, 0.1])

        out = yolo9_utils.postprocess(
            {"predictions": pred, "obb": True},
            conf_thres=0.25,
            iou_thres=0.5,
            input_size=64,
            original_size=(64, 64),
        )

        assert out["num_detections"] == 1
        assert len(out["obb"]) == 1
        torch.testing.assert_close(
            torch.as_tensor(out["obb"])[0, :5],
            torch.tensor([30.0, 30.0, 40.0, 20.0, 0.25]),
        )

    def test_postprocess_obb_uses_letterbox_inverse_for_non_square_images(self):
        pred = torch.zeros(1, 7, 1)
        pred[0, :4, 0] = torch.tensor([100.0, 50.0, 200.0, 150.0])
        pred[0, 4, 0] = 0.25
        pred[0, 5:, 0] = torch.tensor([0.9, 0.1])

        out = yolo9_utils.postprocess(
            {"predictions": pred, "obb": True},
            conf_thres=0.25,
            iou_thres=0.5,
            input_size=640,
            original_size=(1280, 960),
        )

        assert out["num_detections"] == 1
        torch.testing.assert_close(
            torch.as_tensor(out["obb"])[0, :5],
            torch.tensor([300.0, 200.0, 200.0, 200.0, 0.25]),
        )

    def test_postprocess_obb_uses_classwise_rotated_nms(self):
        pred = torch.zeros(1, 7, 3)
        pred[0, :4] = torch.tensor(
            [
                [10.0, 10.0, 10.0],
                [20.0, 20.0, 20.0],
                [50.0, 50.0, 50.0],
                [40.0, 40.0, 40.0],
            ]
        )
        pred[0, 4] = 0.25
        pred[0, 5:] = torch.tensor(
            [
                [0.9, 0.8, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )

        out = yolo9_utils.postprocess(
            {"predictions": pred, "obb": True},
            conf_thres=0.25,
            iou_thres=0.5,
            input_size=64,
            original_size=(64, 64),
        )

        assert out["num_detections"] == 2
        assert out["classes"] == [0, 1]
        assert [round(score, 2) for score in out["scores"]] == [0.9, 0.7]

    def test_postprocess_obb_prefilters_candidates_before_rotated_nms(self, monkeypatch):
        num_candidates = 2000
        pred = torch.zeros(1, 7, num_candidates)
        pred[0, :4] = torch.tensor([[10.0], [20.0], [50.0], [40.0]]).expand(
            4, num_candidates
        )
        pred[0, 4] = 0.25
        pred[0, 5] = torch.linspace(0.9, 0.1, num_candidates)
        pred[0, 6] = 0.01

        exact_candidate_counts = []
        original_rotated_nms = yolo9_postprocess_mod._rotated_nms_keep_indices

        def wrapped_rotated_nms(xywhr, scores, class_ids, iou_thres, max_det):
            exact_candidate_counts.append(int(scores.numel()))
            return original_rotated_nms(xywhr, scores, class_ids, iou_thres, max_det)

        # Patch the postprocess module — that's where postprocess() resolves it.
        monkeypatch.setattr(
            yolo9_postprocess_mod,
            "_rotated_nms_keep_indices",
            wrapped_rotated_nms,
        )

        out = yolo9_utils.postprocess(
            {"predictions": pred, "obb": True},
            conf_thres=0.001,
            iou_thres=0.5,
            input_size=64,
            original_size=(64, 64),
            max_det=50,
        )

        assert out["num_detections"] == 1
        assert exact_candidate_counts
        assert exact_candidate_counts[0] <= yolo9_utils._YOLO9_OBB_MAX_NMS_CANDIDATES

    def test_obb_prefilter_does_not_apply_horizontal_nms(self):
        num_candidates = 400
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]]).expand(
            num_candidates, 4
        )
        scores = torch.linspace(1.0, 0.1, num_candidates)
        classes = torch.zeros(num_candidates, dtype=torch.long)

        keep = yolo9_utils._obb_prefilter_keep_indices(
            boxes,
            scores,
            classes,
            max_det=50,
        )

        assert keep.numel() == num_candidates
        torch.testing.assert_close(scores[keep], scores)

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


def test_yolo9_trainer_disables_proxy_mosaic_for_obb(tmp_path):
    image_dir = tmp_path / "train" / "images"
    label_dir = tmp_path / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text(
        "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: " + str(tmp_path).replace("\\", "/") + "\n"
        "train: train/images\n"
        "val: train/images\n"
        "nc: '1'\n"
        "names:\n"
        "  0: vehicle\n",
        encoding="utf-8",
    )
    wrapper = type(
        "Wrapper",
        (),
        {"task": "obb", "nb_classes": 1, "names": {0: "vehicle"}},
    )()
    trainer = YOLO9Trainer(
        model=torch.nn.Conv2d(3, 3, 1),
        wrapper_model=wrapper,
        data=str(data_yaml),
        epochs=1,
        batch=1,
        imgsz=64,
        workers=0,
        device="cpu",
        mosaic_prob=1.0,
        mixup_prob=1.0,
    )

    train_dataset = trainer._setup_data()

    assert train_dataset.enable_mosaic is False
    assert train_dataset.enable_mixup is False
    assert train_dataset.mosaic_prob == 0.0
    assert train_dataset.mixup_prob == 0.0
    assert train_dataset.preproc.vertical_flip_prob == 0.5
    assert train_dataset.dataset.num_classes == 1


def test_yolo9_trainer_checkpoint_uses_resolved_data_classes_for_obb(tmp_path):
    from libreyolo.utils.serialization import load_trusted_torch_file

    image_dir = tmp_path / "train" / "images"
    label_dir = tmp_path / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text(
        "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: " + str(tmp_path).replace("\\", "/") + "\n"
        "train: train/images\n"
        "val: train/images\n"
        "nc: '1'\n"
        "names:\n"
        "  0: vehicle\n",
        encoding="utf-8",
    )
    wrapper = type(
        "Wrapper",
        (),
        {"task": "obb", "nb_classes": 1, "names": {0: "vehicle"}},
    )()
    trainer = YOLO9Trainer(
        model=torch.nn.Conv2d(3, 3, 1),
        wrapper_model=wrapper,
        data=str(data_yaml),
        epochs=1,
        batch=1,
        imgsz=64,
        workers=0,
        device="cpu",
    )

    trainer._setup_data()
    trainer.save_dir = tmp_path / "run"
    trainer.save_dir.mkdir()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer._save_checkpoint(epoch=0, loss=1.0, is_best=True)

    checkpoint = load_trusted_torch_file(
        trainer.save_dir / "weights" / "last.pt",
        map_location="cpu",
        context="unit test checkpoint",
    )
    assert trainer.config.num_classes == 1
    assert checkpoint["nc"] == 1
    assert checkpoint["config"]["num_classes"] == 1


def test_postprocess_segment_outputs_masks():
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
