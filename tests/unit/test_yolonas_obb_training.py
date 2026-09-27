"""Unit coverage for YOLO-NAS-R (OBB) training: loss, assigner, transform."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from libreyolo.data.augment.yolonas import (
    YOLONASOBBTrainTransform,
    _canonicalize_obb_rows,
)
from libreyolo.data.obb import xywhr_to_corners
from libreyolo.models.yolonas.model import LibreYOLONAS
from libreyolo.models.yolonas.nn import LibreYOLONASOBBModel
from libreyolo.models.yolonas.obb_loss import (
    YOLONASOBBAssigner,
    YOLONASOBBLoss,
    check_points_inside_rboxes,
    cxcywhr_iou,
    pairwise_cxcywhr_iou,
)
from libreyolo.training.config import YOLONASOBBConfig

pytestmark = [pytest.mark.unit, pytest.mark.yolonas]


# ---------------------------------------------------------------------------
# Probabilistic IoU
# ---------------------------------------------------------------------------


def test_probiou_is_one_for_identical_boxes():
    """Identical boxes saturate the metric.

    Not exactly 1.0: the Bhattacharyya distance is clamped at ``eps`` for
    numerical stability, which floors the IoU a hair below one. That clamp is
    upstream's and is what keeps the gradient finite at zero distance.
    """
    box = torch.tensor([[50.0, 50.0, 40.0, 20.0, 0.3]])
    assert cxcywhr_iou(box, box).item() == pytest.approx(1.0, abs=5e-3)


def test_probiou_decays_with_separation_and_rotation():
    ref = torch.tensor([[50.0, 50.0, 40.0, 20.0, 0.0]])
    shifted = torch.tensor([[70.0, 50.0, 40.0, 20.0, 0.0]])
    far = torch.tensor([[300.0, 300.0, 40.0, 20.0, 0.0]])
    rotated = torch.tensor([[50.0, 50.0, 40.0, 20.0, math.pi / 2]])

    same = cxcywhr_iou(ref, ref).item()
    near = cxcywhr_iou(ref, shifted).item()
    away = cxcywhr_iou(ref, far).item()
    turned = cxcywhr_iou(ref, rotated).item()

    assert same > near > away
    assert away == pytest.approx(0.0, abs=1e-3)
    # A 90-degree turn of a 2:1 box is a real mismatch, but still overlapping.
    assert 0.0 < turned < same


def test_probiou_is_differentiable_wrt_angle():
    pred = torch.tensor([[50.0, 50.0, 40.0, 20.0, 0.1]], requires_grad=True)
    target = torch.tensor([[50.0, 50.0, 40.0, 20.0, 0.6]])
    loss = 1 - cxcywhr_iou(pred, target)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0, 4].abs() > 0  # the angle actually receives gradient


def test_pairwise_probiou_shape_and_diagonal():
    boxes = torch.tensor([[[10.0, 10.0, 8.0, 4.0, 0.0], [80.0, 80.0, 8.0, 4.0, 0.5]]])
    ious = pairwise_cxcywhr_iou(boxes, boxes)
    assert ious.shape == (1, 2, 2)
    assert ious[0, 0, 0].item() == pytest.approx(1.0, abs=5e-3)
    assert ious[0, 0, 1].item() < 1e-3


def test_points_inside_rboxes_uses_the_inradius():
    rboxes = torch.tensor([[[50.0, 50.0, 40.0, 20.0, 0.0]]])
    points = torch.tensor([[50.0, 50.0], [50.0, 59.0], [50.0, 100.0]])
    mask = check_points_inside_rboxes(points, rboxes)[0, 0]
    assert mask.tolist() == [1.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# Assigner
# ---------------------------------------------------------------------------


def test_assigner_returns_background_for_empty_ground_truth():
    assigner = YOLONASOBBAssigner(topk=4)
    result = assigner(
        pred_scores=torch.rand(1, 16, 3),
        pred_rboxes=torch.rand(1, 16, 5),
        anchor_points=torch.rand(16, 2),
        gt_labels=torch.zeros(1, 0, 1, dtype=torch.long),
        gt_rboxes=torch.zeros(1, 0, 5),
        bg_index=3,
    )
    assert (result.assigned_labels == 3).all()
    assert result.assigned_scores.abs().sum() == 0


def test_assigner_picks_anchors_near_the_target():
    assigner = YOLONASOBBAssigner(topk=2)
    anchors = torch.tensor([[50.0, 50.0], [52.0, 50.0], [400.0, 400.0]])
    gt = torch.tensor([[[50.0, 50.0, 40.0, 30.0, 0.0]]])
    preds = gt.repeat(1, 3, 1).clone()
    preds[0, 2, :2] = torch.tensor([400.0, 400.0])
    scores = torch.full((1, 3, 2), 0.9)

    result = assigner(
        pred_scores=scores,
        pred_rboxes=preds,
        anchor_points=anchors,
        gt_labels=torch.zeros(1, 1, 1, dtype=torch.long),
        gt_rboxes=gt,
        bg_index=2,
    )
    assert result.assigned_labels[0, 0].item() == 0
    assert result.assigned_labels[0, 1].item() == 0
    # The far anchor is outside the box and must stay background.
    assert result.assigned_labels[0, 2].item() == 2


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return LibreYOLONASOBBModel(config="s", nb_classes=2)


def _targets(device="cpu"):
    targets = torch.zeros(2, 4, 6, device=device)
    targets[:, 0] = torch.tensor([0.0, 128.0, 128.0, 60.0, 30.0, 0.4], device=device)
    targets[:, 1] = torch.tensor([1.0, 60.0, 80.0, 40.0, 20.0, -0.2], device=device)
    return targets


def test_loss_components_are_finite_and_positive(tiny_model):
    loss_fn = YOLONASOBBLoss(num_classes=2)
    outputs = tiny_model(torch.rand(2, 3, 256, 256))
    loss, log = loss_fn(outputs, _targets())
    assert torch.isfinite(loss)
    assert log.shape == (4,)
    assert (log[:3] > 0).all()
    assert log[3].item() == pytest.approx(log[:3].sum().item(), rel=1e-5)


def test_loss_backward_gives_finite_gradients_everywhere():
    torch.manual_seed(0)
    model = LibreYOLONASOBBModel(config="s", nb_classes=2)
    loss_fn = YOLONASOBBLoss(num_classes=2)
    loss, _ = loss_fn(model(torch.rand(1, 3, 256, 256)), _targets()[:1])
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    # The rotation branch must be trained, not left dangling.
    assert model.heads.head1.rot_pred.weight.grad is not None
    assert model.heads.head1.offset_pred.weight.grad is not None


def test_loss_decreases_when_predictions_move_toward_targets(tiny_model):
    """A hand-built output closer to the target must score a lower loss."""
    loss_fn = YOLONASOBBLoss(num_classes=2)
    anchors = torch.tensor([[16.0, 16.0], [8.0, 10.0]])
    strides = torch.tensor([[8.0], [8.0]])

    def build(boxes, logit):
        scores = torch.full((1, 2, 2), -6.0)
        scores[0, 0, 0] = logit
        raw = {
            "score_logits": scores,
            "size_dist": torch.zeros(1, 2, 2 * 17),
            "size_reduced": boxes[..., 2:4] / strides,
            "angles": boxes[..., 4:5],
            "offsets": boxes[..., :2] / strides - anchors,
            "anchor_points": anchors,
            "strides": strides,
            "reg_max": 16,
        }
        return (boxes, scores.sigmoid()), raw

    targets = torch.zeros(1, 1, 6)
    targets[0, 0] = torch.tensor([0.0, 128.0, 128.0, 40.0, 20.0, 0.2])

    good = torch.tensor(
        [[[128.0, 128.0, 40.0, 20.0, 0.2], [64.0, 80.0, 8.0, 8.0, 0.0]]]
    )
    bad = torch.tensor(
        [[[128.0, 128.0, 90.0, 90.0, -1.2], [64.0, 80.0, 8.0, 8.0, 0.0]]]
    )

    loss_good, _ = loss_fn(build(good, 4.0), targets)
    loss_bad, _ = loss_fn(build(bad, 4.0), targets)
    assert loss_good.item() < loss_bad.item()


def test_loss_rejects_wrong_target_width(tiny_model):
    loss_fn = YOLONASOBBLoss(num_classes=2)
    outputs = tiny_model(torch.rand(1, 3, 128, 128))
    with pytest.raises(ValueError, match=r"\(B, max_labels, 6\)"):
        loss_fn(outputs, torch.zeros(1, 3, 5))


def test_loss_rejects_traced_output(tiny_model):
    loss_fn = YOLONASOBBLoss(num_classes=2)
    with pytest.raises(TypeError, match="eager output"):
        loss_fn((torch.zeros(1, 2, 5), torch.zeros(1, 2, 2)), _targets()[:1])


def test_loss_handles_an_all_background_image(tiny_model):
    loss_fn = YOLONASOBBLoss(num_classes=2)
    outputs = tiny_model(torch.rand(1, 3, 128, 128))
    loss, log = loss_fn(outputs, torch.zeros(1, 4, 6))
    assert torch.isfinite(loss)
    assert log[1].item() == pytest.approx(0.0, abs=1e-6)  # no IoU term without GT


# ---------------------------------------------------------------------------
# Train transform
# ---------------------------------------------------------------------------


def _dataset_row(cx, cy, w, h, angle, cls=0):
    """A dataset row: proxy xyxy (which encodes cx/cy/w/h) + class + angle."""
    return np.array(
        [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cls, angle]],
        dtype=np.float32,
    )


def test_transform_emits_class_first_six_column_targets():
    transform = YOLONASOBBTrainTransform(max_labels=4, flip_prob=0.0, hsv_prob=0.0)
    image = np.zeros((512, 1024, 3), dtype=np.uint8)
    out_img, targets = transform(
        image, _dataset_row(100, 200, 40, 20, 0.3, cls=3), 1024
    )

    assert out_img.shape == (3, 1024, 1024)
    assert targets.shape == (4, 6)
    assert targets[0].tolist() == pytest.approx(
        [3.0, 100.0, 200.0, 40.0, 20.0, 0.3], abs=1e-4
    )
    assert targets[1:].sum() == 0  # padding


def test_transform_rescales_with_the_longest_side():
    transform = YOLONASOBBTrainTransform(max_labels=2, flip_prob=0.0, hsv_prob=0.0)
    image = np.zeros((1024, 2048, 3), dtype=np.uint8)  # ratio 0.5 to a 1024 canvas
    _img, targets = transform(image, _dataset_row(100, 200, 40, 20, 0.3), 1024)
    assert targets[0][1:5].tolist() == pytest.approx(
        [50.0, 100.0, 20.0, 10.0], abs=1e-3
    )
    assert targets[0][5] == pytest.approx(
        0.3, abs=1e-5
    )  # uniform scale keeps the angle


def test_horizontal_flip_mirrors_the_centre_and_negates_the_angle():
    transform = YOLONASOBBTrainTransform(max_labels=2, flip_prob=1.0, hsv_prob=0.0)
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    _img, targets = transform(image, _dataset_row(100, 200, 40, 20, 0.3), 1024)
    assert targets[0][1].item() == pytest.approx(1024 - 100, abs=1e-3)
    assert targets[0][2].item() == pytest.approx(200, abs=1e-3)
    assert targets[0][5].item() == pytest.approx(-0.3, abs=1e-5)


def test_vertical_flip_mirrors_the_centre_and_negates_the_angle():
    transform = YOLONASOBBTrainTransform(
        max_labels=2, flip_prob=0.0, hsv_prob=0.0, flipud=1.0
    )
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    _img, targets = transform(image, _dataset_row(100, 200, 40, 20, 0.3), 1024)
    assert targets[0][1].item() == pytest.approx(100, abs=1e-3)
    assert targets[0][2].item() == pytest.approx(1024 - 200, abs=1e-3)
    assert targets[0][5].item() == pytest.approx(-0.3, abs=1e-5)


def test_transform_handles_an_empty_label_file():
    transform = YOLONASOBBTrainTransform(max_labels=3, flip_prob=0.5, hsv_prob=0.5)
    image = np.zeros((320, 480, 3), dtype=np.uint8)
    out_img, targets = transform(image, np.zeros((0, 6), dtype=np.float32), 640)
    assert out_img.shape == (3, 640, 640)
    assert targets.shape == (3, 6) and targets.sum() == 0


def test_transform_rejects_detection_shaped_rows():
    transform = YOLONASOBBTrainTransform()
    with pytest.raises(ValueError, match="six-column"):
        transform(
            np.zeros((64, 64, 3), dtype=np.uint8),
            np.zeros((1, 5), dtype=np.float32),
            640,
        )


def test_canonicalize_rows_preserves_polygons():
    rows = np.array(
        [
            [50.0, 60.0, 10.0, 30.0, 0.3],  # h > w: must swap
            [50.0, 60.0, 30.0, 10.0, 2.5],  # angle outside the range
        ],
        dtype=np.float32,
    )
    out = _canonicalize_obb_rows(rows)
    assert (out[:, 2] >= out[:, 3]).all()
    assert ((out[:, 4] >= -math.pi / 2) & (out[:, 4] < math.pi / 2)).all()
    for before, after in zip(rows, out):
        assert np.allclose(
            np.sort(xywhr_to_corners(before).reshape(-1)),
            np.sort(xywhr_to_corners(after).reshape(-1)),
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# Config + trainer wiring
# ---------------------------------------------------------------------------


def test_obb_config_matches_the_upstream_recipe():
    config = YOLONASOBBConfig(size="s", num_classes=18, data="x.yaml")
    assert (
        config.classification_loss_weight,
        config.iou_loss_weight,
        config.dfl_loss_weight,
        config.bbox_assigner_topk,
    ) == (2.5, 2.0, 0.5, 12)
    assert (config.lr0, config.weight_decay, config.min_lr_ratio) == (
        5e-5,
        3.5e-6,
        0.1,
    )
    assert config.optimizer == "adamw"
    assert config.amp is False
    assert config.ema_decay == 0.9997
    # Rotated boxes get no mosaic/mixup.
    assert config.mosaic_prob == 0.0 and config.mixup_prob == 0.0


def test_obb_trainer_selects_the_rotated_metric():
    from libreyolo.models.yolonas.obb_trainer import YOLONASOBBTrainer

    assert YOLONASOBBTrainer.best_metric_key == "metrics/mAP50-95(OBB)"
    assert YOLONASOBBTrainer._config_class() is YOLONASOBBConfig


def test_detect_to_obb_transfer_copies_shared_weights_only():
    torch.manual_seed(0)
    detect = LibreYOLONAS(None, size="s", task="detect", nb_classes=18)
    obb = LibreYOLONAS(None, size="s", task="obb", nb_classes=18)

    before = obb.model.heads.head1.rot_pred.weight.detach().clone()
    report = obb.load_detect_weights_for_obb(detect.model.state_dict())

    assert any(k.startswith("backbone.") for k in report["transferred"])
    assert any(k.startswith("neck.") for k in report["transferred"])
    # cls_pred has the same shape at equal class counts and does transfer.
    assert "heads.head1.cls_pred.weight" in report["transferred"]
    # The rotated regression branch changes width and must NOT be copied.
    assert "heads.head1.reg_pred.weight" in report["skipped_shape"]
    # rot/offset do not exist upstream at all, so they stay fresh.
    assert "heads.head1.rot_pred.weight" in report["missing"]
    assert torch.equal(obb.model.heads.head1.rot_pred.weight, before)

    stem = "backbone.stem.conv.branch_3x3.conv.weight"
    assert torch.equal(obb.model.state_dict()[stem], detect.model.state_dict()[stem])


def test_detect_to_obb_transfer_rejects_a_rotated_source():
    obb = LibreYOLONAS(None, size="s", task="obb", nb_classes=18)
    with pytest.raises(ValueError, match="got a rotated"):
        obb.load_detect_weights_for_obb(obb.model.state_dict())


def test_detect_to_obb_transfer_rejects_a_detect_target():
    detect = LibreYOLONAS(None, size="s", task="detect", nb_classes=18)
    with pytest.raises(ValueError, match="only valid on a task='obb'"):
        detect.load_detect_weights_for_obb(detect.model.state_dict())


def test_obb_uses_its_own_validation_preprocessor():
    """Validation geometry must match the OBB inference recipe, not detect's.

    Detect resizes the longest side to 636 and centre-pads; OBB resizes to
    1024 and pads bottom-right. ``postprocess_obb`` inverts the OBB recipe, so
    validating through the detect preprocessor mis-maps every box back onto
    the canvas -- observed as OBB mAP collapsing to ~0 while the training loss
    fell normally.
    """
    from libreyolo.validation.preprocessors import (
        YOLONASOBBValPreprocessor,
        YOLONASValPreprocessor,
    )

    obb = LibreYOLONAS(None, size="s", task="obb", nb_classes=2)
    detect = LibreYOLONAS(None, size="s", task="detect", nb_classes=2)

    assert isinstance(obb._get_val_preprocessor(), YOLONASOBBValPreprocessor)
    assert type(detect._get_val_preprocessor()) is YOLONASValPreprocessor

    # Bottom-right padding means no offset to subtract on the way back.
    ratio, off_x, off_y = obb._get_val_preprocessor().letterbox_scale(512, 1024, 1024)
    assert (off_x, off_y) == (0.0, 0.0)
    assert ratio == pytest.approx(1.0)


def test_obb_val_preprocessor_matches_the_inference_preprocessing():
    """Byte-identical to the predict path, which is why it delegates to it."""
    from libreyolo.preprocess.yolonas import preprocess_obb_numpy
    from libreyolo.validation.preprocessors import YOLONASOBBValPreprocessor

    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 255, (300, 500, 3), dtype=np.uint8)

    processed, targets = YOLONASOBBValPreprocessor(img_size=(1024, 1024))(
        bgr, np.zeros((0, 6), dtype=np.float32), (1024, 1024)
    )
    expected, ratio = preprocess_obb_numpy(
        np.ascontiguousarray(bgr[:, :, ::-1]), input_size=1024
    )
    assert np.array_equal(processed, expected)
    assert targets.shape[1] == 6
    assert ratio == pytest.approx(1024 / 500)


def test_obb_val_preprocessor_scales_targets_without_touching_the_angle():
    from libreyolo.validation.preprocessors import YOLONASOBBValPreprocessor

    bgr = np.zeros((512, 1024, 3), dtype=np.uint8)  # ratio 1.0 to a 1024 canvas
    targets = np.array([[10.0, 20.0, 50.0, 60.0, 1.0, 0.4]], dtype=np.float32)
    _img, out = YOLONASOBBValPreprocessor(img_size=(1024, 1024))(
        bgr, targets, (1024, 1024)
    )
    assert out[0][:4].tolist() == pytest.approx([10.0, 20.0, 50.0, 60.0])
    assert out[0][5] == pytest.approx(0.4)
