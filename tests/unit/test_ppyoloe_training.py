"""PP-YOLOE training path: loss, gradients, assigner schedule, augmentation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.models.ppyoloe.nn import LibrePPYOLOEModel
from libreyolo.models.ppyoloe.trainer import (
    SOURCE_RECIPE_LR0,
    SOURCE_STATIC_ASSIGNER_EPOCHS,
    SOURCE_TOTAL_EPOCHS,
    PPYOLOETrainer,
    resolve_static_assigner_epochs,
)
from libreyolo.models.ppyoloe.transforms import PPYOLOETrainTransform, rot90_with_boxes
from libreyolo.models.yolonas.loss import PPYoloELoss
from libreyolo.training.config import PPYOLOEConfig

pytestmark = [pytest.mark.unit, pytest.mark.ppyoloe]

NUM_CLASSES = 3
IMGSZ = 128


def _model():
    return LibrePPYOLOEModel(size="s", nb_classes=NUM_CLASSES).train()


def _targets(num_boxes: int = 2, batch: int = 2) -> torch.Tensor:
    """Padded ``(B, max_labels, 5)`` of ``[cls, cx, cy, w, h]`` pixel targets."""
    targets = torch.zeros(batch, 4, 5)
    for b in range(batch):
        for i in range(num_boxes):
            targets[b, i] = torch.tensor([i % NUM_CLASSES, 40.0 + 10 * i, 50.0, 30.0, 20.0])
    return targets


# ---------------------------------------------------------------------------
# Rung 0: loss, backward, optimizer step
# ---------------------------------------------------------------------------


def test_head_prediction_convs_are_zero_initialized_like_the_source():
    """The source zero-inits pred_cls/pred_reg weights and biases them instead.

    A consequence worth pinning: on a freshly built (non-pretrained) model the
    first backward reaches no backbone parameter, because the zero head kernels
    kill the gradient on their inputs. Fine-tuning starts from released weights
    where these kernels are non-zero, so this only affects step 0 from scratch.
    """
    model = LibrePPYOLOEModel(size="s", nb_classes=NUM_CLASSES)
    for conv in model.head.pred_cls:
        assert float(conv.weight.abs().sum()) == 0.0
        assert float(conv.bias.std()) == pytest.approx(0.0, abs=1e-6)
    for conv in model.head.pred_reg:
        assert float(conv.weight.abs().sum()) == 0.0
        torch.testing.assert_close(conv.bias, torch.ones_like(conv.bias))


def test_rung0_loss_backward_and_step_on_cpu():
    torch.manual_seed(0)
    model = _model()
    # Stand in for released weights: the source zero-inits the head kernels, so
    # a from-scratch model gives no backbone gradient on the very first step.
    with torch.no_grad():
        for conv in list(model.head.pred_cls) + list(model.head.pred_reg):
            conv.weight.normal_(0.0, 0.01)
    loss_fn = PPYoloELoss(num_classes=NUM_CLASSES, distributed_normalize=False)

    imgs = torch.randn(2, 3, IMGSZ, IMGSZ)
    total, components = loss_fn(model(imgs), _targets())

    assert torch.isfinite(total)
    assert torch.isfinite(components).all()
    total.backward()

    for name in ("backbone", "neck", "head.pred_cls", "head.pred_reg"):
        grads = [
            p.grad
            for n, p in model.named_parameters()
            if n.startswith(name) and p.grad is not None
        ]
        assert grads, f"no gradients reached {name}"
        assert any(float(g.abs().sum()) > 0 for g in grads), f"zero gradients in {name}"

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.head.pred_cls[0].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.head.pred_cls[0].weight.detach())


def test_targets_are_pixel_cxcywh_not_normalized():
    """Encode the target contract the loss expects, rather than trusting a comment."""
    from libreyolo.models.yolonas.loss import flatten_yolonas_targets
    from libreyolo.utils.general import cxcywh_to_xyxy

    flat = flatten_yolonas_targets(_targets(num_boxes=1, batch=1))
    xyxy = cxcywh_to_xyxy(flat[:, 2:6])
    # A normalized target would decode to a sub-pixel box; a pixel target does not.
    assert float(xyxy[0, 2] - xyxy[0, 0]) == pytest.approx(30.0)
    assert float(xyxy[0, 3] - xyxy[0, 1]) == pytest.approx(20.0)


def test_empty_and_mixed_batches_stay_finite():
    torch.manual_seed(1)
    model = _model()
    loss_fn = PPYoloELoss(num_classes=NUM_CLASSES, distributed_normalize=False)
    imgs = torch.randn(2, 3, IMGSZ, IMGSZ)
    outputs = model(imgs)

    empty = torch.zeros(2, 4, 5)
    total_empty, _ = loss_fn(outputs, empty)
    assert torch.isfinite(total_empty)

    mixed = torch.zeros(2, 4, 5)
    mixed[0, 0] = torch.tensor([1.0, 40.0, 50.0, 30.0, 20.0])
    total_mixed, _ = loss_fn(outputs, mixed)
    assert torch.isfinite(total_mixed)


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_multiple_batch_sizes_flow_through_loss(batch):
    torch.manual_seed(2)
    model = _model()
    loss_fn = PPYoloELoss(num_classes=NUM_CLASSES, distributed_normalize=False)
    total, _ = loss_fn(
        model(torch.randn(batch, 3, IMGSZ, IMGSZ)), _targets(batch=batch)
    )
    assert torch.isfinite(total)


def test_multi_scale_inputs_do_not_reuse_stale_anchors():
    torch.manual_seed(3)
    model = _model()
    loss_fn = PPYoloELoss(num_classes=NUM_CLASSES, distributed_normalize=False)
    for imgsz in (320, 448, 640):
        outputs = model(torch.randn(1, 3, imgsz, imgsz))
        anchors_per_level = list(outputs[4])
        assert anchors_per_level == [(imgsz // s) ** 2 for s in (32, 16, 8)]
        total, _ = loss_fn(outputs, _targets(batch=1))
        assert torch.isfinite(total)


# ---------------------------------------------------------------------------
# Two-stage assigner schedule
# ---------------------------------------------------------------------------


def test_source_switch_point_on_the_reference_schedule():
    resolved = resolve_static_assigner_epochs(
        SOURCE_STATIC_ASSIGNER_EPOCHS, SOURCE_TOTAL_EPOCHS
    )
    assert resolved == SOURCE_STATIC_ASSIGNER_EPOCHS

    trainer = PPYOLOETrainer.__new__(PPYOLOETrainer)
    trainer.static_assigner_epochs = resolved
    # Epoch 149 is still ATSS; epoch 150 is TaskAligned.
    assert trainer.uses_static_assigner(148) is True
    assert trainer.uses_static_assigner(149) is True
    assert trainer.uses_static_assigner(150) is False
    assert trainer.uses_static_assigner(499) is False


def test_default_scales_the_switch_to_the_requested_budget():
    # 30% of the budget, the same fraction as 150 of 500.
    assert resolve_static_assigner_epochs(None, 500) == 150
    assert resolve_static_assigner_epochs(None, 100) == 30
    assert resolve_static_assigner_epochs(None, 10) == 3


def test_explicit_value_is_honored_and_clamped():
    assert resolve_static_assigner_epochs(5, 100) == 5
    assert resolve_static_assigner_epochs(0, 100) == 0
    assert resolve_static_assigner_epochs(500, 10) == 10
    assert resolve_static_assigner_epochs(-3, 10) == 0


def test_assigner_phase_is_derived_from_the_epoch_so_resume_is_safe():
    trainer = PPYOLOETrainer.__new__(PPYOLOETrainer)
    trainer.static_assigner_epochs = 4
    # Jumping straight to a late epoch (as resume does) lands in the right phase
    # without replaying the earlier ones.
    assert trainer.uses_static_assigner(9) is False
    assert trainer.uses_static_assigner(1) is True


def test_both_assigners_produce_finite_losses():
    torch.manual_seed(4)
    model = _model()
    outputs = model(torch.randn(2, 3, IMGSZ, IMGSZ))
    targets = _targets()
    for static in (True, False):
        loss_fn = PPYoloELoss(
            num_classes=NUM_CLASSES,
            use_static_assigner=static,
            distributed_normalize=False,
        )
        total, _ = loss_fn(outputs, targets)
        assert torch.isfinite(total), f"use_static_assigner={static}"


# ---------------------------------------------------------------------------
# Config and recipe constants
# ---------------------------------------------------------------------------


def test_config_matches_the_source_recipe_where_it_claims_to():
    config = PPYOLOEConfig()
    assert config.optimizer == "adamw"
    assert config.weight_decay == 1e-4
    assert config.warmup_lr_start == 1e-6
    assert config.min_lr_ratio == 0.1
    assert config.ema_decay == 0.9997
    assert config.mixup_prob == 0.5
    assert config.mixup_scale == (0.5, 1.5)
    assert config.translate == 0.25
    assert config.mosaic_scale == (0.5, 1.5)
    assert config.degrees == 0.0 and config.shear == 0.0
    assert config.flip_prob == 0.5
    assert config.rot90_prob == 0.5
    assert config.rgb2bgr_prob == 0.25
    assert config.static_assigner_epochs is None
    assert SOURCE_RECIPE_LR0 == {"s": 2e-3, "m": 1e-3, "l": 1e-3, "x": 2e-3}


# ---------------------------------------------------------------------------
# Augmentation knobs measurably fire
# ---------------------------------------------------------------------------


def _sample():
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(80, 120, 3), dtype=np.uint8)
    targets = np.array([[10.0, 20.0, 60.0, 70.0, 1.0]], dtype=np.float32)
    return img, targets


def test_rot90_moves_pixels_and_boxes_consistently():
    img, targets = _sample()
    rotated, boxes = rot90_with_boxes(img, targets[:, :4].copy())
    assert rotated.shape[:2] == (img.shape[1], img.shape[0])
    # The rotated box must still bound the same pixels: check a corner maps.
    x1, y1, x2, y2 = targets[0, :4]
    np.testing.assert_allclose(boxes[0], [y1, img.shape[1] - x2, y2, img.shape[1] - x1])
    assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()


def test_flip_and_rot90_knobs_change_pixels_when_enabled():
    img, targets = _sample()
    off = PPYOLOETrainTransform(flip_prob=0.0, hsv_prob=0.0, rot90_prob=0.0, rgb2bgr_prob=0.0)
    baseline, base_labels = off(img.copy(), targets.copy(), (IMGSZ, IMGSZ))

    always = PPYOLOETrainTransform(flip_prob=1.0, hsv_prob=0.0, rot90_prob=0.0, rgb2bgr_prob=0.0)
    flipped, flip_labels = always(img.copy(), targets.copy(), (IMGSZ, IMGSZ))
    assert not np.allclose(baseline, flipped)
    assert not np.allclose(base_labels[0, 1], flip_labels[0, 1])

    rotated_t = PPYOLOETrainTransform(flip_prob=0.0, hsv_prob=0.0, rot90_prob=1.0, rgb2bgr_prob=0.0)
    rotated, rot_labels = rotated_t(img.copy(), targets.copy(), (IMGSZ, IMGSZ))
    assert not np.allclose(baseline, rotated)
    assert not np.allclose(base_labels[0, 1:], rot_labels[0, 1:])


def test_rgb2bgr_swaps_raw_channels_before_normalization():
    """The swap must land on raw pixels, not on the normalized tensor.

    Normalization statistics are per-channel, so swapping afterwards would pair
    each channel with the wrong mean and std. Undoing the normalization has to
    recover a channel-reversed copy of the un-swapped raw canvas.
    """
    from libreyolo.models.ppyoloe.utils import PPYOLOE_MEAN, PPYOLOE_STD

    img, targets = _sample()
    off = PPYOLOETrainTransform(flip_prob=0.0, hsv_prob=0.0, rot90_prob=0.0, rgb2bgr_prob=0.0)
    baseline, base_labels = off(img.copy(), targets.copy(), (IMGSZ, IMGSZ))
    on = PPYOLOETrainTransform(flip_prob=0.0, hsv_prob=0.0, rot90_prob=0.0, rgb2bgr_prob=1.0)
    swapped, swap_labels = on(img.copy(), targets.copy(), (IMGSZ, IMGSZ))

    mean = np.array(PPYOLOE_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(PPYOLOE_STD, dtype=np.float32).reshape(3, 1, 1)
    raw_baseline = baseline * std + mean
    raw_swapped = swapped * std + mean
    np.testing.assert_allclose(raw_swapped, raw_baseline[::-1], rtol=1e-4, atol=1e-2)
    # A post-normalization swap would instead satisfy swapped == baseline[::-1].
    assert not np.allclose(swapped, baseline[::-1])
    np.testing.assert_allclose(base_labels, swap_labels)


def test_train_transform_canvas_matches_the_val_normalization():
    """Train and val must land in the same colour space (landmines 1 and 17)."""
    from libreyolo.models.ppyoloe.utils import preprocess_numpy

    img, targets = _sample()
    transform = PPYOLOETrainTransform(
        flip_prob=0.0, hsv_prob=0.0, rot90_prob=0.0, rgb2bgr_prob=0.0
    )
    chw, _ = transform(img.copy(), targets.copy(), (IMGSZ, IMGSZ))
    infer_chw, _ = preprocess_numpy(img[:, :, ::-1].copy(), IMGSZ)
    np.testing.assert_allclose(chw, infer_chw, rtol=1e-5, atol=1e-4)


def test_targets_land_on_the_stretched_canvas():
    img, targets = _sample()
    transform = PPYOLOETrainTransform(
        flip_prob=0.0, hsv_prob=0.0, rot90_prob=0.0, rgb2bgr_prob=0.0
    )
    _, labels = transform(img.copy(), targets.copy(), (IMGSZ, IMGSZ))
    src_h, src_w = img.shape[:2]
    cx = (10.0 + 60.0) / 2 * (IMGSZ / src_w)
    cy = (20.0 + 70.0) / 2 * (IMGSZ / src_h)
    np.testing.assert_allclose(labels[0, 1:3], [cx, cy], rtol=1e-5)
    assert labels[0, 0] == 1.0
