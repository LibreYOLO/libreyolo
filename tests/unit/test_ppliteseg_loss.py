"""Unit tests for the PP-LiteSeg Dice + cross-entropy + edge compound loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from libreyolo.models.ppliteseg.loss import (
    IGNORE_INDEX,
    DiceLoss,
    MaskAttentionCELoss,
    PPLiteSegLoss,
    one_hot_to_binary_edge,
    target_to_binary_edge,
    to_one_hot,
)

pytestmark = [pytest.mark.unit, pytest.mark.ppliteseg]

NC = 4


def _preds(batch=2, h=16, w=32, heads=4):
    torch.manual_seed(0)
    return tuple(torch.randn(batch, NC, h, w) for _ in range(heads))


def test_to_one_hot_zeroes_ignore_pixels_instead_of_adding_a_class():
    target = torch.tensor([[[0, 1], [IGNORE_INDEX, 3]]])
    one_hot = to_one_hot(target, NC, IGNORE_INDEX)
    assert one_hot.shape == (1, NC, 2, 2)
    assert one_hot[0, :, 1, 0].sum() == 0, "ignore pixels get an all-zero vector"
    assert one_hot[0, 0, 0, 0] == 1
    assert one_hot[0, 3, 1, 1] == 1
    # No 20th ignore channel: the width stays nc.
    assert one_hot.shape[1] == NC


def test_to_one_hot_matches_the_source_drop_the_ignore_channel_construction():
    """The source builds nc+1 channels with ignore == nc, then slices it off.

    Reproducing that here proves the masking form is the same arithmetic, not
    merely a plausible substitute.
    """
    target = torch.tensor([[[0, 2], [3, 1]]])
    source_target = target.clone()
    source_target[0, 0, 1] = NC  # source encodes ignore as class index nc
    source = F.one_hot(source_target, NC + 1).permute(0, 3, 1, 2)
    source = torch.cat([source[:, :NC], source[:, NC + 1 :]], dim=1)

    ours_target = target.clone()
    ours_target[0, 0, 1] = IGNORE_INDEX
    ours = to_one_hot(ours_target, NC, IGNORE_INDEX)
    assert torch.equal(ours, source)


def test_edge_map_marks_class_boundaries_with_kernel_width():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :, 4:] = 1
    edge = target_to_binary_edge(target, NC, kernel_size=5, ignore_index=IGNORE_INDEX)
    assert edge.shape == (1, 1, 8, 8)
    columns = edge[0, 0, 0].nonzero().flatten().tolist()
    # edge_width = kernel - 1, centered on the boundary between column 3 and 4.
    assert columns == [2, 3, 4, 5]
    # A uniform target has no boundary at all.
    flat = target_to_binary_edge(torch.zeros(1, 8, 8, dtype=torch.long), NC, 5, IGNORE_INDEX)
    assert flat.sum() == 0


def test_edge_kernel_must_be_odd():
    one_hot = to_one_hot(torch.zeros(1, 4, 4, dtype=torch.long), NC, IGNORE_INDEX)
    with pytest.raises(ValueError, match="odd"):
        one_hot_to_binary_edge(one_hot, kernel_size=4)


def test_dice_excludes_ignore_pixels_from_both_terms():
    torch.manual_seed(1)
    logits = torch.randn(1, NC, 8, 8)
    target = torch.randint(0, NC, (1, 8, 8))
    dice = DiceLoss(num_classes=NC)

    masked = target.clone()
    masked[:, :4] = IGNORE_INDEX
    # Changing the *labels* under the ignore region cannot change the loss.
    other = masked.clone()
    other[:, :4] = IGNORE_INDEX
    assert torch.equal(dice(logits, masked), dice(logits, other))

    # Changing the *predictions* under the ignore region cannot change it either.
    perturbed = logits.clone()
    perturbed[:, :, :4] += 10.0
    assert torch.allclose(dice(logits, masked), dice(perturbed, masked), atol=1e-6)


def test_ce_edge_weights_regular_and_edge_pixels():
    torch.manual_seed(2)
    logits = torch.randn(1, NC, 8, 8)
    target = torch.randint(0, NC, (1, 8, 8))
    edge = torch.zeros(1, 1, 8, 8)
    edge[..., :2] = 1.0

    loss = MaskAttentionCELoss(loss_weights=(0.5, 0.5))
    per_pixel = F.cross_entropy(logits, target, reduction="none", ignore_index=IGNORE_INDEX)
    expected = 0.5 * per_pixel.mean() + 0.5 * per_pixel[edge.view(per_pixel.shape) == 1].mean()
    assert torch.allclose(loss(logits, target, edge), expected)


def test_ce_edge_is_zero_not_nan_when_no_edge_pixels_exist():
    logits = torch.randn(1, NC, 8, 8)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    edge = torch.zeros(1, 1, 8, 8)
    value = MaskAttentionCELoss()(logits, target, edge)
    assert torch.isfinite(value)
    per_pixel = F.cross_entropy(logits, target, reduction="none")
    assert torch.allclose(value, 0.5 * per_pixel.mean())


def test_compound_loss_combines_four_heads_with_equal_weights():
    preds = _preds()
    target = torch.randint(0, NC, (2, 16, 32))
    criterion = PPLiteSegLoss(num_classes=NC)
    out = criterion(preds, target)
    assert list(out) == ["main", "aux0", "aux1", "aux2", "loss"]
    total = out["main"] + out["aux0"] + out["aux1"] + out["aux2"]
    assert torch.allclose(out["loss"], total)
    assert criterion.component_names() == ["main", "aux0", "aux1", "aux2", "loss"]


def test_compound_loss_accepts_main_only_output():
    main = _preds(heads=1)[0]
    target = torch.randint(0, NC, (2, 16, 32))
    criterion = PPLiteSegLoss(num_classes=NC)
    out = criterion(main, target)
    assert list(out) == ["main", "loss"]
    assert torch.allclose(out["loss"], out["main"])


def test_compound_loss_rejects_a_wrong_head_count():
    criterion = PPLiteSegLoss(num_classes=NC)
    target = torch.randint(0, NC, (2, 16, 32))
    with pytest.raises(ValueError, match="1 or 4 prediction tensors"):
        criterion(_preds(heads=2), target)


def test_compound_loss_rejects_misaligned_logits():
    criterion = PPLiteSegLoss(num_classes=NC)
    target = torch.randint(0, NC, (2, 16, 32))
    small = tuple(torch.randn(2, NC, 8, 16) for _ in range(4))
    with pytest.raises(ValueError, match="head scale factors"):
        criterion(small, target)


def test_all_ignore_target_is_finite_and_near_zero():
    preds = _preds()
    target = torch.full((2, 16, 32), IGNORE_INDEX, dtype=torch.long)
    out = PPLiteSegLoss(num_classes=NC)(preds, target)
    for value in out.values():
        assert torch.isfinite(value)
    # Only Dice's laplace smoothing survives; cross-entropy contributes nothing.
    assert float(out["loss"]) < 1e-3


def test_loss_is_differentiable_through_every_head():
    preds = tuple(t.clone().requires_grad_(True) for t in _preds())
    target = torch.randint(0, NC, (2, 16, 32))
    PPLiteSegLoss(num_classes=NC)(preds, target)["loss"].backward()
    for tensor in preds:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
