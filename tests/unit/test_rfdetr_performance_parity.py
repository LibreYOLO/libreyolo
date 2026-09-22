"""Numerical and resume contracts for RF-DETR performance paths."""

from copy import deepcopy

import pytest
import torch

from libreyolo.models.rfdetr import box_ops

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [0, 1, 37])
def test_aligned_iou_matches_pairwise_values_and_gradients(dtype, count):
    generator = torch.Generator().manual_seed(4)
    boxes = [
        box_ops.box_cxcywh_to_xyxy(
            torch.rand(count, 4, generator=generator, dtype=dtype) + 0.1
        ).requires_grad_()
        for _ in range(2)
    ]
    reference = torch.diag(box_ops.generalized_box_iou(*boxes))
    expected_grad = torch.autograd.grad(reference.sum(), boxes)
    actual = box_ops.elementwise_generalized_box_iou(*boxes)
    actual_grad = torch.autograd.grad(actual.sum(), boxes)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for actual_g, expected_g in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_g, expected_g, rtol=1e-6, atol=1e-6)
    iou, union = box_ops.box_iou(*boxes)
    aligned_iou, aligned_union = box_ops.elementwise_box_iou(*boxes)
    torch.testing.assert_close(aligned_iou, iou.diag(), rtol=0, atol=0)
    torch.testing.assert_close(aligned_union, union.diag(), rtol=0, atol=0)


def test_aligned_iou_preserves_degenerate_values_and_rejects_broadcasting():
    boxes = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    actual = box_ops.elementwise_generalized_box_iou(boxes, boxes)
    reference = box_ops.generalized_box_iou(boxes, boxes).diag()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)
    with pytest.raises(ValueError, match="same shape"):
        box_ops.elementwise_box_iou(boxes, boxes[:1])


@pytest.mark.parametrize("budget", [16, 10000])
def test_bounded_l1_preserves_broadcast_math(monkeypatch, budget):
    monkeypatch.setattr(box_ops, "_L1_ELEMENT_BUDGET", budget)
    generator = torch.Generator().manual_seed(42)
    boxes = torch.rand(3, 13, 4, generator=generator)
    targets = torch.rand(3, 17, 4, generator=generator)
    expected = (boxes[:, :, None] - targets[:, None]).abs().sum(-1)
    torch.testing.assert_close(
        box_ops.pairwise_box_l1_cost(boxes, targets), expected, rtol=0, atol=0
    )
    assert box_ops.pairwise_box_l1_cost(boxes, targets[:, :0]).shape == (3, 13, 0)


def test_merged_optimizer_resumes_legacy_moments_and_next_update():
    from libreyolo.models.rfdetr.optim import build_rfdetr_optimizer
    from libreyolo.training.optim import restore_optimizer_state

    parameters = [torch.nn.Parameter(torch.tensor([float(i)])) for i in range(6)]

    def groups(params):
        return [
            {
                "params": p,
                "lr": 0.001 * (1 + i % 2),
                "lr_mult": float(1 + i % 2),
                "weight_decay": 0.01,
            }
            for i, p in enumerate(params)
        ]

    reference = torch.optim.AdamW(groups(parameters))
    for p in parameters:
        p.grad = torch.ones_like(p)
    reference.step()
    state = deepcopy(reference.state_dict())
    resumed_params = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    candidate = build_rfdetr_optimizer(groups(resumed_params))
    assert len(candidate.param_groups) == 2
    restore_optimizer_state(candidate, state)
    for left, right in zip(parameters, resumed_params):
        left.grad = torch.full_like(left, 0.3)
        right.grad = left.grad.clone()
    reference.step()
    candidate.step()
    for left, right in zip(parameters, resumed_params):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        for key in reference.state[left]:
            torch.testing.assert_close(
                reference.state[left][key], candidate.state[right][key], rtol=0, atol=0
            )
    # Migration must not mutate the caller's checkpoint.
    assert len(state["param_groups"]) == 6
    assert all(len(g["params"]) == 1 for g in state["param_groups"])


@pytest.mark.parametrize("counts", [[0, 0, 0], [0, 7, 1], [15, 12, 10]])
@pytest.mark.parametrize("groups", [1, 3])
def test_compact_matching_preserves_full_matrix_assignments(counts, groups):
    from libreyolo.models.rfdetr.matcher import HungarianMatcher

    generator = torch.Generator().manual_seed(95)
    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
    targets = [
        {
            "labels": torch.randint(0, 5, (count,), generator=generator),
            "boxes": torch.rand(count, 4, generator=generator) + 0.01,
        }
        for count in counts
    ]
    levels = [
        {
            "pred_logits": torch.randn(3, 21 * groups, 5, generator=generator),
            "pred_boxes": torch.rand(3, 21 * groups, 4, generator=generator) + 0.01,
        }
        for _ in range(3)
    ]
    expected = [
        matcher.solve(matcher.compute_cost_matrix(level, targets), targets, groups)
        for level in levels
    ]
    actual = matcher.match_many(levels, targets, groups)
    for expected_layer, actual_layer in zip(expected, actual):
        for expected_pair, actual_pair in zip(expected_layer, actual_layer):
            assert all(
                torch.equal(left, right)
                for left, right in zip(expected_pair, actual_pair)
            )
    full = matcher.compute_cost_matrix(levels[0], targets)
    diagonal = torch.cat(
        [block[i] for i, block in enumerate(full.split(counts, dim=-1))], dim=-1
    )
    torch.testing.assert_close(
        matcher._compact_cost(levels[0], targets), diagonal, rtol=0, atol=0
    )


@pytest.mark.parametrize("width", [32, 673])
def test_mask_sampling_preserves_native_rounding(width):
    from libreyolo.models.rfdetr.mask_ops import sample_target_masks_at_points
    from libreyolo.models.rfdetr.segmentation import point_sample

    generator = torch.Generator().manual_seed(8)
    masks = torch.rand(4, width, width, generator=generator) > 0.5
    targets = [{"masks": masks}, {"masks": masks.flip(0)}]
    indices = [(torch.arange(3), torch.tensor([2, 0, 2]))] * 2
    coords = torch.rand(6, 129, 2, generator=generator)
    coords[:, 0] = 0.5
    coords[:, 1] = -0.1
    coords[:, 2] = 1.1
    reference = point_sample(
        torch.cat([t["masks"][j] for t, (_, j) in zip(targets, indices)])[
            :, None
        ].float(),
        coords,
        align_corners=False,
        mode="nearest",
    ).squeeze(1)
    torch.testing.assert_close(
        sample_target_masks_at_points(targets, indices, coords),
        reference,
        rtol=0,
        atol=0,
    )


def test_mask_loss_runs_sampling_and_backward():
    from libreyolo.models.rfdetr.loss import SetCriterion
    from libreyolo.models.rfdetr.matcher import HungarianMatcher

    criterion = SetCriterion(2, HungarianMatcher(), {}, 0.25, ["masks"])
    masks = torch.randn(2, 4, 8, 8, requires_grad=True)
    targets = [{"masks": torch.ones(1, 16, 16, dtype=torch.bool)} for _ in range(2)]
    indices = [(torch.tensor([0]), torch.tensor([0])) for _ in targets]
    losses = criterion.loss_masks(
        {"pred_masks": masks}, targets, indices, torch.tensor(2.0)
    )
    sum(losses.values()).backward()
    assert torch.isfinite(masks.grad).all()
    assert masks.grad[:, 0].abs().sum() > 0


@pytest.mark.parametrize("threshold", [0.0, 0.8, 1.0])
def test_mask_postprocess_filters_before_resize_without_changing_results(threshold):
    from libreyolo.postprocess.rfdetr import postprocess

    generator = torch.Generator().manual_seed(19)
    outputs = {
        "pred_logits": torch.randn(2, 40, 3, generator=generator),
        "pred_boxes": torch.rand(2, 40, 4, generator=generator),
        "pred_masks": torch.randn(2, 40, 8, 8, generator=generator),
    }
    sizes = torch.tensor([[57, 83], [64, 77]])
    reference = postprocess(outputs, sizes, num_select=40)
    candidate = postprocess(
        outputs, sizes, num_select=40, mask_score_threshold=threshold
    )
    for expected, actual in zip(reference, candidate):
        keep = expected["scores"] > threshold
        for key in expected:
            assert torch.equal(actual[key], expected[key][keep])


def test_compact_matching_preserves_nonfinite_fallback():
    from libreyolo.models.rfdetr.matcher import HungarianMatcher

    matcher = HungarianMatcher()
    targets = [
        {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}
    ] * 2
    outputs = {
        "pred_logits": torch.tensor([[[float("nan")], [1.0]], [[2.0], [3.0]]]),
        "pred_boxes": torch.full((2, 2, 4), 0.5),
    }
    expected = matcher.solve(matcher.compute_cost_matrix(outputs, targets), targets)
    actual = matcher.match_many([outputs], targets)[0]
    for left, right in zip(actual, expected):
        assert all(torch.equal(a, b) for a, b in zip(left, right))


@pytest.mark.parametrize("nonfinite", [False, True])
def test_compact_segmentation_preserves_assignments_and_rng(nonfinite):
    from libreyolo.models.rfdetr.matcher import HungarianMatcher

    generator = torch.Generator().manual_seed(83)
    matcher = HungarianMatcher()
    targets = [
        {
            "labels": torch.tensor([0, 1]),
            "boxes": torch.rand(2, 4, generator=generator) + 0.1,
            "masks": torch.rand(2, 8, 8, generator=generator) > 0.5,
        }
        for _ in range(2)
    ]
    levels = [
        {
            "pred_logits": torch.randn(2, 12, 2, generator=generator),
            "pred_boxes": torch.rand(2, 12, 4, generator=generator) + 0.1,
            "pred_masks": torch.randn(2, 12, 8, 8, generator=generator),
        }
        for _ in range(2)
    ]
    if nonfinite:
        levels[0]["pred_logits"][0, 0, 0] = float("nan")
    torch.manual_seed(123)
    expected = [
        matcher.solve(matcher.compute_cost_matrix(level, targets), targets, 3)
        for level in levels
    ]
    expected_rng = torch.get_rng_state()
    torch.manual_seed(123)
    actual = matcher.match_many(levels, targets, 3)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for expected_layer, actual_layer in zip(expected, actual):
        for expected_pair, actual_pair in zip(expected_layer, actual_layer):
            assert all(torch.equal(a, b) for a, b in zip(expected_pair, actual_pair))


@pytest.mark.parametrize("sizes", [[0, 0], [0, 3, 5], [12, 1]])
def test_batched_assignment_adapter_matches_scipy(monkeypatch, sizes):
    import sys
    from types import SimpleNamespace

    from scipy.optimize import linear_sum_assignment

    from libreyolo.models.rfdetr.assignment import assign_compact

    # An independent SciPy oracle exercises the adapter without requiring the
    # optional dependency in the hermetic PR gate.
    calls = []

    def solve(costs):
        calls.append(tuple(costs.shape))
        result = torch.full(costs.shape[:2], -1, dtype=torch.int64)
        for index, matrix in enumerate(costs):
            rows, columns = linear_sum_assignment(matrix.numpy())
            result[index, rows] = torch.as_tensor(columns)
        return result

    monkeypatch.setitem(
        sys.modules,
        "torch_linear_assignment",
        SimpleNamespace(batch_linear_assignment=solve),
    )
    generator = torch.Generator().manual_seed(875)
    matrices = [
        torch.rand(21, sum(sizes), generator=generator),
        torch.rand(15, sum(sizes), generator=generator),
    ]
    actual = assign_compact(matrices, sizes, 3)
    for layer, matrix in zip(actual, matrices):
        width = matrix.shape[0] // 3
        for pair, image in zip(layer, matrix.split(sizes, dim=1)):
            expected = [
                linear_sum_assignment(part.numpy()) for part in image.split(width)
            ]
            rows = torch.cat(
                [torch.as_tensor(p[0]) + i * width for i, p in enumerate(expected)]
            )
            columns = torch.cat([torch.as_tensor(p[1]) for p in expected])
            assert torch.equal(pair[0], rows)
            assert torch.equal(pair[1], columns)
    if not any(sizes):
        assert not calls


@pytest.mark.parametrize("reparam", [True, False])
def test_batched_selection_preserves_loop_outputs_and_gradients(reparam):
    from libreyolo.models.rfdetr.lwdetr import MLP
    from libreyolo.models.rfdetr.selection import BatchedSelection

    class Selection(BatchedSelection, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.group_detr, self.d_model, self.num_queries = 3, 16, 5
            self.bbox_reparam = reparam
            self.enc_output = torch.nn.ModuleList(
                [torch.nn.Linear(16, 16) for _ in range(3)]
            )
            self.enc_output_norm = torch.nn.ModuleList(
                [torch.nn.LayerNorm(16) for _ in range(3)]
            )
            self.enc_out_class_embed = torch.nn.ModuleList(
                [torch.nn.Linear(16, 4) for _ in range(3)]
            )
            self.enc_out_bbox_embed = torch.nn.ModuleList(
                [MLP(16, 16, 4, 3) for _ in range(3)]
            )

    torch.manual_seed(314)
    candidate = Selection().double()
    reference = deepcopy(candidate)
    x = torch.randn(2, 11, 16, dtype=torch.float64)
    anchors = torch.rand(2, 11, 4, dtype=torch.float64)
    refs, memories, boxes, logits = [], [], [], []
    for group in range(3):
        memory = reference.enc_output_norm[group](reference.enc_output[group](x))
        classes = reference.enc_out_class_embed[group](memory)
        delta = reference.enc_out_bbox_embed[group](memory)
        coords = (
            torch.cat(
                (
                    delta[..., :2] * anchors[..., 2:] + anchors[..., :2],
                    delta[..., 2:].exp() * anchors[..., 2:],
                ),
                dim=-1,
            )
            if reparam
            else delta + anchors
        )
        top = classes.max(-1)[0].topk(5, dim=1)[1]
        selected = memory.gather(1, top[..., None].expand(-1, -1, 16))
        coord = coords.gather(1, top[..., None].expand(-1, -1, 4))
        refs.append(coord.detach())
        memories.append(selected)
        boxes.append(coord)
        logits.append(reference.enc_out_class_embed[group](selected))
    expected = [torch.cat(parts, dim=1) for parts in (refs, memories, boxes, logits)]
    assert candidate._two_stage_batching_eligible()
    actual = candidate._two_stage_group_selection(x, anchors, 3)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)
    sum(value.sum() for value in actual[1:]).backward()
    sum(value.sum() for value in expected[1:]).backward()
    for left, right in zip(candidate.parameters(), reference.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-10, atol=1e-10)
    handle = candidate.enc_output[0].register_forward_hook(lambda *args: None)
    assert not candidate._two_stage_batching_eligible()
    handle.remove()
