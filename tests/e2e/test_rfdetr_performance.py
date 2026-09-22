"""CUDA checks for RF-DETR's optional and default performance paths.

Run separately from other e2e files, with libreyolo[rfdetr-accel] installed.
These are opt-in regression checks, not a throughput benchmark or an accuracy
claim. The CPU PR gate covers the same contracts with portable fixtures.
"""

import pytest
import torch

pytestmark = [pytest.mark.e2e, pytest.mark.rfdetr, pytest.mark.extended_training]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("segmentation", [False, True])
def test_cuda_compact_costs_and_assignments_match_full_matrix(dtype, segmentation):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("requires BF16 CUDA support")
    from libreyolo.models.rfdetr.matcher import HungarianMatcher

    generator = torch.Generator().manual_seed(698)
    sizes = [0, 7, 12]
    targets = [
        {
            "labels": torch.randint(0, 5, (n,), generator=generator).cuda(),
            "boxes": (torch.rand(n, 4, generator=generator) + 0.1).cuda(),
        }
        for n in sizes
    ]
    outputs = {
        "pred_logits": torch.randn(3, 39, 5, generator=generator).cuda().to(dtype),
        "pred_boxes": (torch.rand(3, 39, 4, generator=generator) + 0.1)
        .cuda()
        .to(dtype),
    }
    if segmentation:
        for target in targets:
            target["masks"] = (
                torch.rand(len(target["boxes"]), 16, 16, generator=generator) > 0.5
            ).cuda()
        outputs["pred_masks"] = torch.randn(3, 39, 8, 8, generator=generator).cuda()
    matcher = HungarianMatcher()
    torch.cuda.manual_seed(699)
    full = matcher.compute_cost_matrix(outputs, targets)
    expected = matcher.solve(full.cpu(), targets, 3)
    diagonal = torch.cat(
        [block[i] for i, block in enumerate(full.split(sizes, -1))], -1
    )
    torch.cuda.manual_seed(699)
    compact = matcher._compact_cost(outputs, targets)
    torch.testing.assert_close(compact, diagonal, rtol=0, atol=0)
    torch.cuda.manual_seed(699)
    actual = matcher.match_many([outputs], targets, 3)[0]
    for left, right in zip(actual, expected):
        assert all(torch.equal(a, b) for a, b in zip(left, right))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("sizes", [[0, 0], [0, 3, 7], [13, 2]])
@pytest.mark.parametrize("tied", [False, True])
def test_cuda_assignment_matches_scipy(sizes, tied):
    pytest.importorskip(
        "torch_linear_assignment", reason="install libreyolo[rfdetr-accel]"
    )
    from scipy.optimize import linear_sum_assignment

    from libreyolo.models.rfdetr.assignment import assign_compact

    generator = torch.Generator().manual_seed(701)
    matrices = [torch.rand(q, sum(sizes), generator=generator) for q in (21, 15)]
    if tied:
        matrices = [torch.zeros_like(matrix) for matrix in matrices]
    actual = assign_compact([matrix.cuda() for matrix in matrices], sizes, 3)
    for output_layer, matrix in zip(actual, matrices):
        width = matrix.shape[0] // 3
        for output_pair, image in zip(output_layer, matrix.split(sizes, dim=1)):
            expected = [
                linear_sum_assignment(block.numpy()) for block in image.split(width)
            ]
            rows = torch.cat(
                [
                    torch.as_tensor(pair[0]) + index * width
                    for index, pair in enumerate(expected)
                ]
            )
            columns = torch.cat([torch.as_tensor(pair[1]) for pair in expected])
            assert torch.equal(output_pair[0], rows)
            assert torch.equal(output_pair[1], columns)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_pairwise_loss_values_and_gradients(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("requires BF16 CUDA support")
    from libreyolo.models.rfdetr import box_ops

    generator = torch.Generator().manual_seed(711)
    boxes = [
        box_ops.box_cxcywh_to_xyxy(torch.rand(41, 4, generator=generator) + 0.1)
        .cuda()
        .to(dtype)
        .requires_grad_()
        for _ in range(2)
    ]
    reference = box_ops.generalized_box_iou(*boxes).diag()
    reference_grad = torch.autograd.grad(reference.sum(), boxes)
    actual = box_ops.elementwise_generalized_box_iou(*boxes)
    actual_grad = torch.autograd.grad(actual.sum(), boxes)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for left, right in zip(actual_grad, reference_grad):
        torch.testing.assert_close(
            left,
            right,
            rtol=1e-4 if dtype == torch.float32 else 5e-3,
            atol=1e-5 if dtype == torch.float32 else 5e-3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_mask_sampling_matches_batch_reference():
    from libreyolo.models.rfdetr.mask_ops import sample_target_masks_at_points
    from libreyolo.models.rfdetr.segmentation import point_sample

    generator = torch.Generator().manual_seed(719)
    targets = [
        {"masks": (torch.rand(5, 257, 321, generator=generator) > 0.5).cuda()}
        for _ in range(3)
    ]
    indices = [(torch.arange(3), torch.tensor([0, 3, 0])) for _ in targets]
    coords = torch.rand(9, 127, 2, generator=generator).cuda()
    reference = point_sample(
        torch.cat([t["masks"][j] for t, (_, j) in zip(targets, indices)])[
            :, None
        ].float(),
        coords,
        align_corners=False,
        mode="nearest",
    ).squeeze(1)
    assert torch.equal(
        sample_target_masks_at_points(targets, indices, coords), reference
    )
