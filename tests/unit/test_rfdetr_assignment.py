"""Batched assignment packing parity, independent of the optional GPU package."""

import sys
from types import SimpleNamespace

import pytest
import torch
from scipy.optimize import linear_sum_assignment

from libreyolo.models.rfdetr._assignment import assign_many_bucketed

pytestmark = pytest.mark.unit


@pytest.fixture
def scipy_batch_backend(monkeypatch):
    def batch_linear_assignment(costs):
        assignments = torch.full(costs.shape[:2], -1, dtype=torch.long)
        for i, cost in enumerate(costs):
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
            assignments[i, torch.as_tensor(rows)] = torch.as_tensor(cols)
        return assignments.to(costs.device)

    monkeypatch.setitem(
        sys.modules,
        "torch_linear_assignment",
        SimpleNamespace(batch_linear_assignment=batch_linear_assignment),
    )


@pytest.mark.parametrize(
    "sizes,widths",
    [([0, 1, 2], [4, 4]), ([2, 7, 0], [3, 3]), ([1, 3], [2, 5]), ([0, 0], [4, 6])],
)
@pytest.mark.parametrize("groups", [1, 3, 13])
@pytest.mark.parametrize("tied", [False, True])
def test_batched_packing_preserves_assignment_objective(
    scipy_batch_backend, sizes, widths, groups, tied
):
    generator = torch.Generator().manual_seed(17)
    costs = [
        torch.zeros(width * groups, sum(sizes))
        if tied
        else torch.rand(width * groups, sum(sizes), generator=generator)
        for width in widths
    ]
    result = assign_many_bucketed(costs, sizes, groups)
    for cost, width, layer in zip(costs, widths, result):
        offset = 0
        for size, (rows, cols) in zip(sizes, layer):
            n = min(size, width)
            assert rows.numel() == cols.numel() == groups * n
            assert rows.dtype == cols.dtype == torch.int64
            for group in range(groups):
                r, c = (
                    rows[group * n : (group + 1) * n] - group * width,
                    cols[group * n : (group + 1) * n],
                )
                block = cost[
                    group * width : (group + 1) * width, offset : offset + size
                ]
                rr, cc = linear_sum_assignment(block.numpy())
                assert r.unique().numel() == c.unique().numel() == n
                assert ((r >= 0) & (r < width)).all() and ((c >= 0) & (c < size)).all()
                torch.testing.assert_close(
                    block[r, c].sum(), block[rr, cc].sum(), rtol=0, atol=0
                )
            offset += size


def test_padding_budget_uses_exact_size_buckets(scipy_batch_backend, monkeypatch):
    from libreyolo.models.rfdetr import _assignment

    costs = [torch.rand(12, 6)]
    expected = assign_many_bucketed(costs, [1, 2, 3], 3)
    monkeypatch.setattr(_assignment, "_PADDED_ASSIGNMENT_ELEMENT_LIMIT", 0)
    actual = assign_many_bucketed(costs, [1, 2, 3], 3)
    for a, b in zip(actual[0], expected[0]):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
