"""ExecuTorch export decomposes grid sampling and leaves other ops alone.

The Linux ExecuTorch 1.2 runtime has no ``aten::grid_sampler_2d`` kernel, so
RF-DETR programs that kept the op failed to load. CPU only: needs neither
ExecuTorch nor downloaded weights.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from libreyolo.export.executorch import _decompose_grid_sampler

pytestmark = pytest.mark.unit


class _Sampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x, grid):
        return F.grid_sample(self.conv(x), grid, mode="bilinear", align_corners=False)


def _targets(exported):
    return {str(n.target) for n in exported.graph_module.graph.nodes if n.op == "call_function"}


def test_grid_sampling_is_decomposed_and_matches_eager():
    torch.manual_seed(0)
    model = _Sampler().eval()
    x = torch.randn(1, 3, 16, 20)
    grid = torch.rand(1, 8, 9, 2) * 2.2 - 1.1  # some samples fall outside the input
    exported = torch.export.export(model, (x, grid), strict=True)
    assert any("grid_sampler" in t for t in _targets(exported))

    decomposed = _decompose_grid_sampler(exported)

    targets = _targets(decomposed)
    assert not any("grid_sampler" in t for t in targets)
    assert "aten.conv2d.default" in targets  # unrelated ops stay for XNNPACK
    with torch.no_grad():
        torch.testing.assert_close(
            decomposed.module()(x, grid), model(x, grid), atol=1e-5, rtol=1e-5
        )
