"""FDR numerical-correctness tests for the D-FINE port.

These tests exist to catch numerical drift in the three functions that are
*load-bearing for box decoding*: ``weighting_function``, ``distance2bbox``, and
``Integral``. A 1e-6 drift in any of them shifts every predicted box by a few
pixels and silently collapses mAP.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

from libreyolo.models.dfine.fdr import Integral, distance2bbox, weighting_function


class TestWeightingFunction:
    @pytest.mark.parametrize("reg_scale", [4.0, 8.0])
    def test_shape_and_center(self, reg_scale):
        up = torch.tensor([0.5])
        rs = torch.tensor([reg_scale])
        w = weighting_function(32, up, rs)
        assert w.shape == (33,)
        assert w[16].abs().item() < 1e-7

    @pytest.mark.parametrize("reg_scale", [4.0, 8.0])
    def test_monotonic(self, reg_scale):
        up = torch.tensor([0.5])
        rs = torch.tensor([reg_scale])
        w = weighting_function(32, up, rs)
        diffs = w[1:] - w[:-1]
        assert (diffs > 0).all(), "W(n) must be strictly increasing"

    @pytest.mark.parametrize("reg_scale,expected_end", [(4.0, 4.0), (8.0, 8.0)])
    def test_endpoints(self, reg_scale, expected_end):
        up = torch.tensor([0.5])
        rs = torch.tensor([reg_scale])
        w = weighting_function(32, up, rs)
        # Endpoints = ± up * reg_scale * 2 = ± 0.5 * reg_scale * 2 = ± reg_scale
        assert w[0].item() == pytest.approx(-expected_end, abs=1e-6)
        assert w[-1].item() == pytest.approx(expected_end, abs=1e-6)


class TestIntegral:
    def test_output_shape(self):
        integral = Integral(reg_max=32)
        x = torch.randn(2, 300, 4 * 33)
        project = weighting_function(32, torch.tensor([0.5]), torch.tensor([4.0]))
        out = integral(x, project)
        assert out.shape == (2, 300, 4)

    def test_peaked_distribution_hits_center_bin(self):
        """A distribution peaked at bin 16 (center) decodes to ~0 offset."""
        integral = Integral(reg_max=32)
        project = weighting_function(32, torch.tensor([0.5]), torch.tensor([4.0]))
        logits = torch.full((1, 1, 4, 33), -1e4)
        logits[..., 16] = 1e4  # all four edges point at the center bin
        out = integral(logits.reshape(1, 1, 4 * 33), project)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-3)


class TestDistance2Bbox:
    def test_zero_distances_return_point_box(self):
        """Zero distances should decode to a zero-size box centered at the point."""
        points = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
        distance = torch.tensor([[[-2.0, -2.0, -2.0, -2.0]]])
        # With reg_scale=4 and distance=-0.5*reg_scale=-2, (0.5*reg_scale + distance)=0,
        # so the box collapses to the center point.
        out = distance2bbox(points, distance, reg_scale=4.0)
        assert torch.allclose(out[..., 2:], torch.zeros_like(out[..., 2:]), atol=1e-6)
        assert torch.allclose(out[..., :2], points[..., :2], atol=1e-6)
