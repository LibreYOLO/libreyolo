"""Real-ESRGAN compact network (x4t) INT8 preparation: exact cross-layer equalization."""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = pytest.mark.unit


def _net():
    from libreyolo.models.realesrgan.nn import SRVGGNetCompact

    torch.manual_seed(0)
    net = SRVGGNetCompact(num_conv=4).eval()
    with torch.no_grad():  # PReLU slopes as trained networks have them (not the 0.25 default)
        for m in net.body:
            if isinstance(m, torch.nn.PReLU):
                m.weight.uniform_(-0.2, 0.5)
            if isinstance(m, torch.nn.Conv2d):  # uneven channel ranges
                m.weight.mul_(torch.rand(m.weight.shape[0], 1, 1, 1) * 4 + 0.05)
    return net


def test_equalization_keeps_the_function():
    from libreyolo.models.realesrgan.quantize import equalize_conv_activation_ranges

    net = _net()
    reference = copy.deepcopy(net)
    assert equalize_conv_activation_ranges(net) == 5
    x = torch.rand(2, 3, 20, 24)
    with torch.no_grad():
        torch.testing.assert_close(net(x), reference(x), rtol=1e-4, atol=1e-5)


def test_equalization_balances_producer_and_consumer_ranges():
    from libreyolo.models.realesrgan.quantize import equalize_conv_activation_ranges

    net = _net()
    convs = [m for m in net.body if isinstance(m, torch.nn.Conv2d)]

    def spread():
        out = []
        for a, b in zip(convs[:-1], convs[1:]):
            r1 = a.weight.abs().reshape(a.weight.shape[0], -1).amax(1)
            r2 = b.weight.abs().transpose(0, 1).reshape(b.weight.shape[1], -1).amax(1)
            out.append(float((r1 / r2).log().abs().max()))
        return max(out)

    before = spread()
    equalize_conv_activation_ranges(net, rounds=10)
    assert spread() < 0.1 * before


def test_non_homogeneous_activation_is_rejected():
    from libreyolo.models.realesrgan.quantize import equalize_conv_activation_ranges

    net = _net()
    net.body[1] = torch.nn.Hardtanh()
    with pytest.raises(ValueError, match="homogeneous"):
        equalize_conv_activation_ranges(net)
