"""Axelera recipe ``network_learned_rounding``: learned INT8 rounding of every Conv2d/Linear of a network."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.unit


def _net():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 6, 1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(6, 4),
    ).eval()


def _randomize_bn(net):
    g = torch.Generator().manual_seed(1)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(m.num_features, generator=g) + 0.5)


def test_rounded_weights_sit_on_the_int8_grid_and_keep_the_outputs_close():
    from libreyolo.export.learned_rounding import learn_int8_rounding

    net = _net()
    _randomize_bn(net)
    reference = copy.deepcopy(net)
    layers = [m for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    peaks = [m.weight.detach().reshape(m.weight.shape[0], -1).abs().amax(1) for m in layers]
    samples = [torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(i)) for i in range(3)]
    assert learn_int8_rounding(net, samples, steps=30) == len(layers) == 4
    for layer, peak in zip(layers, peaks):
        weight = layer.weight.detach().reshape(layer.weight.shape[0], -1)
        codes = weight / (peak / 127.0)[:, None]
        torch.testing.assert_close(codes, torch.round(codes), rtol=0, atol=1e-3)
        # Each row's largest weight keeps its value: the compiler's per-channel scale is unchanged.
        torch.testing.assert_close(weight.abs().amax(1), peak)
    x = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        a, b = reference(x), net(x)
    assert float((a - b).abs().max()) < 0.05 * float(a.abs().max()) + 1e-3
    # Deterministic.
    again = copy.deepcopy(reference)
    learn_int8_rounding(again, samples, steps=30)
    assert all(torch.equal(p, q) for p, q in zip(again.state_dict().values(), net.state_dict().values()))


def test_without_samples_nothing_is_rounded():
    from libreyolo.export.learned_rounding import learn_int8_rounding

    net = _net()
    before = copy.deepcopy(net.state_dict())
    assert learn_int8_rounding(net, []) == 0
    assert all(torch.equal(before[k], v) for k, v in net.state_dict().items())
