"""MobileNetV4 INT8 preparation: residual-stream equalization and bounds."""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = pytest.mark.unit


def _net():
    from libreyolo import LibreMobileNetV4

    torch.manual_seed(0)
    net = LibreMobileNetV4(size="s", nb_classes=5, device="cpu").model.eval()
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():  # non-trivial BatchNorm statistics
        for module in net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                n = module.num_features
                module.running_mean.copy_(torch.randn(n, generator=generator) * 0.1)
                module.running_var.copy_(torch.rand(n, generator=generator) + 0.5)
                module.weight.copy_(torch.rand(n, generator=generator) * 2 + 0.1)
                module.bias.copy_(torch.randn(n, generator=generator) * 0.1)
    return net


def _samples(n=3, size=64):
    generator = torch.Generator().manual_seed(2)
    return [torch.randn(1, 3, size, size, generator=generator) for _ in range(n)]


def test_equalization_is_exact_and_touches_only_linear_stages():
    from libreyolo.models.mobilenetv4.quantize import equalize_residual_streams

    net = _net()
    reference = copy.deepcopy(net)
    samples = _samples()
    stages = equalize_residual_streams(net, samples, alpha=0.5)
    assert stages == [2, 3]  # MobileNetV4-s: stages 0, 1 and 4 are plain convs
    probe = _samples(2)
    with torch.no_grad():
        for x in probe:
            torch.testing.assert_close(net(x), reference(x), rtol=1e-4, atol=1e-4)
    changed = [k for k, v in net.state_dict().items() if not torch.equal(v, reference.state_dict()[k])]
    assert changed and all(k.startswith(("blocks.2", "blocks.3", "blocks.4")) for k in changed)


def test_bound_at_the_largest_value_is_exact_and_adds_a_hardtanh():
    from libreyolo.models.mobilenetv4.quantize import bound_stage_output

    net = _net()
    reference = copy.deepcopy(net)
    samples = _samples()
    bound = bound_stage_output(net, 2, samples, percentile=100.0)
    assert bound > 0
    assert isinstance(list(net.blocks[2])[-1], torch.nn.Hardtanh)
    with torch.no_grad():
        for x in samples:  # the bound is the max over these, so nothing clips
            torch.testing.assert_close(net(x), reference(x), rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError, match="no linear residual stream"):
        bound_stage_output(net, 0, samples, percentile=99.9)
