"""QuickSRNet INT8 preparation: Hardtanh-bounded channel equalization."""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = pytest.mark.unit


def _net():
    from libreyolo import LibreQuickSRNet

    torch.manual_seed(0)
    return LibreQuickSRNet(None, size="m2", device="cpu").model.eval()


def _samples(n=4, size=24, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(1, 3, size, size, generator=generator) for _ in range(n)]


def test_equalization_keeps_outputs_on_calibration_images():
    from libreyolo.models.quicksrnet.quantize import equalize_hardtanh_ranges

    net = _net()
    reference = copy.deepcopy(net)
    samples = _samples()
    assert len(equalize_hardtanh_ranges(net, samples)) == 6
    with torch.no_grad():
        for x in samples:
            torch.testing.assert_close(net(x), reference(x), rtol=1e-5, atol=1e-5)


def test_equalization_scales_quiet_channels_up_only():
    from libreyolo.models.quicksrnet.quantize import equalize_hardtanh_ranges

    net = _net()
    before = net.cnn[0].weight.detach().clone()
    equalize_hardtanh_ranges(net, _samples(), alpha=1.0)
    ratio = (net.cnn[0].weight / before).reshape(before.shape[0], -1)[:, 0]
    assert bool((ratio >= 1.0 - 1e-6).all())
    # Every hidden channel now peaks at 1 on the calibration images (alpha 1).
    peaks = []
    hook = net.cnn[1].register_forward_hook(lambda _m, _i, out: peaks.append(out.amax(dim=(0, 2, 3))))
    with torch.no_grad():
        for x in _samples():
            net(x)
    hook.remove()
    assert torch.stack(peaks).amax(0).min() > 0.99


def test_unexpected_activation_is_rejected():
    from libreyolo.models.quicksrnet.quantize import prepare_for_int8

    net = _net()
    net.cnn[1] = torch.nn.Hardtanh(0.0, 2.0)
    with pytest.raises(ValueError, match="Hardtanh"):
        prepare_for_int8(net, _samples(1))


def test_bias_correction_moves_outputs_only_slightly():
    from libreyolo.models.quicksrnet.quantize import prepare_for_int8

    net = _net()
    reference = copy.deepcopy(net)
    samples = _samples()
    summary = prepare_for_int8(net, samples)
    assert summary["bias_corrected"] == 7 and summary["learned_rounding"] == 0
    with torch.no_grad():
        drift = max(float((net(x) - reference(x)).abs().max()) for x in samples)
    assert 0.0 < drift < 0.1


def test_learned_rounding_writes_int8_grid_weights_deterministically():
    from libreyolo.models.quicksrnet.quantize import equalize_hardtanh_ranges, learn_int8_rounding

    nets = []
    for _ in range(2):
        net = _net()
        equalize_hardtanh_ranges(net, _samples())
        assert learn_int8_rounding(net, _samples(n=6), steps=20) == 7
        nets.append(net)
    convs = [m for m in nets[0].modules() if isinstance(m, torch.nn.Conv2d)]
    for conv in convs:
        weight = conv.weight.detach()
        step = weight.abs().reshape(weight.shape[0], -1).amax(1).reshape(-1, 1, 1, 1) / 127.0
        levels = weight / step
        torch.testing.assert_close(levels, torch.round(levels), rtol=0, atol=1e-3)
        assert float(levels.abs().max()) <= 127.0 + 1e-3
    for a, b in zip(nets[0].state_dict().values(), nets[1].state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_learned_rounding_keeps_fp32_outputs_close():
    from libreyolo.models.quicksrnet.quantize import prepare_for_int8

    net = _net()
    reference = copy.deepcopy(net)
    samples = _samples()
    summary = prepare_for_int8(net, samples, learned_rounding=True, rounding_samples=_samples(n=6, seed=2))
    assert summary["learned_rounding"] == 7
    with torch.no_grad():
        drift = max(float((net(x) - reference(x)).abs().max()) for x in samples)
    assert 0.0 < drift < 0.1
