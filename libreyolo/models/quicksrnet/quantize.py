"""INT8 preparation of QuickSRNet for integer accelerators.

QuickSRNet's hidden activations are ``Hardtanh(0, 1)``, so every activation
already has the ideal INT8 range. Its loss under INT8 comes from the WEIGHTS:
with per-output-channel scales, the input channels inside one kernel differ
by an order of magnitude (channels that rarely exceed 0.1 next to channels
that saturate at 1), so the small ones round away.

:func:`equalize_hardtanh_ranges` rescales hidden channel ``i`` by ``s_i`` at
its producer (weights and bias of the convolution before the ``Hardtanh``)
and by ``1 / s_i`` at its consumer (the next convolution's input channel).
With ``1 <= s_i <= 1 / m_i``, where ``m_i`` is the channel's largest
activation on the calibration images, the scaled channel still clips at 1
only where the original exceeded ``m_i``, so outputs are unchanged on every
input whose activations stay within the calibration maxima.
"""

from __future__ import annotations

import torch
from torch import nn

#: Exponent of the equalization scale ``s_i = (1 / m_i) ** alpha``, measured on
#: a Metis M.2 Max (docs/axelera.md).
INT8_EQUALIZATION_ALPHA = 0.5
#: Fold the expected weight-rounding error into the biases after equalizing.
INT8_BIAS_CORRECTION = True


def _hidden_pairs(net: nn.Module) -> list[tuple[nn.Conv2d, nn.Hardtanh, nn.Conv2d]]:
    """``(producer, Hardtanh, consumer)`` for every hidden activation of ``net``."""
    layers = list(net.cnn)
    convs = [m for m in layers if isinstance(m, nn.Conv2d)] + [net.conv_last]
    acts = [m for m in layers if isinstance(m, nn.Hardtanh)]
    if len(acts) != len(convs) - 1:
        raise ValueError("Expected one Hardtanh after every hidden QuickSRNet convolution.")
    for act in acts:
        if (float(act.min_val), float(act.max_val)) != (0.0, 1.0):
            raise ValueError("QuickSRNet equalization needs Hardtanh(0, 1) activations.")
    return [(convs[k], acts[k], convs[k + 1]) for k in range(len(acts))]


def equalize_hardtanh_ranges(net: nn.Module, samples, alpha: float = INT8_EQUALIZATION_ALPHA) -> list[float]:
    """Equalize every hidden channel in place; returns the smallest calibration max per layer."""
    pairs = _hidden_pairs(net)
    peaks: dict[int, list[torch.Tensor]] = {k: [] for k in range(len(pairs))}
    hooks = [
        act.register_forward_hook(
            lambda _m, _i, out, k=k: peaks[k].append(out.detach().amax(dim=(0, 2, 3)))
        )
        for k, (_, act, _) in enumerate(pairs)
    ]
    device = next(net.parameters()).device
    try:
        with torch.no_grad():
            for sample in samples:
                net(torch.as_tensor(sample, dtype=torch.float32, device=device))
    finally:
        for hook in hooks:
            hook.remove()
    if not all(peaks.values()):
        return []
    smallest = []
    with torch.no_grad():
        for k, (producer, _, consumer) in enumerate(pairs):
            peak = torch.stack(peaks[k]).amax(0).clamp(1e-3, 1.0)
            scale = (1.0 / peak) ** float(alpha)
            producer.weight.mul_(scale.reshape(-1, 1, 1, 1))
            if producer.bias is not None:
                producer.bias.mul_(scale)
            consumer.weight.div_(scale.reshape(1, -1, 1, 1))
            smallest.append(float(peak.min()))
    return smallest


def _int8_per_channel(weight: torch.Tensor) -> torch.Tensor:
    """Symmetric per-output-channel INT8 rounding (the scheme the weights get on Metis)."""
    peak = weight.abs().reshape(weight.shape[0], -1).amax(1).clamp_min(1e-12)
    step = (peak / 127.0).reshape(-1, *[1] * (weight.dim() - 1))
    return torch.clamp(torch.round(weight / step), -127, 127) * step


def correct_biases(net: nn.Module, samples) -> int:
    """Fold the expected INT8 weight-rounding error into each convolution's bias.

    For input channel means ``mu`` over the calibration images, rounding the
    weights shifts output channel ``o`` by ``sum((W_q - W)[o] * mu)`` on average;
    subtracting that shift from the bias removes the systematic part of the
    error (FP32 outputs move by the same small amount). Returns the number of
    corrected convolutions.
    """
    convs = [m for m in net.modules() if isinstance(m, nn.Conv2d) and m.bias is not None]
    means: dict[int, list[torch.Tensor]] = {k: [] for k in range(len(convs))}
    hooks = [
        conv.register_forward_pre_hook(
            lambda _m, inputs, k=k: means[k].append(inputs[0].detach().mean(dim=(0, 2, 3)))
        )
        for k, conv in enumerate(convs)
    ]
    device = next(net.parameters()).device
    try:
        with torch.no_grad():
            for sample in samples:
                net(torch.as_tensor(sample, dtype=torch.float32, device=device))
    finally:
        for hook in hooks:
            hook.remove()
    with torch.no_grad():
        for k, conv in enumerate(convs):
            if not means[k] or conv.groups != 1:
                continue
            mu = torch.stack(means[k]).mean(0)
            shift = (_int8_per_channel(conv.weight) - conv.weight).sum(dim=(2, 3)) @ mu
            conv.bias.sub_(shift)
    return len(convs)


def prepare_for_int8(net: nn.Module, samples) -> dict:
    """Equalize ``net`` in place, then correct its biases for INT8 weights; returns a summary."""
    peaks = equalize_hardtanh_ranges(net, samples)
    corrected = correct_biases(net, samples) if INT8_BIAS_CORRECTION and peaks else 0
    return {"layer_min_peaks": peaks, "bias_corrected": corrected}


__all__ = [
    "INT8_BIAS_CORRECTION",
    "INT8_EQUALIZATION_ALPHA",
    "correct_biases",
    "equalize_hardtanh_ranges",
    "prepare_for_int8",
]
