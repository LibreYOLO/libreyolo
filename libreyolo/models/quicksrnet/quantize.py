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

:func:`learn_int8_rounding` then chooses, per weight, whether to round up or
down on that per-channel INT8 grid (learned rounding in the style of AdaRound,
one convolution at a time, matching each layer's FP32 output from the INT8
network's inputs), writes the grid values into the weights so the compiler
keeps them, and folds the remaining mean output shift into the bias. It is
deterministic: a fixed seed drives its own generator.
"""

from __future__ import annotations

import torch
from torch import nn

#: Exponent of the equalization scale ``s_i = (1 / m_i) ** alpha``, measured on
#: a Metis M.2 Max (docs/axelera.md).
INT8_EQUALIZATION_ALPHA = 0.5
#: Fold the expected weight-rounding error into the biases after equalizing.
INT8_BIAS_CORRECTION = True
#: Learned rounding (opt-in per size: the ``learned_rounding`` key of
#: ``AXELERA_SIZE_RECIPES``): optimisation steps per convolution and the seed
#: of its sample draws.
INT8_ROUNDING_STEPS = 300
INT8_ROUNDING_SEED = 0


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


def _activation_grid(x: torch.Tensor) -> torch.Tensor:
    """The uint8 grid of a ``[0, 1]`` activation."""
    return torch.round(x.clamp(0.0, 1.0) * 255.0) / 255.0


def learn_int8_rounding(
    net: nn.Module,
    samples,
    steps: int = INT8_ROUNDING_STEPS,
    seed: int = INT8_ROUNDING_SEED,
) -> int:
    """Learn each weight's INT8 rounding direction in place; returns the number of convolutions.

    Layer by layer, a soft up/down choice per weight on the symmetric
    per-output-channel grid (the channel's largest weight stays at +-127, so the
    compiler's scale is unchanged) is optimised to reproduce the layer's FP32
    output from the already-rounded network's uint8 inputs, with a regulariser
    that drives every choice to 0 or 1. The rounded weights are then written
    back as grid values, and each bias absorbs the mean output shift of its
    rounding over the calibration inputs of the FP32 network (as in
    :func:`correct_biases`).
    """
    pairs = _hidden_pairs(net)
    convs = [producer for producer, _, _ in pairs] + [pairs[-1][2]]
    device = next(net.parameters()).device
    x = torch.cat([torch.as_tensor(sample, dtype=torch.float32, device=device) for sample in samples])
    if x.shape[0] == 0:
        return 0
    generator = torch.Generator().manual_seed(int(seed))
    batch = min(16, int(x.shape[0]))
    weights = [conv.weight.detach().clone() for conv in convs]
    biases = [conv.bias.detach().clone() for conv in convs]
    x_fp, x_q = x, _activation_grid(x)
    rounded, means = [], []
    for weight, bias in zip(weights, biases):
        means.append(x_fp.mean(dim=(0, 2, 3)))
        step = (weight.abs().reshape(weight.shape[0], -1).amax(1).clamp_min(1e-12) / 127.0).reshape(-1, 1, 1, 1)
        floor = torch.floor(weight / step)
        rest = (weight / step - floor).clamp(1e-4, 1.0 - 1e-4)
        logits = nn.Parameter(torch.log(rest / (1.0 - rest)))
        optimizer = torch.optim.Adam([logits], lr=1e-2)
        with torch.no_grad():
            target = nn.functional.conv2d(x_fp, weight, bias, padding=1).clamp(0.0, 1.0)
        for it in range(int(steps)):
            beta = 20.0 - 18.0 * it / max(1, int(steps) - 1)
            soft = (torch.sigmoid(logits) * 1.2 - 0.1).clamp(0.0, 1.0)
            candidate = torch.clamp(floor + soft, -127, 127) * step
            index = torch.randint(0, x_q.shape[0], (batch,), generator=generator).to(device)
            out = nn.functional.conv2d(x_q[index], candidate, bias, padding=1).clamp(0.0, 1.0)
            loss = ((out - target[index]) ** 2).mean()
            if it > int(steps) * 0.2:
                loss = loss + 1e-4 * (1.0 - (2.0 * soft - 1.0).abs().pow(beta)).sum() / soft.numel()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            hard = ((torch.sigmoid(logits) * 1.2 - 0.1).clamp(0.0, 1.0) >= 0.5).to(weight.dtype)
            grid = torch.clamp(floor + hard, -127, 127) * step
            rounded.append(grid)
            x_fp = nn.functional.conv2d(x_fp, weight, bias, padding=1).clamp(0.0, 1.0)
            x_q = _activation_grid(nn.functional.conv2d(x_q, grid, bias, padding=1))
    with torch.no_grad():
        for conv, weight, grid, bias, mu in zip(convs, weights, rounded, biases, means):
            conv.weight.copy_(grid)
            conv.bias.copy_(bias - (grid - weight).sum(dim=(2, 3)) @ mu)
    return len(convs)


def prepare_for_int8(net: nn.Module, samples, *, learned_rounding: bool = False, rounding_samples=None) -> dict:
    """Equalize ``net`` in place, then prepare its weights for INT8; returns a summary.

    With ``learned_rounding`` the weights get :func:`learn_int8_rounding` (on
    ``rounding_samples``, default ``samples``), otherwise only the bias
    correction of :func:`correct_biases`.
    """
    peaks = equalize_hardtanh_ranges(net, samples)
    rounded = corrected = 0
    if peaks and learned_rounding:
        rounded = corrected = learn_int8_rounding(net, samples if rounding_samples is None else rounding_samples)
    elif peaks and INT8_BIAS_CORRECTION:
        corrected = correct_biases(net, samples)
    return {"layer_min_peaks": peaks, "bias_corrected": corrected, "learned_rounding": rounded}


__all__ = [
    "INT8_BIAS_CORRECTION",
    "INT8_EQUALIZATION_ALPHA",
    "INT8_ROUNDING_SEED",
    "INT8_ROUNDING_STEPS",
    "correct_biases",
    "equalize_hardtanh_ranges",
    "learn_int8_rounding",
    "prepare_for_int8",
]
