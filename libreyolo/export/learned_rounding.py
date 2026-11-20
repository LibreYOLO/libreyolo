"""Learned INT8 weight rounding for whole networks exported to integer accelerators.

The Metis toolchain quantizes every convolution's weights to a symmetric INT8
grid per output channel (step = the row's largest ``|w|`` / 127; batch norms
fold into the convolution first, which scales a row and its largest weight
together, so the same integer codes survive the fold). Rounding each weight
to its nearest grid value is not the best choice for the network output:
:func:`learn_int8_rounding` chooses, per weight, whether to round up or down
(learned rounding in the style of AdaRound), layer by layer in forward order,
so that each layer's output after its batch norm matches the FP32 network's
output for that layer on calibration images. Each layer sees the inputs the
already-rounded layers before it produce, so it also absorbs part of their
error. The mean output shift that remains goes into the batch norm's running
mean (or the bias). Each row's largest weight stays at +-127, so the
compiler derives the same scale and keeps the chosen codes.

Generalises ``models/rtmdet/quantize.learn_pointwise_int8_rounding`` (one
kind of layer) to every ``Conv2d`` (with its following ``BatchNorm2d``) and
``Linear`` of a network. Deterministic: a fixed seed drives its own generator.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

#: Optimisation steps per layer, images per step, regulariser weight, seed.
LEARNED_ROUNDING_STEPS = 500
LEARNED_ROUNDING_BATCH = 4
LEARNED_ROUNDING_LAMBDA = 0.01
LEARNED_ROUNDING_SEED = 0


def _forward_order(net: nn.Module, sample: torch.Tensor) -> list[tuple[str, nn.Module, nn.Module | None]]:
    """``(name, layer, batch norm or None)`` for every Conv2d/Linear, in forward call order."""
    order: list[nn.Module] = []
    seen: set[int] = set()
    names = {id(m): n for n, m in net.named_modules()}

    def hook(module, _inputs):
        if id(module) not in seen:
            seen.add(id(module))
            order.append(module)

    kinds = (nn.Conv2d, nn.Linear, nn.BatchNorm2d)
    handles = [m.register_forward_pre_hook(hook) for m in net.modules() if isinstance(m, kinds)]
    try:
        with torch.no_grad():
            net(sample)
    finally:
        for handle in handles:
            handle.remove()
    layers = []
    for i, module in enumerate(order):
        if isinstance(module, nn.BatchNorm2d):
            continue
        bn = order[i + 1] if i + 1 < len(order) and isinstance(order[i + 1], nn.BatchNorm2d) else None
        if bn is not None and (not isinstance(module, nn.Conv2d) or bn.num_features != module.out_channels):
            bn = None
        layers.append((names[id(module)], module, bn))
    return layers


def _layer(module: nn.Module, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return F.linear(x, weight, module.bias)
    return F.conv2d(x, weight, module.bias, module.stride, module.padding, module.dilation, module.groups)


def _norm(bn: nn.Module | None, y: torch.Tensor) -> torch.Tensor:
    if bn is None:
        return y
    return F.batch_norm(y, bn.running_mean, bn.running_var, bn.weight, bn.bias, False, 0.0, bn.eps)


def _inputs(net: nn.Module, module: nn.Module, samples) -> list[torch.Tensor]:
    captured: list[torch.Tensor] = []
    handle = module.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
    try:
        with torch.no_grad():
            for sample in samples:
                net(sample)
    finally:
        handle.remove()
    return captured


def _learn(module, bn, xs, refs, steps, batch, lam, generator) -> None:
    weight = module.weight.detach().clone()
    rows = weight.reshape(weight.shape[0], -1)
    step = (rows.abs().amax(1).clamp_min(1e-12) / 127.0).reshape(-1, *([1] * (weight.dim() - 1)))
    floor = torch.floor(weight / step)
    rest = (weight / step - floor).clamp(1e-4, 1.0 - 1e-4)
    logits = nn.Parameter(torch.log(rest / (1.0 - rest)))
    optimizer = torch.optim.Adam([logits], lr=1e-2)
    with torch.no_grad():
        targets = [_norm(bn, _layer(module, r, weight)) for r in refs]
        scale = torch.cat([t.reshape(-1) for t in targets]).var().clamp_min(1e-12)
    count = len(xs)
    for it in range(int(steps)):
        beta = 20.0 - 18.0 * it / max(1, int(steps) - 1)
        soft = (torch.sigmoid(logits) * 1.2 - 0.1).clamp(0.0, 1.0)
        candidate = torch.clamp(floor + soft, -127, 127) * step
        picks = torch.randint(0, count, (min(int(batch), count),), generator=generator).tolist()
        loss = sum(((_norm(bn, _layer(module, xs[i], candidate)) - targets[i]) ** 2).mean() for i in picks)
        loss = loss / len(picks) / scale
        if it > int(steps) * 0.2:
            loss = loss + lam * (1.0 - (2.0 * soft - 1.0).abs().pow(beta)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        hard = ((torch.sigmoid(logits) * 1.2 - 0.1).clamp(0.0, 1.0) >= 0.5).to(weight.dtype)
        codes = torch.clamp(floor + hard, -127, 127).reshape(rows.shape)
        index = torch.arange(rows.shape[0])
        peak = rows.abs().argmax(1)
        # Pin each row's largest weight at +-127 so the compiler derives the same scale.
        codes[index, peak] = torch.sign(rows[index, peak]) * 127.0
        rounded = codes.reshape(weight.shape) * step
        dims = (0,) if isinstance(module, nn.Linear) else (0, 2, 3)
        shift = torch.stack(
            [(_layer(module, x, rounded) - _layer(module, r, weight)).mean(dims) for x, r in zip(xs, refs)]
        ).mean(0)
        module.weight.copy_(rounded)
        if bn is not None:
            bn.running_mean.add_(shift)
        elif module.bias is not None:
            module.bias.sub_(shift)
        else:
            module.bias = nn.Parameter(-shift)


def learn_int8_rounding(
    net: nn.Module,
    samples,
    *,
    steps: int = LEARNED_ROUNDING_STEPS,
    batch: int = LEARNED_ROUNDING_BATCH,
    lam: float = LEARNED_ROUNDING_LAMBDA,
    seed: int = LEARNED_ROUNDING_SEED,
) -> int:
    """Learn the INT8 rounding of every Conv2d/Linear of ``net`` in place; returns how many layers.

    ``samples`` are ``[1, C, H, W]`` calibration images. ``net`` must be a copy
    the caller owns, in eval mode.
    """
    samples = [torch.as_tensor(s, dtype=torch.float32) for s in samples]
    if not samples:
        return 0
    net.eval()
    reference = copy.deepcopy(net).eval()
    reference_modules = dict(reference.named_modules())
    generator = torch.Generator().manual_seed(int(seed))
    layers = _forward_order(net, samples[0])
    for name, module, bn in layers:
        xs = _inputs(net, module, samples)
        refs = _inputs(reference, reference_modules[name], samples)
        _learn(module, bn, xs, refs, steps, batch, lam, generator)
    return len(layers)


__all__ = ["learn_int8_rounding"]
