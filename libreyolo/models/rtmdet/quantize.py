"""INT8 preparation of RTMDet's CSPNeXt pointwise convolutions for integer accelerators.

On RTMDet-s almost all of the INT8 loss comes from the rounding of one
tensor's weights: the pointwise convolution of the first CSPNeXt block of
backbone stage 2. :func:`learn_pointwise_int8_rounding` chooses, per weight,
whether to round up or down on the symmetric per-output-channel INT8 grid the
compiler uses (learned rounding in the style of AdaRound), so that the
convolution's output after its batch norm and SiLU matches the FP32 output on
calibration activations. It writes the grid values into the weights (the
channel's largest weight stays at +-127, so the compiler's scale and the
chosen codes are kept) and folds the mean output shift into the batch norm's
running mean. The batch norm folds into the convolution per output channel
later, which keeps the integer codes. It is deterministic: a fixed seed
drives its own generator.
"""

from __future__ import annotations

import torch
from torch import nn

#: Optimisation steps per convolution, pixels drawn per step, the most
#: calibration pixels kept per convolution, and the seed of the draws.
INT8_ROUNDING_STEPS = 3000
INT8_ROUNDING_BATCH = 4096
INT8_ROUNDING_MAX_PIXELS = 400_000
INT8_ROUNDING_SEED = 0


def _pointwise(block) -> nn.Module:
    """The ``ConvBNAct`` of a depthwise-separable block (after input equalization, the second step)."""
    module = block.pointwise_conv
    return module[1] if isinstance(module, nn.Sequential) else module


def _learn(cbn: nn.Module, x: torch.Tensor, steps: int, generator: torch.Generator) -> None:
    conv, bn = cbn.conv, cbn.bn
    if conv.kernel_size != (1, 1) or conv.groups != 1 or conv.bias is not None:
        raise ValueError(f"expected a 1x1 convolution without bias, got {conv}")
    weight = conv.weight.detach()[:, :, 0, 0].clone()
    scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    shift = (bn.bias - bn.running_mean * scale).detach()
    step = weight.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127.0
    floor = torch.floor(weight / step)
    rest = (weight / step - floor).clamp(1e-4, 1.0 - 1e-4)
    logits = nn.Parameter(torch.log(rest / (1.0 - rest)))
    optimizer = torch.optim.Adam([logits], lr=1e-2)

    def output(w, inputs):
        return nn.functional.silu((inputs @ w.t()) * scale + shift)

    count = int(x.shape[0])
    for it in range(int(steps)):
        beta = 20.0 - 18.0 * it / max(1, int(steps) - 1)
        soft = (torch.sigmoid(logits) * 1.2 - 0.1).clamp(0.0, 1.0)
        candidate = torch.clamp(floor + soft, -127, 127) * step
        batch = x[torch.randint(0, count, (min(INT8_ROUNDING_BATCH, count),), generator=generator)]
        with torch.no_grad():
            target = output(weight, batch)
        loss = ((output(candidate, batch) - target) ** 2).mean()
        if it > int(steps) * 0.2:
            loss = loss + 1e-4 * (1.0 - (2.0 * soft - 1.0).abs().pow(beta)).sum() / soft.numel()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        hard = ((torch.sigmoid(logits) * 1.2 - 0.1).clamp(0.0, 1.0) >= 0.5).to(weight.dtype)
        codes = torch.clamp(floor + hard, -127, 127)
        # Pin each channel's largest weight at +-127 so the compiler derives the same scale.
        rows = torch.arange(weight.shape[0])
        peak = weight.abs().argmax(1)
        codes[rows, peak] = torch.sign(weight[rows, peak]) * 127.0
        rounded = codes * step
        bn.running_mean.add_((x @ (rounded - weight).t()).mean(0))
        conv.weight.copy_(rounded[:, :, None, None])


def learn_pointwise_int8_rounding(
    model: nn.Module, samples, steps: int = INT8_ROUNDING_STEPS, seed: int = INT8_ROUNDING_SEED
) -> int:
    """Learn the INT8 rounding of every CSPNeXt pointwise conv of ``model`` in place; returns how many.

    The convolutions are handled in forward order, each on the inputs the
    already-rounded network produces for ``samples`` (``[1, C, H, W]``
    calibration images). ``model`` must be a copy the caller owns.
    """
    from .nn import DepthwiseSeparableConv

    blocks = [m for m in model.modules() if isinstance(m, DepthwiseSeparableConv)]
    if not blocks or not samples:
        return 0
    generator = torch.Generator().manual_seed(int(seed))
    for block in blocks:
        cbn = _pointwise(block)
        inputs: list[torch.Tensor] = []

        def record(_module, args):
            feature = args[0].detach()
            inputs.append(feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1]))

        handle = cbn.register_forward_pre_hook(record)
        try:
            with torch.no_grad():
                for sample in samples:
                    model(torch.as_tensor(sample, dtype=torch.float32))
        finally:
            handle.remove()
        x = torch.cat(inputs)
        if x.shape[0] > INT8_ROUNDING_MAX_PIXELS:
            x = x[torch.randperm(x.shape[0], generator=generator)[:INT8_ROUNDING_MAX_PIXELS]]
        _learn(cbn, x, steps, generator)
    return len(blocks)


__all__ = ["learn_pointwise_int8_rounding"]
