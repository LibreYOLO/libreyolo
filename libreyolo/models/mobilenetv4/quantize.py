"""INT8 preparation of MobileNetV4 for integer accelerators.

Inside a stage of ``UniversalInvertedResidual`` blocks the residual stream is
linear: each block adds ``pw_proj`` (convolution + BatchNorm, no activation)
to its input. Per-tensor INT8 gives that stream one scale, set by its widest
channels; on MobileNetV4-s the stream feeding stage 3 spans about +-400 while
its median channel stays within +-14, and the device lost 6 top-1 points.

:func:`equalize_residual_streams` rescales stream channel ``c`` by ``1 / s_c``
at every producer (each block's ``pw_proj`` BatchNorm) and by ``s_c`` at every
consumer (the first convolution of the stage's later blocks and of whatever
reads the stage output), so FP32 outputs are unchanged.
:func:`bound_stage_output` additionally clamps one stage's output at a
percentile of its calibration magnitudes through ``Hardtanh(-1, 1)`` on the
stream scaled by ``1 / T``; it changes FP32 outputs only above that bound.
"""

from __future__ import annotations

import torch
from torch import nn

from .nn import UniversalInvertedResidual


def _first_conv(block: nn.Module) -> nn.Conv2d:
    if isinstance(block, nn.Conv2d):
        return block
    for name in ("dw_start", "pw_exp"):
        part = getattr(block, name, None)
        if part is not None and hasattr(part, "conv"):
            return part.conv
    if hasattr(block, "conv"):
        return block.conv
    raise ValueError(f"No input convolution found in {type(block).__name__}.")


def _scale_input_channels(conv: nn.Conv2d, scale: torch.Tensor) -> None:
    if conv.groups > 1 and conv.groups == conv.in_channels:  # depthwise
        conv.weight.mul_(scale[:, None, None, None])
    else:
        conv.weight.mul_(scale[None, :, None, None])


def _linear_stream_stage(stage: nn.Module) -> bool:
    blocks = list(stage)
    if not blocks or not all(isinstance(b, UniversalInvertedResidual) for b in blocks):
        return False
    if blocks[0].has_skip:
        return False
    return all(
        isinstance(b.dw_end, nn.Identity)
        and isinstance(b.layer_scale, nn.Identity)
        and not b.pw_proj.apply_act
        for b in blocks
    )


def _next_consumer(net: nn.Module, stage_index: int) -> nn.Module:
    if stage_index + 1 < len(net.blocks):
        return net.blocks[stage_index + 1][0]
    return net.conv_head


def _rescale_stream(net: nn.Module, stage_index: int, scale: torch.Tensor) -> None:
    """Stream channel ``c`` of the stage becomes ``x_c / scale_c``; FP32-exact."""
    blocks = list(net.blocks[stage_index])
    for block in blocks:
        block.pw_proj.bn.weight.div_(scale)
        block.pw_proj.bn.bias.div_(scale)
    for block in blocks[1:]:
        _scale_input_channels(_first_conv(block), scale)
    _scale_input_channels(_first_conv(_next_consumer(net, stage_index)), scale)


def _stage_outputs(net: nn.Module, stages, samples) -> dict[int, list[torch.Tensor]]:
    outputs: dict[int, list[torch.Tensor]] = {k: [] for k in stages}
    hooks = [
        net.blocks[k].register_forward_hook(
            lambda _m, _i, out, k=k: outputs[k].append(out.detach())
        )
        for k in stages
    ]
    device = next(net.parameters()).device
    try:
        with torch.no_grad():
            for sample in samples:
                net(torch.as_tensor(sample, dtype=torch.float32, device=device))
    finally:
        for hook in hooks:
            hook.remove()
    return outputs


def equalize_residual_streams(net: nn.Module, samples, alpha: float = 0.5) -> list[int]:
    """Equalize every linear residual-stream stage in place; returns their indices."""
    stages = [k for k, stage in enumerate(net.blocks) if _linear_stream_stage(stage)]
    if not stages or not samples:
        return []
    outputs = _stage_outputs(net, stages, samples)
    with torch.no_grad():
        for k in stages:
            peak = torch.stack([o.abs().amax(dim=(0, 2, 3)) for o in outputs[k]]).amax(0)
            peak = peak.clamp_min(1e-3 * float(peak.max()))
            _rescale_stream(net, k, (peak / peak.max()) ** float(alpha))
    return stages


def bound_stage_output(net: nn.Module, stage_index: int, samples, percentile: float) -> float:
    """Clamp a linear stage's output at the ``percentile`` of ``|x|`` over ``samples``.

    The stream is scaled by ``1 / T`` (FP32-exact), ``Hardtanh(-1, 1)`` is
    appended to the stage and the next consumer takes ``T`` back. Returns ``T``.
    """
    if not _linear_stream_stage(net.blocks[stage_index]):
        raise ValueError(f"Stage {stage_index} has no linear residual stream to bound.")
    values = torch.cat([o.abs().flatten() for o in _stage_outputs(net, [stage_index], samples)[stage_index]])
    k = max(1, int(round(values.numel() * (1.0 - float(percentile) / 100.0))))
    bound = float(values.topk(k).values[-1])
    channels = list(net.blocks[stage_index])[0].pw_proj.bn.weight.shape[0]
    with torch.no_grad():
        _rescale_stream(net, stage_index, torch.full((channels,), bound))
    net.blocks[stage_index] = nn.Sequential(*net.blocks[stage_index], nn.Hardtanh(-1.0, 1.0))
    return bound


#: Defaults measured on a Metis M.2 Max (docs/axelera.md): equalize every linear
#: stream, then bound the output of stage 2 at this |x| percentile.
INT8_EQUALIZATION_ALPHA = 0.5
INT8_STAGE_BOUNDS = {2: 99.9}


def prepare_for_int8(net: nn.Module, samples) -> dict:
    """Equalize and bound ``net`` in place for per-tensor INT8; returns a summary."""
    stages = equalize_residual_streams(net, samples, INT8_EQUALIZATION_ALPHA)
    bounds = {}
    for stage, percentile in INT8_STAGE_BOUNDS.items():
        if stage in stages:
            bounds[stage] = bound_stage_output(net, stage, samples, percentile)
    return {"equalized_stages": stages, "stage_bounds": bounds}
