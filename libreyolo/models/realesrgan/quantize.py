"""INT8 preparation of Real-ESRGAN's compact network (SRVGGNetCompact) for integer accelerators.

The compact network is a chain of 3x3 convolutions with a per-channel
activation between them (PReLU in the released x4 v3 weights). Under
per-output-channel INT8 weights, a convolution whose output channels have very
different weight ranges keeps its quiet channels on a coarse grid, and the next
convolution sees those channels with the same per-tensor activation step as the
loud ones.

:func:`equalize_conv_activation_ranges` rescales every hidden channel ``c`` by
``1 / s_c`` at its producer (weights and bias) and by ``s_c`` at its consumer's
input channel, with ``s_c = sqrt(r1_c / r2_c)`` from the producer's output
channel range ``r1_c`` and the consumer's input channel range ``r2_c`` (cross
layer equalization). ReLU, LeakyReLU and PReLU are positively homogeneous per
channel, so the network computes the same function in FP32.
"""

from __future__ import annotations

import torch
from torch import nn

#: Equalization sweeps over the whole chain.
EQUALIZATION_ROUNDS = 3

_HOMOGENEOUS = (nn.ReLU, nn.LeakyReLU, nn.PReLU)


def _chain(net: nn.Module) -> tuple[list[nn.Conv2d], list[nn.Module]]:
    """The trunk's convolutions and the activations between them, in order."""
    layers = list(net.body)
    convs = [m for m in layers if isinstance(m, nn.Conv2d)]
    acts = [m for m in layers if not isinstance(m, nn.Conv2d)]
    if len(acts) != len(convs) - 1 or len(layers) != len(convs) + len(acts):
        raise ValueError("Expected a conv/activation chain ending in a convolution.")
    for k, act in enumerate(acts):
        if not isinstance(act, _HOMOGENEOUS):
            raise ValueError(f"Activation {k} ({type(act).__name__}) is not positively homogeneous.")
        if layers[2 * k + 1] is not act:
            raise ValueError("Expected convolutions and activations to alternate.")
    return convs, acts


def equalize_conv_activation_ranges(net: nn.Module, rounds: int = EQUALIZATION_ROUNDS) -> int:
    """Cross-layer equalize every hidden channel of ``net`` in place; returns the number of equalized pairs."""
    convs, _ = _chain(net)
    with torch.no_grad():
        for _ in range(int(rounds)):
            for producer, consumer in zip(convs[:-1], convs[1:]):
                r1 = producer.weight.abs().reshape(producer.weight.shape[0], -1).amax(1).clamp_min(1e-8)
                r2 = consumer.weight.abs().transpose(0, 1).reshape(consumer.weight.shape[1], -1).amax(1).clamp_min(1e-8)
                scale = torch.sqrt(r1 / r2)
                producer.weight.div_(scale.reshape(-1, 1, 1, 1))
                if producer.bias is not None:
                    producer.bias.div_(scale)
                consumer.weight.mul_(scale.reshape(1, -1, 1, 1))
    return len(convs) - 1


__all__ = ["EQUALIZATION_ROUNDS", "equalize_conv_activation_ranges"]
