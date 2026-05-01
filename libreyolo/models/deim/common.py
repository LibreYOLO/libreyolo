"""Shared nn building blocks for the DEIM family.

Ported from DEIM (https://github.com/Intellindust-AI-Lab/DEIM) / RT-DETR.
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with frozen batch statistics and affine params.

    Copy-paste from torchvision.misc.ops with added ``eps`` before rsqrt — without
    it, backbones other than torchvision's resnet[18, 34, 50, 101] produce NaNs.
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}"


def freeze_batch_norm2d(module: nn.Module) -> nn.Module:
    """Recursively convert BatchNorm2d layers to FrozenBatchNorm2d in-place."""
    if isinstance(module, nn.BatchNorm2d):
        module = FrozenBatchNorm2d(module.num_features)
    else:
        for name, child in module.named_children():
            _child = freeze_batch_norm2d(child)
            if _child is not child:
                setattr(module, name, _child)
    return module
