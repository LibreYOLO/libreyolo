"""ConvNeXt (V1) — native LibreYOLO implementation.

The architecture is a native re-implementation derived from timm's ConvNeXt
(``timm/models/convnext.py``) and the original Meta implementation
(``facebookresearch/ConvNeXt``, MIT). Module and attribute names mirror timm
exactly so a timm pretrained ``state_dict`` loads into this module with
``strict=True`` and produces bit-identical outputs (enforced by the parity
test). See the family NOTICE for attribution.

Only the V1 ``convnext_{tiny,small,base}.fb_in1k`` configuration is implemented
(``conv_mlp=False``, ``ls_init_value=1e-6``, patch stem, FB head ordering
pool -> norm -> fc). ConvNeXt-V2 (GRN instead of layer-scale) is intentionally
handled by the separate ``convnextv2`` family; its official pretrained
weights retain CC-BY-NC-4.0 independently of the MIT architecture code.

Upstream paper: "A ConvNet for the 2020s" (https://arxiv.org/abs/2201.03545).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Per-variant architecture: stage depths + channel dims (timm V1 fb_in1k).
# tiny/small share dims; they differ only in stage-2 depth (9 vs 27).
# --------------------------------------------------------------------------- #
ARCH_DEFS: Dict[str, Dict[str, Tuple[int, ...]]] = {
    "t": {"depths": (3, 3, 9, 3), "dims": (96, 192, 384, 768)},
    "s": {"depths": (3, 3, 27, 3), "dims": (96, 192, 384, 768)},
    "b": {"depths": (3, 3, 27, 3), "dims": (128, 256, 512, 1024)},
}

LS_INIT_VALUE = 1e-6  # layer-scale (gamma) init; value irrelevant once weights load
NORM_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Norm layers. Names ('weight'/'bias') match nn.LayerNorm so state dicts map 1:1.
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dim of an NCHW tensor (timm ``LayerNorm2d``).

    Permute to channels-last, apply ``F.layer_norm`` over C, permute back. timm's
    fast-norm path is disabled by default (``_USE_FAST_NORM = False``), so plain
    ``F.layer_norm`` reproduces timm bit-for-bit.
    """

    def __init__(self, num_channels: int, eps: float = NORM_EPS):
        super().__init__(num_channels, eps=eps, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class Mlp(nn.Module):
    """timm ``Mlp`` (channels-last): Linear -> GELU -> Linear. Submodules ``fc1``/``fc2``."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # exact (erf) GELU — timm 'gelu' maps to nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# --------------------------------------------------------------------------- #
# Block. Submodule names mirror timm: conv_dw / norm / mlp / gamma.
# --------------------------------------------------------------------------- #
class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (``conv_mlp=False`` path).

    DwConv7x7 -> permute NHWC -> LayerNorm -> MLP(Linear-GELU-Linear) ->
    permute NCHW -> layer-scale (``gamma``) -> residual. ``drop_path`` and the
    dim-matching ``shortcut`` are Identity here (stride 1, in==out), so they add
    no parameters and are folded into the plain residual.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=True)
        self.norm = nn.LayerNorm(dim, eps=NORM_EPS)
        self.mlp = Mlp(dim, int(mlp_ratio * dim))
        self.gamma = nn.Parameter(LS_INIT_VALUE * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return x + shortcut


class ConvNeXtStage(nn.Module):
    """A stage: optional (LayerNorm2d + 2x2 stride-2 conv) downsample then N blocks.

    timm names the downsample children ``downsample.0`` (norm) and
    ``downsample.1`` (conv); stage 0 has no downsample (Identity).
    """

    def __init__(self, in_chs: int, out_chs: int, depth: int, downsample: bool):
        super().__init__()
        if downsample:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=2, stride=2, bias=True),
            )
        else:
            self.downsample = nn.Identity()
        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_chs) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(x))


class ConvNeXtHead(nn.Module):
    """FB head ordering: global avg pool -> LayerNorm2d -> flatten -> Linear.

    Submodules ``norm`` (LayerNorm2d) and ``fc`` (Linear) carry the only
    parameters, matching timm's ``head.norm`` / ``head.fc`` keys.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = LayerNorm2d(in_features)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        if pre_logits:
            return x
        return self.fc(x)


# --------------------------------------------------------------------------- #
# The model.
# --------------------------------------------------------------------------- #
class ConvNeXt(nn.Module):
    """ConvNeXt V1 backbone + classification head (timm-compatible)."""

    def __init__(self, size: str = "t", num_classes: int = 1000):
        super().__init__()
        if size not in ARCH_DEFS:
            raise ValueError(f"Unknown ConvNeXt size {size!r}; choose from {list(ARCH_DEFS)}.")
        spec = ARCH_DEFS[size]
        depths, dims = spec["depths"], spec["dims"]
        self.size = size
        self.num_classes = num_classes

        # Patch stem: 4x4 stride-4 conv (patchify) then channels-first LayerNorm.
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4, bias=True),
            LayerNorm2d(dims[0]),
        )

        stages = []
        prev_chs = dims[0]
        for i in range(4):
            stages.append(
                ConvNeXtStage(prev_chs, dims[i], depth=depths[i], downsample=(i > 0))
            )
            prev_chs = dims[i]
        self.stages = nn.Sequential(*stages)

        self.num_features = prev_chs
        self.head = ConvNeXtHead(prev_chs, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return self.head(x, pre_logits=pre_logits)

    def reset_classifier(self, num_classes: int) -> None:
        """Swap the final Linear for fine-tuning to a new class count.

        Preserves the existing head's device AND dtype so half/bfloat16 flows
        survive a class-count change.
        """
        self.num_classes = num_classes
        w = self.head.fc.weight
        self.head.fc = nn.Linear(self.num_features, num_classes).to(device=w.device, dtype=w.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))
