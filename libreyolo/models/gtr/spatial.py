"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Backbone for the Spatial SwiGLU distilled students.

``vit_adapter.py`` is left untouched apart from the ``_build_vit`` hook this file
overrides.
"""

# Adapted for LibreYOLO from Intellindust-AI-Lab/GTR
# revision 782e737efe2e6437ac537fbdcee089673d3376c1 (MIT).
# Changes: native construction, local imports and portable attention. See NOTICE.

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import VisionTransformer, ViTAdapter

__all__ = [
    "ViTAdapterSpatialSwiGLU",
]


class SpatialSwiGLU(nn.Module):
    """Spatial SwiGLU: a SwiGLU (GLAMLP) whose value branch is restored to a
    2D feature map and passed through a 3x3 depthwise conv before gating
    """

    def __init__(self, hidden_size, hidden_ratio=4, intermediate_size=None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.dwconv = nn.Conv2d(
            intermediate_size, intermediate_size, 3, padding=1, groups=intermediate_size
        )
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def convert_to_deploy(self):
        # Split the packed gate/value GEMM so the value half is a standalone [B, L, C]
        # tensor: viewed as [B, C, gs, gs] it is already channels_last, so the depthwise
        # conv consumes and returns it without the NCHW re-layout copy that the `chunk`
        # view of the packed output forced (one ATen transpose-copy per block).
        if getattr(self, "_deploy_split_gemm", False):
            return
        w = self.gate_proj.weight
        half = w.shape[0] // 2
        self.gate_w = nn.Parameter(w[:half].detach().clone(), requires_grad=False)
        self.value_w = nn.Parameter(w[half:].detach().clone(), requires_grad=False)
        del self.gate_proj
        self.dwconv.weight.data = self.dwconv.weight.data.contiguous(
            memory_format=torch.channels_last
        )
        self._deploy_split_gemm = True

    def forward(self, x):
        if getattr(self, "_deploy_split_gemm", False):
            B, L, _ = x.shape
            gs = int(L**0.5)
            gate = F.linear(x, self.gate_w)
            y = F.linear(x, self.value_w)  # [B, L, C] contiguous
            y = y.view(B, gs, gs, -1).permute(
                0, 3, 1, 2
            )  # channels_last view of [B, C, gs, gs]
            y = self.dwconv(y)  # channels_last in -> channels_last out
            y = y.permute(0, 2, 3, 1).reshape(B, L, -1)  # free view back to [B, L, C]
            z = F.silu(gate) * y
            return F.linear(z, self.down_proj.weight, self.down_proj.bias)
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        B, L, C = y.shape
        gs = int(L**0.5)
        y = y.transpose(1, 2).reshape(B, C, gs, gs)
        y = self.dwconv(y).flatten(2).transpose(1, 2)
        # Use pure PyTorch SwiGLU to avoid Triton autotuner failures in swiglu_linear.
        z = F.silu(gate) * y
        return F.linear(z, self.down_proj.weight, self.down_proj.bias)


class VisionTransformerSpatialSwiGLU(VisionTransformer):
    """VisionTransformer without pos_embed / CLS / mask token, whose block MLPs
    are SpatialSwiGLU (the depthwise conv supplies positional information)."""

    def __init__(self, mlp_ratio=4.0, **kwargs):
        kwargs["use_pos_embed"] = False
        super().__init__(mlp_ratio=mlp_ratio, **kwargs)
        del self._model.cls_token
        del self._model.mask_token
        for blk in self._model.blocks:
            blk.mlp = SpatialSwiGLU(
                hidden_size=self.embed_dim, hidden_ratio=int(mlp_ratio)
            )
            # Re-apply weight init for the swapped-in modules
            # (Linear -> normal(0.02), Conv2d -> default reset_parameters).
            blk.mlp.apply(self._init_vit_weights)

    def _apply_bid_scan(self, x, layer_idx):
        # CLS-free version: the whole sequence is the patch grid.
        x = x.flip(1)
        if layer_idx % 2 == 1 and self.if_quad_dir:
            gs = int(x.shape[1] ** 0.5)
            x = x.reshape(x.shape[0], gs, gs, -1).transpose(1, 2).flatten(1, 2)
        return x

    def forward(self, x, return_layer_indexes=None, n=1, mask_ratio=0.0, num_stages=0):
        # The base forward_distill slices off a CLS token this variant does not have.
        assert num_stages == 0, (
            "VisionTransformerSpatialSwiGLU has no CLS token; distillation path unsupported"
        )

        if return_layer_indexes is None:
            rl_list = list(range(self.depth - n, self.depth))
        else:
            rl_list = sorted({int(i) for i in return_layer_indexes})
        rl_set = set(rl_list)

        x = self._model.patch_embed(x)

        L = x.shape[1]
        if self.bid_scan:
            perm = torch.arange(L, device=x.device)
            period = 4 if self.if_quad_dir else 2

        outs_by_idx = {}
        for i, blk in enumerate(self._model.blocks):
            x = blk(x)
            if self.bid_scan:
                x = self._apply_bid_scan(x, i)
                perm = self._update_perm(perm, i)
            if i in rl_set:
                if self.bid_scan and ((i + 1) % period != 0):
                    patch_out = x[:, torch.argsort(perm), :]
                else:
                    patch_out = x
                # No CLS in this variant: patch mean stands in for the summary token.
                outs_by_idx[i] = (patch_out, patch_out.mean(dim=1))

        return [outs_by_idx[i] for i in rl_list]


class ViTAdapterSpatialSwiGLU(ViTAdapter):
    """ViTAdapter whose ViT is the Spatial SwiGLU student. Same config keys as ViTAdapter."""

    @staticmethod
    def _build_vit(**kwargs):
        return VisionTransformerSpatialSwiGLU(**kwargs)
