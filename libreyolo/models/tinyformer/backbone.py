"""TinyFormer SSA backbone: DINOv3/ViT-Tiny plus the Spatial Semantic Adapter.

Ported from the official TinyFormer release (mmpmmpmmpjosh/TinyFormer,
Apache-2.0), which builds on DEIMv2. Attribute names (``dinov3``, ``sda``,
``proj_c1``..``proj_c4``) mirror upstream so released checkpoints load with a
metadata wrap only.

The DINOv3 ViT tower is vendored under
``libreyolo/models/deimv2/engine/backbone/dinov3/`` and carries Meta's DINOv3
License Agreement; the s/m sizes use the DEIMv2-distilled ViT-Tiny towers
instead (see the family docstring in ``model.py`` for the license split).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..deimv2.engine.backbone.dinov3.vision_transformer import DinoVisionTransformer
from ..deimv2.engine.backbone.vit_tiny import VisionTransformer


class DINOv3SSAs_4Scale(nn.Module):
    """4-scale SSA backbone (the PBM variant, TinyFormer's release config).

    Emits (C1, C2, C3, C4) at strides (4, 8, 16, 32). C1 and C2 carry the
    Spatial Detail Extractor stem features; C2 additionally fuses the first
    interaction-layer ViT tokens. The stride-4 level is consumed inside
    ``HybridEncoder_4Scale`` only — the decoder still sees 3 levels.
    """

    def __init__(
        self,
        name: str,
        interaction_indexes: list[int],
        embed_dim: int = 192,
        num_heads: int = 3,
        conv_inplane: int = 16,
        hidden_dim: int | None = None,
        finetune: bool = True,
        patch_size: int = 16,
    ):
        super().__init__()

        if "dinov3" in name:
            self.dinov3 = DinoVisionTransformer(name=name)
        else:
            self.dinov3 = VisionTransformer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                return_layers=interaction_indexes,
            )

        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # Spatial Detail Extractor: three stride-2 conv stages (1/2, 1/4, 1/8).
        self.sda = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, conv_inplane, 3, 2, 1, bias=False),
                nn.SyncBatchNorm(conv_inplane),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(conv_inplane, conv_inplane, 3, 2, 1, bias=False),
                nn.SyncBatchNorm(conv_inplane),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(conv_inplane, 2 * conv_inplane, 3, 2, 1, bias=False),
                nn.SyncBatchNorm(2 * conv_inplane),
                nn.GELU(),
            ),
        )
        c1_dim = conv_inplane
        sda_dim = 2 * conv_inplane

        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim

        self.proj_c1 = nn.Sequential(
            nn.Conv2d(c1_dim, hidden_dim, 1, bias=False),
            nn.SyncBatchNorm(hidden_dim),
            nn.GELU(),
        )
        self.proj_c2 = nn.Sequential(
            nn.Conv2d(sda_dim + embed_dim, hidden_dim, 1, bias=False),
            nn.SyncBatchNorm(hidden_dim),
            nn.GELU(),
        )
        self.proj_c3 = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 1, bias=False),
            nn.SyncBatchNorm(hidden_dim),
        )
        self.proj_c4 = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 1, bias=False),
            nn.SyncBatchNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        H_16, W_16 = x.shape[2] // 16, x.shape[3] // 16

        if len(self.interaction_indexes) > 0 and not isinstance(
            self.dinov3, VisionTransformer
        ):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:
            l0, l1, l2 = all_layers[0], all_layers[0], all_layers[0]
        else:
            l0, l1, l2 = all_layers[0], all_layers[1], all_layers[2]

        feat0 = l0[0].transpose(1, 2).view(B, -1, H_16, W_16).contiguous()
        feat1 = l1[0].transpose(1, 2).view(B, -1, H_16, W_16).contiguous()
        feat2 = l2[0].transpose(1, 2).view(B, -1, H_16, W_16).contiguous()

        s_1_2 = self.sda[0](x)
        s_1_4 = self.sda[1](s_1_2)
        s_1_8 = self.sda[2](s_1_4)

        c1 = self.proj_c1(s_1_4)

        target_h8, target_w8 = s_1_8.shape[2:]
        feat0_up = F.interpolate(
            feat0, size=(target_h8, target_w8), mode="bilinear", align_corners=False
        )
        c2 = self.proj_c2(torch.cat([s_1_8, feat0_up], dim=1))

        c3 = self.proj_c3(feat1)

        feat2_down = F.interpolate(
            feat2, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        c4 = self.proj_c4(feat2_down)

        return c1, c2, c3, c4
