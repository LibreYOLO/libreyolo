"""Top-level RT-DETRv2 module: v1 backbone + v1 encoder + v2 decoder."""

from __future__ import annotations

import torch.nn as nn

from ..rtdetr.backbone import PResNet
from ..rtdetr.nn import HybridEncoder
from .decoder import RTDETRTransformerv2


class RTDETRv2Model(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        backbone: nn.Module | None = None,
        backbone_depth: int = 18,
        backbone_variant: str = "d",
        backbone_pretrained: bool = False,
        backbone_freeze_norm: bool = False,
        backbone_freeze_at: int = 0,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        decoder_hidden_dim: int | None = None,
        decoder_dim_feedforward: int | None = None,
        expansion: float = 0.5,
        dropout: float = 0.0,
        num_denoising: int = 100,
        num_decoder_points=4,
        feat_strides=(8, 16, 32),
        num_levels: int = 3,
        eval_spatial_size=None,
        aux_loss: bool = True,
        eval_idx: int = -1,
        cross_attn_method: str = "default",
        query_select_method: str = "default",
        **kwargs,
    ):
        super().__init__()
        decoder_hidden_dim = decoder_hidden_dim or hidden_dim
        decoder_dim_feedforward = decoder_dim_feedforward or dim_feedforward

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = PResNet(
                depth=backbone_depth,
                variant=backbone_variant,
                return_idx=[1, 2, 3],
                pretrained=backbone_pretrained,
                freeze_norm=backbone_freeze_norm,
                freeze_at=backbone_freeze_at,
            )

        self.encoder = HybridEncoder(
            in_channels=self.backbone.out_channels,
            feat_strides=list(feat_strides),
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            expansion=expansion,
            dropout=dropout,
            eval_spatial_size=eval_spatial_size,
        )

        encoder_out_channels = [hidden_dim] * len(self.backbone.out_channels)
        self.decoder = RTDETRTransformerv2(
            num_classes=num_classes,
            hidden_dim=decoder_hidden_dim,
            num_queries=num_queries,
            feat_channels=encoder_out_channels,
            feat_strides=list(feat_strides),
            num_levels=num_levels,
            num_points=num_decoder_points,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=decoder_dim_feedforward,
            dropout=dropout,
            num_denoising=num_denoising,
            eval_spatial_size=eval_spatial_size,
            aux_loss=aux_loss,
            eval_idx=eval_idx,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
        )

    def forward(self, x, targets=None):
        feats = self.backbone(x)
        feats = self.encoder(feats)
        return self.decoder(feats, targets=targets)
