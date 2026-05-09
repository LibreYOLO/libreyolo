"""Build a DAMO-YOLO Detector from a size code."""

from __future__ import annotations

import torch.nn as nn

from .nn import Detector, GiraffeNeckV2, TinyNAS, ZeroHead
from .structures import SIZES, FamilyConfig


def _build_backbone(cfg: FamilyConfig) -> TinyNAS:
    return TinyNAS(
        structure_info=cfg.structure,
        out_indices=cfg.backbone_out_indices,
        with_spp=cfg.backbone_with_spp,
        use_focus=cfg.backbone_use_focus,
        act=cfg.backbone_act,
        reparam=cfg.backbone_reparam,
    )


def _build_neck(cfg: FamilyConfig) -> GiraffeNeckV2:
    return GiraffeNeckV2(
        depth=cfg.neck_depth,
        hidden_ratio=cfg.neck_hidden_ratio,
        in_channels=cfg.neck_in_channels,
        out_channels=cfg.neck_out_channels,
        act=cfg.neck_act,
        spp=cfg.neck_spp,
        block_name="BasicBlock_3x3_Reverse",
    )


def _build_head(cfg: FamilyConfig, num_classes: int) -> ZeroHead:
    return ZeroHead(
        num_classes=num_classes,
        in_channels=cfg.head_in_channels,
        stacked_convs=cfg.head_stacked_convs,
        feat_channels=cfg.head_feat_channels,
        reg_max=cfg.head_reg_max,
        strides=(8, 16, 32),
        act=cfg.head_act,
        legacy=cfg.head_legacy,
    )


def build_damoyolo(size: str = "t", num_classes: int = 80) -> nn.Module:
    """Build a DAMO-YOLO Detector for the given size."""
    if size not in SIZES:
        raise ValueError(f"Unknown DAMO-YOLO size {size!r}. Available: {sorted(SIZES)}")
    cfg = SIZES[size]
    backbone = _build_backbone(cfg)
    neck = _build_neck(cfg)
    head = _build_head(cfg, num_classes=num_classes)
    return Detector(backbone, neck, head)
