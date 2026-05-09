"""LibreRTDETRv2 — RT-DETRv2 detectors."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import torch.nn as nn

from ...validation.preprocessors import RTDETRv2ValPreprocessor
from ..rtdetr.model import LibreRTDETR, RTDETR_CONFIGS
from .nn import RTDETRv2Model


class LibreRTDETRv2(LibreRTDETR):
    FAMILY = "rtdetrv2"
    FILENAME_PREFIX = "LibreRTDETRv2"
    INPUT_SIZES = {"r18": 640, "r34": 640, "r50": 640, "r50m": 640, "r101": 640}
    val_preprocessor_class = RTDETRv2ValPreprocessor

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # State-dict shape is identical to v1's, so we delegate to v1's check.
        # Disambiguation against v1 in the factory happens via:
        #   (1) the ``model_family`` metadata gate (converted ckpts);
        #   (2) the ``rtdetrv2_`` filename hint (raw upstream ckpts);
        #   (3) registry order — ``LibreRTDETR`` is imported BEFORE
        #       ``LibreRTDETRv2`` so that ckpts lacking both signals route
        #       to v1 by default. v1 cannot be silently shadowed.
        return LibreRTDETR.can_load(weights_dict)

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        detected = super().detect_size_from_filename(filename)
        if detected is not None:
            return detected
        basename = os.path.basename(filename).lower()
        m = re.search(r"rtdetrv2_r(\d+)vd(_m)?_", basename)
        if m:
            depth, m_suffix = m.group(1), m.group(2)
            return f"r{depth}m" if m_suffix else f"r{depth}"
        return None

    @classmethod
    def _get_trainer_class(cls):
        from .trainer import RTDETRv2Trainer

        return RTDETRv2Trainer

    def _init_model(self) -> nn.Module:
        if self.size not in RTDETR_CONFIGS:
            raise ValueError(f"Unknown RT-DETRv2 size: {self.size!r}")
        cfg: Dict[str, Any] = RTDETR_CONFIGS[self.size]
        # v2 ResNet sizes only — HGNetv2 backbones are skipped at this layer
        # (v1 already ships HGNetv2-l/x; v2's HGNetv2 numbers are within ~0.1
        # AP and not worth the duplicate weights).
        if cfg.get("backbone_type") == "hgnetv2":
            raise ValueError(
                f"LibreRTDETRv2 size {self.size!r} uses an HGNetv2 backbone; "
                f"use LibreRTDETR for HGNetv2 variants."
            )
        return RTDETRv2Model(
            num_classes=self.nb_classes,
            backbone_depth=cfg["backbone_depth"],
            backbone_freeze_at=cfg["backbone_freeze_at"],
            backbone_freeze_norm=cfg["backbone_freeze_norm"],
            backbone_pretrained=False,
            hidden_dim=cfg["encoder_hidden_dim"],
            dim_feedforward=cfg["encoder_dim_feedforward"],
            expansion=cfg["encoder_expansion"],
            decoder_hidden_dim=cfg["decoder_hidden_dim"],
            decoder_dim_feedforward=cfg.get("decoder_dim_feedforward", 1024),
            num_decoder_layers=cfg["num_decoder_layers"],
            eval_idx=cfg["eval_idx"],
            eval_spatial_size=(self.input_size, self.input_size),
        )
