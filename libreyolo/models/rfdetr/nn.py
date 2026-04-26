"""
RF-DETR Neural Network for LibreYOLO.

This module provides the RF-DETR model by importing from the original
rfdetr package to ensure 100% weight compatibility.
"""

import torch
import torch.nn as nn

from rfdetr.detr import _build_model_context
from rfdetr.models.lwdetr import LWDETR, MLP, PostProcess
from rfdetr.config import (
    RFDETRLargeConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    RFDETRMediumConfig,
)


RFDETR_CONFIGS = {
    "n": RFDETRNanoConfig,
    "s": RFDETRSmallConfig,
    "m": RFDETRMediumConfig,
    "l": RFDETRLargeConfig,
}


class LibreRFDETRModel(nn.Module):
    """
    RF-DETR Detection Transformer model wrapper.

    This wraps the original RF-DETR model to provide a consistent interface
    while maintaining 100% weight compatibility.
    """

    def __init__(
        self,
        config: str = "s",
        nb_classes: int = 80,
        pretrain_weights: str | None = None,
        device: str = "cpu",
    ):
        """
        Initialize RF-DETR model.

        Args:
            config: Model size variant ('n', 's', 'm', 'l')
            nb_classes: Number of object classes (use 80 for COCO)
            pretrain_weights: Path to pretrained weights (optional)
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        super().__init__()

        if config not in RFDETR_CONFIGS:
            raise ValueError(
                f"Invalid config: {config}. Must be one of: {list(RFDETR_CONFIGS.keys())}"
            )

        self.config_name = config
        self.nb_classes = nb_classes

        config_cls = RFDETR_CONFIGS[config]
        model_config = config_cls(
            num_classes=nb_classes,
            pretrain_weights=pretrain_weights,
            device=device,
        )

        self.resolution = model_config.resolution
        self.hidden_dim = model_config.hidden_dim
        self.num_queries = getattr(model_config, "num_queries", 300)

        ctx = _build_model_context(model_config)
        self.model = ctx.model
        self.postprocess = ctx.postprocess

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Dictionary with 'pred_logits' and 'pred_boxes' (inference mode),
            or tuple of (pred_boxes, pred_logits) when in export mode.
        """
        out = self.model(x)
        # In export mode, forward_export returns (coord, class, masks)
        # where masks may be None (not traceable). Return only tensors.
        if isinstance(out, tuple):
            coord, cls = out[0], out[1]
            return coord, cls
        return out

    def load_state_dict(self, state_dict, strict=True):
        if "model" in state_dict:
            state_dict = state_dict["model"]
        return self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self


def create_rfdetr_model(
    config: str = "s",
    nb_classes: int = 80,
    pretrain_weights: str | None = None,
    device: str = "cpu",
) -> LibreRFDETRModel:
    """
    Create an RF-DETR model.

    Args:
        config: Model size variant ('n', 's', 'm', 'l')
        nb_classes: Number of object classes
        pretrain_weights: Path to pretrained weights
        device: Device to use

    Returns:
        LibreRFDETRModel instance
    """
    return LibreRFDETRModel(
        config=config,
        nb_classes=nb_classes,
        pretrain_weights=pretrain_weights,
        device=device,
    )


__all__ = [
    "LibreRFDETRModel",
    "create_rfdetr_model",
    "RFDETR_CONFIGS",
    "LWDETR",
    "MLP",
    "PostProcess",
]
