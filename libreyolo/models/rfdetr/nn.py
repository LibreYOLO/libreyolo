"""
RF-DETR Neural Network for LibreYOLO.

This module provides the RF-DETR model by importing from the original
rfdetr package to ensure 100% weight compatibility.
"""

import torch
import torch.nn as nn

from rfdetr.main import Model as RFDETRMainModel
from rfdetr.models.lwdetr import LWDETR, MLP, PostProcess
from rfdetr.config import (
    RFDETRLargeConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    RFDETRMediumConfig,
    RFDETRSegLargeConfig,
    RFDETRSegNanoConfig,
    RFDETRSegSmallConfig,
    RFDETRSegMediumConfig,
)


RFDETR_CONFIGS = {
    "n": RFDETRNanoConfig,
    "s": RFDETRSmallConfig,
    "m": RFDETRMediumConfig,
    "l": RFDETRLargeConfig,
}

RFDETR_SEG_CONFIGS = {
    "n": RFDETRSegNanoConfig,
    "s": RFDETRSegSmallConfig,
    "m": RFDETRSegMediumConfig,
    "l": RFDETRSegLargeConfig,
}


class LibreRFDETRModel(nn.Module):
    """
    RF-DETR Detection Transformer model wrapper.

    This wraps the original RF-DETR model to provide a consistent interface
    while maintaining 100% weight compatibility. Supports both detection and
    instance segmentation variants.
    """

    def __init__(
        self,
        config: str = "s",
        nb_classes: int = 80,
        pretrain_weights: str | None = None,
        device: str = "cpu",
        segmentation: bool = False,
    ):
        """
        Initialize RF-DETR model.

        Args:
            config: Model size variant ('n', 's', 'm', 'l')
            nb_classes: Number of object classes (use 80 for COCO)
            pretrain_weights: Path to pretrained weights (optional)
            device: Device to use ('cpu', 'cuda', 'mps')
            segmentation: If True, use segmentation config with mask head
        """
        super().__init__()

        configs = RFDETR_SEG_CONFIGS if segmentation else RFDETR_CONFIGS
        if config not in configs:
            raise ValueError(
                f"Invalid config: {config}. Must be one of: {list(configs.keys())}"
            )

        self.config_name = config
        self.nb_classes = nb_classes
        self.segmentation = segmentation

        config_cls = configs[config]
        model_config = config_cls(
            num_classes=nb_classes,
            pretrain_weights=pretrain_weights,
        )

        self.resolution = model_config.resolution
        self.hidden_dim = model_config.hidden_dim
        self.num_queries = getattr(model_config, "num_queries", 300)

        config_dict = model_config.dict()
        config_dict["device"] = device  # Override device
        self._rfdetr = RFDETRMainModel(**config_dict)

        self.model = self._rfdetr.model
        self.postprocess = self._rfdetr.postprocess

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Dictionary with 'pred_logits', 'pred_boxes', and optionally
            'pred_masks' (inference mode), or tuple of tensors in export mode.
        """
        out = self.model(x)
        # In export mode, forward_export returns (coord, class, masks).
        # Pass through all available tensors.
        if isinstance(out, tuple):
            coord, cls = out[0], out[1]
            if len(out) >= 3 and out[2] is not None:
                return coord, cls, out[2]
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
    segmentation: bool = False,
) -> LibreRFDETRModel:
    """
    Create an RF-DETR model.

    Args:
        config: Model size variant ('n', 's', 'm', 'l')
        nb_classes: Number of object classes
        pretrain_weights: Path to pretrained weights
        device: Device to use
        segmentation: If True, create segmentation variant with mask head

    Returns:
        LibreRFDETRModel instance
    """
    return LibreRFDETRModel(
        config=config,
        nb_classes=nb_classes,
        pretrain_weights=pretrain_weights,
        device=device,
        segmentation=segmentation,
    )


__all__ = [
    "LibreRFDETRModel",
    "create_rfdetr_model",
    "RFDETR_CONFIGS",
    "LWDETR",
    "MLP",
    "PostProcess",
]
