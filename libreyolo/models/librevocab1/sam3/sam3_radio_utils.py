"""
Utility functions for replacing SAM3's vision encoder with RADIO.

This module provides functionality to:
1. Load RADIO models from torch.hub
2. Replace SAM3's vision encoder with RADIO
3. Create appropriate Sam3Processor with correct image sizes for RADIO
"""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from timm.layers.helpers import to_2tuple

from libreyolo.models.librevocab1.sam3.model.sam3_image_processor import Sam3Processor


DEFAULT_NECK_NAME = 'sam3'
DEFAULT_RADIO_RESOLUTION = 1008 * 16 // 14  # RADIO uses 16x16 patches vs SAM3's 14x14


class RADIO_Adaptor(nn.Module):
    """Replaces SAM3's ViT trunk with student features."""

    def __init__(self,
                 student: nn.Module,
                 input_size: int,
                 output_channels: int,
                 student_neck_name: str = DEFAULT_NECK_NAME,
    ):
        super().__init__()

        self.student = student
        self.input_size = to_2tuple(input_size)
        self.student_neck_name = student_neck_name

        # SAM3's ViT has a channel_list attribute that the neck uses
        self.channel_list = [output_channels]

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """
        Forward pass that mimics ViT trunk output.

        Args:
            images: Input tensor [B, C, H, W]

        Returns:
            List of feature tensors in [B, C, H, W] format (ViT output format)
        """
        # Normalize inputs to [0, 1]
        images = (images + 1) / 2

        if images.shape[-2:] != self.input_size:
            images = F.interpolate(images, self.input_size, mode='bilinear', align_corners=False)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            student_output = self.student(images)
            if isinstance(student_output, dict):
                student_output = student_output[self.student_neck_name]
            features = student_output[1]

        patch_size = int(round(math.sqrt(images.shape[-2] * images.shape[-1] / features.shape[1])))

        rows = images.shape[-2] // patch_size
        cols = images.shape[-1] // patch_size

        # Reshape from [B, N, C] to [B, C, H, W] to match ViT output format
        features = rearrange(features, 'b (r c) d -> b d r c', r=rows, c=cols)

        # Return as a list (ViT returns list of outputs from global attention blocks)
        return [features.float()]


def load_radio_model(model_version: str, device: str = 'cuda', vitdet: Optional[int] = None):
    """
    Load RADIO model from torch.hub.

    Args:
        checkpoint_path: Path to the RADIO model checkpoint
        device: Device to load model on

    Returns:
        RADIO model with SAM3 adaptor
    """
    extra = dict()
    if vitdet:
        extra['vitdet_window_size'] = vitdet

    print(f"Loading RADIO model from {model_version}...")
    model: torch.nn.Module = torch.hub.load(
        'NVlabs/RADIO',
        'radio_model',
        model_version,
        adaptor_names='sam3',
        **extra,
    )
    model = model.to(device)
    model.eval()
    print("RADIO model loaded successfully!")
    return model


def replace_sam3_encoder(sam3_model, radio_model, device: str = 'cuda'):
    """
    Replace SAM3's vision encoder with RADIO model.

    Args:
        sam3_model: The SAM3 image model
        radio_model: The RADIO model with SAM3 adaptor
        device: Device for computations

    Returns:
        Modified SAM3 model with RADIO encoder
    """
    print("Replacing SAM3 vision encoder with RADIO...")

    # Get the original SAM3 vision encoder to extract configuration
    original_encoder = sam3_model.backbone.vision_backbone

    sam3_dim = original_encoder.trunk.patch_embed.proj.out_channels

    # Create the adaptor
    input_size = 1152  # SAM3 standard input size
    adaptor = RADIO_Adaptor(
        student=radio_model,
        input_size=input_size,
        output_channels=sam3_dim,
        student_neck_name=DEFAULT_NECK_NAME,
    )

    # Replace the trunk in SAM3's vision encoder
    sam3_model.backbone.vision_backbone.trunk = adaptor

    print("Vision encoder replaced successfully!")
    return sam3_model


def create_sam3_radio_processor(sam3_model,
                                confidence_threshold: float = 0.5,
                                resolution: Optional[int] = None,
                                device: str = "cuda") -> Sam3Processor:
    """
    Create a Sam3Processor configured for use with RADIO encoder.

    RADIO uses 16x16 patches while SAM3's default ViT uses 14x14 patches,
    so we need a different resolution for optimal performance.

    Args:
        sam3_model: The SAM3 model (potentially with RADIO encoder)
        confidence_threshold: Confidence threshold for predictions
        resolution: Image resolution to use. If None, uses DEFAULT_RADIO_RESOLUTION (1152)
                   for RADIO-based models. Set to 1008 for original SAM3 encoder.

    Returns:
        Configured Sam3Processor instance
    """
    if resolution is None:
        resolution = DEFAULT_RADIO_RESOLUTION

    print(f"Creating Sam3Processor with resolution={resolution}, device={device}")
    processor = Sam3Processor(
        sam3_model,
        resolution=resolution,
        confidence_threshold=confidence_threshold,
        device=device,
    )

    return processor
