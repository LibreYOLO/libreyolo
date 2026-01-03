"""
RT-DETR utility functions for LibreYOLO.

Provides preprocessing utilities with ImageNet normalization.
"""

from typing import Tuple, Union
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

from ..common.image_loader import ImageLoader, ImageInput
from ..common.utils import draw_boxes  # Re-export shared draw_boxes


# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

__all__ = ['preprocess_image', 'draw_boxes', 'denormalize_image', 'IMAGENET_MEAN', 'IMAGENET_STD']


def preprocess_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto"
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
    """
    Preprocess an image for RT-DETR inference.

    RT-DETR requires ImageNet normalization unlike YOLO models.

    Args:
        image: Input image (path, PIL, numpy, tensor, bytes, etc.)
        input_size: Target size for model input (default: 640)
        color_format: Color format hint for numpy arrays ('auto', 'rgb', 'bgr')

    Returns:
        Tuple of:
            - input_tensor: Preprocessed tensor [1, 3, H, W]
            - original_image: Original PIL image
            - original_size: Original (width, height)
    """
    # Load image using common ImageLoader
    original_image = ImageLoader.load(image, color_format=color_format)
    original_size = original_image.size  # (width, height)

    # Create transform pipeline
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Apply transforms
    input_tensor = transform(original_image).unsqueeze(0)

    return input_tensor, original_image, original_size


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize an ImageNet-normalized tensor back to [0, 255] range.

    Args:
        tensor: Normalized tensor [C, H, W] or [B, C, H, W]

    Returns:
        Numpy array in [0, 255] uint8 format
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    tensor = (tensor * 255).byte()

    return tensor.permute(1, 2, 0).numpy()
