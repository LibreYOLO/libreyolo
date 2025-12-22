"""
GradCAM implementation for YOLO model interpretability.

GradCAM (Gradient-weighted Class Activation Mapping) uses the gradients
flowing into the target layer to produce a coarse localization map highlighting
important regions in the image for predicting the concept.

Reference: https://arxiv.org/abs/1610.02391
"""

from typing import List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn

from .base import BaseCAM


class GradCAM(BaseCAM):
    """
    GradCAM: Gradient-weighted Class Activation Mapping.
    
    Weights the 2D activations by the average gradient to produce
    class-discriminative localization maps.
    
    Reference:
        Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from
        Deep Networks via Gradient-based Localization. arXiv:1610.02391
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        reshape_transform: Optional[Callable] = None
    ) -> None:
        """
        Initialize GradCAM.

        Args:
            model: The neural network model.
            target_layers: List of target layers for CAM computation.
            reshape_transform: Optional transform for activation shapes.
        """
        super().__init__(
            model,
            target_layers,
            reshape_transform,
            uses_gradients=True  # GradCAM requires gradients
        )

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        targets: Optional[List],
        activations: np.ndarray,
        grads: np.ndarray
    ) -> np.ndarray:
        """
        Compute GradCAM weights by global average pooling the gradients.
        
        The weight for each channel is the mean gradient value across
        the spatial dimensions (H, W).
        
        Args:
            input_tensor: The input image tensor.
            target_layer: The layer being processed.
            targets: Optional target specifications.
            activations: The layer activations of shape (B, C, H, W).
            grads: The gradients of shape (B, C, H, W).
        
        Returns:
            Weights array of shape (B, C).
        """
        if grads is None:
            # Fallback if no gradients available
            return np.ones((activations.shape[0], activations.shape[1]), dtype=np.float32)
        
        # Global average pooling of gradients: (B, C, H, W) -> (B, C)
        return np.mean(grads, axis=(2, 3))

