"""
EigenGradCAM implementation for YOLO model interpretability.

EigenGradCAM combines the gradient information with SVD-based dimensionality
reduction, providing class-discriminative explanations like GradCAM but
with cleaner results.

Reference: Similar to EigenCAM but with gradient weighting.
"""

from typing import List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn

from .base import BaseCAM


class EigenGradCAM(BaseCAM):
    """
    EigenGradCAM: Eigen-based Gradient-weighted Class Activation Mapping.
    
    Like EigenCAM but with class discrimination: computes the first principal
    component of (Activations * Gradients). Looks like GradCAM but cleaner.
    
    This combines the benefits of:
    - EigenCAM: SVD-based dimensionality reduction for cleaner heatmaps
    - GradCAM: Gradient-based class discrimination
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        reshape_transform: Optional[Callable] = None
    ) -> None:
        """
        Initialize EigenGradCAM.

        Args:
            model: The neural network model.
            target_layers: List of target layers for CAM computation.
            reshape_transform: Optional transform for activation shapes.
        """
        super().__init__(
            model,
            target_layers,
            reshape_transform,
            uses_gradients=True  # EigenGradCAM requires gradients
        )

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        targets: Optional[List],
        activations: np.ndarray,
        grads: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        EigenGradCAM doesn't use pooled weights - returns ones.
        Actual computation happens in get_cam_image.
        """
        return np.ones((activations.shape[0], activations.shape[1]), dtype=np.float32)

    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        targets: Optional[List],
        activations: np.ndarray,
        grads: Optional[np.ndarray],
        eigen_smooth: bool = False
    ) -> np.ndarray:
        """
        Compute EigenGradCAM using SVD on activation*gradient product.
        
        Args:
            input_tensor: The input image tensor.
            target_layer: The layer being processed.
            targets: Optional target specifications.
            activations: The layer activations of shape (B, C, H, W).
            grads: The gradients of shape (B, C, H, W).
            eigen_smooth: Ignored (always uses eigen method).
        
        Returns:
            CAM array of shape (B, H, W).
        """
        if grads is None:
            # Fallback to regular EigenCAM
            return self._get_2d_projection(activations)
        
        # Element-wise product of activations and gradients
        weighted_activations = activations * grads
        
        # Apply SVD to get first principal component
        return self._get_2d_projection(weighted_activations)

