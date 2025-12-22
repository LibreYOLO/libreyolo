"""
GradCAM++ implementation for YOLO model interpretability.

GradCAM++ uses second-order gradients to compute weighted importance,
providing better localization especially for multiple instances of the
same class in an image.

Reference: https://arxiv.org/abs/1710.11063
"""

from typing import List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn

from .base import BaseCAM


class GradCAMPlusPlus(BaseCAM):
    """
    GradCAM++: Improved Gradient-weighted Class Activation Mapping.
    
    Uses second-order gradients (squared gradients) for better weighting,
    particularly effective when multiple instances of the same class
    appear in the image.
    
    Reference:
        Chattopadhyay, A., et al. (2018). Grad-CAM++: Improved Visual
        Explanations for Deep Convolutional Networks. arXiv:1710.11063
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        reshape_transform: Optional[Callable] = None
    ) -> None:
        """
        Initialize GradCAM++.

        Args:
            model: The neural network model.
            target_layers: List of target layers for CAM computation.
            reshape_transform: Optional transform for activation shapes.
        """
        super().__init__(
            model,
            target_layers,
            reshape_transform,
            uses_gradients=True  # GradCAM++ requires gradients
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
        Compute GradCAM++ weights using second-order gradient information.
        
        GradCAM++ formula:
            alpha_kc = grad^2 / (2*grad^2 + sum(A * grad^3))
            weights = sum(alpha * ReLU(grad))
        
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
            return np.ones((activations.shape[0], activations.shape[1]), dtype=np.float32)
        
        # Second-order gradients
        grads_2 = grads ** 2
        grads_3 = grads ** 3
        
        # Sum of (activations * grad^3) over spatial dimensions
        # Shape: (B, C)
        sum_activations_grads = np.sum(activations * grads_3, axis=(2, 3))
        
        # Compute alpha (pixel-wise importance weights)
        # alpha = grad^2 / (2*grad^2 + sum(A*grad^3) + eps)
        eps = 1e-7
        
        # We need to keep spatial dimensions for alpha
        # Shape after sum_activations_grads: (B, C)
        # We need to broadcast it: (B, C, 1, 1)
        sum_term = sum_activations_grads[:, :, np.newaxis, np.newaxis]
        
        # Alpha shape: (B, C, H, W)
        alpha = grads_2 / (2 * grads_2 + sum_term * grads + eps)
        
        # Handle NaN/Inf
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        
        # ReLU on gradients
        positive_grads = np.maximum(grads, 0)
        
        # Weighted sum: alpha * ReLU(grad), then sum over spatial dims
        # Shape: (B, C)
        weights = np.sum(alpha * positive_grads, axis=(2, 3))
        
        return weights

