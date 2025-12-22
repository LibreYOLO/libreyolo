"""
XGradCAM implementation for YOLO model interpretability.

XGradCAM (Axiom-based Grad-CAM) scales the gradients by the normalized
activations, providing better axiom satisfaction than standard GradCAM.

Reference: https://arxiv.org/abs/2008.02312
"""

from typing import List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn

from .base import BaseCAM


class XGradCAM(BaseCAM):
    """
    XGradCAM: Axiom-based Gradient-weighted Class Activation Mapping.
    
    Like GradCAM but scales the gradients by the normalized activations,
    improving axiom satisfaction and sensitivity.
    
    Reference:
        Fu, R., et al. (2020). Axiom-based Grad-CAM: Towards Accurate
        Visualization and Explanation of CNNs. arXiv:2008.02312
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        reshape_transform: Optional[Callable] = None
    ) -> None:
        """
        Initialize XGradCAM.

        Args:
            model: The neural network model.
            target_layers: List of target layers for CAM computation.
            reshape_transform: Optional transform for activation shapes.
        """
        super().__init__(
            model,
            target_layers,
            reshape_transform,
            uses_gradients=True  # XGradCAM requires gradients
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
        Compute XGradCAM weights by scaling gradients with normalized activations.
        
        XGradCAM formula: 
            weights = sum(grad * activation) / (sum(activation) + eps)
        
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
        
        # Sum of activations per channel for normalization
        # Shape: (B, C)
        sum_activations = np.sum(activations, axis=(2, 3))
        
        # Prevent division by zero
        eps = 1e-7
        
        # Element-wise product of gradients and activations, summed spatially
        # Shape: (B, C)
        grad_activation_product = np.sum(grads * activations, axis=(2, 3))
        
        # Normalize by sum of activations
        weights = grad_activation_product / (sum_activations + eps)
        
        return weights

