"""
HiResCAM implementation for YOLO model interpretability.

HiResCAM uses element-wise multiplication of activations and gradients,
providing provably guaranteed faithfulness for certain model architectures.

Reference: https://arxiv.org/abs/2011.08891
"""

from typing import List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn

from .base import BaseCAM


class HiResCAM(BaseCAM):
    """
    HiResCAM: High-Resolution Class Activation Mapping.
    
    Element-wise multiplies activations with gradients, providing
    higher resolution explanations than GradCAM.
    
    Reference:
        Draelos, R. L., & Carin, L. (2020). Use HiResCAM instead of Grad-CAM
        for faithful explanations of convolutional neural networks.
        arXiv:2011.08891
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        reshape_transform: Optional[Callable] = None
    ) -> None:
        """
        Initialize HiResCAM.

        Args:
            model: The neural network model.
            target_layers: List of target layers for CAM computation.
            reshape_transform: Optional transform for activation shapes.
        """
        super().__init__(
            model,
            target_layers,
            reshape_transform,
            uses_gradients=True  # HiResCAM requires gradients
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
        HiResCAM doesn't use pooled weights - returns ones.
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
        Compute HiResCAM using element-wise multiplication.
        
        Args:
            input_tensor: The input image tensor.
            target_layer: The layer being processed.
            targets: Optional target specifications.
            activations: The layer activations of shape (B, C, H, W).
            grads: The gradients of shape (B, C, H, W).
            eigen_smooth: Whether to apply SVD smoothing.
        
        Returns:
            CAM array of shape (B, H, W).
        """
        if grads is None:
            # Fallback to activation mean
            return activations.mean(axis=1)
        
        # Element-wise multiplication of activations and gradients
        elementwise_product = activations * grads
        
        if eigen_smooth:
            cam = self._get_2d_projection(elementwise_product)
        else:
            # Sum over channel dimension
            cam = elementwise_product.sum(axis=1)
        
        return cam

