"""
LayerCAM implementation for YOLO model interpretability.

LayerCAM spatially weights the activations by positive gradients,
working better especially in lower layers of the network.

Reference: http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf
"""

from typing import List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn

from .base import BaseCAM


class LayerCAM(BaseCAM):
    """
    LayerCAM: Exploring Hierarchical Class Activation Maps for Localization.
    
    Spatially weights the activations by positive gradients, providing
    better results especially in lower (earlier) layers of the network.
    
    Reference:
        Jiang, P. T., et al. (2021). LayerCAM: Exploring Hierarchical Class
        Activation Maps for Localization. IEEE TIP.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        reshape_transform: Optional[Callable] = None
    ) -> None:
        """
        Initialize LayerCAM.

        Args:
            model: The neural network model.
            target_layers: List of target layers for CAM computation.
            reshape_transform: Optional transform for activation shapes.
        """
        super().__init__(
            model,
            target_layers,
            reshape_transform,
            uses_gradients=True  # LayerCAM requires gradients
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
        LayerCAM doesn't use pooled weights - returns ones.
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
        Compute LayerCAM using positive gradient spatial weighting.
        
        LayerCAM formula:
            CAM = sum_c(ReLU(grad_c) * activation_c)
        
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
            return activations.mean(axis=1)
        
        # Apply ReLU to gradients (keep only positive gradients)
        positive_grads = np.maximum(grads, 0)
        
        # Spatial weighting: element-wise multiplication
        spatial_weights = positive_grads * activations
        
        if eigen_smooth:
            cam = self._get_2d_projection(spatial_weights)
        else:
            # Sum over channel dimension
            cam = spatial_weights.sum(axis=1)
        
        return cam

