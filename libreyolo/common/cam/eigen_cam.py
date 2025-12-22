"""
EigenCAM implementation for YOLO model interpretability.

EigenCAM computes the first principal component of 2D activations using SVD,
producing class-agnostic saliency maps without requiring backpropagation.

Reference: https://arxiv.org/abs/2008.00299
"""

from typing import List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn

from .base import BaseCAM


class EigenCAM(BaseCAM):
    """
    EigenCAM: Class Activation Map using Principal Components.
    
    This is a gradient-free method that uses SVD to find the first
    principal component of the 2D activations. It produces class-agnostic
    saliency maps that highlight generally important regions.
    
    Reference:
        Muhammad, M. B., & Yeasin, M. (2020). Eigen-CAM: Class Activation Map
        using Principal Components. arXiv:2008.00299
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        reshape_transform: Optional[Callable] = None
    ) -> None:
        """
        Initialize EigenCAM.

        Args:
            model: The neural network model.
            target_layers: List of target layers for CAM computation.
            reshape_transform: Optional transform for activation shapes.
        """
        super().__init__(
            model,
            target_layers,
            reshape_transform,
            uses_gradients=False  # EigenCAM doesn't need gradients
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
        EigenCAM doesn't use weights - it directly computes SVD projection.
        
        This method returns ones since the actual computation happens
        in get_cam_image which is overridden.
        """
        # Return uniform weights - actual computation in get_cam_image
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
        Compute EigenCAM using SVD on activations.
        
        Args:
            input_tensor: The input image tensor.
            target_layer: The layer being processed.
            targets: Ignored for EigenCAM.
            activations: The layer activations of shape (B, C, H, W).
            grads: Ignored for EigenCAM.
            eigen_smooth: Ignored (always uses eigen method).
        
        Returns:
            CAM array of shape (B, H, W).
        """
        return self._get_2d_projection(activations)


# Standalone functions for backward compatibility with existing code

def compute_eigen_cam(activations: np.ndarray) -> np.ndarray:
    """
    Compute EigenCAM heatmap from layer activations using SVD.
    
    This is a standalone function for backward compatibility.
    
    Args:
        activations: Feature map tensor of shape (C, H, W) where C is channels.
    
    Returns:
        Normalized heatmap of shape (H, W) with values in [0, 1].
    """
    # Handle NaN and Inf values
    activations = np.nan_to_num(activations, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Reshape: (C, H, W) -> (H*W, C)
    reshaped = activations.reshape(activations.shape[0], -1).T.astype(np.float64)
    
    # Center the data (important for SVD)
    reshaped = reshaped - reshaped.mean(axis=0)
    
    # Compute SVD and project onto first principal component
    try:
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            _, S, VT = np.linalg.svd(reshaped, full_matrices=False)
            # Check if singular values are valid
            if len(S) == 0 or S[0] < 1e-10:
                return np.zeros(activations.shape[1:], dtype=np.float32)
            projection = reshaped @ VT[0]
            
            # Sign correction heuristic:
            # SVD sign is arbitrary. Ensure the projection is positively correlated 
            # with the mean activation to avoid inverted heatmaps.
            if np.corrcoef(projection, reshaped.mean(axis=1))[0, 1] < 0:
                projection = -projection
    except (np.linalg.LinAlgError, ValueError):
        return np.zeros(activations.shape[1:], dtype=np.float32)
    
    # Reshape back to spatial dimensions
    heatmap = projection.reshape(activations.shape[1:])
    
    # Handle any NaN/Inf from computation
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ReLU: keep only positive activations
    heatmap = np.maximum(heatmap, 0)
    
    # Normalize to [0, 1]
    heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
    if heatmap_max - heatmap_min > 1e-8:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap = np.zeros_like(heatmap)
    
    return heatmap.astype(np.float32)

