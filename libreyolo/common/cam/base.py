"""
Base CAM class for all Class Activation Mapping methods.

This module provides an abstract base class that defines the interface
and common functionality for all CAM implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
import cv2

from .activations import ActivationsAndGradients


def scale_cam_image(cam: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Scale and normalize a CAM image.
    
    Args:
        cam: The CAM array of shape (H, W) or (B, H, W).
        target_size: Optional (width, height) to resize to.
    
    Returns:
        Normalized CAM in range [0, 1].
    """
    result = []
    for img in cam:
        img = img - np.min(img)
        max_val = np.max(img)
        if max_val > 1e-8:
            img = img / max_val
        else:
            img = np.zeros_like(img)
        
        if target_size is not None:
            img = cv2.resize(img, target_size)
        
        result.append(img)
    
    return np.array(result, dtype=np.float32)


class BaseCAM(ABC):
    """
    Abstract base class for all CAM methods.
    
    Provides common infrastructure for activation/gradient capture,
    multi-layer aggregation, and heatmap generation.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        reshape_transform: Optional[Callable] = None,
        uses_gradients: bool = True
    ) -> None:
        """
        Initialize the BaseCAM.

        Args:
            model: The neural network model.
            target_layers: List of target layers for CAM computation.
            reshape_transform: Optional transform for activation shapes.
            uses_gradients: Whether this CAM method requires gradients.
        """
        self.model = model
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.uses_gradients = uses_gradients
        
        # Set up activation and gradient capture
        self.activations_and_grads = ActivationsAndGradients(
            model, target_layers, reshape_transform
        )
        
        # Store for computing target size
        self._input_tensor: Optional[torch.Tensor] = None

    @abstractmethod
    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        targets: Optional[List],
        activations: np.ndarray,
        grads: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Get the weights for each channel in the activation map.
        
        This is the core method that differentiates CAM variants.
        
        Args:
            input_tensor: The input image tensor.
            target_layer: The layer being processed.
            targets: Optional target classes/objects.
            activations: The layer activations of shape (B, C, H, W).
            grads: The gradients of shape (B, C, H, W), or None.
        
        Returns:
            Weights array of shape (B, C).
        """
        raise NotImplementedError

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
        Compute the CAM image from activations and optional gradients.
        
        Args:
            input_tensor: The input image tensor.
            target_layer: The layer being processed.
            targets: Optional target classes/objects.
            activations: The layer activations.
            grads: The gradients, or None for gradient-free methods.
            eigen_smooth: Whether to apply SVD smoothing.
        
        Returns:
            CAM array of shape (B, H, W).
        """
        weights = self.get_cam_weights(
            input_tensor, target_layer, targets, activations, grads
        )
        
        # Weighted sum of activation channels
        # weights: (B, C), activations: (B, C, H, W)
        weighted_activations = weights[:, :, None, None] * activations
        
        if eigen_smooth:
            cam = self._get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        
        return cam

    def _get_2d_projection(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute the first principal component of activations using SVD.
        
        Args:
            activations: Array of shape (B, C, H, W).
        
        Returns:
            2D projection of shape (B, H, W).
        """
        batch_size = activations.shape[0]
        result = []
        
        for i in range(batch_size):
            act = activations[i]  # (C, H, W)
            # Reshape to (H*W, C)
            reshaped = act.reshape(act.shape[0], -1).T.astype(np.float64)
            # Center the data
            reshaped = reshaped - reshaped.mean(axis=0)
            
            try:
                _, S, VT = np.linalg.svd(reshaped, full_matrices=False)
                if len(S) == 0 or S[0] < 1e-10:
                    projection = np.zeros(act.shape[1:])
                else:
                    projection = reshaped @ VT[0]
                    # Sign correction
                    if np.corrcoef(projection, reshaped.mean(axis=1))[0, 1] < 0:
                        projection = -projection
                    projection = projection.reshape(act.shape[1:])
            except (np.linalg.LinAlgError, ValueError):
                projection = np.zeros(act.shape[1:])
            
            result.append(projection)
        
        return np.array(result, dtype=np.float32)

    def forward(
        self,
        input_tensor: torch.Tensor,
        targets: Optional[List] = None,
        eigen_smooth: bool = False
    ) -> np.ndarray:
        """
        Compute CAM for the given input.
        
        Args:
            input_tensor: Input image tensor of shape (B, C, H, W).
            targets: Optional target specifications.
            eigen_smooth: Whether to apply SVD smoothing.
        
        Returns:
            CAM array of shape (B, H, W).
        """
        # Enable gradients if needed
        if self.uses_gradients:
            for p in self.model.parameters():
                p.requires_grad_(True)
        
        # Forward pass
        outputs = self.activations_and_grads(input_tensor)
        
        # Compute loss and backward if using gradients
        if self.uses_gradients:
            self.model.zero_grad()
            loss = self._compute_loss(outputs, targets)
            if loss is not None:
                loss.backward(retain_graph=True)
        
        # Compute CAM for each target layer
        cam_per_layer = self._compute_cam_per_layer(
            input_tensor, targets, eigen_smooth
        )
        
        # Aggregate across layers
        return self._aggregate_multi_layers(cam_per_layer)

    def _compute_loss(self, outputs: torch.Tensor, targets: Optional[List]) -> Optional[torch.Tensor]:
        """
        Compute the loss for backpropagation.
        
        For object detection, we typically sum the class confidence scores
        of detected objects.
        
        Args:
            outputs: Model outputs (detection results).
            targets: Optional target specifications.
        
        Returns:
            Loss tensor for backward pass, or None.
        """
        # Default: sum all class scores for YOLO detection output
        # This can be overridden by subclasses for specific targets
        if isinstance(outputs, dict):
            # YOLO output format: {'x8': {'cls': ..., 'box': ...}, ...}
            loss_parts = []
            for scale in ['x8', 'x16', 'x32']:
                if scale in outputs and 'cls' in outputs[scale]:
                    cls_scores = outputs[scale]['cls']
                    loss_parts.append(cls_scores.sum())
            if loss_parts:
                return sum(loss_parts)
        elif isinstance(outputs, torch.Tensor):
            # Direct tensor output
            return outputs.sum()
        return None

    def _compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: Optional[List],
        eigen_smooth: bool
    ) -> List[np.ndarray]:
        """
        Compute CAM for each target layer.
        
        Args:
            input_tensor: The input tensor.
            targets: Optional target specifications.
            eigen_smooth: Whether to apply SVD smoothing.
        
        Returns:
            List of CAM arrays, one per layer.
        """
        activations_list = [
            a.cpu().numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().numpy() for g in self.activations_and_grads.gradients
        ] if self.uses_gradients else [None] * len(activations_list)
        
        cam_per_layer = []
        
        for i, target_layer in enumerate(self.target_layers):
            layer_activations = activations_list[i] if i < len(activations_list) else None
            layer_grads = grads_list[i] if i < len(grads_list) else None
            
            if layer_activations is None:
                continue
            
            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth
            )
            
            # ReLU
            cam = np.maximum(cam, 0)
            cam_per_layer.append(cam)
        
        return cam_per_layer

    def _aggregate_multi_layers(self, cam_per_layer: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate CAMs from multiple layers.
        
        Args:
            cam_per_layer: List of CAM arrays from each layer.
        
        Returns:
            Aggregated CAM array.
        """
        if not cam_per_layer:
            # Return empty array with expected shape
            if self._input_tensor is not None:
                h, w = self._input_tensor.shape[2:]
                return np.zeros((1, h, w), dtype=np.float32)
            return np.zeros((1, 640, 640), dtype=np.float32)
        
        # Get target size from input
        if self._input_tensor is not None:
            target_size = (self._input_tensor.shape[3], self._input_tensor.shape[2])
        else:
            target_size = None
        
        # Scale each layer's CAM to target size
        scaled_cams = []
        for cam in cam_per_layer:
            scaled = scale_cam_image(cam, target_size)
            scaled_cams.append(scaled[:, None, :, :])
        
        # Concatenate and average
        cam_concat = np.concatenate(scaled_cams, axis=1)
        result = np.mean(cam_concat, axis=1)
        
        # Final scaling
        return scale_cam_image(result, target_size)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: Optional[List] = None,
        eigen_smooth: bool = False
    ) -> np.ndarray:
        """
        Compute CAM for the given input.
        
        Args:
            input_tensor: Input image tensor of shape (B, C, H, W).
            targets: Optional target specifications.
            eigen_smooth: Whether to apply SVD smoothing.
        
        Returns:
            CAM array of shape (B, H, W) with values in [0, 1].
        """
        self._input_tensor = input_tensor
        return self.forward(input_tensor, targets, eigen_smooth)

    def release(self) -> None:
        """Release resources and remove hooks."""
        self.activations_and_grads.release()

    def __del__(self):
        """Cleanup on deletion."""
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.release()
        return False

