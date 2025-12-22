"""
EigenCAM implementation for YOLO model interpretability.

EigenCAM computes the first principal component of 2D activations using SVD,
producing class-agnostic saliency maps without requiring backpropagation.

Reference: https://arxiv.org/abs/2008.00299
"""

import numpy as np
import cv2


def compute_eigen_cam(activations: np.ndarray) -> np.ndarray:
    """
    Compute EigenCAM heatmap from layer activations using SVD.
    
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
    except np.linalg.LinAlgError:
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


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a heatmap on an image using a colormap.
    
    Args:
        image: Original image as RGB numpy array of shape (H, W, 3).
        heatmap: Grayscale heatmap of shape (H', W') with values in [0, 1].
        alpha: Blending factor for the heatmap (default: 0.5).
    
    Returns:
        Blended RGB image as numpy array of shape (H, W, 3).
    """
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap (JET: blue=cold, red=hot)
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    blended = (image.astype(np.float32) * (1 - alpha) + heatmap_color.astype(np.float32) * alpha)
    return blended.astype(np.uint8)

