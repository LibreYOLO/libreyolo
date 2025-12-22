"""
EigenCAM implementation for YOLO model interpretability.

This module is kept for backward compatibility. The main implementation
is now in libreyolo.common.cam.eigen_cam.

EigenCAM computes the first principal component of 2D activations using SVD,
producing class-agnostic saliency maps without requiring backpropagation.

Reference: https://arxiv.org/abs/2008.00299
"""

# Re-export from new location for backward compatibility
from .cam.eigen_cam import compute_eigen_cam

import numpy as np
import cv2


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
