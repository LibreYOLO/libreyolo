"""
Utility functions for YOLO9.

Provides preprocessing and postprocessing functions for YOLOv9 inference.
"""

import numpy as np
import torch
from typing import Tuple, Dict
from PIL import Image

# Import shared utilities
from ...utils.general import (
    postprocess_detections,
)
from ...utils.image_loader import ImageLoader, ImageInput


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 640,
) -> Tuple[np.ndarray, float]:
    """
    Preprocess RGB HWC uint8 image for YOLOv9 inference.

    Simple resize + normalize to 0-1 range.

    Args:
        img_rgb_hwc: Input image as RGB HWC uint8 numpy array.
        input_size: Target size for the model.

    Returns:
        Tuple of (preprocessed CHW float32 array in RGB 0-1, ratio).
    """
    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), 1.0


def preprocess_image(
    image: ImageInput, input_size: int = 640, color_format: str = "auto"
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
    """
    Preprocess image for YOLOv9 inference.

    Args:
        image: Input image (path, PIL, numpy, tensor, bytes, etc.)
        input_size: Target size for resizing (default: 640)
        color_format: Color format hint ("auto", "rgb", "bgr")

    Returns:
        Tuple of (preprocessed_tensor, original_image, original_size)
    """
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size  # (width, height)
    original_img = img.copy()

    img_chw, _ = preprocess_numpy(np.array(img), input_size)
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
    return img_tensor, original_img, original_size


def postprocess(
    output: Dict,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 640,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    letterbox: bool = False,
) -> Dict:
    """
    Postprocess YOLOv9 model outputs to get final detections.

    Args:
        output: Model output dictionary with 'predictions' key
        conf_thres: Confidence threshold (default: 0.25)
        iou_thres: IoU threshold for NMS (default: 0.45)
        input_size: Input image size (default: 640)
        original_size: Original image size (width, height) for scaling
        max_det: Maximum number of detections to return (default: 300)

    Returns:
        Dictionary with boxes, scores, classes, num_detections
    """
    # Get predictions from model output
    # Shape: (batch, 4+nc, total_anchors)
    predictions = output["predictions"]

    # Take first batch
    if predictions.dim() == 3:
        pred = predictions[0]  # (4+nc, total_anchors)
    else:
        pred = predictions

    # Transpose to (total_anchors, 4+nc)
    pred = pred.transpose(0, 1)

    # Split boxes and class scores
    boxes = pred[:, :4]  # xyxy format
    scores = pred[:, 4:]  # class scores (already sigmoid applied in model)

    # Get max class score and class id
    max_scores, class_ids = torch.max(scores, dim=1)

    # Apply confidence threshold
    mask = max_scores > conf_thres
    if not mask.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    # Use shared postprocess pipeline
    return postprocess_detections(
        boxes=boxes[mask],
        scores=max_scores[mask],
        class_ids=class_ids[mask],
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        input_size=input_size,
        original_size=original_size,
        max_det=max_det,
        letterbox=letterbox,
    )


def decode_boxes(
    box_preds: torch.Tensor, anchors: torch.Tensor, stride_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Decode box predictions to xyxy coordinates.

    Args:
        box_preds: Box predictions [l, t, r, b] distances from anchors (B, N, 4)
        anchors: Anchor points (N, 2)
        stride_tensor: Stride values (N, 1)

    Returns:
        Decoded boxes in xyxy format (B, N, 4)
    """
    anchors = anchors.unsqueeze(0)
    stride_tensor = stride_tensor.unsqueeze(0)

    # Decode: xyxy = [x - l, y - t, x + r, y + b] * stride
    x1 = (anchors[..., 0:1] - box_preds[..., 0:1]) * stride_tensor[..., 0:1]
    y1 = (anchors[..., 1:2] - box_preds[..., 1:2]) * stride_tensor[..., 0:1]
    x2 = (anchors[..., 0:1] + box_preds[..., 2:3]) * stride_tensor[..., 0:1]
    y2 = (anchors[..., 1:2] + box_preds[..., 3:4]) * stride_tensor[..., 0:1]

    decoded_boxes = torch.cat([x1, y1, x2, y2], dim=-1)
    return decoded_boxes
