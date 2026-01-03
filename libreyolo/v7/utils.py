"""
Utility functions for LibreYOLO v7.

Provides preprocessing and postprocessing functions for YOLOv7 inference.
YOLOv7 uses anchor-based detection, which requires different decoding from anchor-free models.
"""

import torch
from typing import Tuple, Dict, List

# Import shared utilities
from ..common.utils import (
    preprocess_image, draw_boxes, COCO_CLASSES, get_class_color,
    nms, postprocess_detections
)


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from xywh format to xyxy format.

    Args:
        x: Boxes in xywh format (N, 4)

    Returns:
        Boxes in xyxy format (N, 4)
    """
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def postprocess(
    output: Dict,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 640,
    original_size: Tuple[int, int] = None,
    max_det: int = 300
) -> Dict:
    """
    Postprocess YOLOv7 model outputs to get final detections.

    YOLOv7 outputs are already decoded in the model's forward pass.
    Output format: (batch, total_anchors, 5 + num_classes)
    where 5 = [x, y, w, h, objectness]

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
    # Shape: (batch, total_anchors, 5+nc)
    predictions = output['predictions']

    # Take first batch
    if predictions.dim() == 3:
        pred = predictions[0]  # (total_anchors, 5+nc)
    else:
        pred = predictions

    # Split boxes, objectness, and class scores
    # Format: [x, y, w, h, obj, cls1, cls2, ...]
    boxes_xywh = pred[:, :4]  # xywh format
    objectness = pred[:, 4]  # objectness score
    class_scores = pred[:, 5:]  # class probabilities (already sigmoid applied)

    # Compute final scores: objectness * class_prob
    scores = objectness.unsqueeze(1) * class_scores
    max_scores, class_ids = torch.max(scores, dim=1)

    # Apply confidence threshold
    mask = max_scores > conf_thres
    if not mask.any():
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0
        }

    # Convert xywh to xyxy for the valid boxes
    valid_boxes = xywh2xyxy(boxes_xywh[mask])

    # Use shared postprocess pipeline for scaling, clipping, NMS, and max_det
    return postprocess_detections(
        boxes=valid_boxes,
        scores=max_scores[mask],
        class_ids=class_ids[mask],
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        input_size=input_size,
        original_size=original_size,
        max_det=max_det
    )


# Default anchors for YOLOv7
V7_ANCHORS = {
    8: [[12, 16], [19, 36], [40, 28]],      # P3/8
    16: [[36, 75], [76, 55], [72, 146]],    # P4/16
    32: [[142, 110], [192, 243], [459, 401]],  # P5/32
}


def decode_boxes_v7(
    predictions: torch.Tensor,
    anchors: Dict[int, List[List[int]]],
    strides: List[int],
    grid_size: Tuple[int, int] = (80, 80)
) -> torch.Tensor:
    """
    Decode anchor-based predictions from YOLOv7.

    This is typically done in the model's forward pass, but provided here
    for reference and potential post-processing needs.

    For each anchor:
    - tx, ty -> center offset (sigmoid * 2 - 0.5 + grid)
    - tw, th -> size scaling (sigmoid * 2)^2 * anchor

    Args:
        predictions: Raw predictions (batch, anchors, h, w, 5+nc)
        anchors: Anchor boxes per stride
        strides: Stride values for each scale
        grid_size: Grid size (height, width)

    Returns:
        Decoded boxes in xyxy format
    """
    device = predictions.device
    dtype = predictions.dtype

    # Extract components
    tx = predictions[..., 0]  # x offset
    ty = predictions[..., 1]  # y offset
    tw = predictions[..., 2]  # width scale
    th = predictions[..., 3]  # height scale

    # Apply sigmoid and decode
    xy = torch.sigmoid(predictions[..., :2]) * 2 - 0.5
    wh = (torch.sigmoid(predictions[..., 2:4]) * 2) ** 2

    return predictions  # Placeholder - actual decoding done in model


def make_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a grid of (x, y) coordinates.

    Args:
        h: Grid height
        w: Grid width
        device: Torch device
        dtype: Torch dtype

    Returns:
        Grid tensor of shape (1, 1, h, w, 2)
    """
    yv, xv = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing='ij'
    )
    grid = torch.stack((xv, yv), 2).view(1, 1, h, w, 2)
    return grid
