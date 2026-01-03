"""
Utility functions for YOLO11 preprocessing, postprocessing, and visualization.
"""

import torch
from typing import Tuple, Dict, List

# Import shared utilities
from ..common.utils import (
    preprocess_image, draw_boxes, COCO_CLASSES, get_class_color,
    nms, make_anchors, postprocess_detections
)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = False, dim: int = -1) -> torch.Tensor:
    """
    Transform distance(ltrb) to box(xywh or xyxy).
    Matches YOLO11's dist2bbox implementation.
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def decode_boxes(box_preds: torch.Tensor, anchors: torch.Tensor, stride_tensor: torch.Tensor) -> torch.Tensor:
    """
    Decode box predictions to xyxy coordinates using YOLO11's method.
    
    Args:
        box_preds: Box predictions [l, t, r, b] distances from anchors (B, N, 4) in grid space
        anchors: Anchor points (N, 2) in grid coordinates
        stride_tensor: Stride values (N, 1)
        
    Returns:
        Decoded boxes in xyxy format (B, N, 4) in pixel coordinates
    """
    # box_preds shape: (B, N, 4) where 4 = [l, t, r, b] (already DFL-decoded, in grid space)
    # anchors shape: (N, 2) where 2 = [x, y] in grid coordinates
    # stride_tensor shape: (N, 1)
    
    # Reshape anchors and strides for broadcasting: (1, N, 2) and (1, N, 1)
    anchors = anchors.unsqueeze(0)
    stride_tensor = stride_tensor.unsqueeze(0)
    
    # Decode in grid space first
    # dist2bbox expects (B, N, 4) and (B, N, 2) or similar
    decoded_boxes_grid = dist2bbox(box_preds, anchors, xywh=True, dim=-1)  # (B, N, 4) in xywh format, grid space
    
    # Convert to pixel coordinates: multiply center and size by stride
    decoded_boxes_px = decoded_boxes_grid * stride_tensor  # (B, N, 4) in xywh format, pixel space
    
    # Convert xywh to xyxy
    cx, cy, w, h = decoded_boxes_px[..., 0], decoded_boxes_px[..., 1], decoded_boxes_px[..., 2], decoded_boxes_px[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    decoded_boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, N, 4) in xyxy format
    
    return decoded_boxes


def postprocess(output: Dict, conf_thres: float = 0.25, iou_thres: float = 0.45, input_size: int = 640, original_size: Tuple[int, int] = None, max_det: int = 300) -> Dict:
    """
    Postprocess model outputs to get final detections.
    
    Args:
        output: Model output dictionary with 'x8', 'x16', 'x32' keys
        conf_thres: Confidence threshold (default: 0.25)
        iou_thres: IoU threshold for NMS (default: 0.45)
        input_size: Input image size (default: 640)
        original_size: Original image size (width, height) for scaling
        max_det: Maximum number of detections to return (default: 300)
        
    Returns:
        Dictionary with boxes, scores, classes, num_detections
    """
    # Collect outputs from the 3 heads
    box_layers = [output['x8']['box'], output['x16']['box'], output['x32']['box']]
    cls_layers = [output['x8']['cls'], output['x16']['cls'], output['x32']['cls']]
    strides = [8, 16, 32]
    
    # Generate anchors
    anchors, stride_tensor = make_anchors(box_layers, strides)
    
    # Flatten and concatenate predictions
    # Box: (Batch, 4, H, W) -> (Batch, H*W, 4)
    box_preds = torch.cat([x.flatten(2).permute(0, 2, 1) for x in box_layers], dim=1)
    # Cls: (Batch, 80, H, W) -> (Batch, H*W, 80)
    cls_preds = torch.cat([x.flatten(2).permute(0, 2, 1) for x in cls_layers], dim=1)
    
    # Decode boxes
    decoded_boxes = decode_boxes(box_preds, anchors, stride_tensor)  # (1, N, 4)
    decoded_boxes = decoded_boxes[0]  # Remove batch dimension: (N, 4)
    
    # Apply confidence threshold
    scores = cls_preds[0].sigmoid()  # (N, 80)
    max_scores, class_ids = torch.max(scores, dim=1)
    
    mask = max_scores > conf_thres
    if not mask.any():
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0
        }
    
    # Use shared postprocess pipeline for scaling, clipping, NMS, and max_det
    return postprocess_detections(
        boxes=decoded_boxes[mask],
        scores=max_scores[mask],
        class_ids=class_ids[mask],
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        input_size=input_size,
        original_size=original_size,
        max_det=max_det
    )
