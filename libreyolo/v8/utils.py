"""
Utility functions for Libre YOLO8.
"""

import torch
from typing import Tuple, Dict, List

# Import shared utilities
from ..common.utils import (
    preprocess_image,
    draw_boxes,
    COCO_CLASSES,
    get_class_color,
    nms,
    make_anchors,
    resolve_save_path,
    postprocess_detections,
)


def decode_boxes(box_preds: torch.Tensor, anchors: torch.Tensor, stride_tensor: torch.Tensor) -> torch.Tensor:
    """
    Decode box predictions to xyxy coordinates.
    
    Args:
        box_preds: Box predictions [l, t, r, b] distances from anchors (B, N, 4)
        anchors: Anchor points (N, 2)
        stride_tensor: Stride values (N, 1)
        
    Returns:
        Decoded boxes in xyxy format (B, N, 4)
    """
    # box_preds shape: (B, N, 4) where 4 = [l, t, r, b]
    # anchors shape: (N, 2) where 2 = [x, y]
    # stride_tensor shape: (N, 1)
    
    # Reshape anchors and strides for broadcasting: (1, N, 2) and (1, N, 1)
    anchors = anchors.unsqueeze(0)
    stride_tensor = stride_tensor.unsqueeze(0)
    
    # Decode: xyxy = [x - l, y - t, x + r, y + b] * stride
    x1 = (anchors[..., 0:1] - box_preds[..., 0:1]) * stride_tensor[..., 0:1]
    y1 = (anchors[..., 1:2] - box_preds[..., 1:2]) * stride_tensor[..., 0:1]
    x2 = (anchors[..., 0:1] + box_preds[..., 2:3]) * stride_tensor[..., 0:1]
    y2 = (anchors[..., 1:2] + box_preds[..., 3:4]) * stride_tensor[..., 0:1]
    
    decoded_boxes = torch.cat([x1, y1, x2, y2], dim=-1)
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
    box_layers = [output['x8']['box'], output['x16']['box'], output['x32']['box']]
    cls_layers = [output['x8']['cls'], output['x16']['cls'], output['x32']['cls']]
    strides = [8, 16, 32]
    
    anchors, stride_tensor = make_anchors(box_layers, strides)
    
    box_preds = torch.cat([x.flatten(2).permute(0, 2, 1) for x in box_layers], dim=1)
    cls_preds = torch.cat([x.flatten(2).permute(0, 2, 1) for x in cls_layers], dim=1)
    
    decoded_boxes = decode_boxes(box_preds, anchors, stride_tensor)
    decoded_boxes = decoded_boxes[0]
    
    scores = cls_preds[0].sigmoid()
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
