"""
Utility functions for LibreYOLO v7.

Provides preprocessing and postprocessing functions for YOLOv7 inference.
YOLOv7 uses anchor-based detection, which requires different decoding from anchor-free models.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Union
from PIL import Image

# Import shared utilities
from ..common.utils import preprocess_image, draw_boxes, COCO_CLASSES, get_class_color


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> torch.Tensor:
    """
    Non-Maximum Suppression using torch operations.

    Args:
        boxes: Boxes in xyxy format (N, 4)
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    # Filter out boxes with NaN or Inf values
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
    if not valid_mask.any():
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    if not valid_mask.all():
        valid_indices = torch.where(valid_mask)[0]
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    else:
        valid_indices = None

    # Sort by scores (descending)
    _, order = scores.sort(0, descending=True)
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i.item())

        if len(order) == 1:
            break

        # Calculate IoU with remaining boxes
        box_i = boxes[i]
        boxes_remaining = boxes[order[1:]]

        # Calculate intersection
        x1_i, y1_i, x2_i, y2_i = box_i
        x1_r, y1_r, x2_r, y2_r = boxes_remaining[:, 0], boxes_remaining[:, 1], boxes_remaining[:, 2], boxes_remaining[:, 3]

        x1_inter = torch.max(x1_i, x1_r)
        y1_inter = torch.max(y1_i, y1_r)
        x2_inter = torch.min(x2_i, x2_r)
        y2_inter = torch.min(y2_i, y2_r)

        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)

        # Calculate union
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        area_r = (x2_r - x1_r) * (y2_r - y1_r)
        union_area = area_i + area_r - inter_area

        # Calculate IoU
        iou = inter_area / (union_area + 1e-7)

        # Keep boxes with IoU < threshold
        order = order[1:][iou < iou_threshold]

    keep_tensor = torch.tensor(keep, dtype=torch.long, device=boxes.device)

    # Map back to original indices if we filtered out invalid boxes
    if valid_indices is not None:
        keep_tensor = valid_indices[keep_tensor]

    return keep_tensor


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

    valid_boxes_xywh = boxes_xywh[mask]
    valid_scores = max_scores[mask]
    valid_classes = class_ids[mask]

    # Convert xywh to xyxy
    valid_boxes = xywh2xyxy(valid_boxes_xywh)

    # Scale boxes to original image size
    if original_size is not None:
        scale_x = original_size[0] / input_size
        scale_y = original_size[1] / input_size
        valid_boxes[:, [0, 2]] *= scale_x
        valid_boxes[:, [1, 3]] *= scale_y

        # Clip to image bounds
        valid_boxes[:, [0, 2]] = torch.clamp(valid_boxes[:, [0, 2]], 0, original_size[0])
        valid_boxes[:, [1, 3]] = torch.clamp(valid_boxes[:, [1, 3]], 0, original_size[1])

        # Filter out invalid boxes
        box_widths = valid_boxes[:, 2] - valid_boxes[:, 0]
        box_heights = valid_boxes[:, 3] - valid_boxes[:, 1]
        valid_mask = (box_widths > 0) & (box_heights > 0)

        if not valid_mask.all():
            valid_boxes = valid_boxes[valid_mask]
            valid_scores = valid_scores[valid_mask]
            valid_classes = valid_classes[valid_mask]

    if len(valid_boxes) == 0:
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0
        }

    # Apply NMS per class
    try:
        import torchvision.ops
        use_torchvision_nms = True
    except ImportError:
        use_torchvision_nms = False

    unique_classes = torch.unique(valid_classes)
    keep_indices_list = []

    for cls in unique_classes:
        cls_mask = valid_classes == cls
        cls_boxes = valid_boxes[cls_mask]
        cls_scores = valid_scores[cls_mask]

        if len(cls_boxes) == 0:
            continue

        if use_torchvision_nms:
            max_wh = 7680.0
            boxes_for_nms = cls_boxes + cls.float() * max_wh
            cls_keep = torchvision.ops.nms(boxes_for_nms, cls_scores, iou_thres)
        else:
            cls_keep = nms(cls_boxes, cls_scores, iou_thres)

        cls_indices = torch.where(cls_mask)[0]
        keep_indices_list.append(cls_indices[cls_keep])

    if len(keep_indices_list) == 0:
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0
        }

    keep_indices = torch.cat(keep_indices_list)

    # Limit to max detections
    if len(keep_indices) > max_det:
        final_scores_temp = valid_scores[keep_indices]
        _, top_indices = torch.topk(final_scores_temp, max_det)
        keep_indices = keep_indices[top_indices]

    final_boxes = valid_boxes[keep_indices].cpu().numpy()
    final_scores = valid_scores[keep_indices].cpu().numpy()
    final_classes = valid_classes[keep_indices].cpu().numpy()

    return {
        "boxes": final_boxes.tolist(),
        "scores": final_scores.tolist(),
        "classes": final_classes.tolist(),
        "num_detections": len(final_boxes)
    }


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
