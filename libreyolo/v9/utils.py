"""
Utility functions for LibreYOLO v9.

Provides preprocessing and postprocessing functions for YOLOv9 inference.
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


def postprocess(
    output: Dict,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 640,
    original_size: Tuple[int, int] = None,
    max_det: int = 300
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
    predictions = output['predictions']

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
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0
        }

    valid_boxes = boxes[mask]
    valid_scores = max_scores[mask]
    valid_classes = class_ids[mask]

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


def make_anchors(
    feats: List[torch.Tensor],
    strides: List[int],
    grid_cell_offset: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate anchor points from feature map sizes.

    Args:
        feats: List of feature tensors from different scales
        strides: List of stride values corresponding to each feature map
        grid_cell_offset: Offset for grid cell centers (default: 0.5)

    Returns:
        Tuple of (anchor_points, stride_tensor)
    """
    anchor_points = []
    stride_tensor = []

    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        dtype, device = feat.dtype, feat.device

        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)


def decode_boxes(
    box_preds: torch.Tensor,
    anchors: torch.Tensor,
    stride_tensor: torch.Tensor
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
