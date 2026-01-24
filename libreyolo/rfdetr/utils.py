"""
Utility functions for LibreYOLO RF-DETR.

Postprocessing matches the original rfdetr implementation exactly.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from center-x, center-y, width, height to x1, y1, x2, y2.

    Args:
        boxes: Tensor of shape (..., 4) with cxcywh format

    Returns:
        Tensor of shape (..., 4) with xyxy format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def postprocess(
    outputs: Dict[str, torch.Tensor],
    target_sizes: torch.Tensor,
    num_select: int = 300
) -> List[Dict[str, torch.Tensor]]:
    """
    Postprocess RF-DETR outputs to get final detections.

    This matches the original rfdetr PostProcess class exactly:
    1. Apply sigmoid to logits
    2. Select top-K scores across all (queries × classes)
    3. Convert boxes from cxcywh to xyxy
    4. Scale boxes to original image coordinates

    No NMS is applied - just top-K selection (same as original).

    Args:
        outputs: Model output dictionary with 'pred_logits' and 'pred_boxes'
        target_sizes: Tensor of shape (batch_size, 2) with (height, width) for each image
        num_select: Number of top detections to select (default: 300)

    Returns:
        List of dictionaries, one per image, each containing:
            - scores: Tensor of shape (num_select,) with confidence scores
            - labels: Tensor of shape (num_select,) with class IDs
            - boxes: Tensor of shape (num_select, 4) in xyxy format
    """
    out_logits = outputs['pred_logits']  # (B, num_queries, num_classes)
    out_bbox = outputs['pred_boxes']      # (B, num_queries, 4) in cxcywh [0, 1]

    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    # Apply sigmoid to get probabilities
    prob = out_logits.sigmoid()

    # Get top-K across all (queries × classes)
    # Flatten to (B, num_queries * num_classes) then take topk
    batch_size = out_logits.shape[0]
    num_classes = out_logits.shape[2]

    topk_values, topk_indexes = torch.topk(
        prob.view(batch_size, -1),
        num_select,
        dim=1
    )

    scores = topk_values

    # Convert flat indices to query indices and class indices
    topk_boxes = topk_indexes // num_classes  # Which query
    labels = topk_indexes % num_classes        # Which class

    # Convert boxes from cxcywh to xyxy
    boxes = box_cxcywh_to_xyxy(out_bbox)

    # Gather boxes for the selected queries
    # boxes shape: (B, num_queries, 4)
    # topk_boxes shape: (B, num_select)
    boxes = torch.gather(
        boxes,
        1,
        topk_boxes.unsqueeze(-1).repeat(1, 1, 4)
    )

    # Scale from relative [0, 1] to absolute [0, height/width] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    # Build results list
    results = [
        {'scores': s, 'labels': l, 'boxes': b}
        for s, l, b in zip(scores, labels, boxes)
    ]

    return results
