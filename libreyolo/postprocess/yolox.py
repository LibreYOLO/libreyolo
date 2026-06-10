"""YOLOX postprocessing: grid decode + shared NMS tail.

Moved verbatim from ``libreyolo/models/yolox/utils.py``, which re-exports
everything here for backward compatibility.
"""

import torch
from typing import List, Tuple

from ..utils.general import cxcywh_to_xyxy
from .common import postprocess_detections


def make_grids(
    outputs: List[torch.Tensor], strides: List[int], grid_cell_offset: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate grid anchors for YOLOX output decoding.

    IMPORTANT: YOLOX uses 0-based grid indexing (no offset), matching the training
    code in YOLOXHead.get_output_and_grid. This differs from some other YOLO variants
    that use 0.5 offset for anchor-free detection.

    Args:
        outputs: List of output tensors from each scale
        strides: List of stride values [8, 16, 32]
        grid_cell_offset: Offset for grid cells (default: 0.0 for YOLOX compatibility)

    Returns:
        Tuple of (grids, stride_tensor)
        - grids: (N, 2) tensor of grid coordinates
        - stride_tensor: (N, 1) tensor of stride values
    """
    grids = []
    stride_tensors = []

    for output, stride in zip(outputs, strides):
        _, _, h, w = output.shape
        dtype, device = output.dtype, output.device

        # Create grid WITHOUT offset (matching YOLOX training code)
        xv = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        yv = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        yv, xv = torch.meshgrid(yv, xv, indexing="ij")
        grid = torch.stack((xv, yv), dim=2).view(1, -1, 2)
        grids.append(grid)

        # Create stride tensor
        stride_tensors.append(
            torch.full((1, h * w, 1), stride, dtype=dtype, device=device)
        )

    grids = torch.cat(grids, dim=1)
    stride_tensors = torch.cat(stride_tensors, dim=1)

    return grids, stride_tensors


def decode_outputs(
    outputs: List[torch.Tensor], strides: List[int] = [8, 16, 32]
) -> torch.Tensor:
    """
    Decode YOLOX outputs to absolute coordinates.

    YOLOX output format per anchor: [reg_x, reg_y, reg_w, reg_h, obj, cls0, cls1, ...]
    - reg_x, reg_y: offset from grid cell (decoded: (reg + grid) * stride)
    - reg_w, reg_h: log-scaled width/height (decoded: exp(reg) * stride)
    - obj: objectness score (already sigmoid'd in inference mode)
    - cls: class scores (already sigmoid'd in inference mode)

    Args:
        outputs: List of 3 tensors from head, each (B, 5+num_classes, H, W)
        strides: Stride values for each scale

    Returns:
        Decoded outputs tensor (B, N, 5+num_classes) where:
        - [:, :, 0:2] = center_x, center_y (absolute)
        - [:, :, 2:4] = width, height (absolute)
        - [:, :, 4] = objectness
        - [:, :, 5:] = class probabilities
    """
    # Flatten and concatenate outputs: (B, C, H, W) -> (B, N, C)
    flattened = []
    for output in outputs:
        b, c, h, w = output.shape
        flattened.append(output.view(b, c, -1).permute(0, 2, 1))

    # (B, N_total, 5+num_classes)
    outputs_cat = torch.cat(flattened, dim=1)

    grids, stride_tensor = make_grids(outputs, strides)

    # Center: (offset + grid) * stride
    outputs_cat[..., 0:2] = (outputs_cat[..., 0:2] + grids) * stride_tensor
    # Width/Height: exp(pred) * stride
    outputs_cat[..., 2:4] = torch.exp(outputs_cat[..., 2:4]) * stride_tensor

    return outputs_cat


def postprocess(
    outputs: List[torch.Tensor],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 640,
    original_size: Tuple[int, int] | None = None,
    ratio: float = 1.0,
    max_det: int = 300,
) -> dict:
    """
    Postprocess YOLOX outputs to get final detections.

    Args:
        outputs: List of 3 tensors from head, each (B, 5+num_classes, H, W)
        conf_thres: Confidence threshold (default: 0.25)
        iou_thres: IoU threshold for NMS (default: 0.45)
        input_size: Input image size (default: 640)
        original_size: Original image size (width, height) for scaling
        ratio: Scale ratio from preprocessing
        max_det: Maximum number of detections to return (default: 300)

    Returns:
        Dictionary with boxes, scores, classes, num_detections
    """
    decoded = decode_outputs(outputs)  # (B, N, 5+num_classes)

    decoded = decoded[0]  # (N, 5+num_classes)

    boxes_cxcywh = decoded[:, :4]  # (N, 4) - center_x, center_y, width, height
    objectness = decoded[:, 4]  # (N,) - objectness score
    class_probs = decoded[:, 5:]  # (N, num_classes) - class probabilities

    # Final confidence = objectness * class_prob
    scores = objectness.unsqueeze(-1) * class_probs  # (N, num_classes)

    max_scores, class_ids = torch.max(scores, dim=1)  # (N,)

    mask = max_scores > conf_thres
    if not mask.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    valid_boxes_cxcywh = boxes_cxcywh[mask]
    valid_scores = max_scores[mask]
    valid_classes = class_ids[mask]

    valid_boxes = cxcywh_to_xyxy(valid_boxes_cxcywh)

    if original_size is not None and ratio != 1.0:
        valid_boxes = valid_boxes / ratio

        valid_boxes[:, [0, 2]] = torch.clamp(
            valid_boxes[:, [0, 2]], 0, original_size[0]
        )
        valid_boxes[:, [1, 3]] = torch.clamp(
            valid_boxes[:, [1, 3]], 0, original_size[1]
        )

        # Filter out invalid boxes (zero or negative area)
        box_widths = valid_boxes[:, 2] - valid_boxes[:, 0]
        box_heights = valid_boxes[:, 3] - valid_boxes[:, 1]
        valid_mask = (box_widths > 0) & (box_heights > 0)

        if not valid_mask.all():
            valid_boxes = valid_boxes[valid_mask]
            valid_scores = valid_scores[valid_mask]
            valid_classes = valid_classes[valid_mask]

    return postprocess_detections(
        boxes=valid_boxes,
        scores=valid_scores,
        class_ids=valid_classes,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        input_size=input_size,
        original_size=None,  # already scaled above
        max_det=max_det,
        letterbox=False,
    )
