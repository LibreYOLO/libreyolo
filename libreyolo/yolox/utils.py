"""
Utility functions for YOLOX.

YOLOX uses different preprocessing and postprocessing than YOLOv8/v11:
- Preprocessing: Letterbox with gray padding (114,114,114), NO normalization (0-255 range)
- Postprocessing: Box decoding with exp() for width/height, objectness score
"""

import torch
import numpy as np
from typing import Tuple, List
from PIL import Image

from ..common.image_loader import ImageLoader, ImageInput
from ..common.utils import draw_boxes, COCO_CLASSES, get_class_color, nms


def preprocess_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto"
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """
    Preprocess image for YOLOX inference with letterboxing.

    YOLOX-specific preprocessing:
    - Letterbox resize maintaining aspect ratio
    - Gray padding (114, 114, 114)
    - NO normalization (keeps 0-255 range as float32)

    Args:
        image: Input image (path, PIL, numpy, tensor, bytes, etc.)
        input_size: Target size for the model (default: 640)
        color_format: Color format hint ("auto", "rgb", "bgr")

    Returns:
        Tuple of (preprocessed_tensor, original_image, original_size, ratio)
        - preprocessed_tensor: (1, 3, H, W) float32 tensor in 0-255 range
        - original_image: PIL Image copy of original
        - original_size: (width, height) of original image
        - ratio: scale ratio applied during preprocessing
    """
    # Use unified ImageLoader to handle all input types
    img = ImageLoader.load(image, color_format=color_format)

    original_size = img.size  # (width, height)
    original_img = img.copy()

    # Calculate scale ratio to fit in input_size while maintaining aspect ratio
    orig_w, orig_h = original_size
    ratio = min(input_size / orig_h, input_size / orig_w)

    # Calculate new dimensions
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    # Resize image
    img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Create padded image with gray background (114, 114, 114)
    padded_img = Image.new("RGB", (input_size, input_size), (114, 114, 114))

    # Paste resized image at top-left corner
    padded_img.paste(img_resized, (0, 0))

    # Convert to numpy array (keep 0-255 range, NO normalization)
    img_array = np.array(padded_img, dtype=np.float32)

    # Convert to tensor: HWC -> CHW -> add batch dimension
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    return img_tensor, original_img, original_size, ratio


def make_grids(outputs: List[torch.Tensor], strides: List[int], grid_cell_offset: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
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
        yv, xv = torch.meshgrid(yv, xv, indexing='ij')
        grid = torch.stack((xv, yv), dim=2).view(1, -1, 2)
        grids.append(grid)

        # Create stride tensor
        stride_tensors.append(torch.full((1, h * w, 1), stride, dtype=dtype, device=device))

    grids = torch.cat(grids, dim=1)
    stride_tensors = torch.cat(stride_tensors, dim=1)

    return grids, stride_tensors


def decode_outputs(
    outputs: List[torch.Tensor],
    strides: List[int] = [8, 16, 32]
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

    # Generate grids
    grids, stride_tensor = make_grids(outputs, strides)

    # Decode boxes
    # Center: (offset + grid) * stride
    outputs_cat[..., 0:2] = (outputs_cat[..., 0:2] + grids) * stride_tensor
    # Width/Height: exp(pred) * stride
    outputs_cat[..., 2:4] = torch.exp(outputs_cat[..., 2:4]) * stride_tensor

    return outputs_cat


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).

    Args:
        boxes: Boxes in cxcywh format (..., 4)

    Returns:
        Boxes in xyxy format (..., 4)
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def postprocess(
    outputs: List[torch.Tensor],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 640,
    original_size: Tuple[int, int] = None,
    ratio: float = 1.0,
    max_det: int = 300
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
    # Decode outputs to absolute coordinates
    decoded = decode_outputs(outputs)  # (B, N, 5+num_classes)

    # Take first batch
    decoded = decoded[0]  # (N, 5+num_classes)

    # Extract components
    boxes_cxcywh = decoded[:, :4]  # (N, 4) - center_x, center_y, width, height
    objectness = decoded[:, 4]     # (N,) - objectness score
    class_probs = decoded[:, 5:]   # (N, num_classes) - class probabilities

    # Final confidence = objectness * class_prob
    scores = objectness.unsqueeze(-1) * class_probs  # (N, num_classes)

    # Get max class score and class id for each prediction
    max_scores, class_ids = torch.max(scores, dim=1)  # (N,)

    # Filter by confidence threshold
    mask = max_scores > conf_thres
    if not mask.any():
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0
        }

    valid_boxes_cxcywh = boxes_cxcywh[mask]
    valid_scores = max_scores[mask]
    valid_classes = class_ids[mask]

    # Convert to xyxy format
    valid_boxes = cxcywh_to_xyxy(valid_boxes_cxcywh)

    # Scale boxes back to original image coordinates
    if original_size is not None and ratio != 1.0:
        # Divide by ratio to get back to original scale
        valid_boxes = valid_boxes / ratio

        # Clamp to image boundaries
        valid_boxes[:, [0, 2]] = torch.clamp(valid_boxes[:, [0, 2]], 0, original_size[0])
        valid_boxes[:, [1, 3]] = torch.clamp(valid_boxes[:, [1, 3]], 0, original_size[1])

        # Filter out invalid boxes (zero or negative area)
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

    # Limit to max_det
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
