"""
Utility functions for YOLO9.

Provides preprocessing and postprocessing functions for YOLOv9 inference.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms
from typing import Tuple, Dict
from PIL import Image

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


def _nms_keep_indices(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_thres: float,
    max_det: int,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Drop non-finite rows — batched_nms is undefined on NaN/Inf inputs.
    finite_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
    if not finite_mask.all():
        valid_indices = torch.where(finite_mask)[0]
        if len(valid_indices) == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        boxes = boxes[finite_mask]
        scores = scores[finite_mask]
        class_ids = class_ids[finite_mask]
    else:
        valid_indices = None

    # Shift to non-negative coords — batched_nms's class-offset trick uses
    # (boxes.max() + 1) and only separates classes when all coords are
    # non-negative. Translation-invariant for IoU.
    nms_boxes = boxes - boxes.min().clamp(max=0)
    keep = batched_nms(nms_boxes, scores, class_ids, iou_thres)
    if len(keep) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    if len(keep) > max_det:
        _, order = torch.topk(scores[keep], max_det)
        keep = keep[order]

    # Map back to original indices when we filtered non-finite rows above.
    if valid_indices is not None:
        keep = valid_indices[keep]
    return keep


def _crop_masks(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    n, h, w = masks.shape
    if n == 0:
        return masks
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    rows = torch.arange(h, device=masks.device, dtype=masks.dtype)[None, :, None]
    cols = torch.arange(w, device=masks.device, dtype=masks.dtype)[None, None, :]
    keep = (
        (cols >= x1[:, None, None])
        & (cols < x2[:, None, None])
        & (rows >= y1[:, None, None])
        & (rows < y2[:, None, None])
    )
    return masks * keep


def _process_masks(
    proto: torch.Tensor,
    coeffs: torch.Tensor,
    boxes_input: torch.Tensor,
    input_shape: Tuple[int, int],
    original_size: Tuple[int, int] | None,
    letterbox: bool = False,
) -> torch.Tensor:
    if coeffs.numel() == 0:
        h = original_size[1] if original_size is not None else input_shape[0]
        w = original_size[0] if original_size is not None else input_shape[1]
        return torch.zeros((0, h, w), dtype=torch.bool, device=proto.device)

    c, mask_h, mask_w = proto.shape
    masks = (coeffs @ proto.reshape(c, -1)).sigmoid().reshape(-1, mask_h, mask_w)

    input_h, input_w = input_shape
    boxes_mask = boxes_input.clone()
    boxes_mask[:, [0, 2]] *= mask_w / max(float(input_w), 1.0)
    boxes_mask[:, [1, 3]] *= mask_h / max(float(input_h), 1.0)
    masks = _crop_masks(masks, boxes_mask)

    if original_size is not None and letterbox:
        orig_w, orig_h = original_size
        ratio = min(input_h / orig_h, input_w / orig_w)
        new_h = max(int(orig_h * ratio), 1)
        new_w = max(int(orig_w * ratio), 1)
        masks = F.interpolate(
            masks[:, None],
            size=(int(input_h), int(input_w)),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        masks = masks[:, :new_h, :new_w]
        out_h, out_w = orig_h, orig_w
    elif original_size is not None:
        out_h, out_w = original_size[1], original_size[0]
    else:
        out_h, out_w = input_h, input_w
    masks = F.interpolate(
        masks[:, None],
        size=(int(out_h), int(out_w)),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return masks > 0.5


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
    predictions = output["predictions"]  # (batch, 4+nc, total_anchors)

    if predictions.dim() == 3:
        pred = predictions[0]  # (4+nc, total_anchors)
    else:
        pred = predictions

    # Transpose to (total_anchors, 4+nc)
    pred = pred.transpose(0, 1)

    boxes_input = pred[:, :4]  # xyxy format in model input pixels
    scores = pred[:, 4:]  # class scores (already sigmoid applied in model)

    max_scores, class_ids = torch.max(scores, dim=1)

    mask = max_scores > conf_thres
    if not mask.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    boxes_input = boxes_input[mask]
    boxes = boxes_input.clone()
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    mask_coeffs = output.get("mask_coeffs")
    proto = output.get("proto")
    coeffs = None
    if mask_coeffs is not None and proto is not None:
        coeffs_all = mask_coeffs[0].transpose(0, 1) if mask_coeffs.dim() == 3 else mask_coeffs
        coeffs = coeffs_all[mask]

    if original_size is not None:
        if letterbox:
            orig_w, orig_h = original_size
            ratio = min(input_size / orig_h, input_size / orig_w)
            boxes[:, :4] = boxes[:, :4] / ratio
        else:
            scale_x = original_size[0] / input_size
            scale_y = original_size[1] / input_size
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, original_size[0])
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, original_size[1])

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid = (widths > 0) & (heights > 0)
    if not valid.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}
    if not valid.all():
        boxes = boxes[valid]
        boxes_input = boxes_input[valid]
        max_scores = max_scores[valid]
        class_ids = class_ids[valid]
        if coeffs is not None:
            coeffs = coeffs[valid]

    keep = _nms_keep_indices(boxes, max_scores, class_ids, iou_thres, max_det)
    if len(keep) == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    boxes = boxes[keep]
    scores_out = max_scores[keep]
    classes_out = class_ids[keep]

    result = {
        "boxes": boxes.detach().cpu().numpy().tolist(),
        "scores": scores_out.detach().cpu().numpy().tolist(),
        "classes": classes_out.detach().cpu().numpy().tolist(),
        "num_detections": len(boxes),
    }

    if coeffs is not None and proto is not None:
        proto_i = proto[0] if proto.dim() == 4 else proto
        masks = _process_masks(
            proto_i,
            coeffs[keep],
            boxes_input[keep],
            input_shape=(input_size, input_size),
            original_size=original_size,
            letterbox=letterbox,
        )
        result["masks"] = masks.detach().cpu()

    return result
