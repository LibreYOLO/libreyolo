"""
Utility functions for YOLO9.

Provides preprocessing and postprocessing functions for YOLOv9 inference.
"""

import cv2
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
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Preprocess RGB HWC uint8 image for YOLOv9 inference.

    Centered letterbox resize (matches the reference WongKinYiu/yolov9
    ``utils.augmentations.letterbox``: aspect-preserving resize, the
    remaining padding split evenly between the two sides) + normalize to
    0-1 range.

    Args:
        img_rgb_hwc: Input image as RGB HWC uint8 numpy array.
        input_size: Target size for the model.

    Returns:
        Tuple of ``(preprocessed CHW float32 array in RGB 0-1, ratio,
        (pad_w, pad_h))`` where ``ratio`` is the resize gain and
        ``(pad_w, pad_h)`` is the left/top padding applied (so postprocess
        can undo it as ``(coord - pad) / ratio``).
    """
    orig_h, orig_w = img_rgb_hwc.shape[:2]
    ratio = min(input_size / orig_h, input_size / orig_w)
    new_w = int(round(orig_w * ratio))
    new_h = int(round(orig_h * ratio))

    dw = (input_size - new_w) / 2.0
    dh = (input_size - new_h) / 2.0

    resized = cv2.resize(img_rgb_hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))
    padded[top : top + new_h, left : left + new_w] = resized

    arr = np.ascontiguousarray(padded, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), ratio, (float(left), float(top))


def preprocess_image(
    image: ImageInput, input_size: int = 640, color_format: str = "auto"
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float, Tuple[float, float]]:
    """
    Preprocess image for YOLOv9 inference.

    Args:
        image: Input image (path, PIL, numpy, tensor, bytes, etc.)
        input_size: Target size for resizing (default: 640)
        color_format: Color format hint ("auto", "rgb", "bgr")

    Returns:
        Tuple of ``(preprocessed_tensor, original_image, original_size,
        ratio, (pad_w, pad_h))``.
    """
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size  # (width, height)
    original_img = img.copy()

    img_chw, ratio, pad = preprocess_numpy(np.array(img), input_size)
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
    return img_tensor, original_img, original_size, ratio, pad


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
    letterbox: bool = True,
    pad: Tuple[float, float] | None = None,
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
        # Use int(round(...)) to match the centered letterbox in
        # preprocess_numpy: the resized content occupies exactly
        # [top:top+new_h, left:left+new_w], so truncating here would crop the
        # mask 1px off from where the image was actually placed.
        new_h = max(int(round(orig_h * ratio)), 1)
        new_w = max(int(round(orig_w * ratio)), 1)
        masks = F.interpolate(
            masks[:, None],
            size=(int(input_h), int(input_w)),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        left = int(round(pad[0])) if pad is not None else 0
        top = int(round(pad[1])) if pad is not None else 0
        masks = masks[:, top : top + new_h, left : left + new_w]
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
    letterbox: bool = True,
    pad: Tuple[float, float] | None = None,
    multi_label: bool = True,
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
        letterbox: Whether the input was letterboxed (aspect-preserving).
        pad: ``(pad_w, pad_h)`` left/top padding applied at preprocess time,
            used to undo a centered letterbox. ``None`` falls back to the
            legacy top-left-padding assumption (no pad offset).
        multi_label: When True (the reference WongKinYiu/yolov9 ``val.py``
            default), every class whose score exceeds ``conf_thres`` emits a
            detection for that anchor instead of only the argmax class. This
            matches the original COCO eval protocol and is worth ~+0.7 mAP at
            the low conf thresholds used for benchmarking.

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

    boxes_all = pred[:, :4]  # xyxy format in model input pixels
    scores = pred[:, 4:]  # class scores (already sigmoid applied in model)

    mask_coeffs = output.get("mask_coeffs")
    proto = output.get("proto")
    coeffs_all = None
    if mask_coeffs is not None and proto is not None:
        coeffs_all = (
            mask_coeffs[0].transpose(0, 1) if mask_coeffs.dim() == 3 else mask_coeffs
        )

    # multi_label only helps when masks are not requested (segmentation uses
    # one coeff vector per anchor, so stick to the best-class path there).
    if multi_label and coeffs_all is None:
        # Candidate guard (matches WongKinYiu/yolov9 non_max_suppression): first
        # keep only anchors whose *best* class beats conf_thres. Every class
        # above conf necessarily lives in such an anchor, so the result is
        # identical to scanning the full 8400xnc score matrix — but the
        # ``(scores > conf_thres).nonzero()`` below then runs on a few hundred
        # rows instead of ~600k, bounding memory/time at conf=0.001.
        cand = scores.amax(dim=1) > conf_thres
        if not cand.any():
            return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}
        cand_idx = cand.nonzero(as_tuple=True)[0]
        scores_c = scores[cand_idx]
        sub_anchor, class_ids = (scores_c > conf_thres).nonzero(as_tuple=True)
        anchor_idx = cand_idx[sub_anchor]
        boxes_input = boxes_all[anchor_idx]
        boxes = boxes_input.clone()
        max_scores = scores[anchor_idx, class_ids]
        # Cap to max_nms candidates by score before NMS (upstream uses 30000),
        # so a pathological frame cannot blow up the NMS pairwise IoU.
        max_nms = 30000
        if max_scores.numel() > max_nms:
            topk = torch.topk(max_scores, max_nms).indices
            boxes_input = boxes_input[topk]
            boxes = boxes[topk]
            max_scores = max_scores[topk]
            class_ids = class_ids[topk]
        coeffs = None
    else:
        max_scores, class_ids = torch.max(scores, dim=1)
        mask = max_scores > conf_thres
        if not mask.any():
            return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}
        boxes_input = boxes_all[mask]
        boxes = boxes_input.clone()
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]
        coeffs = coeffs_all[mask] if coeffs_all is not None else None

    if original_size is not None:
        if letterbox:
            orig_w, orig_h = original_size
            ratio = min(input_size / orig_h, input_size / orig_w)
            if pad is not None:
                boxes[:, [0, 2]] -= pad[0]
                boxes[:, [1, 3]] -= pad[1]
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
            pad=pad,
        )
        result["masks"] = masks.detach().cpu()

    return result
