"""
Utility functions for YOLO9.

Provides preprocessing and postprocessing functions for YOLOv9 inference.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict
from PIL import Image

from ...utils.general import (
    postprocess_detections,
)
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


def crop_mask(masks, boxes):
    """Zero out mask pixels outside bounding boxes.

    Args:
        masks: (N, H, W) mask tensor.
        boxes: (N, 4) bounding boxes in xyxy format at mask resolution.

    Returns:
        Cropped masks (N, H, W).
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = boxes.T.unsqueeze(-1).unsqueeze(-1)  # each (N, 1, 1)
    cols = torch.arange(w, device=masks.device).view(1, 1, w)
    rows = torch.arange(h, device=masks.device).view(1, h, 1)
    return masks * ((cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2)).float()


def process_mask(proto, mask_coeffs, bboxes, img_shape):
    """Assemble instance masks from prototypes and coefficients.

    Implements YOLACT mask assembly (Bolya et al., ICCV 2019):
        M = sigmoid(C @ P)
    then crop to predicted bounding box and upsample to original resolution.

    Args:
        proto: (num_masks, H_proto, W_proto) prototype masks.
        mask_coeffs: (N, num_masks) per-detection mask coefficients.
        bboxes: (N, 4) bounding boxes in xyxy format at original image resolution.
        img_shape: (H, W) original image size.

    Returns:
        (N, H, W) boolean instance masks at original image resolution.
    """
    c, mh, mw = proto.shape
    ih, iw = img_shape

    if len(mask_coeffs) == 0:
        return torch.zeros((0, ih, iw), dtype=torch.bool, device=proto.device)

    # Mask assembly: coefficients @ prototypes → raw masks
    masks = (mask_coeffs @ proto.float().view(c, -1)).sigmoid().view(-1, mh, mw)

    # Scale bboxes to proto resolution for cropping
    scale_x = mw / iw
    scale_y = mh / ih
    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= scale_x
    downsampled_bboxes[:, 2] *= scale_x
    downsampled_bboxes[:, 1] *= scale_y
    downsampled_bboxes[:, 3] *= scale_y

    # Crop masks to bounding boxes
    masks = crop_mask(masks, downsampled_bboxes)

    # Upsample to original image resolution
    masks = F.interpolate(
        masks.unsqueeze(0), (ih, iw), mode="bilinear", align_corners=False
    )[0]

    # Threshold to binary
    return masks.gt(0.5)


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
        output: Model output dictionary with 'predictions' key,
            and optionally 'proto' + 'mask_coeffs' for segmentation.
        conf_thres: Confidence threshold (default: 0.25)
        iou_thres: IoU threshold for NMS (default: 0.45)
        input_size: Input image size (default: 640)
        original_size: Original image size (width, height) for scaling
        max_det: Maximum number of detections to return (default: 300)

    Returns:
        Dictionary with boxes, scores, classes, num_detections,
        and optionally masks for segmentation models.
    """
    predictions = output["predictions"]  # (batch, 4+nc, total_anchors)

    if predictions.dim() == 3:
        pred = predictions[0]  # (4+nc, total_anchors)
    else:
        pred = predictions

    # Transpose to (total_anchors, 4+nc)
    pred = pred.transpose(0, 1)

    boxes = pred[:, :4]  # xyxy format
    scores = pred[:, 4:]  # class scores (already sigmoid applied in model)

    max_scores, class_ids = torch.max(scores, dim=1)

    conf_mask = max_scores > conf_thres
    if not conf_mask.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    det = postprocess_detections(
        boxes=boxes[conf_mask],
        scores=max_scores[conf_mask],
        class_ids=class_ids[conf_mask],
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        input_size=input_size,
        original_size=original_size,
        max_det=max_det,
        letterbox=letterbox,
    )

    # Assemble masks for segmentation models
    proto = output.get("proto")
    mc = output.get("mask_coeffs")
    if proto is not None and mc is not None and det["num_detections"] > 0:
        proto_single = proto[0]  # (num_masks, H_proto, W_proto)
        mc_single = mc[0]        # (num_masks, total_anchors)

        # Get mask coefficients for conf-filtered anchors
        mc_filtered = mc_single[:, conf_mask].T  # (N_conf, num_masks)

        # Use keep_indices from NMS to select the correct mask coefficients
        keep_indices = det.get("keep_indices")
        if keep_indices is not None and len(keep_indices) > 0:
            mask_coeffs = mc_filtered[keep_indices]
        else:
            mask_coeffs = mc_filtered[:det["num_detections"]]

        if original_size is not None:
            orig_w, orig_h = original_size
            img_shape = (orig_h, orig_w)
        else:
            img_shape = (input_size, input_size)

        boxes_t = torch.tensor(det["boxes"], dtype=torch.float32, device=proto_single.device)
        masks = process_mask(proto_single, mask_coeffs, boxes_t, img_shape)
        det["masks"] = masks.cpu()

    return det
