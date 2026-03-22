"""
Utility functions for LibreYOLO RF-DETR.

Postprocessing matches the original rfdetr implementation exactly.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from PIL import Image

from ...utils.general import cxcywh_to_xyxy

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 560,
) -> Tuple[np.ndarray, float]:
    """
    Preprocess RGB HWC uint8 image for RF-DETR inference.

    Simple resize + ImageNet normalization.

    Args:
        img_rgb_hwc: Input image as RGB HWC uint8 numpy array.
        input_size: Target size for the model.

    Returns:
        Tuple of (preprocessed CHW float32 array with ImageNet norm, ratio).
    """
    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1), 1.0


def postprocess(
    outputs: Dict[str, torch.Tensor], target_sizes: torch.Tensor, num_select: int = 300
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
    out_logits = outputs["pred_logits"]  # (B, num_queries, num_classes)
    out_bbox = outputs["pred_boxes"]  # (B, num_queries, 4) in cxcywh [0, 1]
    out_masks = outputs.get("pred_masks")  # (B, num_queries, Hm, Wm) or None

    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = out_logits.sigmoid()

    # Top-K across all (queries × classes)
    batch_size = out_logits.shape[0]
    num_classes = out_logits.shape[2]

    topk_values, topk_indexes = torch.topk(prob.view(batch_size, -1), num_select, dim=1)

    scores = topk_values

    topk_boxes = topk_indexes // num_classes  # Which query
    labels = topk_indexes % num_classes  # Which class

    boxes = cxcywh_to_xyxy(out_bbox)

    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # Scale from relative [0, 1] to absolute [0, height/width] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = []
    for i in range(batch_size):
        res_i = {"scores": scores[i], "labels": labels[i], "boxes": boxes[i]}

        if out_masks is not None:
            # Gather masks for top-K queries
            k_idx = topk_boxes[i]
            masks_i = torch.gather(
                out_masks[i],
                0,
                k_idx.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, out_masks.shape[-2], out_masks.shape[-1]),
            )  # (K, Hm, Wm)

            # Resize to original image size
            h, w = target_sizes[i].tolist()
            masks_i = F.interpolate(
                masks_i.unsqueeze(1),
                size=(int(h), int(w)),
                mode="bilinear",
                align_corners=False,
            )  # (K, 1, H, W)

            # Threshold at 0.0 in logit space (= 0.5 probability)
            res_i["masks"] = (masks_i[:, 0] > 0.0).bool()  # (K, H, W)

        results.append(res_i)

    return results
