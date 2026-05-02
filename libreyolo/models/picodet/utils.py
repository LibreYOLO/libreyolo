"""PicoDet preprocessing and postprocessing.

PicoDet upstream uses **non-letterbox** simple resize + ImageNet
normalisation (RGB, mean=[123.675, 116.28, 103.53],
std=[58.395, 57.12, 57.375]). Output decoding follows the GFL/DFL
recipe: softmax-expectation over the discrete distribution buckets,
multiplied by the level stride, then ``distance2bbox`` from each grid
centre.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.general import postprocess_detections


# ImageNet stats Bo's repo uses (shared across all PicoDet sizes)
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 320,
) -> Tuple[np.ndarray, float]:
    """Preprocess an RGB HWC uint8 image for PicoDet inference.

    Returns ``(chw_float32, ratio)``. ``ratio`` is unused by PicoDet's
    non-letterbox resize but kept in the signature so it can flow through
    the same postprocess pipeline as letterbox-based families.
    """
    img = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    arr = np.array(img, dtype=np.float32)
    arr -= np.array(IMAGENET_MEAN, dtype=np.float32)
    arr /= np.array(IMAGENET_STD, dtype=np.float32)
    return arr.transpose(2, 0, 1), 1.0


def preprocess_image(
    image: ImageInput,
    input_size: int = 320,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()
    chw, ratio = preprocess_numpy(np.array(img), input_size)
    return torch.from_numpy(chw).unsqueeze(0), original_img, original_size, ratio


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def _grid_centers(
    h: int, w: int, stride: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """(H*W, 2) grid centres in pixel coords, offset 0.5 like upstream."""
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], dim=-1)


def _integral(bbox_pred: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Softmax-expectation along the last bucket dim. ``bbox_pred`` is
    (..., 4 * (reg_max+1)) and the result is (..., 4) in distance units
    (left, top, right, bottom) before stride scaling.
    """
    shape = bbox_pred.shape
    x = bbox_pred.reshape(-1, reg_max + 1)
    x = F.softmax(x, dim=-1)
    project = torch.linspace(0, reg_max, reg_max + 1, device=x.device, dtype=x.dtype)
    x = (x * project).sum(dim=-1)
    return x.reshape(*shape[:-1], 4)


def decode_outputs(
    cls_scores: List[torch.Tensor],
    bbox_preds: List[torch.Tensor],
    strides: Sequence[int] = (8, 16, 32, 64),
    reg_max: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode PicoHead outputs (training-mode list-of-tensors form).

    Args:
        cls_scores: per-level (B, num_classes, H, W).
        bbox_preds: per-level (B, 4*(reg_max+1), H, W).
        strides: per-level pixel stride.

    Returns:
        scores: (B, N_total, num_classes) **after sigmoid**.
        boxes:  (B, N_total, 4) in xyxy pixel coords on the input canvas.
    """
    assert len(cls_scores) == len(bbox_preds) == len(strides)
    B = cls_scores[0].shape[0]
    device, dtype = cls_scores[0].device, cls_scores[0].dtype
    nc = cls_scores[0].shape[1]

    all_scores: List[torch.Tensor] = []
    all_boxes: List[torch.Tensor] = []
    for cls_score, bbox_pred, stride in zip(cls_scores, bbox_preds, strides):
        _, _, h, w = cls_score.shape
        n = h * w

        scores = torch.sigmoid(cls_score).permute(0, 2, 3, 1).reshape(B, n, nc)

        # (B, n, 4*(reg_max+1)) -> (B, n, 4) distances in pixels
        bp = bbox_pred.permute(0, 2, 3, 1).reshape(B, n, 4 * (reg_max + 1))
        distances = _integral(bp, reg_max) * stride

        centers = _grid_centers(h, w, stride, device, dtype).unsqueeze(0).expand(B, -1, -1)
        x1 = centers[..., 0] - distances[..., 0]
        y1 = centers[..., 1] - distances[..., 1]
        x2 = centers[..., 0] + distances[..., 2]
        y2 = centers[..., 1] + distances[..., 3]
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        all_scores.append(scores)
        all_boxes.append(boxes)

    return torch.cat(all_scores, dim=1), torch.cat(all_boxes, dim=1)


# ---------------------------------------------------------------------------
# Postprocess
# ---------------------------------------------------------------------------


def postprocess(
    output: Tuple[List[torch.Tensor], List[torch.Tensor]],
    conf_thres: float = 0.025,
    iou_thres: float = 0.6,
    input_size: int = 320,
    original_size: Tuple[int, int] | None = None,
    ratio: float = 1.0,  # unused; kept for signature parity
    max_det: int = 100,
    strides: Sequence[int] = (8, 16, 32, 64),
    reg_max: int = 7,
) -> dict:
    """Decode PicoDet head output to a single image's detections.

    Defaults match Bo's ``test_cfg`` (score_thr=0.025, iou_threshold=0.6,
    max_per_img=100). Caller usually overrides ``conf_thres`` to 0.25 for
    interactive inference.
    """
    cls_scores, bbox_preds = output
    scores, boxes = decode_outputs(cls_scores, bbox_preds, strides=strides, reg_max=reg_max)

    # Single-image path (B=1)
    scores = scores[0]  # (N, nc)
    boxes = boxes[0]    # (N, 4)

    # Multi-label per anchor: each (anchor, class) pair above conf is a
    # separate candidate. Matches Bo's ``filter_scores_and_topk`` pipeline,
    # which is what produces the upstream mAP. The argmax-per-anchor path
    # we used previously costs ~1.5 mAP because anchors with two strong
    # classes (e.g. "person" and "skier") only emitted the single max.
    mask = scores > conf_thres
    if not mask.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    nz = mask.nonzero(as_tuple=False)
    anchor_idx = nz[:, 0]
    class_ids = nz[:, 1]
    valid_scores = scores[anchor_idx, class_ids]
    valid_boxes = boxes[anchor_idx]

    return postprocess_detections(
        boxes=valid_boxes,
        scores=valid_scores,
        class_ids=class_ids,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        input_size=input_size,
        original_size=original_size,
        max_det=max_det,
        letterbox=False,  # PicoDet uses simple resize, not letterbox
    )
