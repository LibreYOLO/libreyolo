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


def _per_level_filter_topk(
    cls_scores: List[torch.Tensor],
    bbox_preds: List[torch.Tensor],
    strides: Sequence[int],
    reg_max: int,
    score_thr: float,
    nms_pre: int,
    canvas_size: Tuple[int, int] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bo's ``filter_scores_and_topk`` per level: each level applies
    ``score_thr`` to the *flattened* (anchor*classes) score table, then keeps
    the top ``nms_pre`` (anchor, class) pairs by score, then decodes only
    those boxes. Concatenates across levels.

    Returns ``(scores, class_ids, boxes_xyxy)`` flat across levels.
    """
    assert len(cls_scores) == len(bbox_preds) == len(strides)
    B = cls_scores[0].shape[0]
    assert B == 1, "Per-level top-K only implemented for B=1 inference path."
    device, dtype = cls_scores[0].device, cls_scores[0].dtype
    nc = cls_scores[0].shape[1]

    out_scores: List[torch.Tensor] = []
    out_classes: List[torch.Tensor] = []
    out_boxes: List[torch.Tensor] = []

    for cls_score, bbox_pred, stride in zip(cls_scores, bbox_preds, strides):
        _, _, h, w = cls_score.shape
        n = h * w

        # (n, num_classes) sigmoid scores
        scores = torch.sigmoid(cls_score[0]).permute(1, 2, 0).reshape(n, nc)
        # Flatten to (n*nc,) and pick top candidates above threshold
        flat = scores.reshape(-1)
        keep_mask = flat > score_thr
        if not keep_mask.any():
            continue
        kept_flat_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
        kept_scores = flat[kept_flat_idx]
        if kept_scores.numel() > nms_pre:
            top_scores, top_idx = torch.topk(kept_scores, nms_pre)
            kept_flat_idx = kept_flat_idx[top_idx]
            kept_scores = top_scores

        anchor_idx = kept_flat_idx // nc
        class_idx = kept_flat_idx % nc

        # Decode just the kept anchors
        bp = bbox_pred[0].permute(1, 2, 0).reshape(n, 4 * (reg_max + 1))[anchor_idx]
        bp = bp.reshape(-1, 4, reg_max + 1)
        bp = F.softmax(bp, dim=-1)
        proj = torch.linspace(0, reg_max, reg_max + 1, device=device, dtype=dtype)
        distances = (bp * proj).sum(dim=-1) * stride

        # Per-anchor centers from the original grid
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([xx.flatten(), yy.flatten()], dim=-1)[anchor_idx]

        x1 = centers[:, 0] - distances[:, 0]
        y1 = centers[:, 1] - distances[:, 1]
        x2 = centers[:, 0] + distances[:, 2]
        y2 = centers[:, 1] + distances[:, 3]
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        # Bo's distance2bbox clamps to image_shape (input canvas) before NMS.
        # Skipping this lets boxes extend off-canvas; oversized boxes can
        # distort per-class IoU during NMS and suppress legitimate detections.
        if canvas_size is not None:
            ch, cw = canvas_size
            boxes[:, 0].clamp_(0, cw)
            boxes[:, 1].clamp_(0, ch)
            boxes[:, 2].clamp_(0, cw)
            boxes[:, 3].clamp_(0, ch)

        out_scores.append(kept_scores)
        out_classes.append(class_idx)
        out_boxes.append(boxes)

    if not out_scores:
        return (
            torch.zeros(0, device=device, dtype=dtype),
            torch.zeros(0, device=device, dtype=torch.long),
            torch.zeros((0, 4), device=device, dtype=dtype),
        )
    return (
        torch.cat(out_scores, dim=0),
        torch.cat(out_classes, dim=0),
        torch.cat(out_boxes, dim=0),
    )


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
    import torchvision.ops as tvo

    cls_scores, bbox_preds = output

    # Per-level top-K filter, then a single ``batched_nms`` across the union.
    # Each level keeps the top ``nms_pre`` (anchor, class) pairs above
    # ``conf_thres``. Multi-label per anchor (vs argmax) so anchors with two
    # strong classes emit both candidates.
    valid_scores, class_ids, valid_boxes = _per_level_filter_topk(
        cls_scores, bbox_preds, strides=strides, reg_max=reg_max,
        score_thr=conf_thres, nms_pre=1000,
        canvas_size=(input_size, input_size),
    )
    if valid_scores.numel() == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    # Rescale to original image (PicoDet uses simple resize, not letterbox)
    if original_size is not None:
        scale_x = original_size[0] / input_size
        scale_y = original_size[1] / input_size
        valid_boxes = valid_boxes.clone()
        valid_boxes[:, [0, 2]] *= scale_x
        valid_boxes[:, [1, 3]] *= scale_y
        valid_boxes[:, [0, 2]].clamp_(0, original_size[0])
        valid_boxes[:, [1, 3]].clamp_(0, original_size[1])

    # Drop zero/negative-area boxes
    bw = valid_boxes[:, 2] - valid_boxes[:, 0]
    bh = valid_boxes[:, 3] - valid_boxes[:, 1]
    keep_area = (bw > 0) & (bh > 0)
    if not keep_area.all():
        valid_boxes = valid_boxes[keep_area]
        valid_scores = valid_scores[keep_area]
        class_ids = class_ids[keep_area]

    if valid_scores.numel() == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    # Single batched NMS across all classes (one C++ call).
    keep = tvo.batched_nms(valid_boxes, valid_scores, class_ids, iou_thres)
    if keep.numel() > max_det:
        # Top-by-score among the kept indices
        top = torch.topk(valid_scores[keep], max_det).indices
        keep = keep[top]

    final_boxes = valid_boxes[keep].cpu().numpy()
    final_scores = valid_scores[keep].cpu().numpy()
    final_classes = class_ids[keep].cpu().numpy()
    return {
        "boxes": final_boxes.tolist(),
        "scores": final_scores.tolist(),
        "classes": final_classes.tolist(),
        "num_detections": len(final_boxes),
    }
