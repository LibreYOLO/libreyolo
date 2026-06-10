"""RTMDet postprocessing.

Per-level cls + reg -> sigmoid scores + distance2bbox decode -> per-class NMS.
The reg branch already returns ltrb distances multiplied by stride (with
optional .exp() for m/l/x sizes), so decoding is just
``point - left_top, point + right_bottom``.

Moved verbatim from ``libreyolo/models/rtmdet/utils.py``, which re-exports
everything here for backward compatibility.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from .common import postprocess_detections


def _make_grid_priors(
    feats: List[torch.Tensor], strides: List[int]
) -> torch.Tensor:
    """Build (N, 2) grid of pixel-space prior points for all FPN levels.

    Matches mmdet's ``MlvlPointGenerator(offset=0)`` as configured in the
    RTMDet recipe: priors live at cell *corners*, not cell centers.

        x = i * stride
        y = j * stride

    The default mmdet offset is 0.5 (centers) but the RTMDet config explicitly
    sets ``offset=0``. Using 0.5 here introduces a stride/2 pixel shift in
    decoded boxes and silently costs a couple of mAP points.
    """
    points = []
    for feat, stride in zip(feats, strides):
        h, w = feat.shape[-2:]
        device, dtype = feat.device, feat.dtype
        sx = torch.arange(w, device=device, dtype=dtype) * stride
        sy = torch.arange(h, device=device, dtype=dtype) * stride
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack([xx, yy], dim=-1).reshape(-1, 2))
    return torch.cat(points, dim=0)


def _distance2bbox(points: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """Decode point + (l, t, r, b) distances to xyxy boxes."""
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def postprocess(
    outputs: Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
    input_size: int = 640,
    original_size: Tuple[int, int] | None = None,
    ratio: float = 1.0,
    max_det: int = 300,
    strides: Tuple[int, ...] = (8, 16, 32),
    nms_pre: int = 30000,
) -> dict:
    """Decode RTMDet head outputs to {boxes, scores, classes, num_detections}.

    Outputs format: (cls_scores, bbox_preds), each a tuple of per-level tensors.
        cls_scores[i]: (B, num_classes, H_i, W_i)  — pre-sigmoid logits
        bbox_preds[i]: (B, 4, H_i, W_i)            — already multiplied by stride

    Returns boxes in original-image coordinates (after letterbox inverse).
    """
    cls_scores, bbox_preds = outputs

    # Match mmdet's ``_predict_by_feat_single`` (mmdetection/mmdet/models/dense_heads/
    # base_dense_head.py:359-410): apply ``filter_scores_and_topk`` PER FPN LEVEL,
    # then concatenate. Each level keeps up to ``nms_pre`` candidates above
    # ``conf_thres`` independently, so the high-resolution P3 level (which holds
    # most of the small-object recall) is not starved by the noisy long tail of
    # P5. Doing this once globally on the concatenated tensor lost ~7-8 mAP at
    # COCO eval (conf=0.001).
    mlvl_scores = []
    mlvl_classes = []
    mlvl_distances = []
    mlvl_points = []
    for cls, reg, stride in zip(cls_scores, bbox_preds, strides):
        b, c, h, w = cls.shape
        scores_lvl = cls[0].permute(1, 2, 0).reshape(-1, c).sigmoid()  # (H*W, C)
        dist_lvl = reg[0].permute(1, 2, 0).reshape(-1, 4)               # (H*W, 4)

        # Build priors for this level only (offset=0, cell corners).
        device, dtype = scores_lvl.device, scores_lvl.dtype
        sx = torch.arange(w, device=device, dtype=dtype) * stride
        sy = torch.arange(h, device=device, dtype=dtype) * stride
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        points_lvl = torch.stack([xx, yy], dim=-1).reshape(-1, 2)       # (H*W, 2)

        valid_mask = scores_lvl > conf_thres
        if not valid_mask.any():
            continue
        valid_idxs = torch.nonzero(valid_mask, as_tuple=False)           # (M, 2)
        flat_scores = scores_lvl[valid_mask]                             # (M,)

        num_topk = min(nms_pre, flat_scores.numel())
        sorted_scores, sort_idxs = flat_scores.sort(descending=True)
        sorted_scores = sorted_scores[:num_topk]
        topk_pairs = valid_idxs[sort_idxs[:num_topk]]
        loc_idx = topk_pairs[:, 0]
        cls_idx = topk_pairs[:, 1]

        mlvl_scores.append(sorted_scores)
        mlvl_classes.append(cls_idx)
        mlvl_distances.append(dist_lvl[loc_idx])
        mlvl_points.append(points_lvl[loc_idx])

    if not mlvl_scores:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    max_scores = torch.cat(mlvl_scores, dim=0)
    classes = torch.cat(mlvl_classes, dim=0)
    distances = torch.cat(mlvl_distances, dim=0)
    points = torch.cat(mlvl_points, dim=0)

    boxes = _distance2bbox(points, distances)

    # Match mmdet: clamp boxes to the padded input canvas BEFORE rescale, via
    # ``distance2bbox(max_shape=img_shape)`` (mmdet) — img_shape is the padded
    # 640x640 canvas. This is unconditional; previously gating on ``ratio != 1.0``
    # silently skipped the clamp for the very common case where one image
    # dimension is already 640, leaving boxes that overflow the canvas (e.g.
    # y2=643 for a 640x586 image). Larger boxes inflate the union for COCO
    # IoU and cost ~1 mAP at conf=0.001.
    boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, input_size)
    boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, input_size)

    if original_size is not None:
        boxes = boxes / ratio
        orig_w, orig_h = original_size
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, orig_h)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid = (widths > 0) & (heights > 0)
        if not valid.all():
            boxes = boxes[valid]
            max_scores = max_scores[valid]
            classes = classes[valid]
        if boxes.numel() == 0:
            return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    return postprocess_detections(
        boxes=boxes,
        scores=max_scores,
        class_ids=classes,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        input_size=input_size,
        original_size=None,  # already scaled above
        max_det=max_det,
        letterbox=False,
    )
