"""RTMDet postprocessing.

Per-level cls + reg -> sigmoid scores + distance2bbox decode -> per-class NMS.
The reg branch already returns ltrb distances multiplied by stride (with
optional .exp() for m/l/x sizes), so decoding is just
``point - left_top, point + right_bottom``.

Moved verbatim from ``libreyolo/models/rtmdet/utils.py``, which re-exports
everything here for backward compatibility.
"""

from __future__ import annotations

import contextlib
import math
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.ops

from .common import _input_size_hw, postprocess_detections


def _make_grid_priors(feats: List[torch.Tensor], strides: List[int]) -> torch.Tensor:
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
    outputs: tuple,
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
    if len(outputs) == 4:
        # RTMDet-Ins uses nms_pre=1000 in every official COCO size config,
        # while RTMDet detection uses 30000. Preserve a smaller caller value.
        return _postprocess_segment(
            outputs,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=input_size,
            original_size=original_size,
            ratio=ratio,
            max_det=max_det,
            strides=strides,
            nms_pre=min(nms_pre, 1000),
        )

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
        dist_lvl = reg[0].permute(1, 2, 0).reshape(-1, 4)  # (H*W, 4)

        # Build priors for this level only (offset=0, cell corners).
        device, dtype = scores_lvl.device, scores_lvl.dtype
        sx = torch.arange(w, device=device, dtype=dtype) * stride
        sy = torch.arange(h, device=device, dtype=dtype) * stride
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        points_lvl = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (H*W, 2)

        valid_mask = scores_lvl > conf_thres
        if not valid_mask.any():
            continue
        valid_idxs = torch.nonzero(valid_mask, as_tuple=False)  # (M, 2)
        flat_scores = scores_lvl[valid_mask]  # (M,)

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
    input_h, input_w = _input_size_hw(input_size)
    boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, input_w)
    boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, input_h)

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


def _empty_segment_result(
    original_size: Tuple[int, int] | None,
    input_size: Union[int, Tuple[int, int]],
) -> dict:
    if original_size is None:
        input_h, input_w = _input_size_hw(input_size)
        h, w = input_h, input_w
    else:
        w, h = original_size
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "scores": torch.empty(0, dtype=torch.float32),
        "classes": torch.empty(0, dtype=torch.int64),
        "masks": torch.empty((0, h, w), dtype=torch.bool),
        "num_detections": 0,
    }


def _parse_dynamic_params(
    kernels: torch.Tensor,
    weight_nums: Tuple[int, ...] = (80, 64, 8),
    bias_nums: Tuple[int, ...] = (8, 8, 1),
    dyconv_channels: int = 8,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split the 169 RTMDet-Ins parameters into grouped 1x1 convolutions."""
    num_inst = kernels.shape[0]
    pieces = list(torch.split(kernels, weight_nums + bias_nums, dim=1))
    weights = pieces[: len(weight_nums)]
    biases = pieces[len(weight_nums) :]
    for i in range(len(weights)):
        if i < len(weights) - 1:
            weights[i] = weights[i].reshape(num_inst * dyconv_channels, -1, 1, 1)
            biases[i] = biases[i].reshape(num_inst * dyconv_channels)
        else:
            weights[i] = weights[i].reshape(num_inst, -1, 1, 1)
            biases[i] = biases[i].reshape(num_inst)
    return weights, biases


def _decode_masks(
    mask_feat: torch.Tensor,
    kernels: torch.Tensor,
    priors: torch.Tensor,
) -> torch.Tensor:
    """Apply the per-instance dynamic convolution network in fp32."""
    num_inst = priors.shape[0]
    h, w = mask_feat.shape[-2:]
    if num_inst == 0:
        return mask_feat.new_empty((0, h, w), dtype=torch.float32)

    device_type = mask_feat.device.type
    autocast = (
        torch.autocast(device_type=device_type, enabled=False)
        if device_type in {"cpu", "cuda"}
        else contextlib.nullcontext()
    )
    with autocast:
        mask_feat = mask_feat.float()
        kernels = kernels.float()
        priors = priors.float()

        xs = torch.arange(w, device=mask_feat.device, dtype=torch.float32) * 8
        ys = torch.arange(h, device=mask_feat.device, dtype=torch.float32) * 8
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2)
        points = priors[:, :2].reshape(-1, 1, 2)
        relative = (points - coords).permute(0, 2, 1)
        relative = relative / (priors[:, 2].reshape(-1, 1, 1) * 8)
        relative = relative.reshape(num_inst, 2, h, w)

        dynamic_input = torch.cat(
            (relative, mask_feat.unsqueeze(0).repeat(num_inst, 1, 1, 1)),
            dim=1,
        ).reshape(1, -1, h, w)
        weights, biases = _parse_dynamic_params(kernels)
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            dynamic_input = F.conv2d(dynamic_input, weight, bias=bias, groups=num_inst)
            if i < len(weights) - 1:
                dynamic_input = F.relu(dynamic_input)
        return dynamic_input.reshape(num_inst, h, w)


def _postprocess_segment(
    outputs: tuple,
    *,
    conf_thres: float,
    iou_thres: float,
    input_size: int,
    original_size: Tuple[int, int] | None,
    ratio: float,
    max_det: int,
    strides: Tuple[int, ...],
    nms_pre: int,
) -> dict:
    """Decode RTMDet-Ins boxes, run NMS, then decode survivor masks."""
    cls_scores, bbox_preds, kernel_preds, mask_feats = outputs
    scores_all = []
    classes_all = []
    distances_all = []
    priors_all = []
    kernels_all = []

    for cls, reg, kernel, stride in zip(cls_scores, bbox_preds, kernel_preds, strides):
        _, num_classes, h, w = cls.shape
        scores = cls[0].permute(1, 2, 0).reshape(-1, num_classes).sigmoid()
        distances = reg[0].permute(1, 2, 0).reshape(-1, 4)
        kernels = kernel[0].permute(1, 2, 0).reshape(-1, kernel.shape[1])

        xs = torch.arange(w, device=cls.device, dtype=cls.dtype) * stride
        ys = torch.arange(h, device=cls.device, dtype=cls.dtype) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        strides_lvl = torch.full_like(points, stride)
        priors = torch.cat((points, strides_lvl), dim=1)

        valid = scores > conf_thres
        if not valid.any():
            continue
        pairs = torch.nonzero(valid, as_tuple=False)
        values = scores[valid]
        count = min(nms_pre, values.numel())
        values, order = values.sort(descending=True)
        pairs = pairs[order[:count]]
        loc_idx = pairs[:, 0]

        scores_all.append(values[:count])
        classes_all.append(pairs[:, 1])
        distances_all.append(distances[loc_idx])
        priors_all.append(priors[loc_idx])
        kernels_all.append(kernels[loc_idx])

    if not scores_all:
        return _empty_segment_result(original_size, input_size)

    scores = torch.cat(scores_all)
    classes = torch.cat(classes_all)
    distances = torch.cat(distances_all)
    priors = torch.cat(priors_all)
    kernels = torch.cat(kernels_all)
    boxes = _distance2bbox(priors[:, :2], distances)
    # Basic slicing (a view) so the in-place clamp reaches ``boxes``; list
    # indexing would clamp a copy and silently leave boxes unclamped.
    seg_input_h, seg_input_w = _input_size_hw(input_size)
    boxes[:, 0::2].clamp_(0, seg_input_w)
    boxes[:, 1::2].clamp_(0, seg_input_h)

    finite = (
        torch.isfinite(boxes).all(dim=1)
        & torch.isfinite(scores)
        & torch.isfinite(kernels).all(dim=1)
    )
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid = finite & (widths > 0) & (heights > 0)
    if not valid.all():
        boxes = boxes[valid]
        scores = scores[valid]
        classes = classes[valid]
        priors = priors[valid]
        kernels = kernels[valid]
    if boxes.numel() == 0:
        return _empty_segment_result(original_size, input_size)

    boxes = boxes.float()
    scores = scores.float()
    keep = torchvision.ops.batched_nms(boxes, scores, classes, iou_thres)
    keep = keep[:max_det]
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    priors = priors[keep]
    kernels = kernels[keep]

    mask_logits = _decode_masks(mask_feats[0], kernels, priors)
    mask_logits = F.interpolate(
        mask_logits.unsqueeze(0),
        scale_factor=8,
        mode="bilinear",
        align_corners=False,
    )
    if original_size is not None:
        orig_w, orig_h = original_size
        # Resize each axis independently; with a rectangular canvas the mask
        # tensor is non-square, so a single width-derived size would distort
        # masks and blow up memory at wide aspect ratios.
        resized_h = math.ceil(mask_logits.shape[-2] / ratio)
        resized_w = math.ceil(mask_logits.shape[-1] / ratio)
        mask_logits = F.interpolate(
            mask_logits,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        )[..., :orig_h, :orig_w]
        boxes = boxes / ratio
        boxes[:, 0::2].clamp_(0, orig_w)
        boxes[:, 1::2].clamp_(0, orig_h)

    masks = mask_logits.sigmoid().squeeze(0) > 0.5
    return {
        "boxes": boxes.detach().cpu(),
        "scores": scores.detach().cpu(),
        "classes": classes.detach().cpu(),
        "masks": masks.detach().cpu(),
        "num_detections": int(boxes.shape[0]),
    }
