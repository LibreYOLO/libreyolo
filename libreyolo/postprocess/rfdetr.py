"""RF-DETR postprocessing (DETR-style top-K, no NMS; multi-task heads).

Behavior matches upstream RF-DETR (https://github.com/roboflow/rf-detr) so
weights load and produce numerically equivalent detections.

Moved verbatim from ``libreyolo/models/rfdetr/utils.py``, which re-exports it
for backward compatibility.

The GroupPose keypoint postprocess (``_keypoint_log_mean_trace`` /
``_postprocess_grouppose_keypoints`` and the keypoint branch of ``postprocess``)
is ported from RF-DETR v1.8.0 (https://github.com/roboflow/rf-detr).
Copyright (c) 2025 Roboflow, Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
"""

import torch
import torch.nn.functional as F
from collections.abc import Sequence
from typing import Optional

from ..data.obb import scale_xywhr
from ..utils.general import cxcywh_to_xyxy


def _keypoint_log_mean_trace(active_keypoints: torch.Tensor) -> torch.Tensor:
    """Compute the log findability-weighted mean covariance trace.

    Ported from RF-DETR v1.8.0 (``PostProcess._keypoint_log_mean_trace``,
    https://github.com/roboflow/rf-detr, Apache-2.0). Columns ``4:7`` of
    ``active_keypoints`` are the ``(log_l11, l21, log_l22)`` precision-Cholesky
    parameters and column ``2`` is the findable logit. The computation stays in
    log space for numerical stability. Supports a leading batch dimension so it
    can be called once per keypoint class group.
    """
    log_l11 = active_keypoints[..., 4]
    l21 = active_keypoints[..., 5]
    log_l22 = active_keypoints[..., 6]
    w_find = active_keypoints[..., 2].sigmoid()
    log_t1 = -2.0 * log_l11
    log_t2 = -2.0 * log_l22
    log_t3 = 2.0 * torch.log(l21.abs().clamp(min=1e-12)) + log_t1 + log_t2
    log_trace_sigma = torch.logsumexp(torch.stack([log_t1, log_t2, log_t3], dim=-1), dim=-1)
    log_w_find = torch.log(w_find.clamp(min=1e-12))
    return torch.logsumexp(log_trace_sigma + log_w_find, dim=-1) - torch.logsumexp(log_w_find, dim=-1)


def _postprocess_grouppose_keypoints(
    out_keypoints_i: torch.Tensor,
    topk_boxes_i: torch.Tensor,
    labels_i: torch.Tensor,
    scores_i: torch.Tensor,
    target_size: torch.Tensor,
    num_keypoints_per_class: Sequence[int],
    trace_alpha: float,
):
    """Decode GroupPose keypoints for one image into pixel-space outputs.

    Ported faithfully from RF-DETR v1.8.0
    (``PostProcess._postprocess_keypoints`` / ``_decode_keypoints_for_image`` /
    ``_apply_keypoint_trace_fusion``, https://github.com/roboflow/rf-detr,
    Apache-2.0).

    The raw keypoint tensor for one image has shape
    ``(Q, num_classes * max_K, D)`` where ``D >= 7``. Channels are
    ``[x, y, findable_logit, visible_logit, log_l11, l21, log_l22, ...]`` with
    ``x``/``y`` normalized to ``[0, 1]``. For each selected query the predicted
    class slot is gathered, xy is scaled to original-image pixels, confidence is
    ``findable.sigmoid()``, and (when ``trace_alpha > 0``) object scores are
    multiplied by ``exp(-trace_alpha * log_mean_trace)``.

    Returns ``(scores_i, output_keypoints, output_keypoint_precision)`` where
    ``output_keypoints`` is ``(K, max_K, 3)`` ``(x_px, y_px, vis)`` and
    ``output_keypoint_precision`` is ``(K, max_K, 3)`` raw Cholesky params.
    """
    max_num_keypoints = max(num_keypoints_per_class, default=0)
    num_keypoint_classes = len(num_keypoints_per_class)

    keypoints_i = torch.gather(
        out_keypoints_i,
        0,
        topk_boxes_i.unsqueeze(-1).unsqueeze(-1).repeat(
            1, out_keypoints_i.shape[-2], out_keypoints_i.shape[-1]
        ),
    )

    output_keypoints = keypoints_i.new_zeros((keypoints_i.shape[0], max_num_keypoints, 3))
    output_keypoint_precision = keypoints_i.new_full(
        (keypoints_i.shape[0], max_num_keypoints, 3), float("nan")
    )

    if num_keypoint_classes == 0 or max_num_keypoints == 0:
        return scores_i, output_keypoints, output_keypoint_precision

    total_padded_keypoint_slots = keypoints_i.shape[1]
    if total_padded_keypoint_slots != num_keypoint_classes * max_num_keypoints:
        raise ValueError(
            f"keypoints padded slot dimension ({total_padded_keypoint_slots}) must equal "
            f"num_keypoint_classes ({num_keypoint_classes}) * max_num_keypoints ({max_num_keypoints})."
        )

    reshaped = keypoints_i.view(
        keypoints_i.shape[0], num_keypoint_classes, max_num_keypoints, keypoints_i.shape[-1]
    )
    valid_class_mask = labels_i < num_keypoint_classes
    if not valid_class_mask.any():
        return scores_i, output_keypoints, output_keypoint_precision

    valid_indices = valid_class_mask.nonzero(as_tuple=True)[0]
    selected_labels = labels_i[valid_indices]
    selected_keypoints = reshaped[valid_indices, selected_labels]
    has_precision = selected_keypoints.shape[-1] >= 7

    if trace_alpha > 0 and has_precision:
        log_mean_traces = selected_keypoints.new_zeros(selected_labels.shape[0])
        for class_idx in range(num_keypoint_classes):
            class_mask = selected_labels == class_idx
            if not class_mask.any():
                continue
            num_active = num_keypoints_per_class[class_idx]
            if num_active <= 0:
                continue
            log_mean_traces[class_mask] = _keypoint_log_mean_trace(
                selected_keypoints[class_mask, :num_active]
            )
        scores_i = scores_i.clone()
        scores_i[valid_indices] = scores_i[valid_indices] * torch.exp(-trace_alpha * log_mean_traces)

    img_h, img_w = target_size
    for class_idx in range(num_keypoint_classes):
        class_mask = selected_labels == class_idx
        if not class_mask.any():
            continue
        num_active = num_keypoints_per_class[class_idx]
        if num_active <= 0:
            continue
        out_idx = valid_indices[class_mask]
        active_keypoints = selected_keypoints[class_mask, :num_active]
        output_keypoints[out_idx, :num_active, 0] = active_keypoints[..., 0] * img_w
        output_keypoints[out_idx, :num_active, 1] = active_keypoints[..., 1] * img_h
        output_keypoints[out_idx, :num_active, 2] = active_keypoints[..., 2].sigmoid()
        if has_precision:
            output_keypoint_precision[out_idx, :num_active] = active_keypoints[..., 4:7]

    return scores_i, output_keypoints, output_keypoint_precision


def _postprocess_classic_keypoints(
    out_keypoints_i: torch.Tensor,
    topk_boxes_i: torch.Tensor,
    target_size: torch.Tensor,
) -> torch.Tensor:
    """Gather classic RF-DETR keypoints and scale them to pixel coordinates."""
    keypoints_i = torch.gather(
        out_keypoints_i,
        0,
        topk_boxes_i.unsqueeze(-1).unsqueeze(-1).repeat(
            1, out_keypoints_i.shape[-2], out_keypoints_i.shape[-1]
        ),
    ).clone()
    img_h, img_w = target_size
    keypoints_i[..., 0] = keypoints_i[..., 0] * img_w
    keypoints_i[..., 1] = keypoints_i[..., 1] * img_h
    keypoints_i[..., 2] = keypoints_i[..., 2].sigmoid()
    return keypoints_i


def postprocess(
    outputs: dict[str, torch.Tensor],
    target_sizes: torch.Tensor,
    num_select: int = 300,
    num_keypoints_per_class: Optional[Sequence[int]] = None,
    trace_alpha: float = 0.2,
    mask_score_threshold: Optional[float] = None,
) -> list[dict[str, torch.Tensor]]:
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
        num_keypoints_per_class: GroupPose keypoint schema (e.g. ``[0, 17]``); when
            given, ``pred_keypoints`` is decoded with the RF-DETR GroupPose path
            (predicted-class slot gather + pixel scaling + findable.sigmoid()
            confidence + trace-fusion scoring). Ignored for non-keypoint outputs.
        trace_alpha: Keypoint uncertainty fusion weight (default 0.2, matching
            RF-DETR). ``0`` disables fusion. Only used for GroupPose keypoints.

    Returns:
        List of dictionaries, one per image, each containing:
            - scores: Tensor of shape (num_select,) with confidence scores
            - labels: Tensor of shape (num_select,) with class IDs
            - boxes: Tensor of shape (num_select, 4) in xyxy format
    """
    out_logits = outputs["pred_logits"]  # (B, num_queries, num_classes)
    out_bbox = outputs["pred_boxes"]  # (B, num_queries, 4) in cxcywh [0, 1]
    out_masks = outputs.get("pred_masks")  # (B, num_queries, Hm, Wm) or None
    out_keypoints = outputs.get("pred_keypoints")  # (B, num_queries, K, 3) or None
    out_angles = outputs.get("pred_angles")  # (B, num_queries, 1) or None

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

    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).expand(-1, -1, 4))
    obb = None
    if out_angles is not None:
        obb_cxcywh = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).expand(-1, -1, 4))
        obb_angles = torch.gather(out_angles, 1, topk_boxes.unsqueeze(-1)).squeeze(-1)

    # Scale from relative [0, 1] to absolute [0, height/width] coordinates.
    # RF-DETR resizes rectangular images directly to a square canvas, so OBBs
    # must be transformed through corners before refitting xywhr.
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    if out_angles is not None:
        obb_rel = torch.cat((obb_cxcywh, obb_angles.unsqueeze(-1)), dim=-1)
        obb_rows = []
        for batch_idx in range(batch_size):
            scaled = scale_xywhr(
                obb_rel[batch_idx].detach().cpu().numpy(),
                float(img_w[batch_idx].detach().cpu().item()),
                float(img_h[batch_idx].detach().cpu().item()),
                min_size=1e-4,
            )
            obb_rows.append(
                torch.as_tensor(
                    scaled,
                    dtype=obb_cxcywh.dtype,
                    device=obb_cxcywh.device,
                )
            )
        obb_xywhr = torch.stack(obb_rows, dim=0)
        obb = torch.cat(
            (
                obb_xywhr,
                scores.unsqueeze(-1),
                labels.to(dtype=obb_xywhr.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )

    results = []
    for i in range(batch_size):
        res_i = {"scores": scores[i], "labels": labels[i], "boxes": boxes[i]}
        if obb is not None:
            res_i["obb"] = obb[i]

        if out_masks is not None:
            k_idx = topk_boxes[i]
            if mask_score_threshold is not None:
                keep = scores[i] > mask_score_threshold
                k_idx = k_idx[keep]
                res_i = {key: value[keep] for key, value in res_i.items()}
            masks_i = out_masks[i].index_select(0, k_idx)
            h, w = (int(value) for value in target_sizes[i].tolist())
            # Only one chunk of full-resolution float logits is live at a time.
            chunks = [
                F.interpolate(part.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)[:, 0] > 0.0
                for part in masks_i.split(32)
                if part.shape[0]
            ]
            res_i["masks"] = torch.cat(chunks) if chunks else masks_i.new_empty((0, h, w), dtype=torch.bool)

        if out_keypoints is not None and num_keypoints_per_class:
            # GroupPose keypoint decode (ported from RF-DETR v1.8.0). The raw
            # keypoints are (Q, num_classes * max_K, D); the predicted-class slot
            # is gathered per query, xy scaled to original pixels, confidence =
            # findable.sigmoid(), and trace fusion adjusts the object scores.
            # NOTE: the slot select uses ``labels[i]`` (the model's INTERNAL
            # GroupPose class index) so keypoint xy/conf stay byte-identical to
            # upstream; the emitted detection label is remapped below.
            scores_i, keypoints_i, precision_i = _postprocess_grouppose_keypoints(
                out_keypoints[i],
                topk_boxes[i],
                labels[i],
                scores[i],
                target_sizes[i],
                num_keypoints_per_class or [],
                trace_alpha,
            )

            # Class-index remap (LibreYOLO contiguous pose convention).
            #
            # The GroupPose schema ``num_keypoints_per_class`` (e.g. ``[0, 17]``)
            # makes the keypoint-bearing "person" class the INTERNAL index 1, so
            # the model emits detection label 1. LibreYOLO's person-only pose
            # convention is contiguous index 0 (nc=1, names={0: "person"}). Map
            # the internal predicted class to its position among the
            # keypoint-bearing classes (``kp_classes.index(internal)``), so
            # person -> 0. Detections whose predicted class is NOT a
            # keypoint-bearing class (e.g. internal class 0, the empty slot) are
            # not valid pose detections and are dropped -- upstream returns only
            # the person class. Keypoints/scores/boxes for the kept detections are
            # left byte-identical (they are only re-indexed, not recomputed).
            kp_classes = [
                cls_idx
                for cls_idx, count in enumerate(num_keypoints_per_class or [])
                if count > 0
            ]
            labels_i = labels[i]
            if kp_classes:
                kp_class_tensor = torch.as_tensor(
                    kp_classes, dtype=labels_i.dtype, device=labels_i.device
                )
                # remap[internal] = contiguous index, or -1 when not keypoint-bearing.
                remap = labels_i.new_full((num_classes,), -1)
                remap[kp_class_tensor] = torch.arange(
                    len(kp_classes), dtype=labels_i.dtype, device=labels_i.device
                )
                contiguous_labels = remap[labels_i]
                keep_kp = contiguous_labels >= 0
            else:
                # No keypoint-bearing class in the schema: nothing is a valid
                # pose detection. Keep the (empty-mask) filter consistent.
                contiguous_labels = labels_i
                keep_kp = labels_i.new_zeros(labels_i.shape, dtype=torch.bool)

            res_i["labels"] = contiguous_labels[keep_kp]
            res_i["boxes"] = boxes[i][keep_kp]
            # Trace fusion may rescale scores; keep boxes/scores/keypoints in
            # lockstep so the downstream confidence threshold filters all heads.
            res_i["scores"] = scores_i[keep_kp]
            res_i["keypoints"] = keypoints_i[keep_kp]
            res_i["keypoint_precision_cholesky"] = precision_i[keep_kp]
        elif out_keypoints is not None:
            res_i["keypoints"] = _postprocess_classic_keypoints(
                out_keypoints[i],
                topk_boxes[i],
                target_sizes[i],
            )

        results.append(res_i)

    return results
