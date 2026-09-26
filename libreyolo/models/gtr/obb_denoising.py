"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Oriented Contrastive Denoising (OCD) for the OBB task.

Follows the paper's best "box noise" mode (ai4rs noise_mode='only_xyxy'):
the oriented box (cx, cy, w, h, a) is perturbed on the (x1, y1, x2, y2) vertices
of its axis-aligned (cx, cy, w, h) part while the angle channel stays unchanged.
Positive queries get noise in [0, lambda1), negatives in [lambda1, lambda2)
via the DINO rand_part trick (box_noise_scale acts as lambda1, lambda2 = 2*lambda1).
"""

# Adapted for LibreYOLO from Intellindust-AI-Lab/GTR
# revision 782e737efe2e6437ac537fbdcee089673d3376c1 (MIT).
# Changes: local imports. See NOTICE.

import torch

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .utils import inverse_sigmoid


def get_obb_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    group_detr=1,
):
    """OBB version of DINO-style contrastive denoising; boxes are (N, 5) sigmoid-domain."""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device

    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        num_group = 0
    else:
        num_group = num_denoising // max_gt_num
        num_group = 1 if num_group == 0 else num_group
    bs = len(num_gts)

    total_num_group = num_group * group_detr

    input_query_class = torch.full(
        [bs, max_gt_num], num_classes, dtype=torch.int32, device=device
    )
    input_query_bbox = torch.zeros([bs, max_gt_num, 5], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * total_num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * total_num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * total_num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, total_num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(
        dn_positive_idx, [n * total_num_group for n in num_gts]
    )
    # total denoising queries
    num_dn_per_g = int(max_gt_num * 2 * num_group)
    num_denoising = num_dn_per_g * group_detr

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (
            label_noise_ratio * 0.5
        )
        new_label = torch.randint_like(
            mask, 0, num_classes, dtype=input_query_class.dtype
        )
        input_query_class = torch.where(
            mask & pad_gt_mask, new_label, input_query_class
        )

    if box_noise_scale > 0:
        # box noise mode: perturb (x1,y1,x2,y2) of the axis part; keep the angle channel.
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox[..., :4])
        diff = torch.tile(input_query_bbox[..., 2:4] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(known_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(known_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (
            1 - negative_gt_mask
        )
        known_bbox += rand_sign * rand_part * diff
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        noised_xywh = box_xyxy_to_cxcywh(known_bbox)
        noised_xywh[noised_xywh < 0] *= -1
        input_query_bbox = torch.cat([noised_xywh, input_query_bbox[..., 4:5]], dim=-1)
    input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    input_query_logits = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries * group_detr
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # Per Group DETR group: block other groups' DN plus apply DINO's intra-block mask.
    for g in range(group_detr):
        base = g * num_dn_per_g
        if g > 0:
            attn_mask[base : base + num_dn_per_g, :base] = True
        if g < group_detr - 1:
            attn_mask[
                base : base + num_dn_per_g, (g + 1) * num_dn_per_g : num_denoising
            ] = True
        # reconstruct cannot see each other (DINO contrastive mask inside the block)
        for i in range(num_group):
            row_start = base + max_gt_num * 2 * i
            row_end = base + max_gt_num * 2 * (i + 1)
            if i == 0:
                attn_mask[
                    row_start:row_end,
                    base + max_gt_num * 2 * (i + 1) : base + num_dn_per_g,
                ] = True
            if i == num_group - 1:
                attn_mask[row_start:row_end, base : base + max_gt_num * i * 2] = True
            else:
                attn_mask[
                    row_start:row_end,
                    base + max_gt_num * 2 * (i + 1) : base + num_dn_per_g,
                ] = True
                attn_mask[row_start:row_end, base : base + max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_group_detr": group_detr,
        "dn_num_split": [num_denoising, num_queries * group_detr],
    }

    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta
