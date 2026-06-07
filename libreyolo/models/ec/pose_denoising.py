"""Contrastive denoising (CDN) for EC pose training.

Faithful, device-aware port of DETRPose's ``dn_component.py`` (Apache-2.0),
itself derived from DINO / DN-DETR (both Apache-2.0). EdgeCrafter's ECPose was
trained with this denoising group, which is why the pretrained checkpoints carry
``label_enc`` / ``pose_enc`` embeddings — this module makes a fine-tune use them
instead of leaving them without gradient.

The only changes vs upstream are mechanical: ``.cuda()`` -> ``device=`` so the
group is built on the model's device, and ``inverse_sigmoid`` imported from the
EC utils.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .utils import inverse_sigmoid


def get_sigmas(
    num_keypoints: int, device, sigmas: Sequence[float] | None = None
) -> torch.Tensor:
    """COCO OKS sigmas (with a prepended center sigma), shape ``(1, K+1, 1)``."""
    if sigmas is not None:
        if len(sigmas) != num_keypoints:
            raise ValueError(
                f"sigmas has {len(sigmas)} entries but num_keypoints={num_keypoints}"
            )
        sigmas = np.asarray([float(s) for s in sigmas], dtype=np.float32)
    elif num_keypoints == 17:
        sigmas = np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
             1.07, .87, .87, .89, .89], dtype=np.float32) / 10.0
    elif num_keypoints == 14:
        sigmas = np.array(
            [.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
             .79, .79], dtype=np.float32) / 10.0
    elif num_keypoints == 3:
        sigmas = np.array([1.07, 1.07, 0.67], dtype=np.float32) / 10.0
    else:
        raise ValueError(f"Unsupported keypoints number {num_keypoints}")
    sigmas = np.concatenate([[0.1], sigmas])  # center of the human
    sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
    return sigmas[None, :, None]


def prepare_for_cdn(
    dn_args,
    training,
    num_queries,
    num_classes,
    num_keypoints,
    hidden_dim,
    label_enc,
    pose_enc,
    img_dim,
    device,
    sigmas: Sequence[float] | None = None,
):
    """Build the contrastive denoising query group from GT keypoints.

    Returns ``(input_query_label, input_query_pose, attn_mask, dn_meta)`` where
    ``input_query_label`` is ``(B, pad_size, K+1, hidden)`` and
    ``input_query_pose`` is ``(B, pad_size, K+1, 2)`` in logit space.
    """
    if not training:
        return None, None, None, None

    targets, dn_number, label_noise_ratio = dn_args
    dn_number = dn_number * 2  # positive + negative
    known = [(torch.ones_like(t["labels"])) for t in targets]
    batch_size = len(known)
    known_num = [sum(k) for k in known]

    if int(max(known_num)) == 0:
        return None, None, None, None

    dn_number = dn_number // (int(max(known_num) * 2))
    dn_number = 1 if dn_number == 0 else dn_number

    unmask_bbox = unmask_label = torch.cat(known)

    labels = torch.cat([t["labels"] for t in targets])
    batch_idx = torch.cat(
        [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
    )

    known_indice = torch.nonzero(unmask_label + unmask_bbox).view(-1)
    known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)

    known_labels = labels.repeat(2 * dn_number, 1).view(-1)
    known_labels_expaned = known_labels.clone()
    known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)

    if label_noise_ratio > 0:
        p = torch.rand_like(known_labels_expaned.float())
        chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)
        new_label = torch.randint_like(chosen_indice, 0, num_classes)
        known_labels_expaned.scatter_(0, chosen_indice, new_label)

    # keypoint noise
    boxes = torch.cat([t["boxes"] for t in targets])
    xy = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    keypoints = torch.cat([t["keypoints"] for t in targets])
    if "area" in targets[0]:
        areas = torch.cat([t["area"] for t in targets])
    else:
        areas = boxes[:, 2] * boxes[:, 3] * 0.53
    poses = keypoints[:, 0:(num_keypoints * 2)]
    poses = torch.cat([xy, poses], dim=1)
    non_viz = keypoints[:, (num_keypoints * 2):] == 0
    non_viz = torch.cat((torch.ones_like(non_viz[:, 0:1]).bool(), non_viz), dim=1)
    vars_ = (2 * get_sigmas(num_keypoints, device, sigmas=sigmas)) ** 2

    known_poses = poses.repeat(2 * dn_number, 1).reshape(-1, num_keypoints + 1, 2)
    known_areas = areas.repeat(2 * dn_number)[..., None, None]  # normalized [0,1]
    known_areas = known_areas * img_dim[0] * img_dim[1]  # scaled [0, h*w]
    known_non_viz = non_viz.repeat(2 * dn_number, 1)

    single_pad = int(max(known_num))
    pad_size = int(single_pad * 2 * dn_number)
    positive_idx = (
        torch.arange(len(poses), device=device).long().unsqueeze(0).repeat(dn_number, 1)
    )
    positive_idx += (
        (torch.arange(dn_number, device=device) * len(poses) * 2).long().unsqueeze(1)
    )
    positive_idx = positive_idx.flatten()
    negative_idx = positive_idx + len(poses)

    eps = np.finfo("float32").eps
    rand_vector = torch.rand_like(known_poses)
    rand_vector = F.normalize(rand_vector, dim=-1)  # ||rand_vector|| = 1
    rand_alpha = torch.zeros_like(known_poses[..., :1]).uniform_(-np.log(1), -np.log(0.5))
    rand_alpha[negative_idx] = rand_alpha[negative_idx].uniform_(-np.log(0.5), -np.log(0.1))
    rand_alpha = rand_alpha * 2 * (known_areas + eps) * vars_  # distance**2
    rand_alpha = torch.sqrt(rand_alpha) / max(img_dim)
    rand_alpha[known_non_viz] = 0.0

    known_poses_expand = known_poses + rand_alpha * rand_vector

    m = known_labels_expaned.long().to(device)
    input_label_embed = label_enc(m)
    input_label_pose_embed = pose_enc.weight[None].repeat(known_poses_expand.size(0), 1, 1)
    input_label_embed = torch.cat(
        [input_label_embed.unsqueeze(1), input_label_pose_embed], dim=1
    )
    input_label_embed = input_label_embed.flatten(1)

    input_pose_embed = inverse_sigmoid(known_poses_expand)

    padding_label = torch.zeros(
        pad_size, hidden_dim * (num_keypoints + 1), device=device
    )
    padding_pose = torch.zeros(pad_size, num_keypoints + 1, device=device)

    input_query_label = padding_label.repeat(batch_size, 1, 1)
    input_query_pose = padding_pose[..., None].repeat(batch_size, 1, 1, 2)

    map_known_indice = torch.tensor([], device=device)
    if len(known_num):
        map_known_indice = torch.cat(
            [torch.arange(num, device=device) for num in known_num]
        )
        map_known_indice = torch.cat(
            [map_known_indice + single_pad * i for i in range(2 * dn_number)]
        ).long()
    if len(known_bid):
        input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
        input_query_pose[(known_bid.long(), map_known_indice)] = input_pose_embed

    tgt_size = pad_size + num_queries
    attn_mask = torch.ones(tgt_size, tgt_size, device=device) < 0
    # match query cannot see the reconstruct
    attn_mask[pad_size:, :pad_size] = True
    # reconstruct groups cannot see each other
    for i in range(dn_number):
        if i == 0:
            attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                      single_pad * 2 * (i + 1):pad_size] = True
        if i == dn_number - 1:
            attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                      :single_pad * i * 2] = True
        else:
            attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                      single_pad * 2 * (i + 1):pad_size] = True
            attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                      :single_pad * 2 * i] = True

    dn_meta = {"pad_size": pad_size, "num_dn_group": dn_number}

    return (
        input_query_label.unflatten(-1, (-1, hidden_dim)),
        input_query_pose,
        attn_mask,
        dn_meta,
    )
