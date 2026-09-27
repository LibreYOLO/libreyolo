"""DEKR bottom-up pose decoding.

Decoding arithmetic (hierarchical pooling, centre extraction, offset decode,
joint-score sampling, pose NMS and pose scoring) is adapted from
``src/super_gradients/training/utils/pose_estimation/dekr_decode_callbacks.py``
in ``Deci-AI/super-gradients`` at commit
``63de22c404d5740f34f7706c302b37fce3c8fe5d`` (Apache-2.0).

Two deliberate differences from upstream:

* Upstream's ``decode_one_sized_batch`` raises on any batch larger than one and
  its ``pose_nms`` asserts a single-element proposal list. This module loops
  explicitly at the outer boundary so every batch item is decoded, rather than
  exposing a batch-1 limitation as a public constraint.
* ``topk`` is clamped to the number of grid cells so small synthetic heatmaps
  decode instead of raising.

Everything numeric otherwise follows upstream, including the strict ``>``
comparison in the NMS close-joint count and the mean-over-joints pose score.

DEKR has no box head. :func:`derive_boxes_from_keypoints` is a LibreYOLO-side
adapter that fits a tight ``xyxy`` box to the confident decoded joints so the
family can return standard flat pose ``Results``; it is not an upstream
detection branch and must not be read as one.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = [
    "DEKR_KEYPOINT_THRESHOLD",
    "DEKR_MAX_NUM_PEOPLE",
    "DEKR_NMS_NUM_THRESHOLD",
    "DEKR_NMS_THRESHOLD",
    "DEKR_OUTPUT_STRIDE",
    "decode_poses",
    "derive_boxes_from_keypoints",
    "postprocess_dekr",
]

# Public decode defaults, from the pinned upstream
# ``DEKRPoseEstimationModel.get_post_prediction_callback``.
DEKR_OUTPUT_STRIDE = 4
DEKR_MAX_NUM_PEOPLE = 30
DEKR_KEYPOINT_THRESHOLD = 0.05
DEKR_NMS_THRESHOLD = 0.05
DEKR_NMS_NUM_THRESHOLD = 8

_POOL_THRESHOLD1 = 300
_POOL_THRESHOLD2 = 200


def _hierarchical_pool(heatmap: Tensor) -> Tensor:
    """Local-maximum pooling with a kernel chosen by feature-map scale."""
    map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
    if map_size > _POOL_THRESHOLD1:
        kernel, padding = 7, 3
    elif map_size > _POOL_THRESHOLD2:
        kernel, padding = 5, 2
    else:
        kernel, padding = 3, 1
    return F.max_pool2d(heatmap[None], kernel_size=kernel, stride=1, padding=padding)


def _get_maximum_from_heatmap(
    center_heatmap: Tensor,
    max_num_people: int,
    pose_center_score_threshold: float,
) -> Tuple[Tensor, Tensor]:
    """Return flat indices and scores of surviving centre peaks."""
    pooled = _hierarchical_pool(center_heatmap)
    peaks = torch.eq(pooled, center_heatmap).float()
    suppressed = center_heatmap * peaks
    scores = suppressed.view(-1)
    # Upstream calls topk(max_num_people) unguarded; a 640x640 input always has
    # 25600 cells so it never trips there, but synthetic heads can be smaller.
    k = min(int(max_num_people), scores.numel())
    scores, positions = scores.topk(k)

    selected = (scores > pose_center_score_threshold).nonzero()
    return positions[selected][:, 0], scores[selected][:, 0]


def _up_interpolate(x: Tensor, size: Tuple[int, int]) -> Tensor:
    """Upstream's stride-aware upsample: align-corners resize then edge pad."""
    height, width = x.shape[2], x.shape[3]
    scale_h = int(size[0] / height)
    scale_w = int(size[1] / width)
    resized = F.interpolate(
        x,
        size=[size[0] - scale_h + 1, size[1] - scale_w + 1],
        align_corners=True,
        mode="bilinear",
    )
    return F.pad(resized, (0, scale_w - 1, 0, scale_h - 1), mode="replicate")


def _offsets_to_poses(offset: Tensor) -> Tensor:
    """Turn one ``(2K, H, W)`` offset map into ``(H * W, K, 2)`` grid coordinates."""
    num_offset, height, width = offset.shape
    num_joints = num_offset // 2
    reshaped = offset.permute(1, 2, 0).reshape(height * width, num_joints, 2)

    shifts_x = torch.arange(0, width, dtype=torch.float32, device=offset.device)
    shifts_y = torch.arange(0, height, dtype=torch.float32, device=offset.device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    locations = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1)), dim=1)
    return locations[:, None, :] - reshaped


def _pose_area_squared(poses: Tensor) -> Tensor:
    """Squared diagonal proxy ``w^2 + h^2`` of each pose's keypoint extent."""
    width = poses[:, :, 0].max(-1)[0] - poses[:, :, 0].min(-1)[0]
    height = poses[:, :, 1].max(-1)[0] - poses[:, :, 1].min(-1)[0]
    return width * width + height * height


def _nms_core(
    pose_coord: Tensor,
    heat_score: Tensor,
    nms_threshold: float,
    nms_num_threshold: int,
) -> List[int]:
    """Suppress poses that share too many near-coincident joints."""
    num_people, num_joints, _ = pose_coord.shape
    pose_area = _pose_area_squared(pose_coord)[:, None].repeat(
        1, num_people * num_joints
    )
    pose_area = pose_area.reshape(num_people, num_people, num_joints)

    pose_diff = pose_coord[:, None, :, :] - pose_coord
    pose_dist = pose_diff.pow(2).sum(3).sqrt()
    pose_threshold = nms_threshold * torch.sqrt(pose_area)
    close_joints = (pose_dist < pose_threshold).sum(2)
    # Strictly greater than upstream; keep it that way.
    nms_pose = close_joints > nms_num_threshold

    ignored: set[int] = set()
    keep: List[int] = []
    for i in range(nms_pose.shape[0]):
        if i in ignored:
            continue
        keep_inds = [int(v) for v in nms_pose[i].nonzero().flatten().tolist()]
        if not keep_inds:
            continue
        best = keep_inds[int(torch.argmax(heat_score[keep_inds]))]
        if best in ignored:
            continue
        keep.append(best)
        ignored.update(keep_inds)
    return keep


def _sample_joint_scores(pose_coord: Tensor, heatmap: Tensor) -> Tensor:
    """Read each decoded joint's own heatmap channel at its floored position."""
    _, height, width = heatmap.shape
    per_joint = heatmap[:-1].flatten(1, 2).transpose(0, 1)
    rows = torch.clamp(torch.floor(pose_coord[:, :, 1]), 0, height - 1).long()
    cols = torch.clamp(torch.floor(pose_coord[:, :, 0]), 0, width - 1).long()
    return torch.gather(per_joint, 0, rows * width + cols).unsqueeze(-1)


def _decode_single(
    heatmap: Tensor,
    offset: Tensor,
    *,
    output_stride: int,
    max_num_people: int,
    keypoint_threshold: float,
    nms_threshold: float,
    nms_num_threshold: int,
    min_confidence: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode one image's ``(K + 1, H, W)`` heatmap and ``(2K, H, W)`` offsets."""
    num_joints = offset.shape[0] // 2
    height, width = heatmap.shape[1], heatmap.shape[2]

    posemap = _offsets_to_poses(offset)
    heatmap_full = _up_interpolate(
        heatmap[None], size=(output_stride * height, output_stride * width)
    )[0]

    positions, center_scores = _get_maximum_from_heatmap(
        heatmap[-1:],
        max_num_people=max_num_people,
        pose_center_score_threshold=keypoint_threshold,
    )

    empty = (
        np.zeros((0, num_joints, 3), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )
    if positions.numel() == 0:
        return empty

    pose_coord = output_stride * posemap[positions]
    center_scores = center_scores[:, None, None].expand(-1, num_joints, 1)

    joint_scores = _sample_joint_scores(pose_coord, heatmap_full)
    heat_score = (joint_scores.sum(dim=1) / num_joints)[:, 0]
    pose_score = center_scores * joint_scores

    keep = _nms_core(pose_coord, heat_score, nms_threshold, nms_num_threshold)
    if not keep:
        return empty

    poses = torch.cat([pose_coord, pose_score], dim=2)[keep]
    heat_score = heat_score[keep]
    if len(keep) > max_num_people:
        _, top = torch.topk(heat_score, max_num_people)
        poses = poses[top]

    poses_np = poses.detach().float().cpu().numpy()
    scores_np = poses_np[:, :, 2].mean(axis=1)
    mask = scores_np >= min_confidence
    if not mask.any():
        return empty
    return poses_np[mask], scores_np[mask].astype(np.float32)


@torch.no_grad()
def decode_poses(
    heatmap_logits: Tensor,
    offsets: Tensor,
    *,
    output_stride: int = DEKR_OUTPUT_STRIDE,
    max_num_people: int = DEKR_MAX_NUM_PEOPLE,
    keypoint_threshold: float = DEKR_KEYPOINT_THRESHOLD,
    nms_threshold: float = DEKR_NMS_THRESHOLD,
    nms_num_threshold: int = DEKR_NMS_NUM_THRESHOLD,
    min_confidence: float = 0.0,
    apply_sigmoid: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Decode a whole batch of raw DEKR outputs into per-image poses.

    Returns one ``(poses (P, K, 3), scores (P,))`` pair per batch item, in
    network-input pixel coordinates.
    """
    if heatmap_logits.ndim != 4 or offsets.ndim != 4:
        raise ValueError(
            "DEKR decode expects 4D (B, C, H, W) heatmap and offset tensors, got "
            f"{tuple(heatmap_logits.shape)} and {tuple(offsets.shape)}"
        )
    if heatmap_logits.shape[0] != offsets.shape[0]:
        raise ValueError(
            f"batch mismatch: heatmap has {heatmap_logits.shape[0]} items, "
            f"offsets have {offsets.shape[0]}"
        )
    if offsets.shape[1] % 2 != 0:
        raise ValueError(f"offset channels must be even, got {offsets.shape[1]}")
    if heatmap_logits.shape[1] != offsets.shape[1] // 2 + 1:
        raise ValueError(
            f"heatmap has {heatmap_logits.shape[1]} channels but offsets imply "
            f"{offsets.shape[1] // 2} keypoints plus one centre channel"
        )
    if heatmap_logits.shape[2:] != offsets.shape[2:]:
        raise ValueError(
            f"spatial mismatch: heatmap {tuple(heatmap_logits.shape[2:])} vs "
            f"offsets {tuple(offsets.shape[2:])}"
        )

    # Peak comparison and NMS distances are numerically sensitive; promote
    # half/bfloat16 model output before decoding.
    heatmap = heatmap_logits.float()
    offset = offsets.float()
    if apply_sigmoid:
        heatmap = heatmap.sigmoid()

    return [
        _decode_single(
            heatmap[i],
            offset[i],
            output_stride=output_stride,
            max_num_people=max_num_people,
            keypoint_threshold=keypoint_threshold,
            nms_threshold=nms_threshold,
            nms_num_threshold=nms_num_threshold,
            min_confidence=min_confidence,
        )
        for i in range(heatmap.shape[0])
    ]


def derive_boxes_from_keypoints(
    keypoints: np.ndarray,
    image_size: Tuple[int, int],
    keypoint_threshold: float = DEKR_KEYPOINT_THRESHOLD,
) -> np.ndarray:
    """Fit a tight ``xyxy`` person box to each pose's confident joints.

    DEKR predicts no boxes. This is a LibreYOLO Results adapter, not an upstream
    regression branch. Joints scoring above ``keypoint_threshold`` define the
    extent; a pose with fewer than two such joints falls back to all of its
    finite joints. Boxes are clipped to the original canvas.
    """
    width, height = int(image_size[0]), int(image_size[1])
    boxes = np.zeros((keypoints.shape[0], 4), dtype=np.float32)
    for index, pose in enumerate(keypoints):
        finite = np.isfinite(pose[:, 0]) & np.isfinite(pose[:, 1])
        confident = finite & (pose[:, 2] > keypoint_threshold)
        selected = confident if int(confident.sum()) >= 2 else finite
        if not selected.any():
            continue
        xs, ys = pose[selected, 0], pose[selected, 1]
        boxes[index] = (xs.min(), ys.min(), xs.max(), ys.max())
    np.clip(boxes[:, 0::2], 0, width, out=boxes[:, 0::2])
    np.clip(boxes[:, 1::2], 0, height, out=boxes[:, 1::2])
    return boxes


def postprocess_dekr(
    raw: Sequence[Tensor],
    conf_thres: float,
    original_size: Tuple[int, int],
    ratio: float,
    *,
    max_det: int = DEKR_MAX_NUM_PEOPLE,
    keypoint_threshold: float = DEKR_KEYPOINT_THRESHOLD,
    nms_threshold: float = DEKR_NMS_THRESHOLD,
    nms_num_threshold: int = DEKR_NMS_NUM_THRESHOLD,
    output_stride: int = DEKR_OUTPUT_STRIDE,
    apply_sigmoid: bool = True,
    batch_index: int = 0,
) -> dict:
    """Decode raw DEKR output for one image into the LibreYOLO pose contract.

    ``conf_thres`` is the minimum mean-joint pose score, matching upstream's
    ``min_confidence``. ``keypoint_threshold`` stays a separate knob because it
    gates centre candidates before scoring, exactly as upstream.
    """
    heatmap_logits, offsets = raw[0], raw[1]
    decoded = decode_poses(
        heatmap_logits,
        offsets,
        output_stride=output_stride,
        max_num_people=max(1, int(max_det)),
        keypoint_threshold=keypoint_threshold,
        nms_threshold=nms_threshold,
        nms_num_threshold=nms_num_threshold,
        min_confidence=float(conf_thres),
        apply_sigmoid=apply_sigmoid,
    )
    poses, scores = decoded[batch_index]

    width, height = int(original_size[0]), int(original_size[1])
    keypoints = poses.astype(np.float32, copy=True)
    if keypoints.size:
        # Padding is anchored top-left, so undoing the letterbox is a single
        # division by the resize scale with no offset to subtract.
        keypoints[:, :, :2] /= float(ratio)
        np.clip(keypoints[:, :, 0], 0, width, out=keypoints[:, :, 0])
        np.clip(keypoints[:, :, 1], 0, height, out=keypoints[:, :, 1])

    boxes = derive_boxes_from_keypoints(
        keypoints, (width, height), keypoint_threshold=keypoint_threshold
    )
    return {
        "num_detections": int(keypoints.shape[0]),
        "boxes": boxes,
        "scores": scores.astype(np.float32),
        "classes": np.zeros((keypoints.shape[0],), dtype=np.int64),
        "keypoints": keypoints,
    }
