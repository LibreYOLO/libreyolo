"""Matched-mask sampling adapted from RF-DETR.

Upstream: roboflow/rf-detr, 2d319776673ba840c069b243863a4cbe3a62cb58,
Apache-2.0. Copyright (c) 2025 Roboflow, Inc.
Modified for LibreYOLO's sampler import and non-finite-coordinate fallback.
"""

import torch
from torch import Tensor

from .segmentation import point_sample

# The CPU benchmark first crossed over at 256x256 masks (1,048,576 elements):
# 96x96, 128x128, and 192x192 direct gathers were 1.4x slower than point_sample,
# while 256x256 was 1.06x faster and the 312x312 workload was 2.48x faster.
_MIN_DIRECT_MASK_ELEMENTS = 1 << 20
# Require enough mask work per sampled value to amortize direct indexing's fixed
# bookkeeping.  Sixteen is a conservative workload-ratio heuristic, not a
# standalone crossover measurement; the benchmarked production workload clears it.
_MIN_DIRECT_MASK_ELEMENTS_PER_POINT = 16
# One-match groups were 1.25x-1.5x slower in repeated measurements because the
# per-image slicing, index transfer, and gather overhead was not amortized.
_MIN_DIRECT_MATCHES_PER_GROUP = 2


def sample_target_masks_at_points(
    targets: list[dict[str, Tensor]],
    indices: list[tuple[Tensor, Tensor]],
    point_coords: Tensor,
) -> Tensor:
    """Sample matched ground-truth masks at normalized point coordinates.

    Large contiguous masks on CPU are indexed directly, avoiding the
    full matched-mask copies created by advanced indexing and concatenation.
    Eligible multi-image CUDA boolean masks are sampled per image with the native
    nearest-neighbor ``point_sample`` path, avoiding a full batch of matched float masks.
    Other inputs retain the existing concatenation and sampling path.

    Args:
        targets: Per-image target dictionaries containing ``masks`` tensors.
        indices: Per-image matched source and target indices.
        point_coords: Normalized coordinates with shape ``[matches, points, 2]``.

    Returns:
        Sampled float labels with shape ``[matches, points]``.

    Examples:
        >>> masks = torch.tensor([[[False, True], [True, False]]])
        >>> matched = torch.tensor([0])
        >>> coords = torch.tensor([[[0.75, 0.25]]])
        >>> sample_target_masks_at_points([{"masks": masks}], [(matched, matched)], coords)
        tensor([[1.]])
    """
    use_direct = (
        len(targets) == len(indices)
        and point_coords.device.type == "cpu"
        and point_coords.dtype == torch.float32
        and point_coords.ndim == 3
        and point_coords.shape[-1] == 2
    )
    mask_shape: tuple[int, int] | None = None
    matched_mask_elements = 0
    matched_count = 0
    # The direct path pays a fixed per-image loop-iteration cost (slicing, index computation,
    # a device transfer, a gather). A large AGGREGATE element count can hide many small per-image
    # groups whose individual gather is too cheap to be worth that fixed cost -- tracking the
    # smallest non-empty group lets the guard reject that case even though the total clears the floor.
    # This alone is not enough: a single large mask (e.g. 300x300) with only one match per image
    # clears the element floor on its own while doing negligible gather work, so the fixed
    # per-iteration overhead dominates regardless of resolution -- measured a stable ~1.25-1.5x
    # regression across 1-8 images, all with exactly one match per group. Tracking the smallest
    # non-empty group's MATCH COUNT (independent of mask resolution) catches that case too.
    min_group_elements: int | None = None
    min_group_count: int | None = None

    if use_direct:
        for target, (_, target_indices) in zip(targets, indices):
            masks = target.get("masks")
            current_shape = (
                (masks.shape[-2], masks.shape[-1])
                if isinstance(masks, Tensor) and masks.ndim == 3
                else None
            )
            if (
                masks is None
                or current_shape is None
                or not masks.is_contiguous()
                or masks.device != point_coords.device
                or target_indices.device.type != "cpu"
                or target_indices.dtype != torch.int64
                or target_indices.ndim != 1
                or (mask_shape is not None and current_shape != mask_shape)
            ):
                use_direct = False
                break
            mask_shape = current_shape
            group_count = target_indices.numel()
            matched_count += group_count
            group_elements = group_count * current_shape[0] * current_shape[1]
            matched_mask_elements += group_elements
            if group_count > 0:
                min_group_elements = (
                    group_elements
                    if min_group_elements is None
                    else min(min_group_elements, group_elements)
                )
                min_group_count = (
                    group_count
                    if min_group_count is None
                    else min(min_group_count, group_count)
                )

    sampled_elements = (
        point_coords.shape[0] * point_coords.shape[1] if point_coords.ndim == 3 else 0
    )
    use_direct = (
        use_direct
        and matched_count == point_coords.shape[0]
        and matched_mask_elements >= _MIN_DIRECT_MASK_ELEMENTS
        and matched_mask_elements
        >= _MIN_DIRECT_MASK_ELEMENTS_PER_POINT * sampled_elements
        and (
            min_group_elements is None
            or min_group_elements >= _MIN_DIRECT_MASK_ELEMENTS
        )
        and (
            min_group_count is None or min_group_count >= _MIN_DIRECT_MATCHES_PER_GROUP
        )
    )

    if use_direct:
        use_direct = all(
            not bool((target_indices < 0).any())
            and not bool((target_indices >= target["masks"].shape[0]).any())
            for target, (_, target_indices) in zip(targets, indices)
        )

    if use_direct and not torch.isfinite(point_coords).all():
        use_direct = False

    if use_direct:
        sampled_masks = []
        offset = 0
        for target, (_, target_indices) in zip(targets, indices):
            masks = target["masks"]
            count = target_indices.numel()
            coords = point_coords[offset : offset + count]
            height, width = masks.shape[-2:]

            # Reproduce point_sample's normalization order exactly before applying
            # nearest-neighbor rounding and border padding.
            grid = 2.0 * coords - 1.0
            unnorm_x = ((grid[..., 0] + 1.0) * width - 1.0) / 2.0
            unnorm_y = ((grid[..., 1] + 1.0) * height - 1.0) / 2.0
            x_coords = torch.round(unnorm_x).to(torch.int64)
            y_coords = torch.round(unnorm_y).to(torch.int64)
            x_coords.clamp_(0, width - 1)
            y_coords.clamp_(0, height - 1)

            target_indices_device = target_indices.to(device=masks.device)
            flat_indices = (
                target_indices_device[:, None] * (height * width)
                + y_coords * width
                + x_coords
            )
            sampled = (
                masks.reshape(-1)
                .gather(0, flat_indices.reshape(-1))
                .reshape(count, point_coords.shape[1])
                .float()
            )

            # PyTorch's compiled grid_sampler kernel used by ``point_sample`` does not agree with
            # ``torch.round`` on every (coordinate, mask size) combination at an exact pixel-center tie
            # (fractional part == 0.5) -- both compute the same mathematical formula, but float32
            # evaluation order inside the kernel can round a tie to the opposite integer for some sizes
            # and not others (verified: it agrees for width=96, not for width=673, on the identical
            # unnormalized value 0.5). A fine sweep around a known divergence found mismatches only where
            # the computed value was bit-exact at the tie, never in its neighborhood, and 2,000,000 generic
            # random coordinates produced zero mismatches -- so exact ties are the only risk, and real
            # point sets of a few hundred points routinely contain one. Falling back to ``point_sample``
            # for the WHOLE call over one tied point among thousands would give away most of this
            # optimization's benefit for no reason: correct just the tied points instead.
            is_tie = (unnorm_x - torch.floor(unnorm_x) == 0.5) | (
                unnorm_y - torch.floor(unnorm_y) == 0.5
            )
            if bool(is_tie.any()):
                tie_rows, tie_cols = is_tie.nonzero(as_tuple=True)
                tie_masks = masks[target_indices_device[tie_rows]]
                tie_coords = coords[tie_rows, tie_cols]
                corrected = (
                    point_sample(
                        tie_masks.unsqueeze(1).float(),
                        tie_coords.unsqueeze(1),
                        align_corners=False,
                        mode="nearest",
                    )
                    .squeeze(1)
                    .squeeze(1)
                )
                sampled = sampled.clone()
                sampled[tie_rows, tie_cols] = corrected.to(device=sampled.device)

            sampled_masks.append(sampled)
            offset += count

        return torch.cat(sampled_masks, dim=0)

    if (
        point_coords.is_cuda
        and point_coords.ndim == 3
        and len(targets) == len(indices) > 1
        and all(
            target["masks"].ndim == 3
            and target["masks"].dtype == torch.bool
            and target["masks"].device == point_coords.device
            and target["masks"].shape[1:] == targets[0]["masks"].shape[1:]
            and target_indices.ndim == 1
            and target_indices.dtype == torch.int64
            for target, (_, target_indices) in zip(targets, indices)
        )
        and sum(target_indices.numel() for _, target_indices in indices)
        == point_coords.shape[0]
        > 0
    ):
        # Keep the native sampler's rounding. In loss_masks' no_grad context, each
        # image's full float masks can be released before sampling the next image.
        sampled_masks = []
        offset = 0
        for target, (_, target_indices) in zip(targets, indices):
            count = target_indices.numel()
            if count:
                sampled_masks.append(
                    point_sample(
                        target["masks"][target_indices].unsqueeze(1).float(),
                        point_coords[offset : offset + count],
                        align_corners=False,
                        mode="nearest",
                    ).squeeze(1)
                )
            offset += count
        return torch.cat(sampled_masks, dim=0)

    target_masks = torch.cat(
        [
            target["masks"][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ]
    )
    return point_sample(
        target_masks.unsqueeze(1).float(),
        point_coords,
        align_corners=False,
        mode="nearest",
    ).squeeze(1)
