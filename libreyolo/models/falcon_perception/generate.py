# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from https://github.com/tiiuae/Falcon-Perception (Apache-2.0),
# commit c457916c9974efbacfa91f0f6ecc2c49c6543e56:
#   falcon_perception/{batch_inference,aux_output,attention,visualization_utils}.py
#   (pair_bbox_entries), demo/perception_single.py (stop tokens, budgets)
# See libreyolo/models/falcon_perception/NOTICE.
"""Single-image greedy generation for Falcon Perception.

Each decode step emits one token; a ``<|coord|>`` token yields a box centre, a
``<|size|>`` token its extent, and a ``<|seg|>`` token a mask query whose
embedding is dotted with the upsampled image features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor

from .nn import FalconPerception, KVCache
from .preprocess import HR_UPSAMPLE_RATIO

MASK_THRESHOLD = 0.3


@dataclass
class ImgScatterEntry:
    batch_idx: int
    token_start: int
    n_tokens: int
    h_valid_patches: int
    w_valid_patches: int

    def __iter__(self):
        return iter(
            (
                self.batch_idx,
                self.token_start,
                self.n_tokens,
                self.h_valid_patches,
                self.w_valid_patches,
            )
        )


@dataclass
class Generation:
    """Decoded structure of one generation."""

    tokens: Tensor  # (L + n_new,) full token stream
    boxes_raw: List[dict] = field(default_factory=list)  # interleaved {x,y} / {h,w}
    masks: Optional[Tensor] = None  # (K, h, w) bool at feature resolution, or None


def build_prompt_mask(tokens_L: Tensor, soi_id: int, eoi_id: int) -> Tensor:
    """Full (L, L) keep-mask for the prompt: causal OR bidirectional image block.

    The image block spans ``<|image_cls|>`` .. just before ``<|end_of_image|>``.
    """
    L = tokens_L.shape[0]
    idx = torch.arange(L, device=tokens_L.device)
    causal = idx[:, None] >= idx[None, :]

    acc_soi = torch.cumsum(tokens_L == soi_id, dim=0)
    acc_eoi = torch.cumsum(tokens_L == eoi_id, dim=0)
    img_mask = (acc_soi - acc_eoi) > 0
    img_indices = acc_soi * img_mask
    same_image = (
        img_mask[:, None] & img_mask[None, :] & (img_indices[:, None] == img_indices[None, :])
    )
    return causal | same_image


@torch.inference_mode()
def generate(
    model: FalconPerception,
    tokenizer,
    tokens: Tensor,  # (1, L) int64
    pos_t: Tensor,  # (1, L)
    pos_hw: Tensor,  # (1, L, 2)
    pixel_values: Tensor,  # (1, 1, H, W, 3)
    pixel_mask: Tensor,  # (1, 1, H, W) bool
    *,
    segmentation: bool,
    max_new_tokens: int = 2048,
    stop_token_ids: Optional[List[int]] = None,
    coord_dedup_threshold: float = 0.01,
    block_size: int = 128,
    shrink_image: bool = True,
) -> Generation:
    args = model.args
    device = tokens.device
    B, L = tokens.size()
    assert B == 1, "single-image generation only"

    S = (L + max_new_tokens + block_size - 1) // block_size * block_size
    if S > args.max_seq_len:
        raise ValueError(f"max generation length {S} exceeds model max_seq_len {args.max_seq_len}")

    kv_cache = KVCache(
        max_batch_size=B,
        max_seq_length=S,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        num_layers=args.n_layers,
    )

    ps = args.spatial_patch_size
    img_pos = (tokens[0] == args.img_id).nonzero(as_tuple=True)[0]
    img_scatter_info: List[ImgScatterEntry] = []
    if len(img_pos) > 0:
        mask_b = pixel_mask[0]
        h_v = int(mask_b.sum(dim=-2).max()) // ps
        w_v = int(mask_b.sum(dim=-1).max()) // ps
        img_scatter_info.append(ImgScatterEntry(0, int(img_pos[0]), len(img_pos), h_v, w_v))

    prompt_mask = build_prompt_mask(tokens[0], tokenizer.image_cls_token_id, tokenizer.end_of_image_token_id)

    # Prefill
    logits_BSV, h_BSD = model(
        tokens=tokens,
        mask=prompt_mask[None, None],
        kv_cache=kv_cache,
        rope_pos_t=pos_t,
        rope_pos_hw=pos_hw,
        pixel_values=pixel_values,
        img_scatter_info=img_scatter_info or None,
    )

    hr_image_features = None
    if segmentation:
        _, _, H, W, _ = pixel_values.shape
        output_size = (H // ps * HR_UPSAMPLE_RATIO, W // ps * HR_UPSAMPLE_RATIO)
        entry = img_scatter_info[0]
        hr_image_features = model.upsample_single_img_features(
            h_BSD[entry.batch_idx],
            pixel_values[0],
            img_token_start=entry.token_start,
            h_valid=entry.h_valid_patches,
            w_valid=entry.w_valid_patches,
            output_size=output_size,
            shrink_image=shrink_image,
        )[None]

    padded_tokens = torch.full((1, S), tokenizer.pad_token_id, dtype=tokens.dtype, device=device)
    padded_tokens[:, :L] = tokens

    stop_ids = torch.tensor(stop_token_ids or [tokenizer.eos_token_id], device=device)

    boxes_raw: List[dict] = []
    segm_embeds: List[Tensor] = []
    xy_cat: Optional[Tensor] = None
    ic_cat: Optional[Tensor] = None

    should_stop = False
    # S is rounded up for the cache allocation; the budget is what was asked for.
    limit = L + max_new_tokens
    while not should_stop and (pos := kv_cache.get_pos()) < limit:
        tokens_B1 = logits_BSV[:, -1].argmax(dim=-1, keepdim=True)
        padded_tokens[:, pos] = tokens_B1[:, -1]

        h_last = h_BSD[:, -1, :]
        xy_B2, hw_B2, is_coord_B, is_size_B, coord_logits = model.sample_bbox(
            h_last, tokens_B1.squeeze(-1)
        )

        if coord_dedup_threshold > 0 and xy_cat is not None:
            model.dedup_single_coord(
                xy_B2[0], is_coord_B[0], xy_cat, ic_cat, coord_logits[0],
                threshold=coord_dedup_threshold,
            )

        xy_row = xy_B2[0].unsqueeze(0)
        ic_row = is_coord_B[0].unsqueeze(0)
        xy_cat = xy_row if xy_cat is None else torch.cat([xy_cat, xy_row])
        ic_cat = ic_row if ic_cat is None else torch.cat([ic_cat, ic_row])

        if bool(is_coord_B[0]):
            boxes_raw.append({"x": xy_B2[0, 0].item(), "y": xy_B2[0, 1].item()})
        if bool(is_size_B[0]):
            boxes_raw.append({"h": hw_B2[0, 0].item(), "w": hw_B2[0, 1].item()})

        xy_b2 = xy_B2[is_coord_B] if bool(is_coord_B[0]) else xy_B2.new_empty(0, 2)
        hw_b2 = hw_B2[is_size_B] if bool(is_size_B[0]) else hw_B2.new_empty(0, 2)

        if segmentation and int(tokens_B1[0, 0]) == tokenizer.seg_token_id:
            segm_embeds.append(model.proj_segm(h_BSD[0, -1, :]))

        n = kv_cache.get_pos()
        step_mask = torch.ones((1, 1, 1, n + 1), dtype=torch.bool, device=device)
        logits_BSV, h_BSD = model(
            tokens=tokens_B1,
            mask=step_mask,
            kv_cache=kv_cache,
            coord_xy=xy_b2.to(model.dtype),
            size_hw=hw_b2.to(model.dtype),
        )

        should_stop = bool(torch.isin(tokens_B1, stop_ids).any())

    masks = None
    if segmentation and hr_image_features is not None and segm_embeds:
        hr = hr_image_features[0]  # (D, out_H, out_W)
        mask_b = pixel_mask[0, 0]
        h_actual = int(mask_b.any(dim=1).sum())
        w_actual = int(mask_b.any(dim=0).sum())
        h_px = (h_actual // ps) * HR_UPSAMPLE_RATIO
        w_px = (w_actual // ps) * HR_UPSAMPLE_RATIO
        if h_px > 0 and w_px > 0:
            hr = hr[:, :h_px, :w_px]
        embeds = torch.cat([e.unsqueeze(0) for e in segm_embeds], dim=0)
        logits = torch.einsum("dhw,kd->khw", hr, embeds)
        masks = torch.sigmoid(logits.float()) > MASK_THRESHOLD

    out_len = kv_cache.get_pos() + 0
    return Generation(tokens=padded_tokens[0, :out_len].cpu(), boxes_raw=boxes_raw, masks=masks)


def pair_bbox_entries(raw: List[dict]) -> List[dict]:
    """Pair ``[{x, y}, {h, w}, ...]`` into ``[{x, y, h, w}, ...]`` (normalised)."""
    bboxes: List[dict] = []
    current: dict = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        current.update(entry)
        if all(k in current for k in ("x", "y", "h", "w")):
            bboxes.append(dict(current))
            current = {}
    return bboxes
