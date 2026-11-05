# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0. Derived from
# PPDocLayoutV3ImageProcessor.post_process_object_detection and _get_order_seqs
# in huggingface/transformers v5.22.1
# (src/transformers/models/pp_doclayout_v3/image_processing_pp_doclayout_v3.py).
# LibreYOLO keeps the decode and ranking, drops the polygon extraction, and
# adds max_det, the common inference signature and NumPy output.
"""PP-DocLayoutV3 postprocess: top-k decode plus reading-order ranking (no NMS)."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np
import torch


def reading_order_votes(order_logits: torch.Tensor) -> torch.Tensor:
    """Reduce ``(B, Q, Q)`` pairwise order logits to ``(B, Q)`` votes.

    ``sigmoid(order_logits[b, i, j])`` is the probability that query ``i``
    comes before query ``j``. A query's vote counts how many queries are
    predicted to precede it, so sorting by vote gives the reading order.
    """
    scores = torch.sigmoid(order_logits)
    return scores.triu(diagonal=1).sum(dim=1) + (1.0 - scores.transpose(1, 2)).tril(
        diagonal=-1
    ).sum(dim=1)


def order_ranks(votes: torch.Tensor) -> torch.Tensor:
    """Zero-based reading-order rank of every query from its vote, ``(Q,)``."""
    pointers = torch.argsort(votes, dim=-1)
    ranks = torch.empty_like(pointers)
    ranks.scatter_(
        -1,
        pointers,
        torch.arange(
            votes.shape[-1], device=votes.device, dtype=pointers.dtype
        ).expand_as(pointers),
    )
    return ranks


def postprocess(
    outputs: Mapping[str, torch.Tensor],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: Optional[Tuple[int, int]] = None,
    max_det: int = 300,
    **_unused,
) -> dict:
    """Decode one image into LibreYOLO's detection dict, in reading order.

    Scores are per-class sigmoids. As upstream, the top ``Q`` (query, class)
    pairs are taken over the flattened ``Q x nc`` grid, so one query can yield
    two labels. Pairs with ``score >= conf_thres`` are kept and sorted by the
    reading-order rank of their query. ``reading_order`` is one-based and
    counts the regions kept at ``conf_thres``; a ``max_det`` cut keeps the
    highest-scoring regions without renumbering the rest. Set prediction is
    duplicate-aware, so ``iou_thres`` is accepted for API compatibility but
    no NMS is applied.
    """
    del iou_thres
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    votes = outputs.get("order_votes")
    if votes is None:
        votes = reading_order_votes(outputs["order_logits"])
    if logits.ndim == 3:
        logits, boxes, votes = logits[0], boxes[0], votes[0]

    num_queries, num_classes = logits.shape
    ranks = order_ranks(votes)

    scores = torch.sigmoid(logits)
    scores, index = torch.topk(scores.flatten(), num_queries)
    labels = index % num_classes
    query = index // num_classes

    keep = scores >= conf_thres
    scores, labels, query = scores[keep], labels[keep], query[keep]
    order, sort_idx = torch.sort(ranks[query], stable=True)
    del order
    scores, labels, query = scores[sort_idx], labels[sort_idx], query[sort_idx]

    centers, sizes = boxes[query].split(2, dim=-1)
    xyxy = torch.cat([centers - 0.5 * sizes, centers + 0.5 * sizes], dim=-1)
    if original_size is not None:
        width, height = original_size
        xyxy = xyxy * xyxy.new_tensor([width, height, width, height])

    reading_order = torch.arange(1, len(scores) + 1, device=scores.device)
    budget = max(int(max_det), 0)
    if len(scores) > budget:
        top = torch.topk(scores, budget).indices.sort().values
        scores, labels, xyxy, reading_order = (
            scores[top],
            labels[top],
            xyxy[top],
            reading_order[top],
        )

    count = int(scores.shape[0])
    return {
        "num_detections": count,
        "boxes": xyxy.float().cpu().numpy() if count else np.zeros((0, 4), np.float32),
        "scores": scores.float().cpu().numpy() if count else np.zeros((0,), np.float32),
        "classes": labels.cpu().numpy() if count else np.zeros((0,), np.int64),
        "reading_order": (
            reading_order.cpu().numpy() if count else np.zeros((0,), np.int64)
        ),
    }
