"""Standalone detection-fusion primitives.

Pure-torch ops that merge box detections coming from several independent
detectors. They are model-free: inputs are plain tensors, so the ops work on
any detector's output and run on whatever device the inputs live on.

Shared signature
----------------
Every op takes the same stacked inputs and returns the same triple::

    boxes     (N, 4) float  xyxy corner boxes. Pixels or normalized — IoU and
                            weighted means are scale-invariant.
    scores    (N,)   float  detection confidences.
    labels    (N,)   long   class ids in a single shared label space.
    model_ids (N,)   long   index of the model that produced each row.

    -> (boxes (K, 4), scores (K,), labels (K,)) sorted by descending score.

Keyword arguments follow the established fusion vocabulary: ``iou_thr`` is
the cluster threshold, ``skip_box_thr`` drops low-score inputs before
clustering, ``weights`` is one trust factor per model, and ``conf_type``
selects how a cluster's score is reduced (``"avg"`` or ``"max"``).

Consensus
---------
``min_votes`` keeps only clusters whose boxes come from at least that many
distinct models. ``models_per_label`` — a 1-D tensor mapping class id to the
number of models whose label space contains that class — caps the requirement
per class, so consensus stays meaningful when label spaces only partially
overlap: a class that only one model knows about can never collect two votes,
and is not silently erased by ``min_votes=2``.

Weighted Boxes Fusion is implemented from the method described in Solovyev et
al., "Weighted boxes fusion: Ensembling boxes from different object detection
models" (arXiv:1910.13302). With per-model weights the paper's
``min(T, N) / N`` confidence rescale generalizes to ``min(W_T, W_N) / W_N``,
where ``W_T`` is the summed weight of a cluster's boxes and ``W_N`` the summed
weight of all models; unit weights recover the paper exactly. ``label_weights``
(class id → summed weight of the models that know the class) makes ``W_N``
per-class, so a class only some models can detect is not score-penalized for
the models that could never have confirmed it.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from torchvision.ops import batched_nms, box_iou

FusionResult = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
WeightsLike = Union[torch.Tensor, Sequence[float], None]

_EPS = 1e-12


def _validate_stacked(boxes, scores, labels, model_ids):
    """Convert inputs to canonical tensors and check shapes line up."""
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    scores = torch.as_tensor(scores, dtype=torch.float32, device=boxes.device)
    labels = torch.as_tensor(labels, device=boxes.device).long()
    model_ids = torch.as_tensor(model_ids, device=boxes.device).long()

    if boxes.numel() == 0:
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {tuple(boxes.shape)}")
    n = boxes.shape[0]
    for name, t in (("scores", scores), ("labels", labels), ("model_ids", model_ids)):
        if t.reshape(-1).shape[0] != n:
            raise ValueError(
                f"{name} must have {n} entries to match boxes, "
                f"got {t.reshape(-1).shape[0]}"
            )
    return boxes, scores.reshape(-1), labels.reshape(-1), model_ids.reshape(-1)


def _resolve_weights(
    weights: WeightsLike, num_models: Optional[int], model_ids: torch.Tensor
) -> torch.Tensor:
    """Return one positive trust factor per model as a float tensor."""
    if num_models is None:
        if weights is not None:
            num_models = len(weights)
        elif model_ids.numel() > 0:
            num_models = int(model_ids.max()) + 1
        else:
            num_models = 1
    if model_ids.numel() > 0 and (
        int(model_ids.min()) < 0 or int(model_ids.max()) >= num_models
    ):
        bad = int(model_ids.min()) if int(model_ids.min()) < 0 else int(model_ids.max())
        raise ValueError(
            f"model_ids contains index {bad} "
            f"but only {num_models} models are declared"
        )
    if weights is None:
        return torch.ones(num_models, dtype=torch.float32, device=model_ids.device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=model_ids.device)
    if w.reshape(-1).shape[0] != num_models:
        raise ValueError(
            f"weights must have one entry per model ({num_models}), "
            f"got {w.reshape(-1).shape[0]}"
        )
    w = w.reshape(-1)
    # Positivity (not non-negativity) so NaN weights also fail loudly.
    if not bool((w > 0).all()):
        raise ValueError("weights must all be positive")
    return w


def _required_votes(
    min_votes: int,
    num_models: int,
    models_per_label: Optional[torch.Tensor],
    cluster_labels: torch.Tensor,
) -> torch.Tensor:
    """Per-cluster vote requirement, capped by how many models know the class."""
    base = torch.full_like(cluster_labels, min(min_votes, num_models))
    if models_per_label is None:
        return base
    mpl = torch.as_tensor(models_per_label, device=cluster_labels.device).long()
    if cluster_labels.numel() > 0 and int(cluster_labels.max()) >= mpl.numel():
        raise ValueError(
            f"models_per_label covers {mpl.numel()} classes but labels reach "
            f"{int(cluster_labels.max())}"
        )
    return torch.minimum(base, mpl[cluster_labels])


def _rescale_denominator(
    label_weights: Optional[torch.Tensor],
    cluster_labels: torch.Tensor,
    w_model: torch.Tensor,
) -> torch.Tensor:
    """Per-cluster rescale denominator (the ``W_N`` of the score rescale).

    ``label_weights`` maps class id to the summed weight of the models whose
    label space contains that class, so a class only one model knows about is
    not penalized for the models that could never have confirmed it. When
    omitted, every cluster uses the total weight of all models — the paper's
    behavior.
    """
    if label_weights is None:
        return w_model.sum()
    lw = torch.as_tensor(
        label_weights, dtype=torch.float32, device=cluster_labels.device
    )
    if cluster_labels.numel() > 0 and int(cluster_labels.max()) >= lw.numel():
        raise ValueError(
            f"label_weights covers {lw.numel()} classes but labels reach "
            f"{int(cluster_labels.max())}"
        )
    return lw[cluster_labels].clamp(min=_EPS)


def _votes_per_cluster(
    cluster_ids: torch.Tensor, model_ids: torch.Tensor, num_clusters: int, num_models: int
) -> torch.Tensor:
    """Count distinct contributing models per cluster.

    One boolean reduction per model keeps the computation fixed-shape (the
    model count is a small compile-time constant), which the future in-graph
    export path relies on.
    """
    votes = torch.zeros(num_clusters, dtype=torch.long, device=cluster_ids.device)
    for m in range(num_models):
        seen = torch.zeros(num_clusters, dtype=torch.bool, device=cluster_ids.device)
        seen[cluster_ids[model_ids == m]] = True
        votes += seen.long()
    return votes


def _shift_non_negative(boxes: torch.Tensor) -> torch.Tensor:
    """Translate boxes into the non-negative quadrant for ``batched_nms``.

    Its class-offset trick uses ``boxes.max() + 1`` as the per-class step,
    which only separates classes when all coordinates are non-negative.
    Translation does not change IoU.
    """
    if boxes.numel() == 0:
        return boxes
    return boxes - boxes.min().clamp(max=0)


def _empty_result(device) -> FusionResult:
    return (
        torch.zeros((0, 4), dtype=torch.float32, device=device),
        torch.zeros(0, dtype=torch.float32, device=device),
        torch.zeros(0, dtype=torch.long, device=device),
    )


def _sorted_result(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> FusionResult:
    order = torch.argsort(scores, descending=True)
    return boxes[order], scores[order], labels[order]


def _check_conf_type(conf_type: str) -> None:
    if conf_type not in ("avg", "max"):
        raise ValueError(f"conf_type must be 'avg' or 'max', got {conf_type!r}")


def weighted_boxes_fusion(
    boxes,
    scores,
    labels,
    model_ids,
    *,
    weights: WeightsLike = None,
    num_models: Optional[int] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    conf_type: str = "avg",
    min_votes: int = 1,
    models_per_label: Optional[torch.Tensor] = None,
    label_weights: Optional[torch.Tensor] = None,
) -> FusionResult:
    """Paper-faithful sequential Weighted Boxes Fusion.

    Detections are visited in order of decreasing weight-scaled confidence.
    Each one either joins the existing cluster whose running fused box it
    overlaps best (IoU > ``iou_thr``, same label) or starts a new cluster.
    A cluster's fused box is the confidence-weighted average of its members'
    coordinates; its score is the weighted mean (or max) of their confidences,
    rescaled by ``min(W_T, W_N) / W_N`` so boxes confirmed by fewer models
    score lower. ``label_weights`` makes ``W_N`` per-class (see
    :func:`_rescale_denominator`).
    """
    _check_conf_type(conf_type)
    boxes, scores, labels, model_ids = _validate_stacked(
        boxes, scores, labels, model_ids
    )
    w_model = _resolve_weights(weights, num_models, model_ids)
    n_models = w_model.numel()

    keep = scores > skip_box_thr
    boxes, scores, labels, model_ids = (
        boxes[keep], scores[keep], labels[keep], model_ids[keep]
    )
    if boxes.shape[0] == 0:
        return _empty_result(boxes.device)

    w_box = w_model[model_ids]
    order = torch.argsort(scores * w_box, descending=True)

    # Preallocated per-cluster accumulators; k tracks how many are live.
    n = boxes.shape[0]
    fused = boxes.new_empty((n, 4))       # running fused box per cluster
    cl_labels = labels.new_empty(n)
    coord_sum = boxes.new_zeros((n, 4))   # sum of s*w*box
    sw_sum = scores.new_zeros(n)          # sum of s*w
    w_sum = scores.new_zeros(n)           # sum of w  (W_T)
    s_max = scores.new_zeros(n)
    cluster_ids = torch.zeros_like(model_ids)
    k = 0
    for i in order.tolist():
        b, s, w = boxes[i], scores[i], w_box[i]
        j = -1
        if k > 0:
            ious = box_iou(b.unsqueeze(0), fused[:k]).squeeze(0)
            ious = ious.masked_fill(cl_labels[:k] != labels[i], -1.0)
            best_iou, best_j = ious.max(dim=0)
            if best_iou > iou_thr:
                j = int(best_j)
        if j < 0:
            j = k
            k += 1
            fused[j] = b
            cl_labels[j] = labels[i]
        coord_sum[j] += s * w * b
        sw_sum[j] += s * w
        w_sum[j] += w
        s_max[j] = torch.maximum(s_max[j], s)
        cluster_ids[i] = j
        fused[j] = coord_sum[j] / sw_sum[j].clamp(min=_EPS)

    fused, cl_labels = fused[:k], cl_labels[:k]
    sw_sum, w_sum, s_max = sw_sum[:k], w_sum[:k], s_max[:k]
    fused_scores = s_max if conf_type == "max" else sw_sum / w_sum.clamp(min=_EPS)
    denom = _rescale_denominator(label_weights, cl_labels, w_model)
    fused_scores = fused_scores * (w_sum / denom).clamp(max=1.0)

    votes = _votes_per_cluster(cluster_ids, model_ids, k, n_models)
    required = _required_votes(min_votes, n_models, models_per_label, cl_labels)
    keep = votes >= required

    return _sorted_result(fused[keep], fused_scores[keep], cl_labels[keep])


def wbf_seeded(
    boxes,
    scores,
    labels,
    model_ids,
    *,
    weights: WeightsLike = None,
    num_models: Optional[int] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    conf_type: str = "avg",
    min_votes: int = 1,
    models_per_label: Optional[torch.Tensor] = None,
    label_weights: Optional[torch.Tensor] = None,
) -> FusionResult:
    """Parallel one-pass Weighted Boxes Fusion.

    Class-aware NMS at ``iou_thr`` picks the cluster seeds, every detection
    assigns to its best-IoU seed of the same label, and each cluster is then
    reduced exactly as in :func:`weighted_boxes_fusion`. Unlike the sequential
    variant, cluster shapes never shift mid-pass, so the whole op is
    fixed-shape tensor math — the variant the future in-graph export compiles.
    The two variants agree whenever clusters are unambiguous and can differ
    slightly on overlapping cluster chains.
    """
    _check_conf_type(conf_type)
    boxes, scores, labels, model_ids = _validate_stacked(
        boxes, scores, labels, model_ids
    )
    w_model = _resolve_weights(weights, num_models, model_ids)
    n_models = w_model.numel()

    keep = scores > skip_box_thr
    boxes, scores, labels, model_ids = (
        boxes[keep], scores[keep], labels[keep], model_ids[keep]
    )
    if boxes.shape[0] == 0:
        return _empty_result(boxes.device)

    w_box = w_model[model_ids]
    seeds = batched_nms(_shift_non_negative(boxes), scores * w_box, labels, iou_thr)
    k = seeds.shape[0]

    # Assign every detection to its best same-label seed. Greedy NMS only ever
    # suppresses a box through a kept one of the same class with IoU strictly
    # above the threshold, so the assignment is total: non-seeds always clear
    # the strict comparison (matching the sequential variant's join rule) and
    # seeds match themselves at IoU 1.
    ious = box_iou(boxes, boxes[seeds])
    ious = ious.masked_fill(labels.unsqueeze(1) != labels[seeds].unsqueeze(0), -1.0)
    best_iou, cluster_ids = ious.max(dim=1)
    is_seed = torch.zeros_like(scores, dtype=torch.bool)
    is_seed[seeds] = True
    cluster_ids[seeds] = torch.arange(k, device=boxes.device)
    assigned = is_seed | (best_iou > iou_thr)
    boxes, scores, labels, model_ids, w_box, cluster_ids = (
        boxes[assigned], scores[assigned], labels[assigned],
        model_ids[assigned], w_box[assigned], cluster_ids[assigned],
    )

    sw = scores * w_box
    coord_sum = boxes.new_zeros((k, 4)).index_add_(0, cluster_ids, boxes * sw.unsqueeze(1))
    sw_sum = scores.new_zeros(k).index_add_(0, cluster_ids, sw)
    w_sum = scores.new_zeros(k).index_add_(0, cluster_ids, w_box)

    fused = coord_sum / sw_sum.clamp(min=_EPS).unsqueeze(1)
    if conf_type == "max":
        fused_scores = scores.new_zeros(k).scatter_reduce_(
            0, cluster_ids, scores, reduce="amax", include_self=False
        )
    else:
        fused_scores = sw_sum / w_sum.clamp(min=_EPS)

    cl_labels = labels.new_zeros(k).scatter_(0, cluster_ids, labels)
    denom = _rescale_denominator(label_weights, cl_labels, w_model)
    fused_scores = fused_scores * (w_sum / denom).clamp(max=1.0)
    votes = _votes_per_cluster(cluster_ids, model_ids, k, n_models)
    required = _required_votes(min_votes, n_models, models_per_label, cl_labels)
    keep = votes >= required

    return _sorted_result(fused[keep], fused_scores[keep], cl_labels[keep])


def nms_fusion(
    boxes,
    scores,
    labels,
    model_ids,
    *,
    weights: WeightsLike = None,
    num_models: Optional[int] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    min_votes: int = 1,
    models_per_label: Optional[torch.Tensor] = None,
    label_weights: Optional[torch.Tensor] = None,
) -> FusionResult:
    """Concatenate all detections and apply class-aware NMS.

    The simplest fusion: the highest-confidence box of each overlapping group
    survives unchanged. Per-model ``weights`` scale confidences for the
    suppression ranking only — surviving boxes keep their original scores.
    NMS discards cluster membership, so vote counting is not available here;
    use one of the WBF variants for consensus filtering.
    """
    if min_votes > 1:
        raise ValueError(
            "nms fusion cannot count votes; use 'wbf' or 'wbf_seeded' "
            "for min_votes > 1"
        )
    del models_per_label, label_weights  # accepted for signature uniformity, unused
    boxes, scores, labels, model_ids = _validate_stacked(
        boxes, scores, labels, model_ids
    )
    w_model = _resolve_weights(weights, num_models, model_ids)

    keep = scores > skip_box_thr
    boxes, scores, labels, model_ids = (
        boxes[keep], scores[keep], labels[keep], model_ids[keep]
    )
    if boxes.shape[0] == 0:
        return _empty_result(boxes.device)

    ranking = scores * w_model[model_ids]
    kept = batched_nms(_shift_non_negative(boxes), ranking, labels, iou_thr)
    return _sorted_result(boxes[kept], scores[kept], labels[kept])


FUSIONS: Dict[str, Callable[..., FusionResult]] = {
    "wbf": weighted_boxes_fusion,
    "wbf_seeded": wbf_seeded,
    "nms": nms_fusion,
}
