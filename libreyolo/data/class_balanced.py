"""LVIS-style repeat-factor sampling for long-tailed detection sets.

For each class ``c``, ``f(c)`` is the fraction of images that contain it.
Classes rarer than the median frequency get a repeat factor
``r(c) = (t / f(c)) ** alpha``; an image's weight is the max ``r(c)`` of
the classes it contains. Empty images keep weight 1 so they still train
as background.

Opt-in. Off by default; ``create_dataloader(..., class_balanced=False)``
is the historical sampler.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


def class_ids_from_anno(anno) -> np.ndarray:
    """Class column from a YOLO-style annotation array (empty-safe)."""
    arr = np.asarray(anno)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] >= 5:
        return arr[:, 4].astype(np.int64)
    return arr[:, 0].astype(np.int64)


def annotation_source(dataset):
    """Walk mosaic/mixup wrappers to the dataset that owns labels."""
    seen: set[int] = set()
    current = dataset
    while id(current) not in seen:
        seen.add(id(current))
        if callable(getattr(current, "load_anno", None)):
            return current
        if getattr(current, "annotations", None) is not None:
            return current
        inner = getattr(current, "dataset", None)
        if inner is None or inner is current:
            break
        current = inner
    raise TypeError(
        "class_balanced=True needs a dataset with load_anno() or "
        f"annotations; got {type(dataset).__name__}"
    )


def _load_anno(dataset, index: int):
    source = annotation_source(dataset)
    load_anno = getattr(source, "load_anno", None)
    if callable(load_anno):
        return load_anno(index)
    annotations = getattr(source, "annotations", None)
    if annotations is None:
        raise TypeError(
            "class_balanced=True needs a dataset with load_anno() or "
            f"annotations; got {type(dataset).__name__}"
        )
    entry = annotations[index]
    if isinstance(entry, (tuple, list)) and entry:
        return entry[0]
    return entry


def image_repeat_factors(
    dataset,
    *,
    alpha: float = 0.5,
    threshold: float | None = None,
) -> np.ndarray:
    """Return a length-``len(dataset)`` float32 weight per image."""
    n = len(dataset)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    per_image: List[np.ndarray] = []
    max_cls = -1
    source = annotation_source(dataset)
    declared = (
        getattr(source, "num_classes", None)
        or getattr(source, "nc", None)
        or getattr(dataset, "num_classes", None)
        or getattr(dataset, "nc", None)
    )
    for i in range(n):
        ids = class_ids_from_anno(_load_anno(dataset, i))
        ids = ids[ids >= 0]
        per_image.append(ids)
        if ids.size:
            max_cls = max(max_cls, int(ids.max()))
    nc = max(max_cls + 1, int(declared) if declared else 0)
    if nc <= 0:
        return np.ones(n, dtype=np.float32)

    present = np.zeros((n, nc), dtype=np.float32)
    for i, ids in enumerate(per_image):
        if ids.size:
            present[i, np.unique(ids)] = 1.0
    freq = present.mean(axis=0)
    positive = freq[freq > 0]
    if positive.size == 0:
        return np.ones(n, dtype=np.float32)
    t = float(np.median(positive) if threshold is None else threshold)
    if t <= 0:
        return np.ones(n, dtype=np.float32)

    class_repeat = np.ones(nc, dtype=np.float32)
    rare = (freq > 0) & (freq < t)
    class_repeat[rare] = (t / freq[rare]) ** float(alpha)

    weights = np.ones(n, dtype=np.float32)
    for i, ids in enumerate(per_image):
        if ids.size:
            weights[i] = float(class_repeat[np.unique(ids)].max())
    return weights


class DistributedClassBalancedSampler(Sampler):
    """DDP-aware multinomial draw with the same shard contract as DistributedSampler.

    Every rank draws the same ``total_size`` indices from the shared weights
    (seeded with ``seed + epoch``), then keeps ``rank::num_replicas``. That
    keeps collectives aligned while still oversampling rare classes.
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas - 1}], got {rank}")
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.numel() == 0:
            raise ValueError("class_balanced sampler needs a non-empty dataset")
        if torch.any(self.weights < 0) or not bool(torch.any(self.weights > 0)):
            raise ValueError(
                "class_balanced weights must be non-negative and not all zero"
            )
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.num_samples = math.ceil(num_samples / num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=generator
        )
        return iter(indices[self.rank : self.total_size : self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def build_class_balanced_sampler(
    dataset,
    *,
    num_samples: int | None = None,
    distributed_sampler=None,
    alpha: float = 0.5,
    seed: int = 0,
) -> Sampler:
    """Build a single-process or DDP class-balanced sampler from ``dataset``."""
    weights = image_repeat_factors(dataset, alpha=alpha)
    draw = int(num_samples if num_samples is not None else len(dataset))
    if draw < 1:
        raise ValueError(f"class_balanced num_samples must be >= 1, got {draw}")
    if distributed_sampler is not None:
        return DistributedClassBalancedSampler(
            weights,
            draw,
            num_replicas=distributed_sampler.num_replicas,
            rank=distributed_sampler.rank,
            seed=int(getattr(distributed_sampler, "seed", seed) or 0),
        )
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=draw,
        replacement=True,
    )
