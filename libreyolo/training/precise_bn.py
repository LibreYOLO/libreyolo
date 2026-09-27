"""Recompute BatchNorm running stats from a clean forward pass.

Under heavy augmentation the EMA-style BN buffers lag the true train-set
moments. A short forward-only pass accumulates activation moments and
replaces those buffers with an estimate over the requested samples.

Opt-in. Models with no BatchNorm (RF-DETR and other LayerNorm families)
are a no-op, so enabling the flag there cannot change weights.
"""

from __future__ import annotations

import logging
from typing import Collection, Iterable, List

import torch
import torch.nn as nn

from .distributed import is_distributed

logger = logging.getLogger(__name__)


def _batchnorm_modules(
    model: nn.Module,
    excluded_names: Collection[str] = (),
) -> List[nn.modules.batchnorm._BatchNorm]:
    excluded = set(excluded_names)
    return [
        module
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and name not in excluded
        and module.running_mean is not None
        and module.running_var is not None
    ]


def _first_tensor(batch) -> torch.Tensor | None:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and batch:
        return _first_tensor(batch[0])
    if isinstance(batch, dict):
        for value in batch.values():
            found = _first_tensor(value)
            if found is not None:
                return found
    return None


@torch.no_grad()
def compute_precise_bn_stats(
    model: nn.Module,
    loader: Iterable,
    num_samples: int,
    device: torch.device | str | None = None,
    *,
    excluded_names: Collection[str] = (),
) -> int:
    """Overwrite BN running_mean/var from ``num_samples`` loader images.

    ``excluded_names`` keeps frozen BatchNorm modules untouched. Returns the
    number of modules updated (0 if none, so callers can skip work). Module
    modes, momentums, and counters are restored even if the pass cannot run.
    """
    if num_samples <= 0:
        return 0
    excluded = set(excluded_names)
    bns = _batchnorm_modules(model, excluded)
    if not bns:
        return 0

    module_modes = [(module, module.training) for module in model.modules()]
    all_named_bns = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    model.train()
    for name in excluded:
        module = all_named_bns.get(name)
        if module is not None:
            module.eval()

    momentums = [bn.momentum for bn in bns]
    counters = [
        None if bn.num_batches_tracked is None else bn.num_batches_tracked.clone()
        for bn in bns
    ]
    sums = [torch.zeros_like(bn.running_mean, dtype=torch.float32) for bn in bns]
    square_sums = [torch.zeros_like(bn.running_var, dtype=torch.float32) for bn in bns]
    counts = [
        torch.zeros((), dtype=torch.long, device=bn.running_mean.device) for bn in bns
    ]
    handles = []

    def make_hook(index: int):
        def accumulate(_module, inputs) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            values = inputs[0].detach().float()
            if values.ndim < 2 or values.shape[1] != sums[index].numel():
                return
            reduce_dims = (0, *range(2, values.ndim))
            sums[index].add_(values.sum(dim=reduce_dims))
            square_sums[index].add_(values.square().sum(dim=reduce_dims))
            counts[index].add_(values.numel() // values.shape[1])

        return accumulate

    for index, bn in enumerate(bns):
        handles.append(bn.register_forward_pre_hook(make_hook(index)))

    seen = 0
    batches_used = 0
    forward_error: Exception | None = None
    try:
        for bn in bns:
            # Keep the existing buffers stable until every forward succeeds.
            bn.momentum = 0.0
        for batch in loader:
            images = _first_tensor(batch)
            if images is None:
                continue
            remaining = num_samples - seen
            if remaining <= 0:
                break
            if int(images.shape[0]) > remaining:
                images = images[:remaining]
            if device is not None:
                images = images.to(device, non_blocking=True)
            try:
                model(images)
            except Exception as exc:
                forward_error = exc
                break
            seen += int(images.shape[0])
            batches_used += 1
            if seen >= num_samples:
                break
        if is_distributed():
            import torch.distributed as dist

            ok = torch.tensor(
                1 if batches_used > 0 and forward_error is None else 0,
                device=sums[0].device,
                dtype=torch.int32,
            )
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if int(ok.item()) == 0:
                if forward_error is not None:
                    logger.warning(
                        "Precise BN skipped: model.forward(images) failed (%s). "
                        "This flag needs a model that accepts a bare image tensor.",
                        forward_error,
                    )
                return 0
            for tensor in (*sums, *square_sums, *counts):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        else:
            if batches_used == 0 or forward_error is not None:
                if forward_error is not None:
                    logger.warning(
                        "Precise BN skipped: model.forward(images) failed (%s). "
                        "This flag needs a model that accepts a bare image tensor.",
                        forward_error,
                    )
                return 0

        updated = 0
        for i, bn in enumerate(bns):
            count = int(counts[i].item())
            if count <= 0:
                continue
            mean = sums[i] / count
            centered_ss = square_sums[i] - sums[i].square() / count
            denominator = max(count - 1, 1)
            variance = (centered_ss / denominator).clamp_min_(0.0)
            bn.running_mean.copy_(mean.to(dtype=bn.running_mean.dtype))
            bn.running_var.copy_(variance.to(dtype=bn.running_var.dtype))
            updated += 1
    finally:
        for handle in handles:
            handle.remove()
        for bn, momentum, counter in zip(bns, momentums, counters):
            bn.momentum = momentum
            if counter is not None:
                bn.num_batches_tracked.copy_(counter)
        for module, was_training in module_modes:
            module.training = was_training
    logger.info(
        "Precise BN: recomputed running stats for %d BatchNorm modules "
        "from %d images (%d batches)",
        updated,
        seen,
        batches_used,
    )
    return updated
