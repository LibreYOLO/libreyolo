"""Optimizer construction shared by every family trainer.

Promoted from the rfdetr ``_adamw`` helper (PR #761) so all families get the
fused CUDA optimizer path with one implementation and one safety story.
"""

from typing import Any, Dict, Iterable, List, Type, Union

import torch

ParamsLike = Iterable[Union[torch.Tensor, Dict[str, Any]]]

#: Optimizers that must never take the fused path. torch's fused SGD (2.11
#: verified) sets ``_step_supports_amp_scaling`` on the instance, so
#: GradScaler.step delegates overflow handling to the kernel instead of
#: skipping the step itself; the fused SGD kernel then advances the momentum
#: buffer on the very overflow steps the scaler meant to skip entirely. The
#: routine fp16 overflow on the first training steps writes inf into
#: momentum, and the weights explode one step later (yolo9 AMP fine-tuning
#: reached nan loss by step 2). Fused Adam and AdamW honor found_inf and
#: leave their state untouched on skipped steps, so they keep the fused path.
_FUSED_UNSAFE = (torch.optim.SGD,)


def _materialized_groups(params: ParamsLike) -> List[Union[torch.Tensor, Dict[str, Any]]]:
    """Materialize params (and each group's params) so they can be inspected
    for the device gate and still be consumed by the optimizer afterwards.

    A group's ``params`` may legally be one bare tensor (torch wraps it in a
    list); it must NOT be ``list()``-ed, which would iterate its rows into
    non-leaf views (rfdetr builds per-parameter groups exactly this way).
    """
    groups = list(params)
    for group in groups:
        if isinstance(group, dict) and not isinstance(group["params"], torch.Tensor):
            group["params"] = list(group["params"])
    return groups


def _all_params_cuda(groups: List[Union[torch.Tensor, Dict[str, Any]]]) -> bool:
    tensors: List[torch.Tensor] = []
    for group in groups:
        if isinstance(group, dict):
            group_params = group["params"]
            if isinstance(group_params, torch.Tensor):
                group_params = [group_params]
            tensors.extend(p for p in group_params if isinstance(p, torch.Tensor))
        elif isinstance(group, torch.Tensor):
            tensors.append(group)
    return bool(tensors) and all(p.is_cuda for p in tensors)


def build_optimizer(
    optim_cls: Type[torch.optim.Optimizer], params: ParamsLike, **kwargs
) -> torch.optim.Optimizer:
    """Construct ``optim_cls``, preferring the fused implementation on CUDA.

    The fused step is one multi-tensor kernel per dtype instead of the
    foreach implementation's per-op launches, and under a GradScaler it
    consumes the scale/found_inf tensors directly, skipping the per-group
    unscale pass and its device sync. Measured on rfdetr-s 512px b8 fp16:
    optimizer phase 62 -> 24 ms (1.16x on the whole step). Numerics note:
    fused uses a different fp reduction order than foreach (~1e-7 relative
    per step), same algorithm.

    ``fused=True`` is requested only when every parameter is on CUDA. Recent
    torch also ships fused CPU and MPS kernels and would silently accept
    those params, changing the reduction order on machines that never asked
    for it; the explicit device gate keeps non-CUDA construction, and
    therefore non-CUDA training, byte-identical to stock. The gate cannot
    be replaced by the try/except: torch validates fused device support
    lazily at the first ``step()``, not at construction, so a construction-
    time except never sees that error. The except handles older torch
    builds where the kwarg or the fused path itself is missing.
    """
    groups = _materialized_groups(params)
    if _all_params_cuda(groups) and not issubclass(optim_cls, _FUSED_UNSAFE):
        try:
            return optim_cls(groups, **kwargs, fused=True)
        except (RuntimeError, ValueError, TypeError):
            pass
    return optim_cls(groups, **kwargs)


#: Param-group keys that select the step implementation rather than describe
#: the training run. They must reflect the CURRENT construction, never the
#: checkpoint's: a checkpoint written on CUDA carries ``fused: True``, and
#: ``load_state_dict`` copies group hyperparameters wholesale, so resuming on
#: CPU/MPS would re-enable fused on a device the gate above deliberately
#: excluded (step-time RuntimeError on older torch, silent reduction-order
#: change on recent torch).
_IMPL_SELECTION_KEYS = ("fused", "foreach", "capturable", "differentiable")


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer, state_dict: Dict[str, Any]
) -> None:
    """``optimizer.load_state_dict`` that keeps the live optimizer's
    implementation-selection keys (``fused``/``foreach``/...) instead of the
    checkpoint's, so a resume never changes which step implementation runs."""
    snapshots = [
        {key: group[key] for key in _IMPL_SELECTION_KEYS if key in group}
        for group in optimizer.param_groups
    ]
    optimizer.load_state_dict(state_dict)
    for group, snapshot in zip(optimizer.param_groups, snapshots):
        for key in _IMPL_SELECTION_KEYS:
            group.pop(key, None)
        group.update(snapshot)
