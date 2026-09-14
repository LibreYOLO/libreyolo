"""Offline action-error metrics for the LibreVLA tier (pure functions).

These compare predicted action chunks with the chunks a dataset recorded,
in the dataset's own units. They are the offline proxy VLA papers report;
they are not a task success rate (ADR 0028).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["action_error"]


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def action_error(
    predicted: Any,
    target: Any,
    *,
    mask: Any = None,
    names: Optional[List[str]] = None,
    decimals: int = 6,
) -> Dict[str, Any]:
    """Mean L1 and MSE between predicted and recorded action chunks.

    Args:
        predicted: ``(N, T, D)`` or ``(T, D)`` predicted actions.
        target: matching recorded actions.
        mask: optional ``(N, T)`` or ``(T,)`` boolean, True where the target
            step is valid (LeRobot's ``~action_is_pad``).
        names: optional per-dimension names for the per-dimension table.

    Returns:
        ``{"val/action_l1", "val/action_mse", "val/action_l1_first",
        "val/action_l1_dims": {name: value}, "val/steps": int}``.
    """
    pred = _as_array(predicted)
    tgt = _as_array(target)
    if pred.ndim == 2:
        pred = pred[None]
    if tgt.ndim == 2:
        tgt = tgt[None]
    if pred.shape != tgt.shape or pred.ndim != 3:
        raise ValueError(
            f"predicted {tuple(pred.shape)} and target {tuple(tgt.shape)} must "
            "both be (N, T, D) with the same shape."
        )
    n, t, d = pred.shape
    if mask is None:
        valid = np.ones((n, t), dtype=bool)
    else:
        valid = np.asarray(_as_array(mask), dtype=bool)
        if valid.ndim == 1:
            valid = valid[None]
        if valid.shape != (n, t):
            raise ValueError(f"mask must have shape {(n, t)}, got {valid.shape}.")
    steps = int(valid.sum())
    if steps == 0:
        raise ValueError("mask leaves no valid action steps.")
    diff = pred - tgt
    w = valid[..., None].astype(np.float32)
    l1_dims = (np.abs(diff) * w).sum(axis=(0, 1)) / steps
    mse = float(((diff**2) * w).sum() / (steps * d))
    first_valid = valid[:, 0]
    if first_valid.any():
        l1_first = float(np.abs(diff[first_valid, 0, :]).mean())
    else:
        l1_first = float("nan")
    labels = list(names) if names and len(names) == d else [f"a{i}" for i in range(d)]
    return {
        "val/action_l1": round(float(l1_dims.mean()), decimals),
        "val/action_mse": round(mse, decimals),
        "val/action_l1_first": round(l1_first, decimals),
        "val/action_l1_dims": {
            label: round(float(v), decimals) for label, v in zip(labels, l1_dims)
        },
        "val/steps": steps,
    }
