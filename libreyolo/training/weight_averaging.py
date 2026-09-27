"""Metric-gated top-N checkpoint averaging.

Keeps the N best (by the watched validation metric) full state dicts on
CPU and, at the end of training, writes their uniform average. Unlike EMA
this never admits a divergent epoch: a snapshot only enters the pool when
it beats the current worst member.

Opt-in. ``n=0`` is a no-op so default training is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn


def _cpu_float_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().to("cpu").clone()
        for key, value in model.state_dict().items()
    }


class MetricGatedAverager:
    """Rolling pool of the N best snapshots, ranked by a scalar metric."""

    def __init__(self, n: int, *, greater_is_better: bool = True):
        if n < 0:
            raise ValueError(f"average_best must be >= 0, got {n}")
        self.n = int(n)
        self.greater_is_better = bool(greater_is_better)
        self._pool: List[Tuple[float, Dict[str, torch.Tensor]]] = []

    @property
    def size(self) -> int:
        return len(self._pool)

    def consider(self, model: nn.Module, metric: float) -> bool:
        """Offer a snapshot. Returns True if it entered the pool."""
        return self.consider_state(_cpu_float_state(model), metric)

    def consider_state(self, state: Mapping[str, torch.Tensor], metric: float) -> bool:
        """Offer a CPU state dict. Returns True if it entered the pool."""
        if self.n <= 0 or not _finite(metric):
            return False
        snapshot = {
            key: value.detach().to("cpu").clone() if torch.is_tensor(value) else value
            for key, value in state.items()
        }
        if len(self._pool) < self.n:
            self._pool.append((float(metric), snapshot))
            return True
        worst_i = self._worst_index()
        worst_metric = self._pool[worst_i][0]
        better = (
            metric > worst_metric if self.greater_is_better else metric < worst_metric
        )
        if not better:
            return False
        self._pool[worst_i] = (float(metric), snapshot)
        return True

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "n": self.n,
                "greater_is_better": self.greater_is_better,
                "metrics": [float(metric) for metric, _ in self._pool],
                "states": [state for _, state in self._pool],
            },
            path,
        )

    def load(self, path: str | Path) -> int:
        """Replace the pool from a sidecar written by :meth:`save`.

        Keeps this instance's ``n`` (the live config). Extra snapshots from a
        larger saved pool are dropped worst-first. Returns how many loaded.
        """
        blob = torch.load(path, map_location="cpu", weights_only=True)
        metrics = list(blob.get("metrics") or [])
        states = list(blob.get("states") or [])
        self._pool = []
        for metric, state in zip(metrics, states):
            self.consider_state(state, float(metric))
        return len(self._pool)

    def average_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self._pool:
            return None
        keys = self._pool[0][1].keys()
        n = len(self._pool)
        best_i = self._best_index()
        out: Dict[str, torch.Tensor] = {}
        for key in keys:
            acc = self._pool[0][1][key].to(dtype=torch.float32)
            for _, state in self._pool[1:]:
                acc = acc + state[key].to(dtype=torch.float32)
            averaged = acc / n
            ref = self._pool[0][1][key]
            if not ref.dtype.is_floating_point:
                # Integer / bool buffers (e.g. num_batches_tracked) stay as
                # the best snapshot's value, not a fractional mean.
                out[key] = self._pool[best_i][1][key].clone()
            else:
                out[key] = averaged.to(dtype=ref.dtype)
        return out

    def metrics(self) -> List[float]:
        return [metric for metric, _ in self._pool]

    def _worst_index(self) -> int:
        if self.greater_is_better:
            return min(range(len(self._pool)), key=lambda i: self._pool[i][0])
        return max(range(len(self._pool)), key=lambda i: self._pool[i][0])

    def _best_index(self) -> int:
        if self.greater_is_better:
            return max(range(len(self._pool)), key=lambda i: self._pool[i][0])
        return min(range(len(self._pool)), key=lambda i: self._pool[i][0])


def _finite(value: Any) -> bool:
    try:
        return bool(value == value) and abs(float(value)) != float("inf")
    except (TypeError, ValueError):
        return False


def average_state_dicts(
    states: Mapping[str, torch.Tensor] | List[Mapping[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Uniform average of one or more state dicts. Exposed for tests."""
    if isinstance(states, Mapping):
        states = [states]
    if not states:
        raise ValueError("average_state_dicts requires at least one state dict")
    averager = MetricGatedAverager(n=len(states))
    # Bypass consider() so tests can average arbitrary dicts.
    averager._pool = [(0.0, dict(s)) for s in states]
    out = averager.average_state_dict()
    assert out is not None
    return out
