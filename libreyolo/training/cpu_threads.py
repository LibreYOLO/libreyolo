"""Keep PyTorch's CPU thread pool within the process's CPU quota.

PyTorch sizes its intra-op pool from the cores it can see. Containers with a
CPU limit (Docker ``--cpus``, Kubernetes limits, rented GPU hosts) still show
every host core, so a training process on a 256-core host limited to ~31
cores runs 128-255 OpenMP threads. The CPU tensor work in the training step
(Hungarian matching inputs, target preparation, small reductions) then
exhausts the quota within each scheduler period and the whole process is
throttled until the next one. Measured on an RTX 4090 host with a 30.7-core
quota: RF-DETR nano's default recipe took 77.8 s per epoch at 255 threads
and 10.3 s at 16.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_CGROUP_ROOT = Path("/sys/fs/cgroup")


def cpu_quota(root: Path = _CGROUP_ROOT) -> int | None:
    """CPUs the process may use per period under a cgroup limit, else None."""
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        quota, period = (root / "cpu.max").read_text().split()[:2]
        if quota != "max" and int(period) > 0:
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def cap_torch_threads(root: Path = _CGROUP_ROOT) -> int | None:
    """Lower ``torch.set_num_threads`` to the CPUs this process may use.

    Only ever lowers, and only to a limit the kernel enforces (the cgroup
    quota or the affinity mask), split across the local ranks of a
    multi-process run. ``OMP_NUM_THREADS`` wins. Returns the previous thread
    count when it changed it (for :func:`restore_torch_threads`), else None.
    """
    if os.environ.get("OMP_NUM_THREADS"):
        return None
    limits = [cpu_quota(root)]
    if hasattr(os, "sched_getaffinity"):
        limits.append(len(os.sched_getaffinity(0)))
    usable = min((n for n in limits if n), default=None)
    if usable is None:
        return None
    # torchrun sets LOCAL_WORLD_SIZE; LibreYOLO's own single-node spawn sets
    # only WORLD_SIZE, which is then the number of ranks on this machine.
    try:
        local_ranks = int(
            os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("WORLD_SIZE") or 1
        )
    except ValueError:
        local_ranks = 1
    usable = max(1, usable // max(1, local_ranks))
    current = torch.get_num_threads()
    if current <= usable:
        return None
    torch.set_num_threads(usable)
    logger.info(
        "Limited PyTorch CPU threads from %d to %d for training, the CPUs this "
        "process may use (set OMP_NUM_THREADS to override).",
        current,
        usable,
    )
    return current


def restore_torch_threads(previous: int | None) -> None:
    """Put back the thread count :func:`cap_torch_threads` replaced."""
    if previous is not None:
        torch.set_num_threads(previous)
