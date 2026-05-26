"""Distributed training utilities for LibreYOLO.

Thin helpers around ``torch.distributed`` so the rest of the trainer can stay
backend-agnostic. All helpers degrade to no-ops when distributed is not
initialised — single-GPU code paths continue to work unchanged.

User-facing surface mirrors Ultralytics: pass ``device=[0, 1]`` (or
``device="0,1"``) and launch with ``torchrun --nproc_per_node=N``. Inside
each child process ``init_distributed()`` is called by the trainer; outside
DDP everything is a no-op.
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

DeviceArg = Union[str, int, List[int], None]


# =============================================================================
# Distributed state queries
# =============================================================================


def is_distributed() -> bool:
    """True iff a process group is initialised."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Global rank of this process, or 0 outside DDP."""
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    """Local rank from ``LOCAL_RANK`` env (set by torchrun), or 0."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Number of processes participating, or 1 outside DDP."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """True on rank 0 (always True outside DDP)."""
    return get_rank() == 0


def has_torchrun_env() -> bool:
    """True iff this process was spawned by torchrun (LOCAL_RANK is set)."""
    return "LOCAL_RANK" in os.environ


def barrier() -> None:
    """Synchronisation barrier; no-op outside DDP."""
    if is_distributed():
        dist.barrier()


# =============================================================================
# Device argument parsing
# =============================================================================


def parse_device_arg(device: DeviceArg) -> List[int]:
    """Parse a user-facing device argument into a list of CUDA device indices.

    Returns an empty list for CPU / MPS / auto-no-cuda.

    Accepts:
      - ``0`` or ``"0"`` → ``[0]``
      - ``[0, 1]`` or ``"0,1"`` → ``[0, 1]``
      - ``"cpu"``, ``"mps"``, ``"auto"``, ``""`` → ``[]``
      - ``"cuda:0"`` → ``[0]``
    """
    if device is None:
        return []
    if isinstance(device, int):
        return [device] if device >= 0 else []
    if isinstance(device, (list, tuple)):
        return [int(d) for d in device if isinstance(d, int) and d >= 0]
    s = str(device).strip().lower()
    if s in ("", "auto", "cpu", "mps"):
        return []
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip().lstrip("-").isdigit() and int(x.strip()) >= 0]
    if s.startswith("cuda:"):
        s = s.split(":", 1)[1]
    if s.lstrip("-").isdigit():
        idx = int(s)
        return [idx] if idx >= 0 else []
    return []


def wants_distributed(device: DeviceArg) -> bool:
    """True iff the device argument names more than one GPU.

    This is a *user intent* check, separate from whether torchrun launched
    the process. A user calling ``model.train(device=[0, 1])`` from a plain
    Python script (no torchrun) signals intent to do DDP; the trainer can
    then raise a clear error pointing them at torchrun.
    """
    return len(parse_device_arg(device)) > 1


# =============================================================================
# Process-group lifecycle
# =============================================================================


def _select_backend() -> str:
    """Pick NCCL when CUDA + NCCL are available, else Gloo.

    NCCL is the fast GPU backend but isn't built on Windows. Gloo works
    everywhere (CPU and GPU) so it's the safe fallback. Windows users
    get Gloo automatically.
    """
    if torch.cuda.is_available() and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def init_distributed(timeout_seconds: int = 10800) -> None:
    """Initialise the default process group from env vars set by torchrun.

    Safe to call multiple times — second and later calls are no-ops.
    Expects ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE`` to be set in the
    environment (which torchrun does automatically).
    """
    import inspect

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build")
    if dist.is_initialized():
        return
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "init_distributed() called without LOCAL_RANK env var. "
            "Multi-GPU training requires launching with torchrun, e.g. "
            "`torchrun --nproc_per_node=2 your_script.py`."
        )
    backend = _select_backend()
    local_rank = int(os.environ["LOCAL_RANK"])
    init_kwargs: dict = {
        "backend": backend,
        "timeout": timedelta(seconds=timeout_seconds),
        "rank": int(os.environ["RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
    }
    # device_id was added in PyTorch 2.0; guard so we stay compatible with older builds
    pg_sig = inspect.signature(dist.init_process_group)
    if "device_id" in pg_sig.parameters and torch.cuda.is_available():
        init_kwargs["device_id"] = torch.device("cuda", local_rank)
    dist.init_process_group(**init_kwargs)


def shutdown_distributed() -> None:
    """Tear down the default process group if it was initialised."""
    if is_distributed():
        dist.destroy_process_group()


# =============================================================================
# Model unwrapping
# =============================================================================


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP / DataParallel / torch.compile wrappers from a module.

    Idempotent. Returns ``model`` unchanged if no wrappers are present.
    Required when reading ``model.named_parameters()`` for optimizer setup
    after DDP wrap, for state-dict saving, and when model-specific hooks
    need to read attributes that live on the unwrapped module.
    """
    parallel_types = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    while True:
        if isinstance(model, parallel_types):
            model = model.module
            continue
        # torch.compile() wraps modules with an _orig_mod attribute
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        return model


# =============================================================================
# EMA buffer broadcast
# =============================================================================


def broadcast_ema_buffers(ema_module: nn.Module, src: int = 0) -> None:
    """Broadcast all buffers of ``ema_module`` from ``src`` rank to all others.

    Required because EMA is only updated on rank 0 (to match Ultralytics's
    pattern, where the optimizer step also fires per-rank but EMA state
    diverges across ranks if updated everywhere). Before validation runs
    on a non-zero rank, EMA buffers need to be the same as rank 0's.
    """
    if not is_distributed():
        return
    for buf in ema_module.buffers():
        dist.broadcast(buf, src=src)
    for p in ema_module.parameters():
        dist.broadcast(p.data, src=src)


# =============================================================================
# Loss scaling for DDP
# =============================================================================


def scale_loss_for_ddp(loss: torch.Tensor) -> torch.Tensor:
    """Multiply loss by world_size so DDP gradient averaging composes correctly.

    DDP all-reduces gradients during ``backward()`` and divides by world_size
    (an average). For sum-style losses we want the final gradient to be
    ``sum_r dL_r/dθ``, not ``mean_r dL_r/dθ``. Pre-multiplying by world_size
    cancels DDP's 1/N averaging exactly:

        per-rank grad after backward = N * dL_r/dθ
        DDP-averaged grad            = (1/N) * sum_r (N * dL_r/dθ) = sum_r dL_r/dθ

    Matches Ultralytics's pattern (loss *= world_size before backward, no
    no_sync() for accumulation). For mean-normalized losses (yolo9's
    cls_norm, DETR's num_boxes) the result diverges slightly from single-GPU
    semantics — that divergence is the per-rank-normalizer effect and is
    accepted as intentional Ultralytics-mirror behavior.

    No-op outside DDP.
    """
    if not is_distributed():
        return loss
    return loss * float(get_world_size())


# =============================================================================
# Seeding
# =============================================================================


def seed_for_rank(base_seed: int) -> int:
    """Per-rank seed: ``base_seed + 1 + rank``.

    Matches Ultralytics's convention. Ensures different augmentation /
    dataloader shuffling across ranks while keeping the run reproducible
    when ``base_seed`` and ``world_size`` are fixed.
    """
    return base_seed + 1 + get_rank()


# =============================================================================
# Auto-spawn DDP helpers
# =============================================================================


def _find_free_port() -> tuple:
    """Bind to port 0 and return ``(port, socket)``.

    The caller is responsible for closing the socket.  Keeping it open
    until just before ``mp.spawn`` is called minimises the TOCTOU window
    between OS port selection and torch.distributed's TCPStore binding.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", 0))
    port = s.getsockname()[1]
    return port, s


def spawn_ddp_train(
    worker_fn: Callable,
    spawn_args: Tuple,
    nprocs: int,
    result_path: str,
    master_addr: str = "127.0.0.1",
    master_port: Optional[int] = None,
    devices: Optional[List[int]] = None,
) -> None:
    """Spawn *nprocs* DDP workers via :func:`torch.multiprocessing.spawn`.

    Each worker is called as::

        worker_fn(rank, nprocs, master_addr, master_port, result_path, *spawn_args)

    The worker is responsible for setting RANK/LOCAL_RANK/WORLD_SIZE/MASTER_*
    env vars, initialising the process group, running training, and writing a
    result JSON to *result_path* (rank 0 only).

    This is the internal engine behind the auto-spawn path triggered when a
    user calls ``model.train(device="0,1")`` from a plain Python script (no
    torchrun). The model's ``train()`` method calls this helper, collects the
    result JSON from *result_path*, and returns it to the caller — so the user
    gets a clean blocking call without any subprocess plumbing.

    When *devices* is provided, ``CUDA_VISIBLE_DEVICES`` is set to the
    comma-joined device indices before spawning so that ``cuda:N`` inside each
    worker maps to the N-th requested physical GPU.  The original value is
    restored after spawning completes.
    """
    import torch.multiprocessing as mp

    port_sock = None
    if master_port is None:
        master_port, port_sock = _find_free_port()

    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
    try:
        # Close the reservation socket as late as possible — just before
        # spawning — so the OS cannot hand the port to another process in the
        # gap between our bind(0) call and torch.distributed's TCPStore bind.
        if port_sock is not None:
            port_sock.close()
            port_sock = None
        mp.spawn(
            worker_fn,
            args=(nprocs, master_addr, master_port, result_path) + spawn_args,
            nprocs=nprocs,
            join=True,
        )
    finally:
        if port_sock is not None:
            port_sock.close()
        if devices:
            if prev_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd


__all__ = [
    "DeviceArg",
    "barrier",
    "broadcast_ema_buffers",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "has_torchrun_env",
    "init_distributed",
    "is_distributed",
    "is_main_process",
    "parse_device_arg",
    "scale_loss_for_ddp",
    "seed_for_rank",
    "shutdown_distributed",
    "spawn_ddp_train",
    "unwrap_model",
    "wants_distributed",
]
