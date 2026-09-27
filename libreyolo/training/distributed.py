"""Distributed training utilities for LibreYOLO.

Thin helpers around ``torch.distributed`` so the rest of the trainer can stay
backend-agnostic. All helpers degrade to no-ops when distributed is not
initialised — single-GPU code paths continue to work unchanged.

User-facing surface accepts ``device=[0, 1]`` (or ``device="0,1"``). Model
APIs launch local ranks automatically, while an explicit
``torchrun --nproc_per_node=N`` launch remains supported. Inside each child
process ``init_distributed()`` is called by the trainer; outside DDP
everything is a no-op.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

DeviceArg = Union[str, int, List[int], None]
_LEGACY_SPAWN_ENV_LOCK = threading.Lock()


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
    """True iff a DDP launcher provided this process with ``LOCAL_RANK``."""
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
    """Initialise the default process group from launcher-provided env vars.

    Safe to call multiple times — second and later calls are no-ops.
    Expects ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE`` to be set in the
    environment (which torchrun and LibreYOLO's coordinator do automatically).
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
    # device_id was added in PyTorch 2.0; guard so we stay compatible with older builds.
    # inspect.signature() can raise ValueError/TypeError on some C-extension builds,
    # so wrap defensively and omit the kwarg on failure.
    try:
        pg_sig = inspect.signature(dist.init_process_group)
        if "device_id" in pg_sig.parameters and backend == "nccl" and torch.cuda.is_available():
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
    except (TypeError, ValueError):
        pass
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
# Loss scaling for DDP
# =============================================================================


def all_reduce_avg_scalar(
    value: Union[float, torch.Tensor],
    *,
    device: Optional[torch.device] = None,
    min_value: float = 1.0,
) -> float:
    """Average a per-rank scalar count across ranks: ``sum`` then ``/ world_size``.

    Mirrors the DETR ``num_boxes`` reduction. A mean/ratio-normalized loss must
    divide by the *global* count of positives to be numerically equivalent to
    single-GPU training on the same global batch: with each rank dividing by
    ``global_count / world_size`` and DDP averaging the gradients, the two
    cancel to reproduce the single-GPU gradient exactly (see
    ``scale_loss_for_ddp``).

    The global sum is clamped to ``min_value`` BEFORE dividing by world_size.
    Clamping after the divide (the upstream-DETR order) breaks exactness
    whenever the global positive mass is below world_size: an all-background
    batch would train at exactly 1/world_size gradient — worst on the
    background-heavy datasets from issue #484. Clamp-first keeps the
    single-GPU-equivalence exact for every batch, including empty ones.

    Outside DDP this is just ``max(value, min_value)`` — single-GPU behavior is
    unchanged (identical numeric value to the previous ``max(sum, 1)``).

    This is a COLLECTIVE under DDP: every rank must reach it or the callers
    deadlock. Only call it from code that runs symmetrically on all ranks
    (the training loss); never from rank-0-only paths like validation.

    The trailing ``.item()`` is a full pipeline drain every step. Hot loss
    paths that only need a divisor should call
    ``all_reduce_avg_scalar_tensor`` and keep the value on device
    (issue #763); this float form stays for callers that genuinely need a
    Python number.
    """
    return float(
        all_reduce_avg_scalar_tensor(value, device=device, min_value=min_value).item()
    )


def all_reduce_avg_scalar_tensor(
    value: Union[float, torch.Tensor],
    *,
    device: Optional[torch.device] = None,
    min_value: float = 1.0,
) -> torch.Tensor:
    """``all_reduce_avg_scalar`` without the final device sync.

    Returns the clamped global average as a 0-dim fp32 tensor on the input
    tensor's device, so callers that use it as a loss normalizer never force
    a GPU-to-CPU transfer. Same collective contract as the float form.
    """
    if isinstance(value, torch.Tensor):
        # .sum() materializes a fresh tensor even for 0-dim input, so the
        # in-place all_reduce below cannot touch the caller's storage.
        v = value.detach().float().sum().reshape(())
    else:
        v = torch.as_tensor(float(value), dtype=torch.float32, device=device)
    if is_distributed():
        if v.device.type == "cpu" and dist.get_backend() == "nccl":
            raise ValueError(
                "all_reduce_avg_scalar got a CPU scalar under the NCCL "
                "backend; pass device= (or a tensor already on the right "
                "GPU) so the collective can run."
            )
        dist.all_reduce(v)
        return v.clamp_min(min_value) / float(get_world_size())
    return v.clamp_min(min_value)


def scale_loss_for_ddp(loss: torch.Tensor) -> torch.Tensor:
    """Pass the loss through unchanged — DDP needs no per-loss rescaling here.

    DDP all-reduces gradients during ``backward()`` and **averages** them
    (divides by world_size). Every LibreYOLO loss is mean/ratio-normalized:
    each rank already produces a "full-batch magnitude" gradient (normalized by
    a per-positive / per-box count — globally reduced for yolo9's ``cls_norm``
    and the DETR ``num_boxes``), so DDP's averaging composes them into the
    single-GPU-equivalent gradient. Multiplying by world_size on top of that
    over-counts by ~N and inflates the effective learning rate — this was the
    root cause of the multi-GPU accuracy gap in issue #484 (4-GPU trained at
    ~4× LR vs single-GPU).

    This is a CONTRACT each family's loss must satisfy, not a given: a loss
    that normalizes by a globally-summed count WITHOUT dividing by world_size
    (as RT-DETR's ``num_boxes`` did before #484 fixed it) under-scales by 1/N
    once the identity is in place. New families must either use a local
    normalizer or the ``all_reduce_avg_scalar`` reduction (global sum /
    world_size) — never a bare global sum.

    Kept as the single, documented seam where a *genuinely sum-reduced* loss
    (no normalizer) could opt back into ``loss * world_size``. No family
    currently uses one, so this is the identity in every case, DDP or not.
    """
    return loss


# =============================================================================
# Seeding
# =============================================================================


def seed_for_rank(base_seed: int) -> int:
    """Per-rank seed: ``base_seed + 1 + rank``.

    Ensures different augmentation / dataloader shuffling across ranks while
    keeping the run reproducible when ``base_seed`` and ``world_size`` are
    fixed.
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


def _normalized_import_paths(entries) -> list[str]:
    """Return absolute string paths suitable for the private job manifest."""
    normalized: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        try:
            raw = os.fspath(entry)
        except TypeError:
            continue
        if isinstance(raw, bytes):
            raw = os.fsdecode(raw)
        if not isinstance(raw, str):
            continue
        try:
            absolute = os.path.abspath(raw or os.getcwd())
        except (OSError, TypeError, ValueError):
            continue
        key = os.path.normcase(absolute)
        if key not in seen:
            normalized.append(absolute)
            seen.add(key)
    return normalized


def _legacy_spawn_worker(
    rank: int,
    worker_fn: Callable,
    nprocs: int,
    master_addr: str,
    master_port: int,
    result_path: str,
    spawn_args: Tuple,
) -> None:
    """Invoke a standard-pickle job that cannot use by-value transport."""
    os.environ["LIBREYOLO_DDP_COORDINATOR_WORKER"] = "1"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(nprocs)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    worker_fn(
        rank,
        nprocs,
        master_addr,
        master_port,
        result_path,
        *spawn_args,
    )


def _spawn_standard_pickle_fallback(
    worker_fn: Callable,
    spawn_args: Tuple,
    nprocs: int,
    result_path: str,
    master_addr: str,
    master_port: int,
    child_env: dict[str, str],
) -> None:
    """Preserve the guarded-script behavior of the former direct spawn path."""
    import multiprocessing

    if multiprocessing.current_process().name != "MainProcess":
        raise RuntimeError(
            "DDP spawn fell back to standard-pickle transport inside a spawned "
            "subprocess. Put the multi-GPU train() call under "
            "`if __name__ == '__main__':` for this callback or logger."
        )

    import torch.multiprocessing as mp

    warnings.warn(
        "DDP coordinator transport could not represent an otherwise "
        "standard-picklable callback or logger. Using the guarded-script "
        "compatibility launcher; this call must remain under "
        "`if __name__ == '__main__':`.",
        RuntimeWarning,
        stacklevel=3,
    )
    # Standard multiprocessing imports the guarded user module before calling
    # _legacy_spawn_worker. Apply the child mask around the whole spawn so even
    # import-time CUDA setup sees the translated value. This compatibility-only
    # path is serialized because os.environ is process-global; the normal
    # coordinator path continues to use an explicit child-only environment.
    with _LEGACY_SPAWN_ENV_LOCK:
        previous_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        child_has_cvd = "CUDA_VISIBLE_DEVICES" in child_env
        if child_has_cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = child_env["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            mp.spawn(
                _legacy_spawn_worker,
                args=(
                    worker_fn,
                    nprocs,
                    master_addr,
                    master_port,
                    result_path,
                    spawn_args,
                ),
                nprocs=nprocs,
                join=True,
            )
        finally:
            if previous_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_cvd


def spawn_ddp_train(
    worker_fn: Callable,
    spawn_args: Tuple,
    nprocs: int,
    result_path: str,
    master_addr: str = "127.0.0.1",
    master_port: Optional[int] = None,
    devices: Optional[List[int]] = None,
) -> None:
    """Run *nprocs* DDP workers through LibreYOLO's private coordinator.

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

    When *devices* is provided, the coordinator subprocess receives a
    translated ``CUDA_VISIBLE_DEVICES`` so that ``cuda:N`` inside each worker
    maps to the N-th requested physical GPU without mutating the parent. The
    guarded-script compatibility fallback temporarily applies and then
    restores that mask so multiprocessing imports see the same mapping.
    """
    import multiprocessing

    if (
        os.environ.get("LIBREYOLO_DDP_COORDINATOR_WORKER") == "1"
        or multiprocessing.current_process().name != "MainProcess"
    ):
        raise RuntimeError(
            "spawn_ddp_train() was called from a spawned subprocess. Nested "
            "coordinator launches are not supported; put compatibility-only "
            "callbacks or loggers under `if __name__ == '__main__':`."
        )

    port_sock = None
    if master_port is None:
        master_port, port_sock = _find_free_port()

    # A rare standard-pickle fallback temporarily changes the process-global
    # mask while its children start. Serialize environment snapshots with that
    # window so a concurrent normal coordinator launch cannot inherit it.
    with _LEGACY_SPAWN_ENV_LOCK:
        child_env = os.environ.copy()
    prev_cvd = child_env.get("CUDA_VISIBLE_DEVICES")
    if devices:
        if prev_cvd is not None:
            # devices are logical indices into the existing mask — translate to
            # physical GPU IDs so the new mask is correct inside spawned workers.
            existing = [x.strip() for x in prev_cvd.split(",") if x.strip()]
            try:
                new_cvd = ",".join(existing[d] for d in devices)
            except IndexError:
                new_cvd = ",".join(str(d) for d in devices)
        else:
            new_cvd = ",".join(str(d) for d in devices)
        child_env["CUDA_VISIBLE_DEVICES"] = new_cvd

    try:
        # Serialize the job before releasing the reserved rendezvous port.
        from libreyolo.training._ddp_coordinator import (
            JobTransportError,
            job_workspace,
            launch_coordinator,
            write_job,
        )

        package_root = str(Path(__file__).resolve().parents[2])
        import_paths = _normalized_import_paths((package_root, *sys.path))

        with job_workspace() as (job_dir, cleanup_token):
            try:
                write_job(
                    job_dir,
                    worker_fn=worker_fn,
                    spawn_args=spawn_args,
                    nprocs=nprocs,
                    result_path=result_path,
                    master_addr=master_addr,
                    master_port=master_port,
                    import_paths=import_paths,
                )
            except JobTransportError:
                if port_sock is not None:
                    port_sock.close()
                    port_sock = None
                _spawn_standard_pickle_fallback(
                    worker_fn,
                    spawn_args,
                    nprocs,
                    result_path,
                    master_addr,
                    master_port,
                    child_env,
                )
                return
            # Release the port immediately before the coordinator starts and
            # binds its TCPStore, minimizing the unavoidable bind race.
            if port_sock is not None:
                port_sock.close()
                port_sock = None
            launch_coordinator(
                job_dir,
                env=child_env,
                cleanup_token=cleanup_token,
            )
    finally:
        if port_sock is not None:
            port_sock.close()


__all__ = [
    "DeviceArg",
    "all_reduce_avg_scalar",
    "barrier",
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
