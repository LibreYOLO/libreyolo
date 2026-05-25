"""Generic DDP worker + spawn helper for all LibreYOLO model train() methods.

Each model's train() calls spawn_for_model() when device='0,1' (or similar)
is given and we're NOT in a torchrun environment.  spawn_for_model() handles:

  1. Saving model weights to a temp file.
  2. Resolving batch=-1 via autobatch (single-GPU probe, before spawning).
  3. Spawning DDP workers via mp.spawn.
  4. Loading the best checkpoint back into the caller's model instance.

The generic worker (_libreyolo_ddp_worker) re-imports the correct model class
using module/class info packed into init_kw, rebuilds the model from saved
weights, and calls model.train(**train_kw) — which falls through to the
single-device path because has_torchrun_env() returns True inside the spawned
worker (RANK env var is set).
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level worker (must be importable by name for mp.spawn pickling)
# ---------------------------------------------------------------------------


def _libreyolo_ddp_worker(
    rank: int,
    nprocs: int,
    master_addr: str,
    master_port: int,
    result_path: str,
    weights_path: str,
    init_kw: dict,
    train_kw: dict,
) -> None:
    """Generic DDP worker: reconstruct any LibreYOLO model and start training."""
    import importlib

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(nprocs)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    init_kw = dict(init_kw)  # copy — don't mutate caller's dict
    module_path = init_kw.pop("_module")
    class_name = init_kw.pop("_class")
    cls = getattr(importlib.import_module(module_path), class_name)

    model = cls(weights_path, **init_kw)
    result = model.train(**train_kw)

    if rank == 0:
        safe = {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool, type(None)))}
        Path(result_path).write_text(json.dumps(safe))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_init_kw(model_instance: Any) -> dict:
    """Collect __init__ kwargs needed to reconstruct *model_instance* in a worker.

    Uses inspect so each model's specific params are included automatically
    without needing to enumerate them per-model.
    """
    cls = type(model_instance)
    sig = inspect.signature(cls.__init__)
    supported = set(sig.parameters) - {"self", "model_path", "kwargs"}

    kw: dict = {
        "_module": cls.__module__,
        "_class": cls.__name__,
        "device": "auto",
    }
    for attr in ("size", "nb_classes", "reg_max", "task", "num_masks", "proto_channels", "num_keypoints"):
        if attr in supported and hasattr(model_instance, attr):
            kw[attr] = getattr(model_instance, attr)
    return kw


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def spawn_for_model(
    model_instance: Any,
    train_kw: dict,
    nprocs: int,
    *,
    batch_key: str = "batch",
) -> dict:
    """Save weights, optionally probe autobatch, spawn DDP workers, return results.

    Args:
        model_instance: LibreYOLO model object (has .model nn.Module).
        train_kw: All kwargs forwarded to model.train() inside workers.
        nprocs: Number of DDP ranks (= number of GPUs).
        batch_key: Key in train_kw for batch size (default 'batch').

    Returns:
        Training result dict from rank-0 worker.
    """
    from libreyolo.training.distributed import spawn_ddp_train

    fd, tmp_weights = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(
        {"model": {k: v.cpu() for k, v in model_instance.model.state_dict().items()}},
        tmp_weights,
    )

    # Resolve batch=-1 here (main process, before spawning) so every worker
    # receives a concrete integer and needs no inter-process coordination.
    if train_kw.get(batch_key) == -1:
        from libreyolo.training.autobatch import resolve_auto_batch

        probe_device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        model_instance.model.to(probe_device)
        nbs = train_kw.get("nbs") or 64
        imgsz = train_kw.get("imgsz") or getattr(model_instance, "input_size", None) or 640
        resolved = resolve_auto_batch(
            model_instance.model,
            imgsz=int(imgsz),
            amp=bool(train_kw.get("amp", True)),
            world_size=nprocs,
            nbs=nbs,
        )
        train_kw = {**train_kw, batch_key: resolved}
        model_instance.model.cpu()
        torch.cuda.empty_cache()
        logger.info("AutoBatch (pre-spawn): resolved global batch = %d", resolved)

    init_kw = _build_init_kw(model_instance)

    fd, tmp_result = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    try:
        spawn_ddp_train(
            _libreyolo_ddp_worker,
            spawn_args=(tmp_weights, init_kw, train_kw),
            nprocs=nprocs,
            result_path=tmp_result,
        )
    finally:
        Path(tmp_weights).unlink(missing_ok=True)

    result: dict = {}
    tmp_result_path = Path(tmp_result)
    if tmp_result_path.exists():
        result = json.loads(tmp_result_path.read_text())
        tmp_result_path.unlink()

    # Prefer best.pt; fall back to last.pt (e.g. single-epoch runs where
    # validation mAP is 0 so is_best is never set).
    best = result.get("best_checkpoint")
    last = result.get("last_checkpoint")
    ckpt = next((p for p in (best, last) if p and Path(p).exists()), None)
    if ckpt:
        if hasattr(model_instance, "model_path"):
            model_instance.model_path = ckpt
        model_instance._load_weights(ckpt)
        if hasattr(model_instance, "model"):
            target = (
                torch.device("cuda", 0)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            model_instance.model.to(target).eval()

    return result


__all__ = ["spawn_for_model", "_libreyolo_ddp_worker", "_build_init_kw"]
