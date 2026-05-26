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

import functools
import inspect
import json
import logging
import os
import pickle
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

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    init_kw = dict(init_kw)  # copy — don't mutate caller's dict
    module_path = init_kw.pop("_module")
    class_name = init_kw.pop("_class")
    cls = getattr(importlib.import_module(module_path), class_name)

    model = cls(weights_path, **init_kw)
    try:
        result = model.train(**train_kw)
    finally:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0:
        safe = {}
        for k, v in result.items():
            try:
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                pass
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


def _filter_picklable(kw: dict) -> dict:
    """Raise RuntimeError if any value in *kw* is not picklable.

    mp.spawn serialises its args via pickle; callbacks and locally-defined
    functions will crash workers at start-up.  Raising early with a clear
    message is better than a confusing PicklingError deep inside spawn, or
    silently missing training kwargs.
    """
    bad = []
    for k, v in kw.items():
        try:
            pickle.dumps(v)
        except Exception:
            bad.append(f"{k!r} (type: {type(v).__name__})")
    if bad:
        raise RuntimeError(
            "DDP spawn: the following train() kwargs are not picklable and "
            "cannot be passed to worker processes:\n"
            + "\n".join(f"  {b}" for b in bad)
            + "\nRemove or replace them before calling train() with a multi-GPU device."
        )
    return dict(kw)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def spawn_for_model(
    model_instance: Any,
    train_kw: dict,
    nprocs: int,
    *,
    devices: list | None = None,
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

    # When resuming, workers must load from the real checkpoint so that
    # trainer.resume() can read 'epoch', 'optimizer', etc.  The temp-weights
    # file only carries a plain state dict which lacks those keys.
    resuming = bool(train_kw.get("resume")) and getattr(model_instance, "model_path", None)
    if resuming:
        tmp_weights = str(model_instance.model_path)
        tmp_weights_to_delete = None
    else:
        fd, tmp_weights = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save(
            {k: v.cpu() for k, v in model_instance.model.state_dict().items()},
            tmp_weights,
        )
        tmp_weights_to_delete = tmp_weights

    # Resolve batch=-1 here (main process, before spawning) so every worker
    # receives a concrete integer and needs no inter-process coordination.
    if train_kw.get(batch_key) == -1:
        from libreyolo.training.autobatch import resolve_auto_batch

        first_device = devices[0] if devices else 0
        probe_device = torch.device("cuda", first_device) if torch.cuda.is_available() else torch.device("cpu")
        model_instance.model.to(probe_device)
        nbs = train_kw.get("nbs")  # None = uncapped, matches trainer path
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
    safe_train_kw = _filter_picklable(train_kw)

    fd, tmp_result = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    try:
        spawn_ddp_train(
            _libreyolo_ddp_worker,
            spawn_args=(tmp_weights, init_kw, safe_train_kw),
            nprocs=nprocs,
            result_path=tmp_result,
            devices=devices,
        )
    finally:
        if tmp_weights_to_delete:
            Path(tmp_weights_to_delete).unlink(missing_ok=True)

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
            first_device = devices[0] if devices else 0
            target = (
                torch.device("cuda", first_device)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            model_instance.model.to(target).eval()
            model_instance.device = target

    return result


# ---------------------------------------------------------------------------
# DDP-aware decorator
# ---------------------------------------------------------------------------


def ddp_aware(batch_key: str = "batch", experimental_key: str | None = None):
    """Decorator that adds automatic DDP spawn to a model ``train()`` method.

    When the decorated method is called with a multi-GPU device spec from
    outside a torchrun process, it captures all arguments via
    :mod:`inspect`, delegates to :func:`spawn_for_model`, and returns the
    result without running the method body on the calling process.

    Args:
        batch_key: Key in the captured kwargs that holds the batch size.
            Defaults to ``"batch"``; RF-DETR uses ``"batch_size"``.
        experimental_key: If set, names a boolean kwarg (e.g.
            ``"allow_experimental"``) that must be truthy before DDP spawn
            is attempted. When it is falsy the decorator falls through to
            the function body immediately, letting the body raise its own
            validation error on the main process rather than inside a
            spawned worker.
    """
    def decorator(train_fn):
        @functools.wraps(train_fn)
        def wrapper(self, *args, **kwargs):
            import multiprocessing
            from libreyolo.training.distributed import parse_device_arg, has_torchrun_env

            # Prevent recursive spawn: if we're already inside a spawned worker
            # process, fall straight through to the single-device training path.
            if multiprocessing.parent_process() is not None:
                return train_fn(self, *args, **kwargs)

            sig = inspect.signature(train_fn)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()

            train_kw: dict = {}
            for k, v in bound.arguments.items():
                if k == "self":
                    continue
                elif k == "kwargs":  # **kwargs parameter — flatten into train_kw
                    train_kw.update(v)
                else:
                    train_kw[k] = v

            device = train_kw.get("device", "")
            devices = parse_device_arg(device)
            if len(devices) > 1 and not has_torchrun_env():
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        f"Multi-GPU DDP requires CUDA. Got device={device!r} but "
                        "CUDA is not available on this machine."
                    )
                if experimental_key and not train_kw.get(experimental_key, True):
                    # Guard not satisfied — fall through so the function body
                    # raises its validation error cleanly on the main process.
                    return train_fn(self, *args, **kwargs)
                return spawn_for_model(self, train_kw, len(devices), devices=devices, batch_key=batch_key)

            return train_fn(self, *args, **kwargs)

        return wrapper
    return decorator


__all__ = ["ddp_aware", "spawn_for_model", "_libreyolo_ddp_worker", "_build_init_kw"]
