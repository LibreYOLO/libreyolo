"""End-to-end DDP integration test for yolo9 and rf-detr.

Spawns 2 child processes using ``torch.multiprocessing.spawn`` with the
Gloo backend (works on CPU + Windows) and exercises the full BaseTrainer
DDP path: process group init, DDP model wrap, per-rank forward,
``loss * world_size`` backward, optimizer step, parameter sync check
across ranks, and checkpoint round-trip.

This is the load-bearing proof that the DDP plumbing works end-to-end.
Single-GPU regression is covered by the existing trainer smoke tests
(test_dfine_trainer_smoke.py et al.) which keep passing untouched.

Runs on CPU; no GPU required. Slow because of process spawn overhead
(~10–30s on Windows), so the file is intentionally short.
"""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

pytestmark = pytest.mark.unit


# =============================================================================
# Worker entry points (module-level so spawn() can pickle them on Windows)
# =============================================================================


def _setup_pg(rank: int, world_size: int, port: int) -> None:
    """Set env vars + init the gloo process group from inside a child."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _params_match_across_ranks(
    model: nn.Module, atol: float = 1e-6, rtol: float = 1e-5
) -> tuple[bool, str]:
    """All-reduce each parameter and verify all ranks agree.

    For every parameter we sum across ranks; if all ranks hold identical
    values then ``sum == local * world_size`` within fp tolerance. AdamW's
    multiplications introduce small drift, so we allow ~1e-6 atol.

    Returns ``(ok, diagnostic)`` — diagnostic names the first divergent
    parameter and the magnitude of the disagreement.
    """
    world = dist.get_world_size()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        local = p.detach().clone()
        gathered = p.detach().clone()
        dist.all_reduce(gathered, op=dist.ReduceOp.SUM)
        expected = local * world
        if not torch.allclose(gathered, expected, atol=atol, rtol=rtol):
            max_abs = (gathered - expected).abs().max().item()
            max_rel = (
                ((gathered - expected).abs() / expected.abs().clamp_min(1e-12))
                .max()
                .item()
            )
            return False, (
                f"param {name!r} diverged across {world} ranks: "
                f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, "
                f"shape={tuple(p.shape)}"
            )
    return True, "ok"


def _yolo9_ddp_worker(rank: int, world_size: int, port: int, out_dir: str) -> None:
    """One DDP rank that exercises yolo9 forward → backward → step.

    Result: writes a per-rank file ``rank_{rank}.txt`` containing either
    ``ok`` plus diagnostic info, or an ``error: ...`` line. The test then
    checks both files for ``ok``.
    """
    out_path = Path(out_dir) / f"rank_{rank}.txt"
    try:
        _setup_pg(rank, world_size, port)

        from libreyolo import LibreYOLO9
        from libreyolo.training.distributed import (
            get_world_size,
            scale_loss_for_ddp,
            unwrap_model,
        )

        # Tiny model on CPU. Each rank constructs identical weights via
        # the deterministic init path (LibreYOLO9 with no weights).
        torch.manual_seed(0)
        wrapper = LibreYOLO9(None, size="t", device="cpu")
        wrapper.model.train()

        # Wrap with DDP. CPU + gloo: no device_ids.
        ddp_model = nn.parallel.DistributedDataParallel(wrapper.model)

        # Optimiser built from the unwrapped module so named_parameters
        # has no "module." prefix — matches BaseTrainer ordering.
        optimizer = torch.optim.SGD(unwrap_model(ddp_model).parameters(), lr=0.01)

        # Per-rank batches differ so the loss differs across ranks and the
        # all-reduce inside backward() has something non-trivial to do.
        torch.manual_seed(100 + rank)
        imgs = torch.randn(1, 3, 320, 320)
        targets = torch.zeros(1, 30, 5)
        # Real per-rank box so the loss isn't degenerate.
        targets[0, 0] = torch.tensor(
            [float(rank), 160.0, 120.0, 80.0, 60.0]
        )

        # Forward + DDP-scaled backward + step
        out = ddp_model(imgs, targets=targets)
        loss = out["total_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss on rank {rank}: {loss.item()}")
        loss = scale_loss_for_ddp(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # The actual test: after step, all ranks must have identical weights.
        ok, diag = _params_match_across_ranks(ddp_model)
        if not ok:
            raise RuntimeError(f"parameters diverged across ranks after step: {diag}")

        # Checkpoint round-trip on rank 0 (and verify on rank 1 that the
        # checkpoint loads back into a fresh single-process model).
        if rank == 0:
            raw = unwrap_model(ddp_model)
            ckpt_path = Path(out_dir) / "yolo9.pt"
            torch.save({"model": raw.state_dict()}, ckpt_path)
        dist.barrier()
        if rank == 0:
            fresh = LibreYOLO9(None, size="t", device="cpu")
            sd = torch.load(Path(out_dir) / "yolo9.pt", weights_only=False)["model"]
            missing, unexpected = fresh.model.load_state_dict(sd, strict=False)
            if unexpected:
                raise RuntimeError(f"unexpected ckpt keys: {sorted(unexpected)[:5]}")

        dist.barrier()
        world = get_world_size()
        out_path.write_text(
            f"ok world={world} loss={float(loss.detach().item()):.6f}\n"
        )

    except Exception as exc:
        out_path.write_text(f"error: {type(exc).__name__}: {exc}\n")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _rfdetr_ddp_worker(rank: int, world_size: int, port: int, out_dir: str) -> None:
    """Same shape as the yolo9 worker but for RF-DETR."""
    out_path = Path(out_dir) / f"rank_{rank}.txt"
    try:
        _setup_pg(rank, world_size, port)

        from libreyolo import LibreRFDETR
        from libreyolo.models.rfdetr.trainer import RFDETRTrainer
        from libreyolo.training.distributed import (
            get_world_size,
            scale_loss_for_ddp,
            unwrap_model,
        )

        torch.manual_seed(0)
        wrapper = LibreRFDETR(None, size="n", device="cpu", segmentation=False)
        wrapper.model.train()

        # The RF-DETR trainer needs to build its criterion via on_setup, so
        # we instantiate the trainer the same way the existing smoke tests
        # do and let it own the criterion. on_setup is called before DDP
        # wrap (so attribute access on raw model is fine).
        trainer = RFDETRTrainer(
            model=wrapper.model,
            wrapper_model=wrapper,
            size="n",
            num_classes=80,
            data=None,
            epochs=1,
            batch=1,
            imgsz=320,
            device="cpu",
            amp=False,
            ema=False,
            no_aug_epochs=0,
            warmup_epochs=0,
            eval_interval=-1,
        )
        trainer.on_setup()  # builds criterion

        # Wrap with DDP after on_setup. RF-DETR's transformer self-attention
        # produces a non-contiguous gradient layout (Grad strides do not
        # match bucket view strides) that breaks the default DDP reducer
        # under CPU/Gloo. gradient_as_bucket_view=False relaxes the bucket
        # constraint; static_graph=True defers reducer analysis to after
        # the first iteration so the layout is detected correctly.
        find_unused = trainer._ddp_find_unused_parameters()
        ddp_model = nn.parallel.DistributedDataParallel(
            wrapper.model,
            find_unused_parameters=find_unused,
            gradient_as_bucket_view=False,
            static_graph=not find_unused,
        )
        # Replace trainer.model with the wrapped one for on_forward.
        trainer.model = ddp_model

        optimizer = torch.optim.AdamW(
            unwrap_model(ddp_model).parameters(), lr=1e-4
        )

        torch.manual_seed(200 + rank)
        imgs = torch.randn(1, 3, 320, 320)
        targets = torch.zeros(1, 30, 5)
        targets[0, 0] = torch.tensor(
            [float(rank), 160.0, 120.0, 80.0, 60.0]
        )

        out = trainer.on_forward(imgs, targets)
        loss = out["total_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss on rank {rank}: {loss.item()}")
        loss = scale_loss_for_ddp(loss)
        optimizer.zero_grad()
        loss.backward()

        # First check gradients are synced across ranks BEFORE the step. If
        # this fails, the DDP all_reduce isn't reaching the diverged param.
        for name, p in unwrap_model(ddp_model).named_parameters():
            if p.grad is None or not p.requires_grad:
                continue
            local = p.grad.detach().clone()
            gathered = p.grad.detach().clone()
            dist.all_reduce(gathered, op=dist.ReduceOp.SUM)
            expected = local * world_size
            if not torch.allclose(gathered, expected, atol=1e-6, rtol=1e-5):
                max_abs = (gathered - expected).abs().max().item()
                raise RuntimeError(
                    f"GRAD for {name!r} not synced across ranks: max_abs={max_abs:.3e}"
                )

        optimizer.step()

        ok, diag = _params_match_across_ranks(ddp_model)
        if not ok:
            raise RuntimeError(f"parameters diverged across ranks after step: {diag}")

        if rank == 0:
            raw = unwrap_model(ddp_model)
            ckpt_path = Path(out_dir) / "rfdetr.pt"
            torch.save({"model": raw.state_dict()}, ckpt_path)
        dist.barrier()
        if rank == 0:
            fresh = LibreRFDETR(None, size="n", device="cpu", segmentation=False)
            sd = torch.load(Path(out_dir) / "rfdetr.pt", weights_only=False)["model"]
            missing, unexpected = fresh.model.load_state_dict(sd, strict=False)
            if unexpected:
                raise RuntimeError(f"unexpected ckpt keys: {sorted(unexpected)[:5]}")

        dist.barrier()
        world = get_world_size()
        out_path.write_text(
            f"ok world={world} loss={float(loss.detach().item()):.6f}\n"
        )

    except Exception as exc:
        out_path.write_text(f"error: {type(exc).__name__}: {exc}\n")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# =============================================================================
# Helpers
# =============================================================================


def _free_port() -> int:
    """Pick an ephemeral free port for the rendezvous master."""
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _spawn_and_check(worker, n_ranks: int, tmp_path) -> dict:
    """Spawn ``n_ranks`` worker processes and return per-rank result text."""
    port = _free_port()
    out_dir = str(tmp_path)
    try:
        mp.spawn(
            worker,
            args=(n_ranks, port, out_dir),
            nprocs=n_ranks,
            join=True,
        )
    except Exception as exc:
        # Surface child-process errors with whatever the child wrote.
        outputs = {
            rank: (tmp_path / f"rank_{rank}.txt").read_text()
            if (tmp_path / f"rank_{rank}.txt").exists()
            else "<no output>"
            for rank in range(n_ranks)
        }
        raise AssertionError(
            f"spawn failed: {exc}\nper-rank outputs: {outputs}"
        ) from exc

    return {
        rank: (tmp_path / f"rank_{rank}.txt").read_text()
        for rank in range(n_ranks)
    }


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.skipif(
    sys.platform == "win32" and sys.version_info < (3, 8),
    reason="mp.spawn on Windows needs Python 3.8+",
)
def test_yolo9_ddp_2_ranks_cpu_gloo(tmp_path):
    """Two-rank DDP smoke for yolo9. Proves: process group init, DDP wrap,
    forward, loss-scale backward, optimizer step, cross-rank parameter
    equality, checkpoint save+load round-trip.
    """
    outputs = _spawn_and_check(_yolo9_ddp_worker, n_ranks=2, tmp_path=tmp_path)
    for rank, text in outputs.items():
        assert text.startswith("ok "), f"rank {rank} did not finish ok: {text!r}"


@pytest.mark.skipif(
    sys.platform == "win32" and sys.version_info < (3, 8),
    reason="mp.spawn on Windows needs Python 3.8+",
)
def test_rfdetr_ddp_2_ranks_cpu_gloo(tmp_path):
    """Two-rank DDP smoke for rf-detr. Same coverage as the yolo9 test.

    Also exercises RF-DETR's criterion all_reduce(num_boxes) path on the
    backward — that path was already in libreyolo's rfdetr loss.py:518-520
    but it's never exercised without an actual process group.
    """
    outputs = _spawn_and_check(_rfdetr_ddp_worker, n_ranks=2, tmp_path=tmp_path)
    for rank, text in outputs.items():
        assert text.startswith("ok "), f"rank {rank} did not finish ok: {text!r}"


def test_parse_device_arg_and_wants_distributed():
    """Unit-only sanity checks for the device-argument parser. These run in
    the parent test process and don't need spawn."""
    from libreyolo.training.distributed import parse_device_arg, wants_distributed

    assert parse_device_arg(0) == [0]
    assert parse_device_arg([0, 1]) == [0, 1]
    assert parse_device_arg("0,1") == [0, 1]
    assert parse_device_arg("cpu") == []
    assert parse_device_arg("auto") == []
    assert parse_device_arg("cuda:0") == [0]
    assert parse_device_arg(-1) == []

    assert wants_distributed([0, 1])
    assert wants_distributed("0,1,2")
    assert not wants_distributed(0)
    assert not wants_distributed("0")
    assert not wants_distributed("cpu")
    assert not wants_distributed("auto")


def test_syncbn_weights_land_in_no_weight_decay_group():
    """Regression: SyncBatchNorm is a sibling of BatchNorm2d (both subclass
    ``_BatchNorm``), not a subclass of it. An ``isinstance(v, nn.BatchNorm2d)``
    check silently moves SyncBN weights into the weight-decay group post
    conversion — masquerades as a tiny training-quality regression under
    DDP+sync_bn=True. Verify _setup_optimizer's grouping covers all batch-
    norm flavours.
    """
    from libreyolo import LibreYOLO9
    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    wrapper = LibreYOLO9(None, size="t", device="cpu")
    # Convert plain BN to SyncBN as ``setup()`` would under DDP+sync_bn.
    wrapper.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(wrapper.model)

    trainer = YOLO9Trainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="t",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=320,
        device="cpu",
        amp=False,
        ema=False,
        no_aug_epochs=0,
        warmup_epochs=0,
        eval_interval=-1,
    )
    optimizer = trainer._setup_optimizer()

    # Find every SyncBN weight tensor and verify it's in pg0 (no-WD group).
    bn_param_ids = {
        id(m.weight)
        for m in wrapper.model.modules()
        if isinstance(m, torch.nn.SyncBatchNorm) and m.weight is not None
    }
    assert bn_param_ids, "test precondition: expected at least one SyncBN layer"

    pg0_param_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    pg_with_wd_ids = {
        id(p)
        for g in optimizer.param_groups
        if g.get("weight_decay", 0.0) > 0
        for p in g["params"]
    }

    misplaced = bn_param_ids - pg0_param_ids
    assert not misplaced, (
        f"{len(misplaced)} SyncBN weights missing from no-WD pg0 — "
        "the isinstance check probably excludes SyncBN"
    )
    leaked = bn_param_ids & pg_with_wd_ids
    assert not leaked, (
        f"{len(leaked)} SyncBN weights ended up in a weight-decay group"
    )


def test_multi_gpu_device_raises_without_torchrun():
    """If the user passes ``device=[0,1]`` without launching with torchrun
    the trainer must fail with a clear message pointing them at it."""
    # Build a minimal trainer-like setup that exercises _setup_device alone.
    from libreyolo import LibreYOLO9
    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    # CUDA may or may not be available; the parser fails CUDA-missing case
    # before the torchrun check. Skip if CUDA truly absent so we exercise
    # the torchrun-pointer path.
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA to exercise the torchrun-missing path")

    wrapper = LibreYOLO9(None, size="t", device="cpu")
    with pytest.raises(RuntimeError, match="torchrun"):
        YOLO9Trainer(
            model=wrapper.model,
            wrapper_model=wrapper,
            size="t",
            num_classes=80,
            data=None,
            epochs=1,
            batch=2,
            imgsz=320,
            device=[0, 1],  # multi-GPU intent without torchrun env
            amp=False,
            ema=False,
            no_aug_epochs=0,
            warmup_epochs=0,
            eval_interval=-1,
        )
