"""DDP smoke test designed to be launched with torchrun.

Complements hw_ddp_smoke.py (mp.spawn + Gloo on one GPU) by exercising the
real torchrun launch path: LOCAL_RANK env var → has_torchrun_env() → init_distributed().

Launch:
    # 2 ranks on the same machine (shares GPU(s) if only one available)
    torchrun --nproc_per_node=2 tools/ddp_torchrun_smoke.py

    # Single model
    torchrun --nproc_per_node=2 tools/ddp_torchrun_smoke.py --model rfdetr

    # More steps
    torchrun --nproc_per_node=2 tools/ddp_torchrun_smoke.py --steps 5

    # CPU-only (uses Gloo automatically)
    torchrun --nproc_per_node=2 tools/ddp_torchrun_smoke.py --device cpu

What this covers beyond the existing tests:
    - torchrun env path: has_torchrun_env() → init_distributed() (BaseTrainer.__init__ path)
    - NCCL backend when CUDA available (Gloo fallback on CPU or single-GPU)
    - Multiple training steps
    - EMA buffer broadcast from rank-0 to all ranks (broadcast_ema_buffers)
    - DistributedSampler: each rank gets distinct indices, set_epoch changes them
    - Checkpoint save on rank-0 + load round-trip into a single-process model
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_torchrun_env() -> None:
    if "LOCAL_RANK" not in os.environ:
        sys.exit(
            "\nERROR: LOCAL_RANK not in environment.\n"
            "Launch this script with torchrun, not python directly:\n\n"
            "    torchrun --nproc_per_node=2 tools/ddp_torchrun_smoke.py\n"
        )


def _log(rank: int, msg: str) -> None:
    print(f"[rank {rank}] {msg}", flush=True)


def _ok(rank: int, msg: str) -> None:
    print(f"[rank {rank}] OK  {msg}", flush=True)


def _params_match(model: nn.Module, world_size: int, atol: float = 2e-5) -> tuple[bool, str]:
    """All-reduce each param and verify sum == local * world_size within tolerance."""
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        local = p.detach().clone()
        summed = p.detach().clone()
        dist.all_reduce(summed, op=dist.ReduceOp.SUM)
        expected = local * world_size
        if not torch.allclose(summed, expected, atol=atol, rtol=1e-4):
            delta = (summed - expected).abs().max().item()
            return False, f"param {name!r} diverged: max_abs={delta:.3e}"
    return True, "ok"


# ---------------------------------------------------------------------------
# DistributedSampler test
# ---------------------------------------------------------------------------


def run_sampler_test(rank: int, world_size: int) -> None:
    """Verify DistributedSampler gives non-overlapping indices and that
    set_epoch changes the shuffled order."""
    dataset = TensorDataset(torch.arange(200))
    sampler0 = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                   shuffle=True, drop_last=True)

    # Epoch 0 indices for this rank
    sampler0.set_epoch(0)
    indices_e0 = list(sampler0)

    # Epoch 1 should differ
    sampler0.set_epoch(1)
    indices_e1 = list(sampler0)

    # Gather all indices across ranks. NCCL requires CUDA tensors; Gloo works on CPU.
    backend = dist.get_backend()
    tensor_device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
    local_t = torch.tensor(indices_e0, dtype=torch.long, device=tensor_device)
    all_tensors = [torch.zeros_like(local_t) for _ in range(world_size)]
    dist.all_gather(all_tensors, local_t)

    if rank == 0:
        # Check no overlap between any two ranks
        seen: set[int] = set()
        for r, t in enumerate(all_tensors):
            idxs = set(t.tolist())
            overlap = seen & idxs
            assert not overlap, f"ranks share indices: {overlap}"
            seen |= idxs

        # Check epoch shuffle changes indices
        assert indices_e0 != indices_e1, "set_epoch did not change indices"

    _ok(rank, "sampler: non-overlapping indices, set_epoch changes order")


# ---------------------------------------------------------------------------
# EMA broadcast test
# ---------------------------------------------------------------------------


def run_ema_broadcast_test(rank: int, world_size: int, device: torch.device) -> None:
    """Verify broadcast_ema_buffers syncs rank-0's EMA to all other ranks.

    Pattern: all ranks create identical EMA, rank-0 modifies it, all call
    broadcast, then verify values are identical again.
    """
    from libreyolo import LibreYOLO9
    from libreyolo.training.distributed import broadcast_ema_buffers
    from libreyolo.training.ema import ModelEMA

    torch.manual_seed(42)
    wrapper = LibreYOLO9(None, size="t", device=str(device))
    wrapper.model.eval()

    ema = ModelEMA(wrapper.model)

    # Only rank 0 updates EMA — simulates N training steps on rank-0 only
    if rank == 0:
        with torch.no_grad():
            for p in ema.ema.parameters():
                p.data.mul_(1.5).add_(0.3)

    dist.barrier()

    # Before broadcast: verify params differ across ranks (all_reduce sum != local * world)
    if world_size > 1:
        any_param = next(iter(ema.ema.parameters()))
        local = any_param.detach().clone()
        summed = any_param.detach().clone()
        dist.all_reduce(summed, op=dist.ReduceOp.SUM)
        pre_diverged = not torch.allclose(summed, local * world_size, atol=1e-5)
    else:
        pre_diverged = True  # trivially: single rank, nothing to diverge

    # All ranks call broadcast (collective: rank-0 sends, others receive)
    broadcast_ema_buffers(ema.ema, src=0)

    # After broadcast: verify all ranks hold rank-0's EMA values.
    # EMA params have requires_grad=False so _params_match skips them; use
    # all_reduce directly on all params (no grad involved).
    backend = dist.get_backend()
    for name, p in ema.ema.named_parameters():
        local = p.data.detach().clone()
        summed = p.data.detach().clone()
        dist.all_reduce(summed, op=dist.ReduceOp.SUM)
        expected = local * world_size
        if not torch.allclose(summed, expected, atol=1e-5, rtol=1e-4):
            delta = (summed - expected).abs().max().item()
            raise RuntimeError(f"EMA broadcast failed for param {name!r}: max_abs={delta:.3e}")

    _ok(rank, f"EMA broadcast: {'pre-diverged as expected, ' if pre_diverged else ''}post-broadcast synced")


# ---------------------------------------------------------------------------
# YOLO9 multi-step DDP test
# ---------------------------------------------------------------------------


def run_yolo9(rank: int, world_size: int, device: torch.device, steps: int, out_dir: Path) -> None:
    from libreyolo import LibreYOLO9
    from libreyolo.training.distributed import scale_loss_for_ddp, unwrap_model

    _log(rank, f"yolo9: building model on {device}")
    torch.manual_seed(0)
    wrapper = LibreYOLO9(None, size="t", device=str(device))
    wrapper.model.train()

    ddp_kwargs: dict = dict(gradient_as_bucket_view=False, static_graph=True)
    if device.type == "cuda":
        ddp_kwargs["device_ids"] = [device.index]
        ddp_kwargs["output_device"] = device.index
    ddp_model = nn.parallel.DistributedDataParallel(wrapper.model, **ddp_kwargs)
    optimizer = torch.optim.SGD(unwrap_model(ddp_model).parameters(), lr=0.01, momentum=0.9)

    for step in range(steps):
        torch.manual_seed(step * 1000 + rank)
        imgs = torch.randn(1, 3, 320, 320, device=device)
        targets = torch.zeros(1, 30, 5, device=device)
        targets[0, 0] = torch.tensor([float(rank), 160., 120., 80., 60.], device=device)

        out = ddp_model(imgs, targets=targets)
        loss = out["total_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        optimizer.zero_grad()
        scale_loss_for_ddp(loss).backward()
        optimizer.step()

        ok, diag = _params_match(unwrap_model(ddp_model), world_size)
        if not ok:
            raise RuntimeError(f"step {step}: {diag}")
        _log(rank, f"yolo9 step {step + 1}/{steps}  loss={loss.item():.4f}  params: synced")

    # Checkpoint round-trip (rank-0 saves, rank-0 reloads into fresh single-GPU model)
    if rank == 0:
        ckpt = out_dir / "yolo9.pt"
        torch.save({"model": unwrap_model(ddp_model).state_dict()}, ckpt)
    dist.barrier()
    if rank == 0:
        fresh = LibreYOLO9(None, size="t", device="cpu")
        sd = torch.load(out_dir / "yolo9.pt", weights_only=False)["model"]
        _, unexpected = fresh.model.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected checkpoint keys: {sorted(unexpected)[:5]}")
        _ok(rank, "yolo9 checkpoint round-trip: ok")
    dist.barrier()

    _ok(rank, f"yolo9: {steps} steps, param sync, checkpoint — PASS")


# ---------------------------------------------------------------------------
# RF-DETR multi-step DDP test
# ---------------------------------------------------------------------------


def run_rfdetr(rank: int, world_size: int, device: torch.device, steps: int, out_dir: Path) -> None:
    from libreyolo import LibreRFDETR
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer
    from libreyolo.training.distributed import scale_loss_for_ddp, unwrap_model

    _log(rank, f"rfdetr: building model on {device}")
    torch.manual_seed(0)
    wrapper = LibreRFDETR(None, size="n", device=str(device), segmentation=False)
    wrapper.model.train()

    trainer = RFDETRTrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
        num_classes=80,
        data=None,
        epochs=1,
        batch=1,
        imgsz=320,
        device=str(device),
        amp=False,
        ema=False,
        no_aug_epochs=0,
        warmup_epochs=0,
        eval_interval=-1,
    )
    trainer.on_setup()

    find_unused = trainer._ddp_find_unused_parameters()
    ddp_kwargs: dict = dict(
        find_unused_parameters=find_unused,
        gradient_as_bucket_view=False,
        static_graph=not find_unused,
    )
    if device.type == "cuda":
        ddp_kwargs["device_ids"] = [device.index]
        ddp_kwargs["output_device"] = device.index
    ddp_model = nn.parallel.DistributedDataParallel(wrapper.model, **ddp_kwargs)
    trainer.model = ddp_model
    optimizer = torch.optim.AdamW(unwrap_model(ddp_model).parameters(), lr=1e-4)

    for step in range(steps):
        torch.manual_seed(step * 1000 + rank)
        imgs = torch.randn(1, 3, 320, 320, device=device)
        targets = torch.zeros(1, 30, 5, device=device)
        targets[0, 0] = torch.tensor([float(rank), 160., 120., 80., 60.], device=device)

        out = trainer.on_forward(imgs, targets)
        loss = out["total_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        optimizer.zero_grad()
        scale_loss_for_ddp(loss).backward()
        optimizer.step()

        ok, diag = _params_match(unwrap_model(ddp_model), world_size)
        if not ok:
            raise RuntimeError(f"step {step}: {diag}")
        _log(rank, f"rfdetr step {step + 1}/{steps}  loss={loss.item():.4f}  params: synced")

    if rank == 0:
        ckpt = out_dir / "rfdetr.pt"
        torch.save({"model": unwrap_model(ddp_model).state_dict()}, ckpt)
    dist.barrier()
    if rank == 0:
        fresh = LibreRFDETR(None, size="n", device="cpu", segmentation=False)
        sd = torch.load(out_dir / "rfdetr.pt", weights_only=False)["model"]
        _, unexpected = fresh.model.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected checkpoint keys: {sorted(unexpected)[:5]}")
        _ok(rank, "rfdetr checkpoint round-trip: ok")
    dist.barrier()

    _ok(rank, f"rfdetr: {steps} steps, param sync, checkpoint — PASS")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _check_torchrun_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yolo9", "rfdetr", "all"], default="all")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="auto",
                        help="'auto' picks cuda:LOCAL_RANK if available, else cpu")
    args = parser.parse_args()

    from libreyolo.training.distributed import init_distributed, get_rank, get_world_size, get_local_rank

    # This is the production path: init_distributed reads RANK/LOCAL_RANK/WORLD_SIZE
    # set by torchrun. BaseTrainer.__init__ does exactly this via has_torchrun_env().
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    backend = dist.get_backend()
    if rank == 0:
        print(f"\n=== ddp_torchrun_smoke: world={world_size}, backend={backend}, device={device} ===\n",
              flush=True)

    dist.barrier()

    out_dir = Path(__file__).parent / "_ddp_torchrun_smoke"
    if rank == 0:
        out_dir.mkdir(exist_ok=True)
    dist.barrier()

    t0 = time.time()
    families = ["yolo9", "rfdetr"] if args.model == "all" else [args.model]
    passed: list[str] = []
    failed: list[str] = []

    # DistributedSampler correctness (all models share this infrastructure)
    try:
        run_sampler_test(rank, world_size)
        passed.append("sampler")
    except Exception as exc:
        _log(rank, f"FAIL sampler: {exc}")
        failed.append("sampler")

    # EMA broadcast (uses YOLO9 model, model-agnostic)
    try:
        run_ema_broadcast_test(rank, world_size, device)
        passed.append("ema_broadcast")
    except Exception as exc:
        _log(rank, f"FAIL ema_broadcast: {exc}")
        failed.append("ema_broadcast")

    for family in families:
        try:
            if family == "yolo9":
                run_yolo9(rank, world_size, device, args.steps, out_dir)
            else:
                run_rfdetr(rank, world_size, device, args.steps, out_dir)
            passed.append(family)
        except Exception as exc:
            import traceback
            _log(rank, f"FAIL {family}: {exc}")
            traceback.print_exc()
            failed.append(family)

    dist.barrier()
    elapsed = time.time() - t0

    if rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"Results (world={world_size}, backend={backend}, {elapsed:.1f}s):", flush=True)
        for name in passed:
            print(f"  PASS  {name}", flush=True)
        for name in failed:
            print(f"  FAIL  {name}", flush=True)
        print("=" * 60, flush=True)

    dist.destroy_process_group()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
