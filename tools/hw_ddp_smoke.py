"""Real-hardware DDP smoke test: 2 processes sharing one CUDA device.

NCCL isn't available on Windows so this exercises the CUDA + Gloo path
specifically — the production code path Windows users actually hit.

Launch both workers with mp.spawn; each sets LOCAL_RANK=0 and
CUDA_VISIBLE_DEVICES=0 so both processes target the same physical GPU.
This is unusual for production DDP (normally one rank per GPU) but
correctly stresses the distributed plumbing.
"""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _worker(rank: int, world_size: int, port: int, out_dir: str, model_family: str) -> None:
    """One rank that exercises DDP-wrapped training on cuda:0 via Gloo."""
    out_path = Path(out_dir) / f"rank_{rank}.txt"
    try:
        # Both procs see only the 5070 Ti; both pin to cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        torch.cuda.set_device(0)

        from libreyolo.training.distributed import (
            get_world_size,
            scale_loss_for_ddp,
            unwrap_model,
        )

        if model_family == "yolo9":
            from libreyolo import LibreYOLO9
            torch.manual_seed(0)
            wrapper = LibreYOLO9(None, size="t", device="cuda:0")
            wrapper.model.train()
            ddp_model = nn.parallel.DistributedDataParallel(
                wrapper.model,
                device_ids=[0],
                output_device=0,
                gradient_as_bucket_view=False,
                static_graph=True,
            )
            optim = torch.optim.SGD(unwrap_model(ddp_model).parameters(), lr=0.01)

            torch.manual_seed(100 + rank)
            imgs = torch.randn(1, 3, 320, 320, device="cuda:0")
            targets = torch.zeros(1, 30, 5, device="cuda:0")
            targets[0, 0] = torch.tensor([float(rank), 160.0, 120.0, 80.0, 60.0], device="cuda:0")
            out = ddp_model(imgs, targets=targets)
            loss = out["total_loss"]

        elif model_family == "rfdetr":
            from libreyolo import LibreRFDETR
            from libreyolo.models.rfdetr.trainer import RFDETRTrainer
            torch.manual_seed(0)
            wrapper = LibreRFDETR(None, size="n", device="cuda:0", segmentation=False)
            wrapper.model.train()
            trainer = RFDETRTrainer(
                model=wrapper.model, wrapper_model=wrapper,
                size="n", num_classes=80, data=None,
                epochs=1, batch=1, imgsz=320, device="cuda:0",
                amp=False, ema=False, no_aug_epochs=0, warmup_epochs=0,
                eval_interval=-1,
            )
            trainer.on_setup()
            ddp_model = nn.parallel.DistributedDataParallel(
                wrapper.model,
                device_ids=[0],
                output_device=0,
                gradient_as_bucket_view=False,
                static_graph=True,
            )
            trainer.model = ddp_model
            optim = torch.optim.AdamW(unwrap_model(ddp_model).parameters(), lr=1e-4)

            torch.manual_seed(200 + rank)
            imgs = torch.randn(1, 3, 320, 320, device="cuda:0")
            targets = torch.zeros(1, 30, 5, device="cuda:0")
            targets[0, 0] = torch.tensor([float(rank), 160.0, 120.0, 80.0, 60.0], device="cuda:0")
            out = trainer.on_forward(imgs, targets)
            loss = out["total_loss"]
        else:
            raise ValueError(f"unknown family: {model_family}")

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss on rank {rank}: {loss.item()}")
        loss_pre = float(loss.item())
        loss_scaled = scale_loss_for_ddp(loss)
        optim.zero_grad()
        loss_scaled.backward()
        optim.step()

        # Verify cross-rank parameter equality after step.
        world = get_world_size()
        for name, p in unwrap_model(ddp_model).named_parameters():
            if not p.requires_grad:
                continue
            local = p.detach().clone()
            gathered = p.detach().clone()
            dist.all_reduce(gathered, op=dist.ReduceOp.SUM)
            expected = local * world
            if not torch.allclose(gathered, expected, atol=1e-5, rtol=1e-4):
                max_abs = (gathered - expected).abs().max().item()
                raise RuntimeError(
                    f"param {name!r} diverged across ranks: max_abs={max_abs:.3e}"
                )

        gpu_mem_mb = torch.cuda.max_memory_allocated(0) / 1e6
        out_path.write_text(
            f"ok rank={rank} world={world} loss_pre_scale={loss_pre:.4f} "
            f"gpu_mem_mb={gpu_mem_mb:.1f}\n"
        )
    except Exception as exc:
        import traceback
        out_path.write_text(f"error: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(model_family: str, n_ranks: int = 2) -> None:
    out_dir = Path(__file__).parent / f"_hw_smoke_{model_family}"
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("rank_*.txt"):
        f.unlink()
    port = _free_port()
    t0 = time.time()
    print(f"\n=== {model_family} 2-proc DDP on cuda:0 (Gloo) ===")
    try:
        mp.spawn(_worker, args=(n_ranks, port, str(out_dir), model_family), nprocs=n_ranks, join=True)
    except Exception as exc:
        print(f"SPAWN FAILED: {exc}")
    elapsed = time.time() - t0
    for rank in range(n_ranks):
        txt = (out_dir / f"rank_{rank}.txt")
        if txt.exists():
            print(f"  {txt.name}: {txt.read_text().strip()}")
        else:
            print(f"  rank_{rank}.txt MISSING")
    print(f"  total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    families = sys.argv[1:] or ["yolo9", "rfdetr"]
    for fam in families:
        run(fam)
