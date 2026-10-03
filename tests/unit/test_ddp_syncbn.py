"""SyncBatchNorm under DDP for YOLO9 (issue #484, residual multi-GPU gap).

After the loss-scaling fix (``test_ddp_loss_parity.py``), YOLO9 multi-GPU
training still trailed single-GPU by a few points of mAP. The cause is
BatchNorm: the global batch is split across ranks (``batch // world_size``),
so with SyncBatchNorm off each rank's BN running statistics track only its own
small shard, and only rank 0's shard-derived buffers are validated and saved.
YOLO9 is BatchNorm-heavy, so this measurably degrades the converged model.

The fix is ``YOLO9Config.sync_bn = True`` (plus a trainer warning for any
BatchNorm family left with sync_bn off at a small per-rank batch). These tests:

1. Guard the config default so it can never silently regress to False again.
2. Verify the real YOLO9 model's BatchNorm converts to SyncBatchNorm, i.e. the
   conversion the trainer runs under DDP actually reaches every BN layer.
3. Demonstrate, over 2 gloo ranks, the per-rank running-stat divergence that
   SyncBatchNorm removes.

SyncBatchNorm's forward requires GPU modules, so the identical-stats end state
cannot be exercised on CPU; tests 1-2 lock in the config + conversion, test 3
documents the mechanism on CPU. Gloo/CPU only, no GPU required.
"""

from __future__ import annotations

import contextlib
import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

pytestmark = pytest.mark.unit


# =============================================================================
# 1. Config guard
# =============================================================================


def test_yolo9_config_enables_sync_bn():
    """YOLO9 must enable SyncBatchNorm; the base config
    must keep it off. Guards against the config silently disagreeing with its
    documented intent again (issue #484)."""
    from libreyolo.training.config import TrainConfig, YOLO9Config

    assert TrainConfig().sync_bn is False
    assert YOLO9Config().sync_bn is True


def test_bn_heavy_cnn_configs_enable_sync_bn():
    """Every BatchNorm-heavy pure-CNN family must default to SyncBatchNorm
    under DDP, not only yolo9: per-rank BN stats on a small shard degrade the
    converged model the same way for all of them (issue #484)."""
    from libreyolo.training.config import (
        FOMOConfig,
        PICODETConfig,
        RTMDetConfig,
        YOLONASConfig,
        YOLONASPoseConfig,
        YOLOXConfig,
        YOLOv7Config,
    )

    for cfg_cls in (
        FOMOConfig,
        PICODETConfig,
        RTMDetConfig,
        YOLONASConfig,
        YOLONASPoseConfig,
        YOLOXConfig,
        YOLOv7Config,
    ):
        assert cfg_cls().sync_bn is True, cfg_cls.__name__


# =============================================================================
# 2. Real-model conversion
# =============================================================================


def test_yolo9_batchnorm_converts_to_syncbatchnorm():
    """The conversion the trainer performs under DDP
    (``nn.SyncBatchNorm.convert_sync_batchnorm``) must reach every BatchNorm in
    a real YOLO9 model — no BN layer left as a plain per-rank module."""
    from libreyolo import LibreYOLO9

    model = LibreYOLO9(None, size="t", device="cpu").model

    n_bn_before = sum(
        isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()
    )
    assert n_bn_before > 0, "YOLO9-t is expected to contain BatchNorm layers"

    converted = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    n_sync = sum(isinstance(m, nn.SyncBatchNorm) for m in converted.modules())
    n_plain_left = sum(
        isinstance(m, nn.modules.batchnorm._BatchNorm)
        and not isinstance(m, nn.SyncBatchNorm)
        for m in converted.modules()
    )
    assert n_sync == n_bn_before
    assert n_plain_left == 0


# =============================================================================
# 3. Per-rank BN divergence (the mechanism SyncBatchNorm removes)
# =============================================================================


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _setup_pg(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _build_bn_model() -> nn.Sequential:
    """Deterministic conv+BN, identical initial weights in every process."""
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))


def _bn_divergence_worker(rank: int, world_size: int, port: int, out_dir: str) -> None:
    """One rank: forward its OWN data through DDP-wrapped plain BatchNorm, then
    save the BN running_mean. With sync_bn off each rank updates running stats
    from its local shard only, so the saved buffers differ across ranks."""
    out_path = Path(out_dir) / f"rank_{rank}.txt"
    try:
        _setup_pg(rank, world_size, port)

        model = _build_bn_model()
        ddp_model = nn.parallel.DistributedDataParallel(model)
        ddp_model.train()

        # Different, deterministic data per rank → different local BN stats.
        g = torch.Generator().manual_seed(100 + rank)
        x = torch.rand(4, 3, 8, 8, generator=g)
        ddp_model(x)

        running_mean = model[1].running_mean.detach().clone()
        torch.save(running_mean, Path(out_dir) / f"running_mean_{rank}.pt")
        dist.barrier()
        out_path.write_text("ok\n")
    except Exception as exc:  # pragma: no cover - surfaced via the parent assert
        out_path.write_text(f"error: {type(exc).__name__}: {exc}\n")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "win32" and sys.version_info < (3, 8),
    reason="mp.spawn on Windows needs Python 3.8+",
)
@pytest.mark.distributed
def test_ddp_batchnorm_running_stats_diverge_per_rank(tmp_path):
    """Plain BatchNorm under 2-rank DDP: each rank's running_mean reflects only
    its own shard, so the buffers differ across ranks. This is the degradation
    (issue #484) that ``sync_bn=True`` removes by computing BN statistics over
    the global batch."""
    port = _free_port()
    out_dir = str(tmp_path)
    try:
        mp.spawn(
            _bn_divergence_worker,
            args=(2, port, out_dir),
            nprocs=2,
            join=True,
        )
    except Exception as exc:
        outputs = {
            r: (tmp_path / f"rank_{r}.txt").read_text()
            if (tmp_path / f"rank_{r}.txt").exists()
            else "<no output>"
            for r in range(2)
        }
        raise AssertionError(f"spawn failed: {exc}\nper-rank outputs: {outputs}") from exc

    for r in range(2):
        text = (tmp_path / f"rank_{r}.txt").read_text()
        assert text.startswith("ok"), f"rank {r} did not finish ok: {text!r}"

    rm0 = torch.load(tmp_path / "running_mean_0.pt", weights_only=False)
    rm1 = torch.load(tmp_path / "running_mean_1.pt", weights_only=False)

    assert not torch.allclose(rm0, rm1), (
        "BatchNorm running_mean is identical across ranks, but with sync_bn off "
        "each rank should track only its own shard. If this ever passes, the "
        "per-rank BN divergence this test documents has changed."
    )
