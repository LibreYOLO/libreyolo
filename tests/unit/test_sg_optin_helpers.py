"""Opt-in training helpers from issue #768.

Defaults stay off so existing training calls are unchanged. These tests lock
that contract and the helpers' math without running a full training job.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from libreyolo.data.class_balanced import (
    DistributedClassBalancedSampler,
    class_ids_from_anno,
    image_repeat_factors,
)
from libreyolo.data.dataset import create_dataloader
from libreyolo.training.config import TrainConfig
from libreyolo.training.export_check import assert_close, outputs_comparable
from libreyolo.training.precise_bn import compute_precise_bn_stats
from libreyolo.training.weight_averaging import MetricGatedAverager
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Defaults: nothing new turns on by accident
# ---------------------------------------------------------------------------


def test_train_config_new_knobs_default_off():
    cfg = TrainConfig()
    assert cfg.class_balanced is False
    assert cfg.average_best == 0
    assert cfg.export_check is False
    assert cfg.precise_bn == 0
    assert cfg.ema is True
    assert cfg.sync_bn is False
    assert cfg.min_samples == 0


def test_train_config_rejects_negative_knobs():
    with pytest.raises(ValueError, match="average_best"):
        TrainConfig(average_best=-1)
    with pytest.raises(ValueError, match="precise_bn"):
        TrainConfig(precise_bn=-1)


# ---------------------------------------------------------------------------
# Weight averaging
# ---------------------------------------------------------------------------


class _Tiny(nn.Module):
    def __init__(self, fill: float, *, steps: int = 1):
        super().__init__()
        self.w = nn.Parameter(torch.full((2,), fill))
        self.register_buffer("steps", torch.tensor(steps, dtype=torch.long))


def test_averager_off_never_stores():
    averager = MetricGatedAverager(0)
    assert averager.consider(_Tiny(1.0), 0.9) is False
    assert averager.average_state_dict() is None


def test_averager_keeps_top_n_and_means_weights():
    averager = MetricGatedAverager(2)
    assert averager.consider(_Tiny(1.0, steps=1), 0.10)
    assert averager.consider(_Tiny(3.0, steps=3), 0.30)
    assert averager.consider(_Tiny(5.0, steps=5), 0.20) is True  # replaces 0.10
    assert averager.consider(_Tiny(7.0, steps=7), 0.05) is False
    avg = averager.average_state_dict()
    assert avg is not None
    # pool is 3.0 @ 0.30 and 5.0 @ 0.20 → mean 4.0
    assert torch.allclose(avg["w"], torch.tensor([4.0, 4.0]))
    assert avg["steps"].dtype == torch.long
    assert avg["steps"].item() == 3


def test_averager_rejects_nan():
    averager = MetricGatedAverager(2)
    assert averager.consider(_Tiny(1.0), float("nan")) is False
    assert averager.size == 0


def test_averager_pool_roundtrip(tmp_path):
    src = MetricGatedAverager(2)
    src.consider(_Tiny(1.0), 0.10)
    src.consider(_Tiny(3.0), 0.30)
    path = tmp_path / "average_pool.pt"
    src.save(path)
    dst = MetricGatedAverager(2)
    assert dst.load(path) == 2
    assert sorted(dst.metrics()) == [0.10, 0.30]
    avg = dst.average_state_dict()
    assert torch.allclose(avg["w"], torch.tensor([2.0, 2.0]))


def test_averager_load_keeps_live_n(tmp_path):
    src = MetricGatedAverager(3)
    src.consider(_Tiny(1.0), 0.10)
    src.consider(_Tiny(3.0), 0.30)
    src.consider(_Tiny(5.0), 0.50)
    path = tmp_path / "average_pool.pt"
    src.save(path)
    dst = MetricGatedAverager(2)
    dst.load(path)
    assert dst.size == 2
    assert sorted(dst.metrics()) == [0.30, 0.50]


def test_average_validation_restores_live_weights():
    from libreyolo.training.trainer import BaseTrainer

    live = _Tiny(9.0)
    observed = {}
    trainer = SimpleNamespace(model=live, ema_model=None, current_epoch=4)

    def run_validation(epoch, *, save_plots):
        observed["epoch"] = epoch
        observed["save_plots"] = save_plots
        observed["weights"] = live.w.detach().clone()
        return {"best_metric": 0.5}

    trainer._run_validation = run_validation
    metrics = BaseTrainer._validate_average_state(trainer, _Tiny(3.0).state_dict())

    assert metrics == {"best_metric": 0.5}
    assert observed["epoch"] == 4
    assert observed["save_plots"] is False
    assert torch.equal(observed["weights"], torch.tensor([3.0, 3.0]))
    assert torch.equal(live.w, torch.tensor([9.0, 9.0]))


def test_average_checkpoint_records_separate_validation(tmp_path):
    from libreyolo.training.trainer import BaseTrainer

    averager = MetricGatedAverager(2)
    averager.consider(_Tiny(1.0), 0.2)
    averager.consider(_Tiny(3.0), 0.4)
    average_metrics = {
        "best_metric": 0.35,
        "best_metric_key": "metrics/mAP50-95",
    }
    trainer = SimpleNamespace(
        _weight_averager=averager,
        is_distributed=False,
        save_dir=tmp_path,
        wrapper_model=SimpleNamespace(names={0: "object"}, task="detect"),
        config=SimpleNamespace(size="t", imgsz=32, num_classes=1),
        num_classes=1,
        current_epoch=4,
        best_mAP50_95=0.4,
        best_mAP50=0.5,
        best_metric_key="metrics/mAP50-95",
        best_epoch=3,
        ema_model=None,
    )
    trainer.get_model_family = lambda: "yolo9"
    trainer._checkpoint_extra_metadata = lambda: {}
    trainer._validate_average_state = lambda _state: average_metrics
    trainer._best_metric_value = lambda values: float(values["best_metric"])

    path = BaseTrainer._write_average_checkpoint(trainer)

    assert path == tmp_path / "weights" / "average.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["averaged_snapshot_count"] == 2
    assert checkpoint["average_metric_key"] == "metrics/mAP50-95"
    assert checkpoint["average_metric_value"] == pytest.approx(0.35)
    assert checkpoint["is_ema_weights"] is False
    assert torch.equal(checkpoint["model"]["w"], torch.tensor([2.0, 2.0]))


def _restore_pool_trainer(tmp_path, *, start_epoch, best_epoch, fill=1.0):
    from libreyolo.training.trainer import BaseTrainer

    trainer = SimpleNamespace(
        model=_Tiny(fill),
        ema_model=None,
        _weight_averager=MetricGatedAverager(2),
        best_mAP50_95=0.42,
        start_epoch=start_epoch,
        best_epoch=best_epoch,
    )
    trainer._average_seed_snapshot = lambda path: (
        BaseTrainer._average_seed_snapshot(trainer, path)
    )
    trainer._live_average_source_state = lambda: BaseTrainer._live_average_source_state(
        trainer
    )
    trainer._snapshot_from_checkpoint = lambda path: (
        BaseTrainer._snapshot_from_checkpoint(trainer, path)
    )
    trainer.upgrade_resume_checkpoint = lambda checkpoint: (
        BaseTrainer.upgrade_resume_checkpoint(trainer, checkpoint)
    )
    trainer._upgrade_resume_state_dict = lambda state: (
        BaseTrainer._upgrade_resume_state_dict(trainer, state)
    )
    return trainer


def test_restore_average_pool_prefers_sidecar(tmp_path):
    from libreyolo.training.trainer import BaseTrainer

    trainer = _restore_pool_trainer(tmp_path, start_epoch=5, best_epoch=2)
    last = tmp_path / "last.pt"
    last.write_bytes(b"")
    src = MetricGatedAverager(2)
    src.consider(_Tiny(3.0), 0.30)
    src.consider(_Tiny(5.0), 0.50)
    src.save(tmp_path / "average_pool.pt")
    BaseTrainer._restore_average_pool(trainer, last)
    assert sorted(trainer._weight_averager.metrics()) == [0.30, 0.50]
    avg = trainer._weight_averager.average_state_dict()
    assert torch.allclose(avg["w"], torch.tensor([4.0, 4.0]))


def test_restore_average_pool_seeds_sibling_best_not_last(tmp_path):
    from libreyolo.training.trainer import BaseTrainer

    trainer = _restore_pool_trainer(tmp_path, start_epoch=5, best_epoch=2, fill=1.0)
    last = tmp_path / "last.pt"
    last.write_bytes(b"")
    torch.save(
        {"model": _Tiny(9.0).state_dict(), "best_metric_value": 0.73},
        tmp_path / "best.pt",
    )
    BaseTrainer._restore_average_pool(trainer, last)
    assert trainer._weight_averager.metrics() == [0.73]
    avg = trainer._weight_averager.average_state_dict()
    assert torch.allclose(avg["w"], torch.tensor([9.0, 9.0]))


def test_restore_average_pool_upgrades_saved_states_through_resume_hook(tmp_path):
    """Pooled and seeded snapshots get the family's resume key migration."""
    from libreyolo.training.trainer import BaseTrainer

    def _rename(checkpoint):
        state = checkpoint.get("model")
        if isinstance(state, dict):
            checkpoint["model"] = {
                ("w" if key == "legacy_w" else key): value for key, value in state.items()
            }
        return checkpoint

    last = tmp_path / "last.pt"
    last.write_bytes(b"")
    legacy_pool = MetricGatedAverager(2)
    legacy_pool.consider_state({"legacy_w": torch.full((2,), 3.0)}, 0.30)
    legacy_pool.save(tmp_path / "average_pool.pt")

    trainer = _restore_pool_trainer(tmp_path, start_epoch=5, best_epoch=2)
    trainer.upgrade_resume_checkpoint = _rename
    BaseTrainer._restore_average_pool(trainer, last)
    trainer._weight_averager.consider(_Tiny(5.0), 0.50)
    avg = trainer._weight_averager.average_state_dict()
    assert torch.allclose(avg["w"], torch.tensor([4.0, 4.0]))

    (tmp_path / "average_pool.pt").unlink()
    torch.save(
        {"model": {"legacy_w": torch.full((2,), 9.0)}, "best_metric_value": 0.73},
        tmp_path / "best.pt",
    )
    trainer = _restore_pool_trainer(tmp_path, start_epoch=5, best_epoch=2)
    trainer.upgrade_resume_checkpoint = _rename
    BaseTrainer._restore_average_pool(trainer, last)
    assert list(trainer._weight_averager.average_state_dict()) == ["w"]


def test_restore_average_pool_does_not_mislabel_sibling_without_metric(tmp_path):
    from libreyolo.training.trainer import BaseTrainer

    trainer = _restore_pool_trainer(tmp_path, start_epoch=5, best_epoch=2)
    last = tmp_path / "last.pt"
    last.write_bytes(b"")
    torch.save({"model": _Tiny(9.0).state_dict()}, tmp_path / "best.pt")
    BaseTrainer._restore_average_pool(trainer, last)
    assert trainer._weight_averager.size == 0


def test_restore_average_pool_empty_when_last_is_not_best(tmp_path):
    from libreyolo.training.trainer import BaseTrainer

    trainer = _restore_pool_trainer(tmp_path, start_epoch=5, best_epoch=2, fill=1.0)
    last = tmp_path / "last.pt"
    last.write_bytes(b"")
    BaseTrainer._restore_average_pool(trainer, last)
    assert trainer._weight_averager.size == 0


def test_restore_average_pool_seeds_when_resume_is_best_epoch(tmp_path):
    from libreyolo.training.trainer import BaseTrainer

    trainer = _restore_pool_trainer(tmp_path, start_epoch=2, best_epoch=2, fill=4.0)
    last = tmp_path / "last.pt"
    last.write_bytes(b"")
    BaseTrainer._restore_average_pool(trainer, last)
    assert trainer._weight_averager.metrics() == [0.42]
    avg = trainer._weight_averager.average_state_dict()
    assert torch.allclose(avg["w"], torch.tensor([4.0, 4.0]))


def test_restore_average_pool_seeds_resumed_best_pt(tmp_path):
    from libreyolo.training.trainer import BaseTrainer

    trainer = _restore_pool_trainer(tmp_path, start_epoch=5, best_epoch=2, fill=7.0)
    best = tmp_path / "best.pt"
    best.write_bytes(b"")
    BaseTrainer._restore_average_pool(trainer, best)
    assert trainer._weight_averager.metrics() == [0.42]
    avg = trainer._weight_averager.average_state_dict()
    assert torch.allclose(avg["w"], torch.tensor([7.0, 7.0]))


# ---------------------------------------------------------------------------
# Precise BN
# ---------------------------------------------------------------------------


class _BnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.bn.running_mean.fill_(99.0)
        self.bn.running_var.fill_(99.0)

    def forward(self, x):
        return self.bn(x)


def _precise_bn_nonshared_best_worker(rank, world_size, port, out_dir):
    import os
    from datetime import timedelta

    import torch.distributed as dist

    out_path = Path(out_dir) / f"precise_bn_rank_{rank}.txt"
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )

        from libreyolo.training.trainer import BaseTrainer

        rank_dir = Path(out_dir) / f"rank_{rank}"
        if rank == 0:
            weights_dir = rank_dir / "weights"
            weights_dir.mkdir(parents=True)
            historical_best = _BnNet()
            historical_best.bn.running_mean.fill_(11.0)
            checkpoint = wrap_libreyolo_checkpoint(
                historical_best.state_dict(),
                model_family="yolo9",
                size="t",
                task="detect",
                nc=1,
                names={0: "object"},
                imgsz=32,
            )
            torch.save(checkpoint, weights_dir / "best.pt")
        dist.barrier()

        live = _BnNet()
        live.bn.running_mean.fill_(7.0)
        trainer = SimpleNamespace(
            config=SimpleNamespace(precise_bn=2),
            model=live,
            ema_model=None,
            train_loader=[(torch.ones(2, 1, 2, 2) * (1.0 + 4.0 * rank),)],
            device=torch.device("cpu"),
            save_dir=rank_dir,
            current_epoch=4,
            best_epoch=2 if rank == 0 else 0,
            is_distributed=True,
            _stop_training=False,
            _frozen_bn_modules=(),
        )
        trainer._sync_main_bool = lambda value: BaseTrainer._sync_main_bool(
            trainer, value
        )

        assert BaseTrainer._refresh_best_precise_bn_checkpoint(trainer) is True
        assert torch.equal(live.bn.running_mean, torch.tensor([7.0]))
        if rank == 0:
            refreshed = torch.load(
                rank_dir / "weights" / "best.pt",
                map_location="cpu",
                weights_only=True,
            )
            assert torch.allclose(
                refreshed["model"]["bn.running_mean"],
                torch.tensor([3.0]),
                atol=1e-5,
            )
        else:
            assert not (rank_dir / "weights" / "best.pt").exists()
        out_path.write_text("ok\n")
    except Exception as exc:
        out_path.write_text(f"error: {type(exc).__name__}: {exc}\n")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.distributed
def test_precise_bn_historical_best_works_without_shared_rank_paths(tmp_path):
    import contextlib
    import socket

    import torch.multiprocessing as mp

    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    mp.spawn(
        _precise_bn_nonshared_best_worker,
        args=(2, port, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        result = (tmp_path / f"precise_bn_rank_{rank}.txt").read_text()
        assert result == "ok\n", f"rank {rank}: {result!r}"


def test_precise_bn_zero_samples_is_noop():
    net = _BnNet()
    before = net.bn.running_mean.clone()
    assert compute_precise_bn_stats(net, [torch.zeros(2, 1, 4, 4)], 0) == 0
    assert torch.equal(net.bn.running_mean, before)


def test_precise_bn_no_bn_is_noop():
    net = nn.Linear(3, 3)
    assert compute_precise_bn_stats(net, [torch.zeros(2, 3)], 8) == 0


def test_precise_bn_rewrites_running_stats_and_restores_momentum():
    net = _BnNet()
    images = torch.ones(4, 1, 2, 2) * 3.0
    updated = compute_precise_bn_stats(net, [(images, "ignored")], num_samples=4)
    assert updated == 1
    assert net.bn.momentum == 0.1
    assert torch.allclose(net.bn.running_mean, torch.tensor([3.0]), atol=1e-5)


def test_precise_bn_skips_when_forward_rejects_bare_images():
    class _NeedsTargets(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(1)

        def forward(self, x, targets):
            return self.bn(x)

    net = _NeedsTargets()
    mean_before = net.bn.running_mean.clone()
    assert compute_precise_bn_stats(net, [torch.zeros(2, 1, 2, 2)], 4) == 0
    assert torch.equal(net.bn.running_mean, mean_before)
    assert net.bn.momentum == 0.1


def test_precise_bn_rolls_back_partial_forward_updates():
    class _FailsAfterBn(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(1)

        def forward(self, x):
            self.bn(x)
            raise RuntimeError("after BN")

    net = _FailsAfterBn()
    mean_before = net.bn.running_mean.clone()
    var_before = net.bn.running_var.clone()
    count_before = net.bn.num_batches_tracked.clone()
    assert compute_precise_bn_stats(net, [torch.ones(2, 1, 2, 2)], 2) == 0
    assert torch.equal(net.bn.running_mean, mean_before)
    assert torch.equal(net.bn.running_var, var_before)
    assert torch.equal(net.bn.num_batches_tracked, count_before)


def test_precise_bn_excludes_frozen_modules():
    class _TwoBn(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = nn.BatchNorm2d(1)
            self.trainable = nn.BatchNorm2d(1)
            self.frozen.running_mean.fill_(11.0)
            self.trainable.running_mean.fill_(22.0)

        def forward(self, x):
            return self.trainable(self.frozen(x))

    net = _TwoBn()
    updated = compute_precise_bn_stats(
        net,
        [torch.ones(2, 1, 2, 2) * 3.0],
        2,
        excluded_names={"frozen"},
    )
    assert updated == 1
    assert torch.equal(net.frozen.running_mean, torch.tensor([11.0]))
    expected = (3.0 - 11.0) / (1.0 + net.frozen.eps) ** 0.5
    assert torch.allclose(net.trainable.running_mean, torch.tensor([expected]))


def test_precise_bn_hook_runs_on_force_before_last_epoch():
    from libreyolo.training.trainer import BaseTrainer

    trainer = SimpleNamespace(
        config=TrainConfig(precise_bn=4, epochs=10),
        model=_BnNet(),
        train_loader=[(torch.ones(2, 1, 2, 2) * 3.0,)],
        device="cpu",
        _stop_training=False,
    )
    trainer._is_final_epoch = lambda epoch: BaseTrainer._is_final_epoch(trainer, epoch)
    BaseTrainer._maybe_precise_bn(trainer, 2)
    assert not getattr(trainer, "_precise_bn_done", False)
    assert BaseTrainer._maybe_precise_bn(trainer, 2, force=True) is True
    assert trainer._precise_bn_done is True
    assert torch.allclose(trainer.model.bn.running_mean, torch.tensor([3.0]), atol=1e-5)


def test_precise_bn_hook_updates_ema_copy():
    from libreyolo.training.trainer import BaseTrainer

    raw = _BnNet()
    ema = _BnNet()
    trainer = SimpleNamespace(
        config=TrainConfig(precise_bn=4, epochs=1),
        model=raw,
        ema_model=SimpleNamespace(ema=ema),
        train_loader=[(torch.ones(2, 1, 2, 2) * 3.0,)],
        device="cpu",
        _stop_training=False,
    )
    trainer._is_final_epoch = lambda epoch: True
    assert BaseTrainer._maybe_precise_bn(trainer, 0) is True
    assert torch.allclose(raw.bn.running_mean, torch.tensor([3.0]), atol=1e-5)
    assert torch.allclose(ema.bn.running_mean, torch.tensor([3.0]), atol=1e-5)


def test_precise_bn_forces_validation_on_final_epoch():
    from libreyolo.training.trainer import BaseTrainer

    trainer = SimpleNamespace(
        config=TrainConfig(epochs=10, eval_interval=6, precise_bn=4)
    )
    trainer._is_final_epoch = lambda epoch: BaseTrainer._is_final_epoch(trainer, epoch)

    assert BaseTrainer._should_validate_epoch(trainer, 8) is False
    assert BaseTrainer._should_validate_epoch(trainer, 9) is True


@pytest.mark.parametrize("use_ema", [False, True])
def test_precise_bn_refreshes_historical_best_and_restores_live_model(
    tmp_path, use_ema
):
    from libreyolo.training.trainer import BaseTrainer

    live = _BnNet()
    live.bn.running_mean.fill_(7.0)
    live_ema = _BnNet()
    live_ema.bn.running_mean.fill_(8.0)
    historical_best = _BnNet()
    historical_best.bn.running_mean.fill_(11.0)
    historical_ema = _BnNet()
    historical_ema.bn.running_mean.fill_(12.0)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    checkpoint = wrap_libreyolo_checkpoint(
        (
            historical_ema.state_dict()
            if use_ema
            else historical_best.state_dict()
        ),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=1,
        names={0: "object"},
        imgsz=32,
    )
    if use_ema:
        checkpoint["train_model"] = historical_best.state_dict()
        checkpoint["ema"] = historical_ema.state_dict()
    torch.save(checkpoint, weights_dir / "best.pt")

    trainer = SimpleNamespace(
        config=SimpleNamespace(precise_bn=2),
        model=live,
        ema_model=SimpleNamespace(ema=live_ema) if use_ema else None,
        train_loader=[(torch.ones(2, 1, 2, 2) * 3.0,)],
        device="cpu",
        save_dir=tmp_path,
        current_epoch=4,
        best_epoch=2,
        is_distributed=False,
        _stop_training=False,
        _frozen_bn_modules=(),
    )
    trainer._sync_main_bool = lambda value: BaseTrainer._sync_main_bool(
        trainer, value
    )

    assert BaseTrainer._refresh_best_precise_bn_checkpoint(trainer) is True
    refreshed = torch.load(
        weights_dir / "best.pt", map_location="cpu", weights_only=True
    )
    assert torch.allclose(
        refreshed["model"]["bn.running_mean"], torch.tensor([3.0]), atol=1e-5
    )
    if use_ema:
        assert torch.allclose(
            refreshed["train_model"]["bn.running_mean"],
            torch.tensor([3.0]),
            atol=1e-5,
        )
        assert torch.equal(
            refreshed["ema"]["bn.running_mean"],
            refreshed["model"]["bn.running_mean"],
        )
    assert torch.equal(live.bn.running_mean, torch.tensor([7.0]))
    assert torch.equal(live_ema.bn.running_mean, torch.tensor([8.0]))


def test_precise_bn_nonmain_refresh_does_not_require_best_file(tmp_path, monkeypatch):
    from libreyolo.training import trainer as trainer_module
    from libreyolo.training.trainer import BaseTrainer

    live = _BnNet()
    live.bn.running_mean.fill_(7.0)
    trainer = SimpleNamespace(
        config=SimpleNamespace(precise_bn=2),
        model=live,
        ema_model=None,
        train_loader=[(torch.ones(2, 1, 2, 2) * 3.0,)],
        device="cpu",
        save_dir=tmp_path,
        current_epoch=4,
        best_epoch=2,
        is_distributed=True,
        _stop_training=False,
        _frozen_bn_modules=(),
        _sync_main_bool=lambda _value: True,
    )
    barriers = []
    monkeypatch.setattr(trainer_module, "is_main_process", lambda: False)
    monkeypatch.setattr(trainer_module, "barrier", lambda: barriers.append(True))
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda _values, src: None
    )
    monkeypatch.setattr(torch.distributed, "broadcast", lambda _value, src: None)

    assert BaseTrainer._refresh_best_precise_bn_checkpoint(trainer) is True
    assert barriers == [True]
    assert not (tmp_path / "weights" / "best.pt").exists()
    assert torch.equal(live.bn.running_mean, torch.tensor([7.0]))


def test_precise_bn_improvement_cancels_early_stop():
    from libreyolo.training.trainer import BaseTrainer

    trained_epochs = []
    initial_metrics = [0.5, 0.4, 0.55]

    def train_epoch(epoch):
        trained_epochs.append(epoch)
        return 1.0, {"best_metric": initial_metrics[epoch]}, {}, {}

    def no_op(*_args, **_kwargs):
        return None

    callbacks = SimpleNamespace(
        on_train_start=no_op,
        on_train_epoch_end=no_op,
        on_train_end=no_op,
    )
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            epochs=3,
            no_aug_epochs=0,
            patience=1,
            save_plots=False,
            batch=1,
            imgsz=32,
        ),
        is_distributed=False,
        device=torch.device("cpu"),
        start_epoch=0,
        current_epoch=0,
        best_epoch=0,
        best_mAP50_95=0.0,
        best_mAP50=0.0,
        patience_counter=0,
        final_loss=0.0,
        epoch_losses=[],
        epoch_events=[],
        _stop_training=False,
        distiller=None,
        callbacks=callbacks,
        input_size=(32, 32),
        effective_lr=0.01,
    )
    trainer.setup = no_op
    trainer._maybe_export_check = no_op
    trainer.get_model_tag = lambda: "test-model"
    trainer._build_train_start_event = lambda: None
    trainer._dispatch_artifact_callbacks = no_op
    trainer._train_epoch = train_epoch
    trainer._normalize_epoch_result = lambda result: result
    trainer._best_metric_value = lambda metrics: BaseTrainer._best_metric_value(
        trainer, metrics
    )
    trainer._as_float = BaseTrainer._as_float
    trainer._update_best_state = lambda epoch, metrics: BaseTrainer._update_best_state(
        trainer, epoch, metrics
    )
    trainer._sync_main_bool = lambda value: BaseTrainer._sync_main_bool(
        trainer, value
    )
    trainer._maybe_precise_bn = lambda epoch, force=False: force and epoch == 1
    trainer._validate_epoch = lambda epoch, save_plots=False: {"best_metric": 0.6}
    trainer._maybe_offer_average = no_op
    trainer._save_checkpoint = no_op
    trainer._build_train_epoch_event = lambda **kwargs: kwargs
    trainer._refresh_best_precise_bn_checkpoint = no_op
    trainer._write_average_checkpoint = no_op
    trainer._build_train_results = lambda: {"trained_epochs": trained_epochs}
    trainer._record_trained_dataset = no_op
    trainer._build_train_end_event = lambda *_args: None
    trainer._build_train_exception_event = lambda *_args: None

    result = BaseTrainer.train(trainer)

    assert result["trained_epochs"] == [0, 1, 2]
    assert trainer.best_epoch == 2


# ---------------------------------------------------------------------------
# Export-check comparison
# ---------------------------------------------------------------------------


def test_export_check_comparable_and_close():
    a = (torch.zeros(1, 4),)
    b = (torch.zeros(1, 4),)
    assert outputs_comparable(a, b)
    assert_close(a, b)


def test_export_check_mismatch_raises():
    a = (torch.zeros(1, 4),)
    b = (torch.ones(1, 4),)
    with pytest.raises(AssertionError, match="differs"):
        assert_close(a, b)


def test_export_check_layout_mismatch_is_not_comparable():
    a = (torch.zeros(1, 4),)
    b = (torch.zeros(2, 4), torch.zeros(2, 1))
    assert not outputs_comparable(a, b)


# ---------------------------------------------------------------------------
# Class-balanced sampler
# ---------------------------------------------------------------------------


class _AnnoDataset:
    def __init__(self, rows):
        self._rows = rows
        self.num_classes = 2

    def __len__(self):
        return len(self._rows)

    def load_anno(self, index):
        return np.asarray(self._rows[index], dtype=np.float32)


def test_class_ids_from_empty_anno():
    assert class_ids_from_anno(np.zeros((0, 5))).size == 0
    assert class_ids_from_anno([]).size == 0


def test_repeat_factors_boost_rare_class_images():
    # 3 images of class 0, 1 image of class 1 → class 1 is below the median
    dataset = _AnnoDataset(
        [
            [[0, 0, 1, 1, 0]],
            [[0, 0, 1, 1, 0]],
            [[0, 0, 1, 1, 0]],
            [[0, 0, 1, 1, 1]],
        ]
    )
    weights = image_repeat_factors(dataset, alpha=0.5)
    assert weights.shape == (4,)
    assert weights[3] > weights[0]
    assert np.allclose(weights[:3], weights[0])


def test_repeat_factors_unwrap_mosaic_wrapper():
    inner = _AnnoDataset(
        [
            [[0, 0, 1, 1, 0]],
            [[0, 0, 1, 1, 0]],
            [[0, 0, 1, 1, 1]],
        ]
    )

    class _Mosaic:
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

    weights = image_repeat_factors(_Mosaic(inner))
    assert weights.shape == (3,)
    assert weights[2] > weights[0]


def test_create_dataloader_class_balanced_off_keeps_random_sampler():
    dataset = [None] * 6
    loader = create_dataloader(dataset, batch_size=2, num_workers=0)
    assert isinstance(loader.sampler, RandomSampler)
    assert loader.sampler.replacement is False


def test_create_dataloader_class_balanced_uses_weighted_sampler():
    dataset = _AnnoDataset(
        [
            [[0, 0, 1, 1, 0]],
            [[0, 0, 1, 1, 0]],
            [[0, 0, 1, 1, 1]],
        ]
    )
    loader = create_dataloader(
        dataset, batch_size=2, num_workers=0, class_balanced=True
    )
    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert len(loader.sampler) == 3


def test_create_dataloader_class_balanced_rejects_custom_sampler():
    dataset = _AnnoDataset([[[0, 0, 1, 1, 0]]])
    custom = RandomSampler(dataset)
    with pytest.raises(ValueError, match="custom"):
        create_dataloader(
            dataset, batch_size=1, num_workers=0, sampler=custom, class_balanced=True
        )


def test_create_dataloader_class_balanced_replaces_distributed_sampler():
    dataset = _AnnoDataset(
        [
            [[0, 0, 1, 1, 0]],
            [[0, 0, 1, 1, 1]],
        ]
    )
    sampler = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True)
    loader = create_dataloader(
        dataset, batch_size=1, num_workers=0, sampler=sampler, class_balanced=True
    )
    assert isinstance(loader.sampler, DistributedClassBalancedSampler)
    assert loader.sampler.num_replicas == 2
    assert loader.sampler.rank == 0
    assert hasattr(loader.sampler, "set_epoch")


# ---------------------------------------------------------------------------
# DDP calib gather: single-process path is a plain forward
# ---------------------------------------------------------------------------


def test_rfdetr_cli_forwards_all_supported_helpers(tmp_path):
    from libreyolo.cli.config import _build_rfdetr_train_kwargs

    kwargs = _build_rfdetr_train_kwargs(
        {
            "project": str(tmp_path),
            "name": "exp",
            "exist_ok": True,
            "average_best": 5,
            "export_check": True,
            "precise_bn": 64,
            "class_balanced": True,
        }
    )
    assert kwargs["average_best"] == 5
    assert kwargs["export_check"] is True
    assert kwargs["precise_bn"] == 64
    assert kwargs["class_balanced"] is True


def test_build_train_kwargs_forwards_new_knobs():
    from libreyolo.cli.config import build_train_kwargs

    kwargs = build_train_kwargs(
        {
            "class_balanced": True,
            "average_best": 5,
            "export_check": True,
            "precise_bn": 128,
        }
    )
    assert kwargs["class_balanced"] is True
    assert kwargs["average_best"] == 5
    assert kwargs["export_check"] is True
    assert kwargs["precise_bn"] == 128


def test_export_check_uses_output_path_and_restores_live_model(tmp_path, monkeypatch):
    import sys

    import libreyolo.training.export_check as export_check

    class _Tiny(nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3))

    live = _Tiny()
    seen = {}

    class _Wrapper:
        def __init__(self):
            self.model = live
            self.device = torch.device("cpu")

        def export(self, **kwargs):
            seen["kwargs"] = kwargs
            seen["model_id"] = id(self.model)
            path = Path(kwargs["output_path"])
            path.write_bytes(b"onnx")
            return str(path)

    class _Sess:
        def get_inputs(self):
            return [SimpleNamespace(name="x")]

        def run(self, *args, **kwargs):
            return [torch.zeros(1, 1).numpy()]

    class _ORT:
        @staticmethod
        def InferenceSession(*args, **kwargs):
            return _Sess()

    wrapper = _Wrapper()
    monkeypatch.setitem(sys.modules, "onnxruntime", _ORT)
    monkeypatch.setattr(
        export_check,
        "module_has_lora",
        lambda _m: False,
        raising=False,
    )
    # module_has_lora is imported inside the function; patch the lora module.
    import libreyolo.training.lora as lora

    monkeypatch.setattr(lora, "module_has_lora", lambda _m: False)

    out = export_check.run_export_parity_check(wrapper, out_dir=tmp_path, imgsz=32)
    assert seen["kwargs"]["output_path"] == str(tmp_path / "export_check.onnx")
    assert seen["kwargs"]["format"] == "onnx"
    assert "out" not in seen["kwargs"]
    assert wrapper.model is live
    assert Path(out).name == "export_check.onnx"


def test_export_check_exports_a_copy_when_lora_is_live(tmp_path, monkeypatch):
    import sys

    import libreyolo.training.export_check as export_check
    import libreyolo.training.lora as lora

    class _Tiny(nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3))

    live = _Tiny()
    seen = {}

    class _Wrapper:
        def __init__(self):
            self.model = live
            self.device = torch.device("cpu")

        def export(self, **kwargs):
            seen["model_id"] = id(self.model)
            path = Path(kwargs["output_path"])
            path.write_bytes(b"onnx")
            return str(path)

    class _Sess:
        def get_inputs(self):
            return [SimpleNamespace(name="x")]

        def run(self, *args, **kwargs):
            return [torch.zeros(1, 3).numpy()]

    class _ORT:
        @staticmethod
        def InferenceSession(*args, **kwargs):
            return _Sess()

    monkeypatch.setitem(sys.modules, "onnxruntime", _ORT)
    monkeypatch.setattr(lora, "module_has_lora", lambda _m: True)
    wrapper = _Wrapper()
    export_check.run_export_parity_check(wrapper, out_dir=tmp_path, imgsz=32)
    assert wrapper.model is live
    assert seen["model_id"] != id(live)


def test_assert_class_balanced_rejected_on_plain_sampler():
    from libreyolo.training.trainer import BaseTrainer
    from torch.utils.data import RandomSampler

    trainer = SimpleNamespace(
        config=TrainConfig(class_balanced=True),
        train_loader=SimpleNamespace(sampler=RandomSampler([0, 1, 2])),
    )
    trainer.get_model_family = lambda: "dfine"
    with pytest.raises(ValueError, match="class_balanced"):
        BaseTrainer._assert_class_balanced_honored(trainer)
