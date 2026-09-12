"""Imitation-learning trainer and offline validator for the LibreVLA tier.

Deliberately not a ``BaseTrainer`` subclass: that chassis assumes stacked
image tensors, detection losses and mAP validation. This trainer keeps the
repo's user-facing surface (run directories, callbacks, loggers, results
dict) and owns a compact epoch loop over a LeRobot dataset: the upstream
policy's own loss, the family's optimizer and schedule presets, gradient
clipping, best/last checkpoints and the checkpoint contract in
:mod:`..checkpoint`.

The validator computes offline action error (:mod:`..metrics`) on held-out
episodes, which is what CI can run without a robot (ADR 0028).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ....training.callbacks import (
    TrainCallbackList,
    TrainCallbacks,
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from ....training.loggers import resolve_loggers
from ..base import _INSTALL_HINT
from ..checkpoint import read_contract, write_contract
from ..metrics import action_error
from ..observation import action_names_from_features
from .data import (
    camera_names,
    camera_rename_map,
    resolve_data_source,
    split_episodes,
)

logger = logging.getLogger(__name__)

__all__ = ["VLATrainConfig", "VLATrainer", "VLAValidator"]


@dataclass
class VLATrainConfig:
    """Resolved VLA training configuration (user kwargs over preset defaults)."""

    data: str = ""
    epochs: int = 5
    batch: int = 8
    accumulate: int = 1
    lr0: Optional[float] = None
    val_split: float = 0.1
    val_episodes: Optional[List[int]] = None
    train_episodes: Optional[List[int]] = None
    output_dir: str = "runs/act/train"
    project: Optional[str] = None
    name: Optional[str] = None
    exist_ok: bool = False
    workers: int = 0
    seed: int = 0
    device: Optional[str] = None
    max_steps: Optional[int] = None
    val_batches: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _lerobot():
    try:
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.processor import rename_stats
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return (
        LeRobotDataset,
        LeRobotDatasetMetadata,
        resolve_delta_timestamps,
        make_policy,
        make_pre_post_processors,
        rename_stats,
    )


class _DatasetBundle:
    """Metadata, episode split and the train/val LeRobot datasets."""

    def __init__(
        self, source, meta, train_ds, val_ds, train_eps, val_eps, rename_map, cameras
    ):
        self.source = source
        self.meta = meta
        self.train = train_ds
        self.val = val_ds
        self.train_episodes = train_eps
        self.val_episodes = val_eps
        self.rename_map = rename_map
        self.cameras = cameras

    @property
    def action_names(self) -> Optional[List[str]]:
        return action_names_from_features(self.meta.features)

    @property
    def state_names(self) -> Optional[List[str]]:
        state = self.meta.features.get("observation.state", {})
        names = state.get("names") if isinstance(state, dict) else None
        if isinstance(names, (list, tuple)):
            return [str(n) for n in names]
        return None


def _build_datasets(
    wrapper,
    config,
    data,
    *,
    val_split,
    val_episodes,
    train_episodes,
    need_train=True,
    allow_empty_train=False,
    meta=None,
    slots=None,
):
    (
        LeRobotDataset,
        LeRobotDatasetMetadata,
        resolve_delta_timestamps,
        _mp,
        _mpp,
        _rs,
    ) = _lerobot()
    source = resolve_data_source(data)
    if meta is None:
        meta = LeRobotDatasetMetadata(source.repo_id, root=source.root)
    train_eps, val_eps = split_episodes(
        meta.total_episodes,
        val_split=val_split,
        val_episodes=val_episodes,
        train_episodes=train_episodes,
        allow_empty_train=allow_empty_train,
    )
    if slots is None:
        slots = wrapper.camera_slots
    rename_map = camera_rename_map(meta.camera_keys, slots)
    cameras = camera_names(meta.camera_keys, slots)
    delta = resolve_delta_timestamps(config, meta)
    train_ds = (
        LeRobotDataset(
            source.repo_id, root=source.root, episodes=train_eps, delta_timestamps=delta
        )
        if need_train and train_eps
        else None
    )
    val_ds = (
        LeRobotDataset(
            source.repo_id, root=source.root, episodes=val_eps, delta_timestamps=delta
        )
        if val_eps
        else None
    )
    return _DatasetBundle(
        source, meta, train_ds, val_ds, train_eps, val_eps, rename_map, cameras
    )


def _make_loader(dataset, batch, shuffle, workers, seed):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch),
        shuffle=shuffle,
        num_workers=int(workers),
        drop_last=False,
        generator=generator,
    )


class VLATrainer:
    """Epoch-based imitation fine-tuning of a LibreVLA family."""

    def __init__(
        self,
        wrapper,
        *,
        data: str,
        callbacks: TrainCallbacks = None,
        loggers: Any = None,
        **kwargs,
    ):
        known = set(VLATrainConfig.__dataclass_fields__) - {"data", "extra"}
        config_kwargs = {k: v for k, v in kwargs.items() if k in known}
        extra = {k: v for k, v in kwargs.items() if k not in known}
        if extra:
            logger.warning("Ignoring unknown train() kwargs: %s", sorted(extra))
        self.config = VLATrainConfig(data=str(data), extra=extra, **config_kwargs)
        if self.config.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.config.epochs}")
        if self.config.batch < 1 or self.config.accumulate < 1:
            raise ValueError("batch and accumulate must both be >= 1.")
        self.wrapper = wrapper
        self.callbacks = TrainCallbackList(callbacks)
        for logger_cb in resolve_loggers(loggers):
            self.callbacks.append(logger_cb)
        self.save_dir = self._resolve_save_dir()

    def _resolve_save_dir(self) -> Path:
        from ....utils.general import increment_path

        cfg = self.config
        output = Path(cfg.output_dir)
        project = Path(cfg.project) if cfg.project else output.parent
        name = cfg.name if cfg.name else output.name
        return increment_path(
            Path(project) / str(name), exist_ok=cfg.exist_ok, mkdir=True
        )

    def _resolve_device(self):
        import torch

        if self.config.device is not None:
            return self.wrapper._resolve_device(self.config.device)
        return torch.device(self.wrapper.device)

    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        import torch

        cfg = self.config
        wrapper = self.wrapper
        start_time = time.time()
        torch.manual_seed(cfg.seed)
        device = self._resolve_device()

        (_LD, _LM, _rdt, make_policy, make_pre_post_processors, rename_stats) = (
            _lerobot()
        )
        scratch = not wrapper.PRETRAINED_BASE and wrapper._checkpoint_dir is None
        meta = None
        slots = None
        if scratch:
            source = resolve_data_source(cfg.data)
            meta = _LM(source.repo_id, root=source.root)
            config = wrapper._scratch_config(meta)
            base_dir = None
            slots = [
                key.removeprefix("observation.images.") for key in meta.camera_keys
            ]
        else:
            base_dir = wrapper._ensure_weights()
            config = wrapper._pretrained_config(base_dir)
        config.pretrained_path = base_dir
        config.device = str(device)

        # Datasets need the config for delta timestamps; the config needs the
        # wrapper's camera slots, which come from the same base config.
        wrapper._config = config
        bundle = _build_datasets(
            wrapper,
            config,
            cfg.data,
            val_split=cfg.val_split,
            val_episodes=cfg.val_episodes,
            train_episodes=cfg.train_episodes,
            meta=meta,
            slots=slots,
        )
        if bundle.train is None:
            raise ValueError("No training episodes after the split.")

        policy_kwargs = {} if scratch else {"rename_map": bundle.rename_map}
        policy = make_policy(config, ds_meta=bundle.meta, **policy_kwargs)
        policy.to(device)
        stats = rename_stats(bundle.meta.stats, bundle.rename_map)
        features = {**config.input_features, **config.output_features}
        processor_kwargs = (
            {}
            if scratch
            else {
                "pretrained_path": base_dir,
                "preprocessor_overrides": {
                    "device_processor": {"device": device.type},
                    "normalizer_processor": {
                        "features": features,
                        "norm_map": config.normalization_mapping,
                        "stats": stats,
                    },
                    "rename_observations_processor": {"rename_map": bundle.rename_map},
                },
                "postprocessor_overrides": {
                    "unnormalizer_processor": {
                        "features": config.output_features,
                        "norm_map": config.normalization_mapping,
                        "stats": stats,
                    },
                },
            }
        )
        preprocessor, postprocessor = make_pre_post_processors(
            config, dataset_stats=stats, **processor_kwargs
        )

        train_loader = _make_loader(
            bundle.train, cfg.batch, True, cfg.workers, cfg.seed
        )
        val_loader = (
            _make_loader(bundle.val, cfg.batch, False, cfg.workers, cfg.seed)
            if bundle.val is not None
            else None
        )
        steps_per_epoch = len(train_loader)
        if cfg.max_steps is not None:
            steps_per_epoch = min(steps_per_epoch, int(cfg.max_steps))
        total_updates = max(1, math.ceil(steps_per_epoch / cfg.accumulate) * cfg.epochs)

        optimizer_cfg = config.get_optimizer_preset()
        if cfg.lr0 is not None:
            optimizer_cfg.lr = float(cfg.lr0)
        optimizer = optimizer_cfg.build(policy.get_optim_params())
        grad_clip = float(getattr(optimizer_cfg, "grad_clip_norm", 0.0) or 0.0)
        scheduler_cfg = config.get_scheduler_preset()
        scheduler = None
        if scheduler_cfg is not None:
            if hasattr(scheduler_cfg, "num_warmup_steps"):
                scheduler_cfg.num_warmup_steps = min(
                    int(scheduler_cfg.num_warmup_steps), max(1, total_updates // 10)
                )
            if hasattr(scheduler_cfg, "num_decay_steps"):
                scheduler_cfg.num_decay_steps = max(1, total_updates)
            if hasattr(scheduler_cfg, "peak_lr") and cfg.lr0 is not None:
                scheduler_cfg.peak_lr = float(cfg.lr0)
            scheduler = scheduler_cfg.build(optimizer, total_updates)

        weights_dir = self.save_dir / "weights"
        best_metric: Optional[float] = None
        best_epoch: Optional[int] = None
        final_loss = float("nan")
        completed_epochs = 0
        metric_name = "val/loss" if val_loader is not None else "train/loss"
        config_dump = {**asdict(cfg), "family": wrapper.FAMILY, "lr0": optimizer_cfg.lr}
        config_dump.pop("extra", None)

        self.callbacks.on_train_start(
            TrainStartEvent(
                start_epoch=1,
                total_epochs=cfg.epochs,
                model_family=wrapper.FAMILY,
                model_size=wrapper.size,
                task="act",
                save_dir=str(self.save_dir),
                config=config_dump,
            )
        )

        def save(directory: Path) -> None:
            directory.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(directory)
            if hasattr(config, "save_pretrained"):
                config.save_pretrained(directory)
            preprocessor.save_pretrained(directory)
            postprocessor.save_pretrained(directory)
            write_contract(
                directory,
                family=wrapper.FAMILY,
                size=wrapper.size,
                base_repo=wrapper.HF_REPOS.get(wrapper.size),
                base_revision=wrapper.HF_REVISIONS.get(wrapper.size),
                data=bundle.source.label,
                fps=float(bundle.meta.fps) if bundle.meta.fps else None,
                cameras=bundle.cameras,
                action_names=bundle.action_names,
                state_names=bundle.state_names,
                chunk_size=int(getattr(config, "chunk_size", 0)) or None,
            )

        epoch = 0
        try:
            for epoch in range(1, cfg.epochs + 1):
                epoch_start = time.time()
                policy.train()
                running, seen = 0.0, 0
                optimizer.zero_grad(set_to_none=True)
                for step, batch in enumerate(train_loader):
                    if step >= steps_per_epoch:
                        break
                    batch = preprocessor(batch)
                    loss, _out = policy.forward(batch)
                    # Scale by the real window so a partial tail window is not
                    # underweighted when steps_per_epoch % accumulate != 0.
                    window_start = (step // cfg.accumulate) * cfg.accumulate
                    window = min(cfg.accumulate, steps_per_epoch - window_start)
                    (loss / window).backward()
                    running += float(loss.detach()) * 1
                    seen += 1
                    if (step + 1) % cfg.accumulate == 0 or step + 1 == steps_per_epoch:
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                policy.parameters(), grad_clip
                            )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if scheduler is not None:
                            scheduler.step()
                train_loss = running / max(seen, 1)
                final_loss = train_loss

                val_metrics: Dict[str, float] = {}
                validated = False
                if val_loader is not None:
                    val_metrics = self._validate_loss(
                        policy, preprocessor, val_loader, cfg.val_batches
                    )
                    validated = True
                current = val_metrics.get("val/loss", train_loss)
                is_best = best_metric is None or current < best_metric
                if is_best:
                    best_metric, best_epoch = current, epoch
                    save(weights_dir / "best")
                save(weights_dir / "last")
                completed_epochs = epoch
                lr_now = {
                    f"lr/{i}": float(g["lr"])
                    for i, g in enumerate(optimizer.param_groups)
                }
                logger.info(
                    "epoch %d/%d  train/loss %.5f  %s %.5f%s",
                    epoch,
                    cfg.epochs,
                    train_loss,
                    metric_name,
                    current,
                    "  (best)" if is_best else "",
                )
                self.callbacks.on_train_epoch_end(
                    TrainEpochEvent(
                        epoch=epoch,
                        total_epochs=cfg.epochs,
                        model_family=wrapper.FAMILY,
                        model_size=wrapper.size,
                        task="act",
                        save_dir=str(self.save_dir),
                        train_loss=train_loss,
                        train_loss_items={"loss": train_loss},
                        lr=lr_now,
                        val_metrics=val_metrics,
                        validated=validated,
                        is_best=is_best,
                        current_metric=current,
                        current_metric_name=metric_name,
                        best_metric=best_metric,
                        best_metric_name=metric_name,
                        best_epoch=best_epoch,
                        epoch_seconds=time.time() - epoch_start,
                    )
                )
        except BaseException as exc:
            self.callbacks.on_train_exception(
                TrainExceptionEvent(
                    epoch=epoch or None,
                    total_epochs=cfg.epochs,
                    model_family=wrapper.FAMILY,
                    model_size=wrapper.size,
                    task="act",
                    save_dir=str(self.save_dir),
                    exception=exc,
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                    elapsed_seconds=time.time() - start_time,
                )
            )
            raise

        results: Dict[str, Any] = {
            "save_dir": str(self.save_dir),
            "best": str(weights_dir / "best"),
            "last": str(weights_dir / "last"),
            "epochs": completed_epochs,
            "final_loss": final_loss,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "metric_name": metric_name,
            "train_episodes": bundle.train_episodes,
            "val_episodes": bundle.val_episodes,
        }
        if not wrapper.PRETRAINED_BASE:
            wrapper._checkpoint_dir = weights_dir / "last"
            wrapper.model_path = str(wrapper._checkpoint_dir)
            wrapper.contract = read_contract(wrapper._checkpoint_dir)
            wrapper.device = device
            wrapper._policy = policy.eval()
            wrapper._preprocessor = preprocessor
            wrapper._postprocessor = postprocessor
            wrapper.reset()
        self.callbacks.on_train_end(
            TrainEndEvent(
                total_epochs=cfg.epochs,
                completed_epochs=completed_epochs,
                model_family=wrapper.FAMILY,
                model_size=wrapper.size,
                task="act",
                save_dir=str(self.save_dir),
                final_loss=final_loss,
                best_metric=best_metric,
                best_epoch=best_epoch,
                total_seconds=time.time() - start_time,
                results=results,
            )
        )
        return results

    @staticmethod
    def _validate_loss(policy, preprocessor, loader, max_batches) -> Dict[str, float]:
        import torch

        policy.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                if max_batches is not None and idx >= int(max_batches):
                    break
                batch = preprocessor(batch)
                loss, _out = policy.forward(batch)
                total += float(loss)
                count += 1
        policy.train()
        return {"val/loss": total / max(count, 1)}


class VLAValidator:
    """Offline action error of a loaded policy on held-out episodes."""

    def __init__(
        self,
        wrapper,
        *,
        data: str,
        batch: int = 8,
        workers: int = 0,
        val_split: float = 0.1,
        val_episodes: Optional[List[int]] = None,
        split: str = "val",
        max_batches: Optional[int] = None,
        verbose: bool = True,
        **kwargs,
    ):
        if kwargs:
            logger.warning("Ignoring unknown val() kwargs: %s", sorted(kwargs))
        if split not in ("val", "train", "all"):
            raise ValueError("split must be 'val', 'train' or 'all'.")
        self.wrapper = wrapper
        self.data = str(data)
        self.batch = int(batch)
        self.workers = int(workers)
        self.val_split = float(val_split)
        self.val_episodes = val_episodes
        self.split = split
        self.max_batches = max_batches
        self.verbose = verbose

    def run(self) -> Dict[str, Any]:
        import torch

        wrapper = self.wrapper
        wrapper._ensure_loaded()
        config = wrapper.config
        (LeRobotDataset, _LM, resolve_delta_timestamps, *_rest) = _lerobot()
        source = resolve_data_source(self.data)
        bundle = _build_datasets(
            wrapper,
            config,
            self.data,
            val_split=0.0 if self.split == "all" else self.val_split,
            val_episodes=self.val_episodes,
            train_episodes=None,
            need_train=self.split != "val",
            allow_empty_train=self.split == "val",
        )
        if self.split == "val":
            dataset = bundle.val
        elif self.split == "train":
            dataset = bundle.train
        else:
            delta = resolve_delta_timestamps(config, bundle.meta)
            dataset = LeRobotDataset(
                source.repo_id, root=source.root, delta_timestamps=delta
            )
        if dataset is None:
            raise ValueError(f"The {self.split!r} split has no episodes.")
        loader = _make_loader(dataset, self.batch, False, self.workers, 0)
        preds, targets, masks = [], [], []
        wrapper.reset()
        policy = wrapper._policy
        policy.eval()
        with torch.inference_mode():
            for idx, batch in enumerate(loader):
                if self.max_batches is not None and idx >= int(self.max_batches):
                    break
                target = batch["action"].detach().clone()
                pad = batch.get("action_is_pad")
                processed = wrapper._preprocessor(batch)
                chunk = policy.predict_action_chunk(processed)
                chunk = wrapper._postprocessor(chunk)
                dim = target.shape[-1]
                preds.append(chunk[:, : target.shape[1], :dim].detach().cpu())
                targets.append(target.cpu())
                masks.append(
                    (~pad).cpu()
                    if pad is not None
                    else torch.ones(target.shape[:2], dtype=torch.bool)
                )
        if not preds:
            raise ValueError("No batches evaluated.")
        metrics = action_error(
            torch.cat(preds),
            torch.cat(targets),
            mask=torch.cat(masks),
            names=bundle.action_names or wrapper.action_names,
        )
        metrics["episodes"] = (
            bundle.val_episodes
            if self.split == "val"
            else bundle.train_episodes
            if self.split == "train"
            else list(range(bundle.meta.total_episodes))
        )
        if self.verbose:
            logger.info(
                "val/action_l1 %.5f  val/action_mse %.5f  val/action_l1_first %.5f  steps %d",
                metrics["val/action_l1"],
                metrics["val/action_mse"],
                metrics["val/action_l1_first"],
                metrics["val/steps"],
            )
        return metrics
