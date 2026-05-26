"""Base trainer for LibreYOLO models.

Model-specific trainers subclass BaseTrainer and override hooks.
"""

import logging
import math
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .artifacts import TrainingArtifactsCallback
from .callbacks import (
    TrainCallbackList,
    TrainCallbacks,
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from .config import TrainConfig
from .distributed import (
    barrier,
    get_local_rank,
    get_rank,
    get_world_size,
    has_torchrun_env,
    init_distributed,
    is_distributed,
    is_main_process,
    parse_device_arg,
    scale_loss_for_ddp,
    seed_for_rank,
    unwrap_model,
    wants_distributed,
)
from .ema import ModelEMA
from ..data.dataset import YOLODataset, COCODataset, create_dataloader
from ..data import load_data_config, get_img_files, img2label_paths
from ..utils.serialization import (
    SCHEMA_VERSION,
    build_class_names,
    load_trusted_torch_file,
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)


logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Base trainer for all LibreYOLO model families.

    Subclasses override hook methods to customise transforms, schedulers,
    loss extraction, and family-specific behaviour.
    """

    best_metric_key: str = "metrics/mAP50-95"
    artifact_model_families: Tuple[str, ...] = ()

    def __init__(
        self,
        model: nn.Module,
        wrapper_model: Optional[Any] = None,
        callbacks: TrainCallbacks = None,
        **kwargs,
    ):
        self.config = self._config_class().from_kwargs(**kwargs)
        self.model = model
        self.wrapper_model = wrapper_model
        self.callbacks = TrainCallbackList(callbacks)
        self.artifact_callbacks = TrainCallbackList(
            TrainingArtifactsCallback(enabled_families=self.artifact_model_families)
        )

        # Distributed state. We init the process group eagerly when launched
        # under torchrun (LOCAL_RANK set in env) — this also covers the case
        # where the user passed device=[0,1] and ran with torchrun. If the
        # user passed a list-form device but did NOT launch with torchrun,
        # we raise a clear error in _setup_device pointing them at it.
        if has_torchrun_env() and not is_distributed():
            init_distributed()
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.world_size = get_world_size()
        self.is_distributed = is_distributed()

        # Per-rank seed so dataloader/aug RNG differs across ranks.
        if self.is_distributed and getattr(self.config, "seed", None) is not None:
            seed = seed_for_rank(int(self.config.seed))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Device
        self.device = self._setup_device()

        # Training state
        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0

        # Metric tracking
        self.best_mAP50_95 = 0.0
        self.best_mAP50 = 0.0
        self.best_epoch = 0
        self.final_loss = 0.0
        self.epoch_losses: List[float] = []
        self.epoch_events: List[TrainEpochEvent] = []
        self.patience_counter = 0

        # Initialised in setup()
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.ema_model = None
        self.train_loader = None
        self._is_setup = False

    # =========================================================================
    # Config
    # =========================================================================

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        """Return the config dataclass for this trainer. Subclasses override."""
        return TrainConfig

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def effective_lr(self) -> float:
        """Optimizer base learning rate."""
        return self.config.lr0

    @property
    def _accum_steps(self) -> int:
        """Micro-batches accumulated per optimizer step (1 disables accumulation).

        Derived from ``config.nbs`` (nominal batch size) the way Ultralytics
        derives accumulation — ``round(nbs / batch)``. When ``nbs`` is unset the
        trainer runs the standard one-optimizer-step-per-batch loop, unchanged
        from a build without this feature.
        """
        nbs = getattr(self.config, "nbs", None)
        if nbs is None:
            return 1
        return max(1, round(nbs / self.config.batch))

    def _scheduler_steps_per_epoch(self) -> int:
        """Optimizer steps per epoch — the unit the LR schedule advances in.

        Equals ``len(train_loader)`` without accumulation; with accumulation it
        is ``ceil(len / accum)`` so the schedule still advances exactly once per
        optimizer step. Requires ``train_loader`` to be set.
        """
        steps = len(self.train_loader)
        if self._accum_steps > 1:
            steps = max(1, math.ceil(steps / self._accum_steps))
        return steps

    @property
    def input_size(self) -> Tuple[int, int]:
        return (self.config.imgsz, self.config.imgsz)

    # =========================================================================
    # Hook methods — subclasses override these
    # =========================================================================

    @abstractmethod
    def get_model_family(self) -> str:
        """Return canonical model family string for checkpoint metadata."""

    @abstractmethod
    def get_model_tag(self) -> str:
        """Return human-readable model tag for log messages (e.g. 'YOLOX-s')."""

    @abstractmethod
    def create_transforms(self):
        """Return (preproc_transform, mosaic_dataset_class)."""

    @abstractmethod
    def create_scheduler(self, iters_per_epoch: int):
        """Return a scheduler with an ``update_lr(iters)`` method."""

    @abstractmethod
    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        """Extract per-component losses for progress bar and epoch metrics.

        Returns:
            Dict mapping loss name → scalar value.
        """

    def on_setup(self):
        """Called after model is on device, before data setup (e.g. bias init)."""

    def on_mosaic_disable(self):
        """Called when mosaic is disabled for final no-aug epochs."""
        dataset = getattr(self.train_loader, "dataset", None)
        if hasattr(dataset, "close_mosaic"):
            dataset.close_mosaic()

    def on_forward(
        self,
        imgs: torch.Tensor,
        targets: torch.Tensor,
        polygons: Optional[List] = None,
    ) -> Dict:
        """Run the model forward pass. Override if call signature differs.

        When ``load_segments=True`` is enabled, ``polygons`` follows the shared
        preservation contract:

        - list length equals batch size
        - each image entry is a list of instances matching that image's target rows
        - each instance is a list of polygon rings
        - each ring is an ``Nx2`` array in original image pixel coordinates

        Detection rows without polygon labels use an empty ring list for that
        instance. Detection-only trainers may ignore ``polygons``.
        """
        return self.model(imgs, targets)

    # =========================================================================
    # Shared infrastructure
    # =========================================================================

    def _setup_device(self) -> torch.device:
        # Distributed mode: device is dictated by LOCAL_RANK + intent.
        # The user can force CPU/MPS even with CUDA available (useful for
        # CPU-DDP smoke tests with gloo). Otherwise default to cuda:LOCAL_RANK.
        if self.is_distributed:
            cfg_device = str(self.config.device).strip().lower() if not isinstance(self.config.device, (list, tuple, int)) else None
            forced_cpu = cfg_device == "cpu"
            forced_mps = cfg_device == "mps"
            if forced_cpu:
                device = torch.device("cpu")
            elif forced_mps and torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                device = torch.device(f"cuda:{self.local_rank}")
            else:
                device = torch.device("cpu")
            if is_main_process():
                logger.info(
                    f"DDP active: rank={self.rank}/{self.world_size} device={device}"
                )
            return device

        # Single-process mode. Accept list/comma device only as an intent signal
        # — fail loudly with a torchrun pointer rather than silently degrading.
        raw_device = self.config.device

        # Normalise single-element list/tuple to its int (Ultralytics accepts
        # ``device=[0]`` as equivalent to ``device=0``). Multi-element forms
        # fall through to the wants_distributed check below.
        if isinstance(raw_device, (list, tuple)) and len(raw_device) == 1:
            raw_device = raw_device[0]

        if wants_distributed(raw_device):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Multi-GPU requested (device={raw_device!r}) but CUDA is not "
                    "available."
                )
            n = len(parse_device_arg(raw_device))
            raise RuntimeError(
                f"Multi-GPU device {raw_device!r} was passed directly to the trainer "
                "without an active process group. Use the model API instead — it "
                f"spawns DDP workers automatically: model.train(data=..., device={raw_device!r}). "
                f"Alternatively launch with torchrun: "
                f"`torchrun --nproc_per_node={n} your_script.py`."
            )

        device_str = str(raw_device).strip().lower() if not isinstance(raw_device, int) else str(raw_device)
        if isinstance(raw_device, int):
            device_str = f"cuda:{raw_device}"
        elif device_str in ("", "auto"):
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            logger.info(f"Using device: {device}")
            return device
        # YOLO-style "0" -> "cuda:0"
        if device_str.isdigit():
            device_str = f"cuda:{device_str}"
        device = torch.device(device_str)
        logger.info(f"Using device: {device}")
        return device

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        pg0, pg1, pg2 = [], [], []
        # Catch every batch-norm flavour, including SyncBN: BatchNorm{1,2,3}d
        # and SyncBatchNorm are all siblings under ``_BatchNorm``. The naive
        # ``isinstance(v, nn.BatchNorm2d)`` check would silently put SyncBN
        # weights into the weight-decay group post sync_bn conversion.
        bn_types = nn.modules.batchnorm._BatchNorm
        for _k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, bn_types):
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)

        lr = self.effective_lr
        opt_name = self.config.optimizer

        if opt_name == "sgd":
            optimizer = torch.optim.SGD(
                pg0,
                lr=lr,
                momentum=self.config.momentum,
                nesterov=self.config.nesterov,
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(pg0, lr=lr)
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(pg0, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        optimizer.add_param_group(
            {"params": pg1, "lr": lr, "weight_decay": self.config.weight_decay}
        )
        optimizer.add_param_group({"params": pg2, "lr": lr})

        if is_main_process():
            logger.info(f"Optimizer: {opt_name}")
            logger.info(f"  - pg0 (BN): {len(pg0)} params")
            logger.info(f"  - pg1 (Conv, wd={self.config.weight_decay}): {len(pg1)} params")
            logger.info(f"  - pg2 (Bias): {len(pg2)} params")
        return optimizer

    def _get_save_dir(self) -> Path:
        project = Path(self.config.project)
        name = self.config.name

        save_dir = project / name
        if not self.config.exist_ok and save_dir.exists():
            i = 2
            while (project / f"{name}{i}").exists():
                i += 1
            save_dir = project / f"{name}{i}"

        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _setup_data(self):
        img_size = self.input_size
        preproc, MosaicDatasetClass = self.create_transforms()
        load_segments = getattr(self.wrapper_model, "task", "detect") == "segment"

        if self.config.data:
            data_cfg = load_data_config(
                self.config.data,
                allow_scripts=self.config.allow_download_scripts,
            )
            data_dir = data_cfg["root"]
            self.num_classes = data_cfg.get("nc", self.config.num_classes)

            ann_file = Path(data_dir) / "annotations" / "instances_train2017.json"

            # Prefer pre-resolved file lists from load_data_config (.txt format)
            img_files = data_cfg.get("train_img_files")
            label_files = data_cfg.get("train_label_files")

            if img_files:
                train_dataset = YOLODataset(
                    img_files=img_files,
                    label_files=label_files,
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                )
            elif ann_file.exists():
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file="instances_train2017.json",
                    name="train2017",
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                )
            else:
                train_path = data_cfg.get("train", "images/train")
                train_img_dir = Path(train_path)
                if not train_img_dir.is_absolute():
                    train_img_dir = Path(data_dir) / train_img_dir

                try:
                    img_files = get_img_files(train_path, prefix=data_dir)
                except (FileNotFoundError, ValueError):
                    img_files = []

                if len(img_files) == 0:
                    raise FileNotFoundError(f"No images found in {train_img_dir}")

                label_files = img2label_paths(img_files)

                train_dataset = YOLODataset(
                    img_files=img_files,
                    label_files=label_files,
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                )
        elif self.config.data_dir:
            data_dir = self.config.data_dir
            self.num_classes = self.config.num_classes

            if (Path(data_dir) / "annotations").exists():
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file="instances_train2017.json",
                    name="train2017",
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                )
            else:
                train_dataset = YOLODataset(
                    data_dir=data_dir,
                    split="train",
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                )
        else:
            raise ValueError("Either 'data' or 'data_dir' must be specified")

        train_dataset = MosaicDatasetClass(
            dataset=train_dataset,
            img_size=img_size,
            mosaic=True,
            preproc=preproc,
            degrees=self.config.degrees,
            translate=self.config.translate,
            mosaic_scale=self.config.mosaic_scale,
            mixup_scale=self.config.mixup_scale,
            shear=self.config.shear,
            enable_mixup=self.config.mixup_prob > 0,
            mosaic_prob=self.config.mosaic_prob,
            mixup_prob=self.config.mixup_prob,
        )

        # Ultralytics-mirror semantics: ``batch`` is the GLOBAL batch under
        # DDP. Each rank's loader is built with ``batch // world_size``.
        per_rank_batch = max(1, self.config.batch // max(self.world_size, 1))
        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )

        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=per_rank_batch,
            num_workers=self.config.workers,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
            sampler=sampler,
        )

        if is_main_process():
            logger.info(f"Training dataset: {len(train_dataset)} images")
            logger.info(
                f"Iterations per epoch: {len(self.train_loader)} "
                f"(batch_per_rank={per_rank_batch}, world_size={self.world_size})"
            )
        return train_dataset

    # =========================================================================
    # Setup / train / epoch
    # =========================================================================

    def setup(self):
        if self._is_setup:
            return

        if is_main_process():
            logger.info("Setting up training...")
        self.model.to(self.device)

        # SyncBatchNorm conversion: only meaningful under DDP. Single-GPU
        # runs skip this regardless of the flag so single-GPU is unchanged.
        if self.is_distributed and getattr(self.config, "sync_bn", False):
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if is_main_process():
                logger.info("Converted BatchNorm to SyncBatchNorm")

        self.on_setup()

        if getattr(self.config, "batch", 16) == -1:
            from libreyolo.training.autobatch import resolve_auto_batch

            self.config.batch = resolve_auto_batch(
                self.model,
                imgsz=self.config.imgsz,
                amp=self.config.amp,
                world_size=self.world_size,
                nbs=getattr(self.config, "nbs", None),
            )
            if is_main_process():
                logger.info("AutoBatch: resolved global batch size = %d", self.config.batch)

        self._setup_data()
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = self.create_scheduler(self._scheduler_steps_per_epoch())

        # resume() may be called before setup() when the optimizer doesn't exist
        # yet. Apply the deferred state now so momentum buffers are restored before
        # _initialize_scheduler_lr() sets the correct LR on top.
        if getattr(self, "_resume_optimizer_state", None) is not None:
            try:
                self.optimizer.load_state_dict(self._resume_optimizer_state)
                logger.info("Optimizer state restored from resume checkpoint")
            except Exception as e:
                logger.warning(f"Could not load deferred optimizer state: {e}")
            finally:
                self._resume_optimizer_state = None

        self._initialize_scheduler_lr()

        # DDP wrap AFTER optimizer setup so _setup_optimizer's
        # named_parameters() sees the raw model. EMA below also reads the
        # raw model — ModelEMA already unwraps via is_parallel() check.
        if self.is_distributed:
            ddp_kwargs = self._ddp_kwargs()
            if self.device.type == "cuda":
                ddp_kwargs["device_ids"] = [self.local_rank]
                ddp_kwargs["output_device"] = self.local_rank
            self.model = nn.parallel.DistributedDataParallel(self.model, **ddp_kwargs)
            if is_main_process():
                logger.info(
                    "Wrapped model in DDP ("
                    + ", ".join(f"{k}={v}" for k, v in ddp_kwargs.items() if k not in ("device_ids", "output_device"))
                    + ")"
                )

        if self.config.amp and self.device.type == "cuda":
            self.scaler = GradScaler("cuda")
            if is_main_process():
                logger.info("Using mixed precision training (AMP)")
        else:
            self.scaler = None

        if self.config.ema:
            ema_tau = getattr(self.config, "ema_tau", 2000)
            self.ema_model = ModelEMA(
                self.model, decay=self.config.ema_decay, tau=ema_tau
            )
            if is_main_process():
                logger.info(
                    "Using EMA with decay=%s, tau=%s",
                    self.config.ema_decay,
                    ema_tau,
                )

        # Save-dir creation, config dump, and TB writer all live on rank 0.
        # The resolved name (which may include an auto-increment suffix when
        # exist_ok=False and a previous run exists) is broadcast to other
        # ranks so every rank's ``self.save_dir`` agrees and event-level
        # paths emitted by callbacks are consistent.
        if is_main_process():
            self.save_dir = self._get_save_dir()
            self.config.to_yaml(self.save_dir / "train_config.yaml")
            logger.info(f"Saving to: {self.save_dir}")
        else:
            self.save_dir = Path(self.config.project) / self.config.name

        if self.is_distributed:
            import torch.distributed as _dist

            container = [str(self.save_dir)] if is_main_process() else [None]
            _dist.broadcast_object_list(container, src=0)
            self.save_dir = Path(container[0])

        # Wait for rank 0 to finish dir creation before any rank proceeds.
        barrier()
        self._is_setup = True

    def _ddp_find_unused_parameters(self) -> bool:
        """Subclasses override to flip when their forward graph is conditional.

        Default False matches PyTorch's default and Ultralytics. rf-detr
        flips True when a segmentation head is present (the sparse branch
        leaves some params un-grad'd on some batches).
        """
        return False

    def _ddp_static_graph(self) -> bool:
        """Whether to pass ``static_graph=True`` to DDP.

        ``static_graph=True`` defers DDP's reducer analysis until after
        the first iteration, which correctly handles models whose
        gradients land with non-contiguous strides (e.g. multi-head
        attention QKV projections). It can only be combined with
        ``find_unused_parameters=False`` — when the forward graph has
        conditional branches, static_graph is unsound.

        Default: enabled when find_unused is False. Subclasses can
        override for finer control.
        """
        return not self._ddp_find_unused_parameters()

    def _ddp_kwargs(self) -> Dict[str, Any]:
        """Assemble DDP constructor kwargs. Subclasses can override.

        gradient_as_bucket_view defaults False because some flagship
        models (RF-DETR's transformer) produce gradient tensors whose
        strides don't match DDP's bucket view, causing silent sync
        misses. The memory cost is small for the models in scope.
        """
        return {
            "find_unused_parameters": self._ddp_find_unused_parameters(),
            "static_graph": self._ddp_static_graph(),
            "gradient_as_bucket_view": False,
        }

    def train(self) -> Dict:
        start_time = time.time()
        try:
            self.setup()

            if is_main_process():
                logger.info(f"Starting training for {self.config.epochs} epochs")
                logger.info(f"Model: {self.get_model_tag()}")
                logger.info(f"Batch size: {self.config.batch}")
                logger.info(f"Learning rate: {self.effective_lr}")

            start_event = self._build_train_start_event()
            # Artifact + user callbacks fire on rank 0 only — they write
            # files (results.json, TensorBoard, etc.) that would race on
            # shared paths otherwise.
            if is_main_process():
                self._dispatch_artifact_callbacks("on_train_start", start_event)
                self.callbacks.on_train_start(start_event)

            no_aug_start = self.config.epochs - self.config.no_aug_epochs
            if self.config.no_aug_epochs > 0 and self.start_epoch > no_aug_start:
                if is_main_process():
                    logger.info(
                        f"Resumed past no-aug threshold (epoch {self.start_epoch} > {no_aug_start}), "
                        f"disabling mosaic/mixup immediately"
                    )
                self.on_mosaic_disable()

            for epoch in range(self.start_epoch, self.config.epochs):
                self.current_epoch = epoch

                if epoch == no_aug_start:
                    if is_main_process():
                        logger.info(
                            f"Disabling mosaic/mixup for final {self.config.no_aug_epochs} epochs"
                        )
                    self.on_mosaic_disable()

                epoch_start_time = time.time()
                epoch_result = self._train_epoch(epoch)
                epoch_seconds = time.time() - epoch_start_time
                epoch_loss, val_metrics, loss_items, lr = self._normalize_epoch_result(
                    epoch_result
                )
                self.final_loss = epoch_loss
                self.epoch_losses.append(epoch_loss)

                is_best = self._update_best_state(epoch, val_metrics)
                should_save = (
                    (epoch + 1) % self.config.save_period == 0
                    or epoch == self.config.epochs - 1
                    or is_best
                )
                if should_save:
                    self._save_checkpoint(
                        epoch, epoch_loss, val_metrics, is_best=is_best
                    )

                event = self._build_train_epoch_event(
                    epoch=epoch,
                    train_loss=epoch_loss,
                    train_loss_items=loss_items,
                    lr=lr,
                    val_metrics=val_metrics,
                    is_best=is_best,
                    epoch_seconds=epoch_seconds,
                )
                self.epoch_events.append(event)
                if is_main_process():
                    self._dispatch_artifact_callbacks("on_train_epoch_end", event)
                    self.callbacks.on_train_epoch_end(event)

                # Early-stop decision lives on rank 0 only (patience_counter
                # is updated from val_metrics, which only rank 0 receives).
                # We broadcast the stop flag so every rank exits the loop in
                # lockstep — otherwise non-rank-0 ranks proceed into the
                # next epoch's collective backward() and deadlock.
                should_stop = (
                    self.config.patience > 0
                    and self.patience_counter >= self.config.patience
                )
                if self.is_distributed:
                    import torch.distributed as _dist

                    flag = torch.tensor(int(should_stop), dtype=torch.int, device=self.device)
                    _dist.broadcast(flag, src=0)
                    should_stop = bool(flag.item())
                if should_stop:
                    if is_main_process():
                        logger.info(
                            f"Early stopping triggered after {epoch + 1} epochs "
                            f"(patience={self.config.patience}, no improvement for {self.patience_counter} epochs)"
                        )
                    break

            total_time = time.time() - start_time
            if is_main_process():
                logger.info(f"Training complete in {total_time / 3600:.2f} hours")

            results = self._build_train_results()
            end_event = self._build_train_end_event(total_time, results)
            if is_main_process():
                self._dispatch_artifact_callbacks("on_train_end", end_event)
                self.callbacks.on_train_end(end_event)
            return results

        except BaseException as exc:
            elapsed_seconds = time.time() - start_time
            exception_event = self._build_train_exception_event(exc, elapsed_seconds)
            if is_main_process():
                self._dispatch_artifact_callbacks("on_train_exception", exception_event)
                try:
                    self.callbacks.on_train_exception(exception_event)
                except Exception:
                    logger.exception("Training exception callback failed")
            raise

    def _dispatch_artifact_callbacks(self, method_name: str, event) -> None:
        try:
            getattr(self.artifact_callbacks, method_name)(event)
        except Exception:
            logger.exception("Training artifact callback failed")

    def _build_train_results(self) -> Dict[str, Any]:
        weights_dir = self.save_dir / "weights"
        epoch_metrics = [self._event_to_dict(event) for event in self.epoch_events]
        return {
            "final_loss": self.final_loss,
            "epoch_losses": list(self.epoch_losses),
            "epoch_lrs": [dict(event.lr) for event in self.epoch_events],
            "epoch_loss_items": [
                dict(event.train_loss_items) for event in self.epoch_events
            ],
            "val_metrics": [dict(event.val_metrics) for event in self.epoch_events],
            "epoch_metrics": epoch_metrics,
            "best_mAP50": self.best_mAP50,
            "best_mAP50_95": self.best_mAP50_95,
            "best_epoch": self.best_epoch,
            "save_dir": str(self.save_dir),
            "best_checkpoint": str(weights_dir / "best.pt"),
            "last_checkpoint": str(weights_dir / "last.pt"),
        }

    def _event_context(self) -> Dict[str, Any]:
        return {
            "total_epochs": self.config.epochs,
            "model_family": self.get_model_family(),
            "model_size": getattr(self.config, "size", None),
            "task": getattr(self.wrapper_model, "task", "detect"),
            "save_dir": str(getattr(self, "save_dir", "")),
        }

    def _build_train_start_event(self) -> TrainStartEvent:
        return TrainStartEvent(
            start_epoch=self.start_epoch + 1,
            **self._event_context(),
        )

    def _build_train_end_event(
        self, total_seconds: float, results: Mapping[str, Any]
    ) -> TrainEndEvent:
        return TrainEndEvent(
            completed_epochs=len(self.epoch_events),
            final_loss=self.final_loss,
            best_metric=self.best_mAP50_95 if self.best_epoch else None,
            best_epoch=self.best_epoch if self.best_epoch else None,
            total_seconds=total_seconds,
            results=results,
            **self._event_context(),
        )

    def _build_train_exception_event(
        self, exc: BaseException, elapsed_seconds: float
    ) -> TrainExceptionEvent:
        return TrainExceptionEvent(
            epoch=self.current_epoch + 1 if self._is_setup else None,
            exception=exc,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            elapsed_seconds=elapsed_seconds,
            **self._event_context(),
        )

    def _scale_lr(self, base_lr: float, param_group: dict) -> float:
        """Hook for per-group LR scaling. Override in subclasses."""
        return base_lr

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _scalar_mapping(self, values: Optional[Mapping]) -> Dict[str, float]:
        if not isinstance(values, Mapping):
            return {}

        scalars = {}
        for name, value in values.items():
            scalar = self._as_float(value)
            if scalar is not None:
                scalars[str(name)] = scalar
        return scalars

    def _current_lrs(self) -> Dict[str, float]:
        if self.optimizer is None:
            return {}
        return {
            f"group{i}": float(param_group.get("lr", 0.0))
            for i, param_group in enumerate(self.optimizer.param_groups)
        }

    @staticmethod
    def _event_to_dict(event: TrainEpochEvent) -> Dict[str, Any]:
        return {
            "epoch": event.epoch,
            "total_epochs": event.total_epochs,
            "model_family": event.model_family,
            "model_size": event.model_size,
            "task": event.task,
            "save_dir": event.save_dir,
            "train_loss": event.train_loss,
            "train_loss_items": dict(event.train_loss_items),
            "lr": dict(event.lr),
            "val_metrics": dict(event.val_metrics),
            "validated": event.validated,
            "is_best": event.is_best,
            "current_metric": event.current_metric,
            "current_metric_name": event.current_metric_name,
            "best_metric": event.best_metric,
            "best_metric_name": event.best_metric_name,
            "best_epoch": event.best_epoch,
            "epoch_seconds": event.epoch_seconds,
        }

    def _normalize_epoch_result(
        self, epoch_result: Tuple
    ) -> Tuple[float, Optional[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
        if not isinstance(epoch_result, tuple):
            raise TypeError("_train_epoch must return a tuple")

        if len(epoch_result) == 2:
            epoch_loss, val_metrics = epoch_result
            loss_items = {}
            lr = self._current_lrs()
        elif len(epoch_result) == 4:
            epoch_loss, val_metrics, loss_items, lr = epoch_result
            loss_items = self._scalar_mapping(loss_items)
            lr = self._scalar_mapping(lr) or self._current_lrs()
        else:
            raise ValueError(
                "_train_epoch must return (loss, val_metrics) or "
                "(loss, val_metrics, loss_items, lr)"
            )

        return float(epoch_loss), val_metrics, dict(loss_items), dict(lr)

    def _best_metric_value(self, val_metrics: Optional[Dict[str, Any]]) -> float:
        if not val_metrics:
            return 0.0

        value = val_metrics.get("best_metric", val_metrics.get("mAP50_95", 0.0))
        scalar = self._as_float(value)
        return scalar if scalar is not None else 0.0

    def _best_metric_name(self, val_metrics: Optional[Dict[str, Any]]) -> str:
        if val_metrics:
            return str(
                val_metrics.get(
                    "best_metric_key",
                    getattr(self, "best_metric_key", "metrics/mAP50-95"),
                )
            )
        return str(getattr(self, "best_metric_key", "metrics/mAP50-95"))

    def _validation_metrics_for_event(
        self, val_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        if not val_metrics:
            return {}

        raw_metrics = val_metrics.get("metrics")
        if isinstance(raw_metrics, Mapping):
            return self._scalar_mapping(raw_metrics)
        return self._scalar_mapping(val_metrics)

    def _build_train_epoch_event(
        self,
        *,
        epoch: int,
        train_loss: float,
        train_loss_items: Mapping[str, float],
        lr: Mapping[str, float],
        val_metrics: Optional[Dict[str, Any]],
        is_best: bool,
        epoch_seconds: float,
    ) -> TrainEpochEvent:
        current_metric = self._best_metric_value(val_metrics) if val_metrics else None
        current_metric_name = (
            self._best_metric_name(val_metrics) if val_metrics else None
        )
        best_metric = self.best_mAP50_95 if self.best_epoch else None
        best_metric_name = (
            self._best_metric_name(val_metrics) if self.best_epoch else None
        )

        return TrainEpochEvent(
            epoch=epoch + 1,
            total_epochs=self.config.epochs,
            model_family=self.get_model_family(),
            model_size=getattr(self.config, "size", None),
            task=getattr(self.wrapper_model, "task", "detect"),
            save_dir=str(self.save_dir),
            train_loss=float(train_loss),
            train_loss_items=self._scalar_mapping(train_loss_items),
            lr=self._scalar_mapping(lr),
            val_metrics=self._validation_metrics_for_event(val_metrics),
            validated=bool(val_metrics),
            is_best=is_best,
            current_metric=current_metric,
            current_metric_name=current_metric_name,
            best_metric=best_metric,
            best_metric_name=best_metric_name,
            best_epoch=self.best_epoch if self.best_epoch else None,
            epoch_seconds=float(epoch_seconds),
        )

    def _update_best_state(
        self, epoch: int, val_metrics: Optional[Dict[str, Any]]
    ) -> bool:
        if not val_metrics:
            return False

        best_metric = self._best_metric_value(val_metrics)
        is_best = self.best_epoch == 0 or best_metric > self.best_mAP50_95
        if is_best:
            self.best_mAP50_95 = best_metric
            mAP50 = self._as_float(val_metrics.get("mAP50", 0.0))
            self.best_mAP50 = mAP50 if mAP50 is not None else 0.0
            self.best_epoch = epoch + 1
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return is_best

    def _get_clip_max_norm(self) -> float:
        value = getattr(self.config, "clip_max_norm", 0.0)
        if value is None:
            return 0.0
        try:
            max_norm = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"clip_max_norm must be a finite non-negative number, got {value!r}"
            ) from exc
        if max_norm < 0.0 or not math.isfinite(max_norm):
            raise ValueError(
                f"clip_max_norm must be a finite non-negative number, got {value!r}"
            )
        return max_norm

    def _should_clip_gradients(self) -> bool:
        return self._get_clip_max_norm() > 0.0

    def _set_optimizer_lr(self, base_lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self._scale_lr(base_lr, param_group)

    def _initialize_scheduler_lr(self) -> None:
        if self.optimizer is None or self.lr_scheduler is None:
            return
        init_iter = getattr(self, "start_epoch", 0) * self._scheduler_steps_per_epoch()
        self._set_optimizer_lr(self.lr_scheduler.update_lr(init_iter))

    def _gradient_clip_parameters(self) -> List[torch.nn.Parameter]:
        if self.optimizer is None:
            return []
        params = []
        seen = set()
        for group in self.optimizer.param_groups:
            for param in group.get("params", ()):
                if param.grad is None:
                    continue
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                params.append(param)
        return params

    def _clip_gradients(self) -> Optional[torch.Tensor]:
        max_norm = self._get_clip_max_norm()
        if max_norm <= 0.0:
            return None
        return torch.nn.utils.clip_grad_norm_(
            self._gradient_clip_parameters(),
            max_norm,
        )

    def _train_epoch(
        self, epoch: int
    ) -> Tuple[float, Optional[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
        self.model.train()

        # Gradient accumulation is opt-in. When enabled, delegate to the
        # accumulation loop; otherwise fall through to the standard
        # one-optimizer-step-per-batch loop below, unchanged.
        if self._accum_steps > 1:
            return self._train_epoch_accum(epoch)

        # DistributedSampler needs its epoch set so shuffling differs per
        # epoch while staying deterministic for resume.
        if is_distributed() and hasattr(self.train_loader, "sampler"):
            sampler = self.train_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            total=len(self.train_loader),
            disable=not sys.stderr.isatty() or not is_main_process(),
            file=sys.stderr,
        )

        total_loss = 0.0
        num_batches = 0
        loss_component_sums: Dict[str, float] = {}

        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 5:
                imgs, targets, img_infos, img_ids, polygons = batch
            else:
                imgs, targets, img_infos, img_ids = batch
                polygons = None
            self.current_iter = epoch * len(self.train_loader) + batch_idx

            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if hasattr(self, "_apply_multi_scale_batch"):
                imgs, targets, polygons = self._apply_multi_scale_batch(
                    imgs,
                    targets,
                    polygons,
                    step=self.current_iter,
                )

            # Forward + backward. Under DDP we multiply loss by world_size
            # so that the gradient averaging that happens inside backward()
            # produces the same sum-of-per-rank gradients as single-GPU
            # would (Ultralytics-mirror pattern). No-op outside DDP.
            if self.scaler is not None:
                with autocast("cuda"):
                    outputs = self.on_forward(imgs, targets, polygons=polygons)
                    total_loss_raw = outputs["total_loss"]
                loss = scale_loss_for_ddp(total_loss_raw)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                if self._should_clip_gradients():
                    self.scaler.unscale_(self.optimizer)
                    self._clip_gradients()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.on_forward(imgs, targets, polygons=polygons)
                total_loss_raw = outputs["total_loss"]
                loss = scale_loss_for_ddp(total_loss_raw)
                self.optimizer.zero_grad()
                loss.backward()
                self._clip_gradients()
                self.optimizer.step()

            # EMA
            if self.ema_model is not None:
                self.ema_model.update(self.model)

            # Logging captures the pre-scale value so single-GPU and DDP
            # report identical magnitudes (single-GPU semantics). ``.item()``
            # already returns a Python float and detaches from autograd.
            loss_val = float(total_loss_raw.item())
            loss_components = self._scalar_mapping(self.get_loss_components(outputs))
            total_loss += loss_val
            for name, value in loss_components.items():
                loss_component_sums[name] = loss_component_sums.get(name, 0.0) + value

            del outputs, loss

            # LR update
            lr = self.lr_scheduler.update_lr(self.current_iter + 1)
            self._set_optimizer_lr(lr)
            num_batches += 1

            # Progress bar
            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{lr:.6f}"}
            postfix.update({k: f"{v:.4f}" for k, v in loss_components.items()})
            pbar.set_postfix(postfix)

        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_components = {
            name: value / max(num_batches, 1)
            for name, value in loss_component_sums.items()
        }
        if is_main_process():
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

        # Validation
        val_metrics = None
        if (
            self.config.eval_interval > 0
            and (epoch + 1) % self.config.eval_interval == 0
        ):
            val_metrics = self._validate_epoch(epoch)

        return avg_loss, val_metrics, avg_loss_components, self._current_lrs()

    def _train_epoch_accum(
        self, epoch: int
    ) -> Tuple[float, Optional[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
        """``_train_epoch`` variant with gradient accumulation enabled.

        Accumulates gradients over ``accum`` micro-batches before each optimizer
        step. The micro-batch loss is divided by the accumulation window so the
        summed gradient equals the mean over the effective batch; the optimizer
        step, gradient clipping, EMA update and LR scheduler each advance once
        per optimizer step. Reached only when ``_accum_steps > 1``.
        """
        self.model.train()

        if is_distributed() and hasattr(self.train_loader, "sampler"):
            sampler = self.train_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            total=len(self.train_loader),
            disable=not sys.stderr.isatty() or not is_main_process(),
            file=sys.stderr,
        )

        accum = self._accum_steps
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / accum))
        total_loss = 0.0
        num_batches = 0
        loss_component_sums: Dict[str, float] = {}
        actual_window = accum
        lr = self.optimizer.param_groups[0]["lr"]

        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 5:
                imgs, targets, img_infos, img_ids, polygons = batch
            else:
                imgs, targets, img_infos, img_ids = batch
                polygons = None

            is_opt_step = (batch_idx + 1) % accum == 0 or batch_idx == len(self.train_loader) - 1
            opt_step = epoch * steps_per_epoch + batch_idx // accum
            self.current_iter = opt_step

            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if hasattr(self, "_apply_multi_scale_batch"):
                imgs, targets, polygons = self._apply_multi_scale_batch(
                    imgs,
                    targets,
                    polygons,
                    step=opt_step,
                )

            if batch_idx % accum == 0:
                self.optimizer.zero_grad(set_to_none=True)
                actual_window = min(accum, len(self.train_loader) - batch_idx)

            # Forward + backward. Gradients accumulate across the window; the
            # optimizer step, clipping, EMA and LR update fire only on the
            # window boundary (``is_opt_step``). Under DDP we additionally
            # multiply the per-micro-batch loss by world_size so DDP's
            # gradient-averaging composes correctly with the division-by-
            # window scheme (Ultralytics-mirror pattern).
            if self.scaler is not None:
                with autocast("cuda"):
                    outputs = self.on_forward(imgs, targets, polygons=polygons)
                    total_loss_raw = outputs["total_loss"]
                    loss = total_loss_raw / actual_window
                loss = scale_loss_for_ddp(loss)
                self.scaler.scale(loss).backward()
                if is_opt_step:
                    if self._should_clip_gradients():
                        self.scaler.unscale_(self.optimizer)
                        self._clip_gradients()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                outputs = self.on_forward(imgs, targets, polygons=polygons)
                total_loss_raw = outputs["total_loss"]
                loss = total_loss_raw / actual_window
                loss = scale_loss_for_ddp(loss)
                loss.backward()
                if is_opt_step:
                    self._clip_gradients()
                    self.optimizer.step()

            if is_opt_step:
                # EMA
                if self.ema_model is not None:
                    self.ema_model.update(self.model)
                # LR update
                lr = self.lr_scheduler.update_lr(opt_step + 1)
                self._set_optimizer_lr(lr)

            # Logging uses the raw pre-scale value (single-GPU semantics).
            loss_val = float(total_loss_raw.detach().item())
            loss_components = self._scalar_mapping(self.get_loss_components(outputs))
            total_loss += loss_val
            num_batches += 1
            for name, value in loss_components.items():
                loss_component_sums[name] = loss_component_sums.get(name, 0.0) + value

            del outputs, loss

            # Progress bar
            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{lr:.6f}"}
            postfix.update({k: f"{v:.4f}" for k, v in loss_components.items()})
            pbar.set_postfix(postfix)

        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_components = {
            name: value / max(num_batches, 1)
            for name, value in loss_component_sums.items()
        }
        if is_main_process():
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

        # Validation
        val_metrics = None
        if (
            self.config.eval_interval > 0
            and (epoch + 1) % self.config.eval_interval == 0
        ):
            val_metrics = self._validate_epoch(epoch)

        return avg_loss, val_metrics, avg_loss_components, self._current_lrs()

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_epoch(self, epoch: int) -> Optional[Dict[str, Any]]:
        # First-cut policy: validation runs on rank 0 only. Non-zero ranks
        # barrier-wait so the next epoch's set_epoch fires in lockstep.
        # Rank 0 barriers once at the bottom regardless of outcome.
        if self.is_distributed and not is_main_process():
            barrier()
            return None
        try:
            return self._run_validation(epoch)
        finally:
            if self.is_distributed:
                barrier()

    def _run_validation(self, epoch: int) -> Optional[Dict[str, Any]]:
        try:
            from libreyolo.validation import (
                DetectionValidator,
                SegmentationValidator,
                ValidationConfig,
            )

            logger.info(f"Running validation for epoch {epoch + 1}")

            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=self.config.batch,
                imgsz=self.config.imgsz,
                conf_thres=0.001,
                iou_thres=0.65,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                verbose=False,
                num_workers=self.config.workers,
            )

            if self.wrapper_model is None:
                logger.error(
                    "Validation requires wrapper_model to be provided to trainer"
                )
                return None

            # Validator wants the un-DDP-wrapped module.
            eval_pytorch_model = (
                self.ema_model.ema if self.ema_model else unwrap_model(self.model)
            )
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model

            try:
                validator_cls = (
                    SegmentationValidator
                    if getattr(self.wrapper_model, "task", "detect") == "segment"
                    else DetectionValidator
                )
                validator = validator_cls(model=self.wrapper_model, config=val_config)
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            raw_metrics = self._scalar_mapping(results)
            best_key = getattr(self, "best_metric_key", "metrics/mAP50-95")
            best_metric = raw_metrics.get(
                best_key, raw_metrics.get("metrics/mAP50-95", 0.0)
            )
            metrics = {
                "mAP50": raw_metrics.get(
                    "metrics/mAP50", raw_metrics.get("metrics/mAP50(B)", 0.0)
                ),
                "mAP50_95": best_metric,
                "best_metric": best_metric,
                "best_metric_key": best_key,
                "metrics": raw_metrics,
            }

            logger.debug(
                f"Extracted metrics: mAP50={metrics['mAP50']:.4f}, mAP50_95={metrics['mAP50_95']:.4f}"
            )
            logger.info(
                "Validation - mAP50: %.4f, mAP50-95: %.4f",
                metrics["mAP50"],
                metrics["mAP50_95"],
            )
            return metrics

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import traceback

            logger.debug(f"Validation traceback:\n{traceback.format_exc()}")
            return None

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def _save_checkpoint(
        self,
        epoch: int,
        loss: float,
        val_metrics: Optional[Dict[str, Any]] = None,
        is_best: Optional[bool] = None,
    ):
        if is_best is None:
            is_best = self._update_best_state(epoch, val_metrics)

        # Only rank 0 writes checkpoint files. Other ranks skip silently.
        if not is_main_process():
            return

        # Always unwrap DDP/compile wrappers before reading state_dict so the
        # checkpoint is interchangeable with single-GPU runs.
        raw_model = unwrap_model(self.model)
        model_to_save = self.ema_model.ema if self.ema_model else raw_model

        best_metric_key = (
            val_metrics.get(
                "best_metric_key",
                getattr(self, "best_metric_key", "metrics/mAP50-95"),
            )
            if val_metrics
            else getattr(self, "best_metric_key", "metrics/mAP50-95")
        )
        names = (
            self.wrapper_model.names
            if self.wrapper_model is not None and hasattr(self.wrapper_model, "names")
            else build_class_names(self.config.num_classes)
        )
        checkpoint_imgsz = getattr(self.config, "imgsz", None)
        if checkpoint_imgsz is None and self.wrapper_model is not None:
            get_input_size = getattr(self.wrapper_model, "_get_input_size", None)
            if callable(get_input_size):
                checkpoint_imgsz = get_input_size()
        if checkpoint_imgsz is None:
            checkpoint_imgsz = 640
            logger.warning(
                "Training config has no imgsz. Writing checkpoint metadata "
                "imgsz=640; set config.imgsz to avoid this compatibility fallback."
            )

        checkpoint = wrap_libreyolo_checkpoint(
            model_to_save.state_dict(),
            model_family=self.get_model_family(),
            size=self.config.size,
            task=getattr(self.wrapper_model, "task", "detect"),
            nc=self.config.num_classes,
            names=names,
            imgsz=int(checkpoint_imgsz),
            epoch=epoch,
            optimizer=self.optimizer.state_dict(),
            config=self.config.to_dict(),
            loss=loss,
            best_mAP50_95=self.best_mAP50_95,
            best_mAP50=self.best_mAP50,
            best_metric_key=best_metric_key,
            best_metric_value=self.best_mAP50_95,
            best_epoch=self.best_epoch,
            is_ema_weights=self.ema_model is not None,
        )
        checkpoint.update(self._checkpoint_extra_metadata())
        checkpoint["best_metric"] = self.best_mAP50_95
        checkpoint["best_metric_name"] = checkpoint["best_metric_key"]
        if self.ema_model is not None:
            checkpoint["train_model"] = raw_model.state_dict()
            checkpoint["ema"] = self.ema_model.ema.state_dict()
            checkpoint["ema_updates"] = self.ema_model.updates
        validate_checkpoint_metadata(checkpoint, strict=True)

        weights_dir = self.save_dir / "weights"
        weights_dir.mkdir(exist_ok=True)

        latest_path = weights_dir / "last.pt"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = weights_dir / "best.pt"
            torch.save(checkpoint, best_path)
            metric_key = checkpoint["best_metric_key"]
            metric_value = self.best_mAP50_95
            logger.info(
                f"New best model saved - Epoch {epoch + 1}: "
                f"{metric_key}={metric_value:.4f}"
            )

        if (epoch + 1) % self.config.save_period == 0:
            epoch_path = weights_dir / f"epoch_{epoch + 1}.pt"
            torch.save(checkpoint, epoch_path)

        logger.info(f"Checkpoint saved: {latest_path}")

    def _checkpoint_extra_metadata(self) -> Dict[str, Any]:
        return {}

    def resume(self, checkpoint_path: str):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        logger.info(f"Resuming from {checkpoint_path}")
        checkpoint = load_trusted_torch_file(
            checkpoint_path,
            map_location=self.device,
            context="training resume checkpoint",
        )
        metadata_errors = validate_checkpoint_metadata(checkpoint, strict=False)
        if metadata_errors:
            logger.warning(
                "Resume checkpoint %s predates LibreYOLO checkpoint metadata v%s "
                "or is incomplete: %s. Training will resume through compatibility "
                "mode; the next saved checkpoint will be written with v%s metadata.",
                checkpoint_path,
                SCHEMA_VERSION,
                "; ".join(metadata_errors),
                SCHEMA_VERSION,
            )

        try:
            model_state = checkpoint.get("train_model", checkpoint["model"])
            # Checkpoint state dict is always unwrapped — feed it to the
            # unwrapped module so the DDP "module." prefix doesn't trip us.
            unwrap_model(self.model).load_state_dict(model_state)
        except Exception as e:
            raise RuntimeError(f"Cannot resume: model architecture mismatch - {e}")

        self.start_epoch = checkpoint["epoch"] + 1

        if "optimizer" in checkpoint:
            if self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                    logger.info("Optimizer state restored")
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")
            else:
                # setup() hasn't run yet — defer until the optimizer exists.
                self._resume_optimizer_state = checkpoint["optimizer"]
                logger.info("Optimizer state deferred until after setup()")

        if "best_metric_value" in checkpoint or "best_mAP50_95" in checkpoint:
            checkpoint_metric_key = checkpoint.get("best_metric_key", "metrics/mAP50-95")
            current_metric_key = getattr(self, "best_metric_key", "metrics/mAP50-95")
            if checkpoint_metric_key != current_metric_key:
                logger.warning(
                    "Checkpoint best metric key %s differs from current key %s. "
                    "Resetting best metric tracking for this run.",
                    checkpoint_metric_key,
                    current_metric_key,
                )
                self.best_mAP50_95 = 0.0
                self.best_mAP50 = 0.0
                self.best_epoch = 0
            else:
                self.best_mAP50_95 = checkpoint.get(
                    "best_metric_value",
                    checkpoint.get("best_mAP50_95", 0.0),
                )
                self.best_mAP50 = checkpoint.get("best_mAP50", 0.0)
                self.best_epoch = checkpoint.get("best_epoch", 0)
                logger.info(
                    f"Restored best metrics: mAP50={self.best_mAP50:.4f}, "
                    f"mAP50-95={self.best_mAP50_95:.4f} (epoch {self.best_epoch})"
                )
        elif "loss" in checkpoint:
            logger.warning(
                "Old checkpoint format detected (loss-based). Converting to mAP tracking."
            )
            self.best_mAP50_95 = 0.0
            self.best_mAP50 = 0.0
            self.best_epoch = 0

        if self.ema_model and "ema_updates" in checkpoint:
            if "ema" in checkpoint:
                try:
                    self.ema_model.ema.load_state_dict(checkpoint["ema"])
                    logger.info("EMA weights restored")
                except Exception as e:
                    logger.warning(f"Could not load EMA weights: {e}")
            self.ema_model.updates = checkpoint["ema_updates"]
            logger.info(f"EMA updates restored: {self.ema_model.updates}")

        self.patience_counter = 0
        logger.info(
            f"Resumed from epoch {self.start_epoch} "
            f"(will train to epoch {self.config.epochs})"
        )

        # setup() may have already run (immediate path: setup → resume). Re-sync
        # the LR now that start_epoch is known — _initialize_scheduler_lr() is
        # idempotent and fast-forwards to the correct schedule position.
        if self.lr_scheduler is not None and self.train_loader is not None:
            self._initialize_scheduler_lr()
