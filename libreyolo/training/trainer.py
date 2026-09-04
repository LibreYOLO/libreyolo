"""Base trainer for LibreYOLO models.

Model-specific trainers subclass BaseTrainer and override hooks.
"""

import contextlib
import inspect
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

from libreyolo.utils.amp import (
    amp_uses_grad_scaler,
    normalize_amp_dtype,
    torch_amp_dtype,
)

from .artifacts import TrainingArtifactsCallback, TrainingStatusCallback
from .callbacks import (
    TrainCallbackList,
    TrainCallbacks,
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from .config import TrainConfig
from .loggers import resolve_loggers
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
from .optim import build_optimizer, restore_optimizer_state
from .freezing import FreezeGroup, apply_freeze, default_freeze_groups
from .qat_defaults import apply_qat_training_guards
from ..data.dataset import YOLODataset, COCODataset, create_dataloader
from ..data import (
    get_coco_annotation_file,
    get_coco_image_dir,
    get_img_files,
    img2label_paths,
    load_data_config,
    resolve_default_coco_image_dir,
)
from ..utils.image_size import imgsz_to_hw
from ..utils.serialization import (
    SCHEMA_VERSION,
    build_class_names,
    load_trusted_torch_file,
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)


logger = logging.getLogger(__name__)

# Families whose architecture supports rectangular (non-square) training input.
# CNN-based detection families (YOLO variants, RTMDet, PicoDet) have no fixed
# positional embeddings and adapt to any (H, W) divisible by their stride.
# Transformer-based families (DETR variants, RF-DETR, D-FINE, etc.) use
# positional embeddings tied to a square grid and must not receive rectangular
# input. YOLO-NAS is excluded: its preprocessing resizes the longest side to a
# fixed target, which cannot fill a rectangular canvas.
# Values are the family's maximum feature stride; both imgsz dimensions must be
# divisible by it.
RECTANGULAR_TRAINING_FAMILIES = {
    "yolo9": 32,
    "yolo9_e2e": 32,
    "yolo9_p2": 32,
    "yolox": 32,
    "yolo7": 32,
    "rtmdet": 32,
    "picodet": 64,
    # PP-LiteSeg is natively rectangular: its released recipes train at
    # 512x1024 (the 50 sizes) and validate at 512x1024 / 768x1536. A square
    # canvas would not be the model the checkpoints were trained as.
    "ppliteseg": 32,
    # U-Net S5-D16 is natively rectangular (512x1024 Cityscapes canvas);
    # encoder stride product is 16, not 32.
    "unet": 16,
}
# Tasks each family may train on rectangularly. Detection is the historical
# case and stays the default; a family whose rectangular support is not
# detect-shaped declares its own tasks here.
RECTANGULAR_TRAINING_TASKS = {
    "ppliteseg": frozenset({"semantic"}),
    "unet": frozenset({"semantic"}),
}
_DEFAULT_RECTANGULAR_TRAINING_TASKS = frozenset({"detect"})


def ensure_mutation_reaches_workers(loader, target, hook: str) -> None:
    """Fail loudly when a main-process mutation of ``target`` cannot reach workers.

    ``close_mosaic`` / ``set_epoch`` style hooks mutate the dataset (or
    collate) object owned by the main process. Dataloader workers hold their
    own copies: non-persistent workers are respawned every epoch and inherit
    the mutated state, but persistent workers live for the whole run and
    never re-copy it, so the mutation silently never applies and epoch-gated
    augmentation scheduling (no-aug tail, stop_epoch gating) becomes a no-op.

    ``target`` is the object about to be mutated (usually
    ``loader.dataset``, or ``loader.collate_fn`` for collate-side epoch
    gating). No-op when the loader has no workers, when workers are not
    persistent, or when ``target`` does not implement ``hook``.
    """
    if loader is None or target is None:
        return
    if not callable(getattr(target, hook, None)):
        return
    num_workers = int(getattr(loader, "num_workers", 0) or 0)
    if num_workers <= 0:
        return
    if not getattr(loader, "persistent_workers", False):
        return
    raise RuntimeError(
        f"{type(target).__name__}.{hook}() would have no effect: the train "
        f"dataloader runs num_workers={num_workers} with "
        "persistent_workers=True, and persistent workers never observe "
        "main-process dataset mutations, so augmentation scheduling would "
        "silently stop working. Rebuild the dataloader after the mutation "
        "or train with persistent_workers=False."
    )


class BaseTrainer(ABC):
    """Base trainer for all LibreYOLO model families.

    Subclasses override hook methods to customise transforms, schedulers,
    loss extraction, and family-specific behaviour.
    """

    best_metric_key: str = "metrics/mAP50-95"
    artifact_model_families: Tuple[str, ...] = ()
    # Whether this family supports ``lora=True`` fine-tuning. Overridden to True
    # by trainers with LoRA-amenable (transformer/nn.Linear) backbones.
    supports_lora: bool = False

    def __init__(
        self,
        model: nn.Module,
        wrapper_model: Optional[Any] = None,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ):
        self.config = self._config_class().from_kwargs(**kwargs)
        if self.config.single_cls and self.config.class_balanced and is_main_process():
            logger.warning(
                "class_balanced=True has no effect with single_cls=True; all labels "
                "belong to the same class, so sampling remains uniform."
            )
        self.config.amp_dtype = normalize_amp_dtype(
            getattr(self.config, "amp_dtype", "float16")
        )
        self.config.max_det = int(getattr(self.config, "max_det", 300))
        if self.config.max_det < 1:
            raise ValueError(f"max_det must be >= 1, got {self.config.max_det}")
        self.config.eval_max_det = getattr(self.config, "eval_max_det", None)
        if self.config.eval_max_det is not None:
            self.config.eval_max_det = int(self.config.eval_max_det)
            if self.config.eval_max_det < 1:
                raise ValueError(
                    f"eval_max_det must be >= 1, got {self.config.eval_max_det}"
                )
        self.model = model
        self.wrapper_model = wrapper_model
        self.callbacks = TrainCallbackList(callbacks)
        for logger_callback in resolve_loggers(loggers):
            self.callbacks.append(logger_callback)
        # TrainingArtifactsCallback is family-gated (results.csv / summary.json
        # only for opted-in families). TrainingStatusCallback is universal: every
        # run gets a live status.json + train.log so agents and the `libreyolo
        # monitor` web UI can watch any training without extra configuration.
        self.artifact_callbacks = TrainCallbackList(
            [
                TrainingArtifactsCallback(enabled_families=self.artifact_model_families),
                TrainingStatusCallback(),
            ]
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
        self.freeze_summary = None
        self._frozen_bn_modules: Tuple[nn.Module, ...] = ()
        self.train_loader = None
        self._is_setup = False
        self.distiller = None
        self._distill_loss_val = 0.0

        # Profiling (opt-in via config.profile). None = disabled, zero overhead.
        self._profiler = None
        self._stop_training = False
        self._weight_averager = None

        # CUDA graph capture of the training network (opt-in via
        # config.cuda_graph). None = disabled, zero overhead. The family
        # spec is resolved lazily on the first training batch so capture
        # sees the fully-constructed model and criterion.
        self._cuda_graph_manager = None
        self._cuda_graph_spec = None
        self._cuda_graph_spec_resolved = False

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

        Derived from ``config.nbs`` (nominal batch size) as
        ``round(nbs / batch)``. When ``nbs`` is unset the
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
        return imgsz_to_hw(self.config.imgsz, name="imgsz")

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

    def validate_validation_loss_config(self) -> None:
        """Fail early when a family does not implement ``val_loss=True``."""
        if getattr(self.config, "val_loss", False):
            raise ValueError(
                f"val_loss=True is not supported by {self.get_model_family()} training"
            )

    def build_validation_loss_adapter(self, model: nn.Module):
        """Build the family adapter used by rank-0 training validation."""
        raise NotImplementedError

    def _validation_loss_kwargs(self, eval_pytorch_model: nn.Module) -> Dict[str, Any]:
        """Build the validator's ``loss_adapter`` kwarg when ``val_loss`` is on.

        No task check here: ``validate_validation_loss_config`` runs for every
        family at setup, so a run that reaches this point has already been
        cleared by its own family's gate. A failure to construct the adapter
        costs the run its validation loss, never its accuracy metrics.
        """
        if not getattr(self.config, "val_loss", False):
            return {}
        try:
            return {
                "loss_adapter": self.build_validation_loss_adapter(eval_pytorch_model)
            }
        except Exception as exc:
            logger.warning(
                "Validation loss could not be initialized; %s metrics will "
                "continue without it: %s",
                getattr(getattr(self, "wrapper_model", None), "task", "detect"),
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return {}

    def get_freeze_groups(self) -> List[FreezeGroup]:
        """Return integer-addressable freeze groups for this family."""
        return default_freeze_groups(self.model)

    def preserve_freeze_param(self, name: str, param: nn.Parameter) -> bool:
        """Return True for trainable params that freezing must not disable."""
        return False

    def on_num_classes_resolved(self):
        """Called before on_setup() for trainers that pre-sync class counts."""

    def on_mosaic_disable(self):
        """Called when mosaic is disabled for final no-aug epochs."""
        dataset = getattr(self.train_loader, "dataset", None)
        if hasattr(dataset, "close_mosaic"):
            ensure_mutation_reaches_workers(self.train_loader, dataset, "close_mosaic")
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

    def cuda_graph_train_spec(self):
        """Family hook: describe how to capture this family's training step.

        Return a :class:`~libreyolo.training.cuda_graph.CudaGraphTrainSpec`
        splitting the step into a static-shaped network (captured) and an
        eager loss (``assemble``), or ``None`` when the family, task or
        current configuration does not support capture. Called once, on the
        first training batch, only when ``config.cuda_graph`` is enabled.
        """
        return None

    def invalidate_cuda_graph(self, reason: str) -> None:
        """Drop any captured training graph so a later batch re-captures.

        Call this from a family hook that changes what the captured region
        computes mid-run (YOLOX enabling its L1 branch at mosaic close is
        the canonical case). A no-op when capture is off.
        """
        manager = getattr(self, "_cuda_graph_manager", None)
        if manager is not None:
            manager.invalidate(reason)

    def _forward_train(
        self,
        imgs: torch.Tensor,
        targets: torch.Tensor,
        polygons: Optional[List] = None,
    ) -> Dict:
        """``on_forward`` with optional CUDA-graph capture of the network.

        Routing keeps a hard *dispatch* contract: any batch that cannot go
        through a captured graph (no family spec, shape mismatch, capture
        disabled) takes the exact ``on_forward`` path instead. So the flag
        never changes which code computes the loss.

        It is not a bit-identical contract for every family. Most reproduce
        eager exactly; three documented cases do not, for reasons inherent
        to the family rather than to capture. See the exception list in
        ``libreyolo.training.cuda_graph`` and the per-family table in
        ``docs/training_cuda_graphs.md``.
        """
        # getattr defaults keep partially-constructed trainers (test
        # doubles, exotic subclasses skipping BaseTrainer.__init__) on the
        # plain eager path.
        manager = getattr(self, "_cuda_graph_manager", None)
        if manager is not None and not manager.disabled:
            if not getattr(self, "_cuda_graph_spec_resolved", False):
                self._cuda_graph_spec_resolved = True
                try:
                    self._cuda_graph_spec = self.cuda_graph_train_spec()
                except Exception as exc:
                    self._cuda_graph_spec = None
                    manager.disabled = True
                    logger.warning(
                        "cuda_graph=True ignored (family spec failed: %r); "
                        "training runs eager.",
                        exc,
                    )
                if self._cuda_graph_spec is None and not manager.disabled:
                    manager.disabled = True
                    logger.warning(
                        "cuda_graph=True ignored (%s does not support "
                        "training capture for this task); training runs "
                        "eager.",
                        type(self).__name__,
                    )
            spec = getattr(self, "_cuda_graph_spec", None)
            if spec is not None:
                flat = manager.run(spec, imgs)
                if flat is not None:
                    return spec.assemble(flat, imgs, targets, polygons)
        return self.on_forward(imgs, targets, polygons=polygons)

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

        # Normalise single-element list/tuple to its int. Multi-element forms
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
        captured_ids: set = set()

        def add_if_trainable(group: list, param: Optional[nn.Parameter]) -> None:
            if isinstance(param, nn.Parameter) and param.requires_grad:
                group.append(param)
                captured_ids.add(id(param))

        # Catch every batch-norm flavour, including SyncBN: BatchNorm{1,2,3}d
        # and SyncBatchNorm are all siblings under ``_BatchNorm``. The naive
        # ``isinstance(v, nn.BatchNorm2d)`` check would silently put SyncBN
        # weights into the weight-decay group post sync_bn conversion.
        bn_types = nn.modules.batchnorm._BatchNorm
        for _k, v in self.model.named_modules():
            if hasattr(v, "bias"):
                add_if_trainable(pg2, v.bias)
            if isinstance(v, bn_types):
                add_if_trainable(pg0, v.weight)
            elif hasattr(v, "weight"):
                add_if_trainable(pg1, v.weight)

        # Bare nn.Parameters (LayerScale gamma, NAFNet beta/gamma, YOLO-NAS
        # alpha, ...) are not exposed as a module ``.weight``/``.bias``
        # attribute, so the named_modules() sweep above never captures them and
        # they would silently never join a param group (no gradient updates).
        # Add any still-uncaptured trainable parameter to the no-weight-decay
        # group (pg2), matching how biases/norms are handled.
        for _pk, p in self.model.named_parameters():
            if p.requires_grad and id(p) not in captured_ids:
                pg2.append(p)
                captured_ids.add(id(p))

        lr = self.effective_lr
        opt_name = self.config.optimizer
        # BN and bias groups carry an explicit weight_decay=0.0: groups without
        # the key inherit the optimizer's default, which is 0.01 for AdamW --
        # silently decaying norm gammas and biases that every upstream recipe
        # (paramwise norm/bias_decay_mult=0) exempts. No-op for SGD/Adam.
        param_groups = []
        if pg0:
            param_groups.append({"params": pg0, "lr": lr, "weight_decay": 0.0})
        if pg1:
            param_groups.append(
                {"params": pg1, "lr": lr, "weight_decay": self.config.weight_decay}
            )
        if pg2:
            param_groups.append({"params": pg2, "lr": lr, "weight_decay": 0.0})
        if not param_groups:
            raise ValueError(
                "No trainable parameters remain after layer freezing; "
                "reduce the freeze value or choose a narrower selector."
            )

        if opt_name == "sgd":
            optimizer = build_optimizer(
                torch.optim.SGD,
                param_groups,
                lr=lr,
                momentum=self.config.momentum,
                nesterov=self.config.nesterov,
            )
        elif opt_name == "adam":
            optimizer = build_optimizer(torch.optim.Adam, param_groups, lr=lr)
        elif opt_name == "adamw":
            optimizer = build_optimizer(torch.optim.AdamW, param_groups, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        if is_main_process():
            logger.info(f"Optimizer: {opt_name}")
            logger.info(f"  - pg0 (BN): {len(pg0)} params")
            logger.info(f"  - pg1 (Conv, wd={self.config.weight_decay}): {len(pg1)} params")
            logger.info(f"  - pg2 (Bias): {len(pg2)} params")
        return optimizer

    def _apply_freeze_config(self) -> None:
        summary = apply_freeze(
            self.model,
            getattr(self.config, "freeze", None),
            freeze_groups=self.get_freeze_groups(),
            preserve_trainable_param=self.preserve_freeze_param,
        )
        self.freeze_summary = summary
        self._frozen_bn_modules = summary.frozen_bn_modules if summary else ()
        if summary is not None and is_main_process():
            logger.info(
                "Layer freezing: selectors=%s, tensors=%d, params=%d, trainable=%d/%d",
                list(summary.selectors),
                summary.frozen_tensor_count,
                summary.frozen_param_count,
                summary.trainable_param_count,
                summary.total_param_count,
            )

    def _enforce_frozen_bn_eval(self) -> None:
        for module in getattr(self, "_frozen_bn_modules", ()):
            module.eval()

    def _setup_distillation(self):
        """Set up knowledge distillation when ``config.distill_model`` is set."""
        if not getattr(self.config, "distill_model", None):
            return

        from ..distillation import Distiller
        from ..distillation.teachers import is_foundation_teacher

        if self.wrapper_model is None:
            raise ValueError(
                "distill_model requires the trainer to be constructed with "
                "wrapper_model set (the student wrapper provides tap points)."
            )

        if is_foundation_teacher(self.config.distill_model):
            # Foundation teacher (e.g. DINOv2): a frozen semantic ViT supervises
            # a single student backbone stage via feature-MSE. Features come
            # through the teacher's extract_features(), not forward hooks, and
            # the loss handles the teacher/student spatial-grid mismatch.
            from ..distillation.teachers import DINOv2Teacher

            logger.info(f"Loading foundation teacher: {self.config.distill_model}")
            teacher = DINOv2Teacher(self.config.distill_model).to(self.device)

            if not hasattr(self.wrapper_model, "get_backbone_distill_config"):
                family = getattr(self.wrapper_model, "FAMILY", type(self.wrapper_model).__name__)
                raise NotImplementedError(
                    f"Foundation-model distillation into the '{family}' family is "
                    f"not supported yet (no get_backbone_distill_config())."
                )

            self.distiller = Distiller(
                teacher_model=teacher,
                student_model=self.model,
                teacher_config=teacher.get_distill_config(),
                student_config=self.wrapper_model.get_backbone_distill_config(),
                loss_type="feat_mse",
                loss_weight=self.config.dis,
                teacher_feature_fn=teacher.extract_features,
                normalize=getattr(self.config, "distill_normalize", False),
            )
        else:
            from ..models import LibreYOLO

            # Load teacher via the factory (handles family detection, weight loading)
            logger.info(f"Loading teacher model: {self.config.distill_model}")
            teacher_wrapper = LibreYOLO(self.config.distill_model)
            teacher_nn = teacher_wrapper.model.to(self.device)

            # Get distillation configs from the models themselves. Families that
            # don't support distillation raise NotImplementedError here with a
            # message naming the family.
            teacher_cfg = teacher_wrapper.get_distill_config()
            student_cfg = self.wrapper_model.get_distill_config()

            self.distiller = Distiller(
                teacher_model=teacher_nn,
                student_model=self.model,
                teacher_config=teacher_cfg,
                student_config=student_cfg,
                loss_type=self.config.distill_loss_type,
                loss_weight=self.config.dis,
                mask_ratio=self.config.distill_mask_ratio,
                tau=self.config.distill_tau,
            )
        self.distiller.to(self.device)

        # resume() may run before setup() — apply deferred adapter state now.
        deferred_state = getattr(self, "_resume_distiller_state", None)
        if deferred_state is not None:
            try:
                self.distiller.loss_modules.load_state_dict(deferred_state)
                logger.info("Distiller adapter state restored from resume checkpoint")
            except Exception as e:
                logger.warning(f"Could not load deferred distiller state: {e}")
            finally:
                self._resume_distiller_state = None

        # Add distiller's learnable params (align/generation convs) to optimizer
        distill_params = list(self.distiller.loss_modules.parameters())
        if distill_params:
            self.optimizer.add_param_group(
                {"params": distill_params, "lr": self.effective_lr}
            )
            logger.info(
                f"Added {len(distill_params)} distillation params to optimizer"
            )

    def _sync_distiller_grads(self):
        """All-reduce distiller adapter/generator gradients across DDP ranks.

        The distiller's loss modules live outside the DDP-wrapped student, so
        DDP's reducer never sees their gradients; without an explicit sync each
        rank would train its own diverging adapters. No-op outside DDP.
        """
        distiller = getattr(self, "distiller", None)
        if distiller is None or not is_distributed():
            return
        import torch.distributed as dist

        world_size = float(get_world_size())
        for param in distiller.loss_modules.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size

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
        wrapper_task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        if wrapper_task == "classify":
            return self._setup_classify_data()
        if wrapper_task == "semantic":
            return self._setup_semantic_data()
        if wrapper_task == "depth":
            return self._setup_depth_data()
        if wrapper_task == "restore":
            return self._setup_restore_data()

        img_size = self.input_size
        preproc, MosaicDatasetClass = self.create_transforms()
        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        load_segments = task == "segment"
        load_obb = task == "obb"

        if self.config.data:
            data_cfg = load_data_config(
                self.config.data,
                allow_scripts=self.config.allow_download_scripts,
                single_cls=self.config.single_cls,
            )
            data_dir = data_cfg["root"]
            data_nc = data_cfg.get("nc")
            if data_nc is None and data_cfg.get("names") is not None:
                data_nc = len(data_cfg["names"])
            self.num_classes = (
                int(data_nc) if data_nc is not None else self.config.num_classes
            )

            ann_file = Path(data_dir) / "annotations" / "instances_train2017.json"
            coco_ann_file = get_coco_annotation_file(data_cfg, "train")

            # Prefer pre-resolved file lists from load_data_config (.txt format)
            img_files = data_cfg.get("train_img_files")
            label_files = data_cfg.get("train_label_files")

            if coco_ann_file:
                default_image_dir = resolve_default_coco_image_dir(
                    data_dir,
                    "train",
                    coco_ann_file,
                )
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file=coco_ann_file,
                    name=get_coco_image_dir(data_cfg, "train", default_image_dir),
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                    load_obb=load_obb,
                    num_classes=self.num_classes,
                    names=data_cfg.get("_original_names", data_cfg.get("names")),
                    single_cls=self.config.single_cls,
                )
            elif img_files:
                train_dataset = YOLODataset(
                    img_files=img_files,
                    label_files=label_files,
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                    load_obb=load_obb,
                    num_classes=self.num_classes,
                    single_cls=self.config.single_cls,
                )
            elif ann_file.exists():
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file="instances_train2017.json",
                    name=resolve_default_coco_image_dir(
                        data_dir,
                        "train",
                        "instances_train2017.json",
                    ),
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                    load_obb=load_obb,
                    num_classes=self.num_classes,
                    names=data_cfg.get("_original_names", data_cfg.get("names")),
                    single_cls=self.config.single_cls,
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
                    load_obb=load_obb,
                    num_classes=self.num_classes,
                    single_cls=self.config.single_cls,
                )
        elif self.config.data_dir:
            data_dir = self.config.data_dir
            self.num_classes = 1 if self.config.single_cls else self.config.num_classes

            if (Path(data_dir) / "annotations").exists():
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file="instances_train2017.json",
                    name=resolve_default_coco_image_dir(
                        data_dir,
                        "train",
                        "instances_train2017.json",
                    ),
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                    load_obb=load_obb,
                    num_classes=self.num_classes,
                    single_cls=self.config.single_cls,
                )
            else:
                train_dataset = YOLODataset(
                    data_dir=data_dir,
                    split="train",
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                    load_obb=load_obb,
                    num_classes=self.num_classes,
                    single_cls=self.config.single_cls,
                )
        else:
            raise ValueError("Either 'data' or 'data_dir' must be specified")

        self.config.num_classes = int(self.num_classes)

        mosaic_enabled = not load_obb
        if load_obb and is_main_process():
            logger.info(
                "Disabling mosaic/mixup for OBB training until corner-aware "
                "OBB augmentation is implemented."
            )

        # Mosaic-gated mixup: for families whose MixUp only fires on mosaic
        # samples, mixup_prob > 0 with mosaic_prob == 0 silently does nothing.
        # Say so instead (spec-driven; see libreyolo/data/augment/spec.py).
        if mosaic_enabled and is_main_process():
            from ..data.augment.spec import mixup_gating_warning

            gating_msg = mixup_gating_warning(
                self.get_model_family(),
                self.config.mosaic_prob,
                self.config.mixup_prob,
            )
            if gating_msg:
                logger.warning(gating_msg)

        train_dataset.enable_image_cache(getattr(self.config, "cache", False))

        dataset_kwargs = dict(
            dataset=train_dataset,
            img_size=img_size,
            mosaic=mosaic_enabled,
            preproc=preproc,
            degrees=self.config.degrees,
            translate=self.config.translate,
            mosaic_scale=self.config.mosaic_scale,
            mixup_scale=self.config.mixup_scale,
            shear=self.config.shear,
            perspective=getattr(self.config, "perspective", 0.0),
            enable_mixup=mosaic_enabled and self.config.mixup_prob > 0,
            mosaic_prob=self.config.mosaic_prob if mosaic_enabled else 0.0,
            mixup_prob=self.config.mixup_prob if mosaic_enabled else 0.0,
        )
        # Copy-paste is only wired for the mosaic datasets whose constructor
        # accepts it (segmentation-capable pipelines); pass it through only
        # there so the shared instantiation stays valid for every family. OBB
        # has no segments, so leave it off in that case.
        cp_prob = float(getattr(self.config, "copy_paste", 0.0) or 0.0)
        if "copy_paste" in inspect.signature(MosaicDatasetClass).parameters:
            dataset_kwargs["copy_paste"] = 0.0 if load_obb else cp_prob
            dataset_kwargs["copy_paste_mode"] = getattr(
                self.config, "copy_paste_mode", "flip"
            )
        train_dataset = MosaicDatasetClass(**dataset_kwargs)

        # ``batch`` is the global batch under DDP. Each rank's loader is built
        # with ``batch // world_size``.
        per_rank_batch = max(1, self.config.batch // max(self.world_size, 1))
        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(train_dataset) >= self.world_size,
            )

        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=per_rank_batch,
            num_workers=self.config.workers,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
            sampler=sampler,
            min_samples=int(getattr(self.config, "min_samples", 0) or 0),
            class_balanced=bool(getattr(self.config, "class_balanced", False)),
        )

        if is_main_process():
            logger.info(f"Training dataset: {len(train_dataset)} images")
            _hint_task = getattr(
                getattr(self, "wrapper_model", None), "task", "detect"
            )
            if _hint_task == "detect" and self.config.data:
                logger.info(
                    f"Tip: sanity-check your dataset for common issues first with "
                    f"`libreyolo doctor {self.config.data}`"
                )
            logger.info(
                f"Iterations per epoch: {len(self.train_loader)} "
                f"(batch_per_rank={per_rank_batch}, world_size={self.world_size})"
            )
        return train_dataset

    def _setup_classify_data(self):
        """Build the classification train dataloader from an ImageFolder root.

        Classification bypasses the detection mosaic/letterbox pipeline: ``data``
        is a dataset root (or known name) with ``train``/``val`` splits, each a
        folder-per-class. The class count is the source of truth here, so the
        wrapper head is (re)built to match before the optimizer is created.
        """
        from torch.utils.data import DataLoader

        from ..data.classify_dataset import (
            ClassifyDataset,
            build_classify_collate,
            get_class_names,
            resolve_classify_data,
        )

        dataset_root = resolve_classify_data(self.config.data)
        classes = get_class_names(dataset_root, split="train")
        num_classes = len(classes)
        self.num_classes = num_classes
        self.config.num_classes = num_classes
        class_to_idx = {name: i for i, name in enumerate(classes)}

        wrapper = self.wrapper_model
        if wrapper is not None:
            if (
                getattr(wrapper, "nb_classes", None) != num_classes
                and hasattr(wrapper, "_rebuild_for_new_classes")
            ):
                wrapper._rebuild_for_new_classes(num_classes)
                self.model = wrapper.model.to(self.device)
            wrapper.nb_classes = num_classes
            wrapper.names = {i: name for i, name in enumerate(classes)}

        imgsz = self.config.imgsz
        train_dataset = ClassifyDataset(
            dataset_root=dataset_root,
            split="train",
            imgsz=imgsz,
            augment=True,
            class_to_idx=class_to_idx,
            transform_kwargs={
                "crop_pct": getattr(wrapper, "crop_pct", 0.875),
                "interpolation": getattr(wrapper, "interpolation", "bilinear"),
                "auto_augment": getattr(self.config, "auto_augment", None),
                "erasing": getattr(self.config, "erasing", 0.0),
            },
        )

        # Batch-level MixUp / CutMix (soft labels) when requested; otherwise this
        # returns the plain classify collate so default training is unchanged.
        collate_fn = build_classify_collate(
            num_classes,
            mixup=getattr(self.config, "mixup", 0.0),
            cutmix=getattr(self.config, "cutmix", 0.0),
        )

        per_rank_batch = max(1, self.config.batch // max(self.world_size, 1))
        if per_rank_batch < 2:
            raise ValueError(
                "Classification training needs an effective per-rank batch size >= 2 "
                f"(got {per_rank_batch} from batch={self.config.batch}, "
                f"world_size={self.world_size}). A batch of 1 breaks the BatchNorm in "
                "the pooled classifier head (e.g. MobileNetV4/EfficientNetV2 norm_head). "
                "Increase batch (or reduce world_size)."
            )
        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(train_dataset) >= self.world_size,
            )

        # Under DDP each rank only sees ``len(sampler)`` samples, so base the
        # drop_last decision on the per-rank visible count. Otherwise a small
        # dataset split across ranks could drop every rank's only partial
        # batch and leave zero iterations.
        try:
            visible_samples = len(sampler) if sampler is not None else len(train_dataset)
        except TypeError:
            visible_samples = len(train_dataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_batch,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.config.workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_fn,
            drop_last=visible_samples >= per_rank_batch,
        )

        if is_main_process():
            logger.info(
                "Classification dataset: %d images, %d classes",
                len(train_dataset),
                num_classes,
            )
            logger.info(
                "Iterations per epoch: %d (batch_per_rank=%d, world_size=%d)",
                len(self.train_loader),
                per_rank_batch,
                self.world_size,
            )
        return train_dataset

    def _setup_semantic_data(self):
        """Build the semantic-segmentation train dataloader from a dataset YAML.

        Dense masks bypass the detection mosaic/letterbox pipeline. The
        dataset's class space (including the background class appended for
        polygon-derived masks) is the source of truth, so the wrapper head is
        (re)built to match before the optimizer is created.
        """
        from torch.utils.data import DataLoader

        from ..data.semantic_dataset import (
            SemanticDataset,
            resolve_semantic_data,
            semantic_collate_fn,
        )

        if not self.config.data:
            raise ValueError("Semantic training requires data= (a dataset YAML).")
        data_config = resolve_semantic_data(
            self.config.data,
            allow_scripts=self.config.allow_download_scripts,
        )
        resize_mode = getattr(self.wrapper_model, "semantic_resize_mode", "letterbox")
        divisor = getattr(self.wrapper_model, "semantic_imgsz_divisor", None)
        if divisor:
            sides = (
                self.config.imgsz
                if isinstance(self.config.imgsz, (tuple, list))
                else (self.config.imgsz,)
            )
            if any(int(side) % int(divisor) for side in sides):
                raise ValueError(
                    f"Semantic training imgsz={self.config.imgsz} must be divisible "
                    f"by {int(divisor)} for this model family."
                )
        # Family-scoped scale-jitter range. Families that do not define this
        # attribute (default None) keep the SemanticDataset default jitter,
        # unchanged. Input standardization is family-internal (applied in the
        # model's forward on the raw [0, 1] tensor), so the dataset stays
        # /255-only for every family.
        scale_jitter = getattr(self.wrapper_model, "semantic_scale_jitter", None)
        semantic_kwargs = {}
        if scale_jitter is not None:
            semantic_kwargs["scale_jitter"] = tuple(scale_jitter)
        # Same deal for photometric jitter: opt in per family, or keep the
        # SemanticDataset default. Note this deliberately does NOT read
        # config.hsv_prob -- SemanticDataset has always used its own default for
        # every semantic family, so honoring the config here would silently
        # change RF-DETR's and DINOv2's training too.
        hsv_prob = getattr(self.wrapper_model, "semantic_hsv_prob", None)
        if hsv_prob is not None:
            semantic_kwargs["hsv_prob"] = float(hsv_prob)
        # A family whose reference recipe uses a different photometric
        # transform than the shared HSV-gain jitter supplies it here; families
        # that do not define the attribute keep the shared path unchanged.
        photometric = getattr(self.wrapper_model, "semantic_photometric", None)
        if photometric is not None:
            semantic_kwargs["photometric"] = photometric
        train_dataset = SemanticDataset(
            data_config,
            split="train",
            imgsz=self.config.imgsz,
            augment=True,
            resize_mode=resize_mode,
            **semantic_kwargs,
        )

        num_classes = train_dataset.nc
        self.num_classes = num_classes
        self.config.num_classes = num_classes

        wrapper = self.wrapper_model
        if wrapper is not None:
            if (
                getattr(wrapper, "nb_classes", None) != num_classes
                and hasattr(wrapper, "_rebuild_for_new_classes")
            ):
                wrapper._rebuild_for_new_classes(num_classes)
                self.model = wrapper.model.to(self.device)
            wrapper.nb_classes = num_classes
            wrapper.names = dict(train_dataset.names)

        per_rank_batch = max(1, self.config.batch // max(self.world_size, 1))
        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(train_dataset) >= self.world_size,
            )

        try:
            visible_samples = len(sampler) if sampler is not None else len(train_dataset)
        except TypeError:
            visible_samples = len(train_dataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_batch,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.config.workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=semantic_collate_fn,
            drop_last=visible_samples >= per_rank_batch,
        )

        if is_main_process():
            logger.info(
                "Semantic dataset: %d images, %d classes",
                len(train_dataset),
                num_classes,
            )
            logger.info(
                "Iterations per epoch: %d (batch_per_rank=%d, world_size=%d)",
                len(self.train_loader),
                per_rank_batch,
                self.world_size,
            )
        return train_dataset

    def _setup_depth_data(self):
        """Build the depth train dataloader from a dataset YAML."""
        from torch.utils.data import DataLoader

        from ..data.depth_dataset import (
            DepthDataset,
            depth_collate_fn,
            resolve_depth_data,
        )

        if not self.config.data:
            raise ValueError("Depth training requires data= (a dataset YAML).")
        data_config = resolve_depth_data(
            self.config.data,
            allow_scripts=self.config.allow_download_scripts,
        )
        resize_mode = getattr(self.wrapper_model, "depth_resize_mode", "letterbox")
        divisor = getattr(self.wrapper_model, "depth_imgsz_divisor", None)
        if divisor and self.config.imgsz % int(divisor):
            raise ValueError(
                f"Depth training imgsz={self.config.imgsz} must be divisible "
                f"by {int(divisor)} for this model family."
            )
        train_dataset = DepthDataset(
            data_config,
            split="train",
            imgsz=self.config.imgsz,
            augment=True,
            resize_mode=resize_mode,
        )

        self.num_classes = 1
        self.config.num_classes = 1
        if self.wrapper_model is not None:
            self.wrapper_model.nb_classes = 1
            self.wrapper_model.names = {0: "depth"}

        per_rank_batch = max(1, self.config.batch // max(self.world_size, 1))
        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(train_dataset) >= self.world_size,
            )

        try:
            visible_samples = len(sampler) if sampler is not None else len(train_dataset)
        except TypeError:
            visible_samples = len(train_dataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_batch,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.config.workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=depth_collate_fn,
            drop_last=visible_samples >= per_rank_batch,
        )

        if is_main_process():
            logger.info("Depth dataset: %d images", len(train_dataset))
            logger.info(
                "Iterations per epoch: %d (batch_per_rank=%d, world_size=%d)",
                len(self.train_loader),
                per_rank_batch,
                self.world_size,
            )
        return train_dataset

    def _setup_restore_data(self):
        """Build the restoration train dataloader from a paired dataset YAML."""
        from torch.utils.data import DataLoader

        from ..data.restore_dataset import (
            RestoreDataset,
            resolve_restore_data,
            restore_collate_fn,
        )

        if not self.config.data:
            raise ValueError("Restore training requires data= (a dataset YAML).")
        data_config = resolve_restore_data(
            self.config.data,
            allow_scripts=self.config.allow_download_scripts,
        )
        train_dataset = RestoreDataset(
            data_config,
            split="train",
            imgsz=self.config.imgsz,
            augment=True,
        )

        self.num_classes = 1
        self.config.num_classes = 1
        if self.wrapper_model is not None:
            self.wrapper_model.nb_classes = 1
            self.wrapper_model.names = {0: "image"}

        per_rank_batch = max(1, self.config.batch // max(self.world_size, 1))
        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(train_dataset) >= self.world_size,
            )

        try:
            visible_samples = len(sampler) if sampler is not None else len(train_dataset)
        except TypeError:
            visible_samples = len(train_dataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_batch,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.config.workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=restore_collate_fn,
            drop_last=visible_samples >= per_rank_batch,
        )

        if is_main_process():
            logger.info("Restore dataset: %d image pairs", len(train_dataset))
            logger.info(
                "Iterations per epoch: %d (batch_per_rank=%d, world_size=%d)",
                len(self.train_loader),
                per_rank_batch,
                self.world_size,
            )
        return train_dataset

    def _resolve_num_classes_from_data_config(self) -> int:
        """Resolve dataset class count before criterion construction."""
        resolved = 1 if self.config.single_cls else int(self.config.num_classes)
        if self.config.data:
            # Only the YAML's class count is needed here; the dataset itself is
            # downloaded later in _setup_data.
            data_cfg = load_data_config(
                self.config.data,
                autodownload=False,
                allow_scripts=self.config.allow_download_scripts,
                single_cls=self.config.single_cls,
            )
            resolved = int(data_cfg.get("nc", resolved))

        self.num_classes = resolved
        self.config.num_classes = resolved
        return resolved

    def _infer_model_num_classes(self) -> Optional[int]:
        """Best-effort class-count introspection for detector heads."""
        model = unwrap_model(self.model)
        for obj in (getattr(model, "decoder", None), getattr(model, "head", None), model):
            value = getattr(obj, "num_classes", None)
            if value is not None:
                return int(value)
        return None

    def _sync_wrapped_model_num_classes(self, num_classes: int) -> None:
        """Rebuild wrapper-owned heads or fail before criterion/head desync."""
        num_classes = int(num_classes)
        self.num_classes = num_classes
        self.config.num_classes = num_classes

        wrapper = self.wrapper_model
        model_nc = self._infer_model_num_classes()
        wrapper_nc = getattr(wrapper, "nb_classes", None) if wrapper is not None else None
        needs_rebuild = (
            (model_nc is not None and model_nc != num_classes)
            or (wrapper_nc is not None and int(wrapper_nc) != num_classes)
        )

        if not needs_rebuild:
            if wrapper is not None and self.config.single_cls:
                wrapper.names = {0: "object"}
            return

        if wrapper is None or not hasattr(wrapper, "_rebuild_for_new_classes"):
            raise RuntimeError(
                f"{self.get_model_family()} trainer resolved num_classes={num_classes}, "
                f"but the model head exposes num_classes={model_nc}. Pass a "
                "wrapper_model with _rebuild_for_new_classes() or construct the "
                "raw model with the resolved class count."
            )

        wrapper._rebuild_for_new_classes(num_classes)
        self.model = wrapper.model.to(self.device)
        wrapper.device = self.device

        rebuilt_nc = self._infer_model_num_classes()
        if rebuilt_nc is not None and rebuilt_nc != num_classes:
            raise RuntimeError(
                f"{self.get_model_family()} wrapper rebuild did not sync the model "
                f"head to num_classes={num_classes}; got {rebuilt_nc}."
            )
        if self.config.single_cls:
            wrapper.names = {0: "object"}

    # =========================================================================
    # Setup / train / epoch
    # =========================================================================

    def _check_ddp_train_loader(self) -> None:
        """Verify the train loader actually shards across DDP ranks.

        Duck-types on ``num_replicas``/``rank`` rather than requiring
        ``DistributedSampler`` so distributed-aware custom samplers pass.
        """
        loader = getattr(self, "train_loader", None)
        if loader is None:
            return
        sampler = getattr(loader, "sampler", None)
        if (
            getattr(sampler, "num_replicas", None) != self.world_size
            or getattr(sampler, "rank", None) != self.rank
        ):
            raise ValueError(
                f"{type(self).__name__} built a train loader that does not "
                f"shard across DDP ranks (sampler={type(sampler).__name__}): "
                "every rank would train on the full dataset. Build the loader "
                "with a DistributedSampler and batch // world_size per rank, "
                "like BaseTrainer._setup_data."
            )
        per_rank_batch = max(1, self.config.batch // self.world_size)
        loader_batch = getattr(loader, "batch_size", None)
        if loader_batch is not None and loader_batch != per_rank_batch:
            raise ValueError(
                f"{type(self).__name__} built a train loader with "
                f"batch_size={loader_batch}, but batch={self.config.batch} is "
                f"the global batch: world_size={self.world_size} requires "
                f"{per_rank_batch} per rank."
            )

    def setup(self):
        if self._is_setup:
            return

        quant_manifest = getattr(self.wrapper_model, "_quant_manifest", None)
        if quant_manifest and quant_manifest.get("recipe") in ("fp16", "bf16"):
            raise ValueError(
                "Cast-precision quantized models (fp16/bf16) are "
                "inference-only. Train the float model with amp=True "
                "instead, or use a quantizing recipe (int8/fp8/w4a8/nvfp4/"
                "...) for quantization-aware training."
            )
        if quant_manifest and quant_manifest.get("state") == "finalized":
            from ..quant.api import reprepare_model

            reprepare_model(self.wrapper_model)

        if quant_manifest:
            qat_changes = apply_qat_training_guards(self.config)
            if qat_changes and is_main_process():
                logger.warning(
                    "QAT recipe '%s': disabled %s because these features can "
                    "interfere with fake-quant observer and scale state.",
                    quant_manifest.get("recipe", "unknown"),
                    ", ".join(
                        f"{option}={getattr(self.config, option)!r}"
                        for option in qat_changes
                    ),
                )

        if getattr(self.config, "lora", False) and not self.supports_lora:
            family = self.get_model_family() if hasattr(self, "get_model_family") else "this model"
            raise ValueError(
                f"LoRA fine-tuning (lora=True) is not supported for {family}. "
                "LoRA targets transformer components with nn.Linear layers "
                "(e.g. RF-DETR, D-FINE, DEIM)."
            )

        # Validate rectangular imgsz for the selected family and task.
        imgsz = self.config.imgsz
        if isinstance(imgsz, (list, tuple)) and int(imgsz[0]) != int(imgsz[1]):
            family = self.get_model_family() if hasattr(self, "get_model_family") else ""
            if family and family.lower() not in RECTANGULAR_TRAINING_FAMILIES:
                raise ValueError(
                    f"Rectangular imgsz={tuple(imgsz)} is not supported for {family}. "
                    f"Only CNN-based detection families support rectangular training "
                    f"input. Supported families: {sorted(RECTANGULAR_TRAINING_FAMILIES)}."
                )
            task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
            allowed_tasks = RECTANGULAR_TRAINING_TASKS.get(
                family.lower(), _DEFAULT_RECTANGULAR_TRAINING_TASKS
            )
            if task not in allowed_tasks:
                raise ValueError(
                    f"Rectangular imgsz={tuple(imgsz)} is only supported for the "
                    f"{'/'.join(sorted(allowed_tasks))} task for {family}, "
                    f"got task='{task}'."
                )
            stride = RECTANGULAR_TRAINING_FAMILIES.get(family.lower(), 32)
            h, w = int(imgsz[0]), int(imgsz[1])
            if h % stride != 0 or w % stride != 0:
                raise ValueError(
                    f"imgsz=({h}, {w}): both height and width must be divisible by "
                    f"{stride} (the {family} maximum feature stride), got remainders "
                    f"({h % stride}, {w % stride})."
                )

        if is_main_process():
            logger.info("Setting up training...")
        self.model.to(self.device)
        if self.wrapper_model is not None:
            self.wrapper_model.device = self.device

        self.on_num_classes_resolved()

        # SyncBatchNorm conversion: only meaningful under DDP. Single-GPU
        # runs skip this regardless of the flag so single-GPU is unchanged.
        if self.is_distributed and getattr(self.config, "sync_bn", False):
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if is_main_process():
                logger.info("Converted BatchNorm to SyncBatchNorm")

        self.validate_validation_loss_config()
        self.on_setup()

        if getattr(self.config, "batch", 16) == -1:
            from libreyolo.training.autobatch import resolve_auto_batch, _DEFAULT_FRACTION

            self.config.batch = resolve_auto_batch(
                self.model,
                imgsz=self.config.imgsz,
                amp=self.config.amp,
                amp_dtype=self.config.amp_dtype,
                world_size=self.world_size,
                nbs=getattr(self.config, "nbs", None),
                fraction=getattr(self.wrapper_model, "autobatch_fraction", _DEFAULT_FRACTION),
            )
            if is_main_process():
                logger.info("AutoBatch: resolved global batch size = %d", self.config.batch)

        if int(getattr(self.config, "batch", 16)) < 1:
            raise ValueError(
                f"batch={self.config.batch} is invalid: use a positive global "
                "batch size, or -1 for AutoBatch."
            )

        # ``batch`` is the global batch under DDP: every rank trains at
        # batch // world_size, so a non-divisible value would silently train
        # at a different global batch than requested (e.g. batch=6 on 4 GPUs
        # trains at 4). Fail fast instead of silently changing the batch.
        # AutoBatch above already returns world_size-divisible values.
        if self.is_distributed and self.config.batch % self.world_size != 0:
            lower = (self.config.batch // self.world_size) * self.world_size
            higher = lower + self.world_size
            options = (
                f"batch={higher}" if lower == 0 else f"batch={lower} or batch={higher}"
            )
            raise ValueError(
                f"batch={self.config.batch} is the global batch and must be "
                f"divisible by world_size={self.world_size}: each rank trains at "
                f"batch // world_size, so this value would silently train at a "
                f"different global batch than requested. Use {options}."
            )

        # BN statistics quality under DDP: with SyncBatchNorm off, each rank's
        # BatchNorm tracks only its own per-rank shard (batch // world_size).
        # A small per-rank batch produces noisy running stats and degrades the
        # converged model (issue #484). Warn (do not silently change behavior)
        # so users of BatchNorm families can enable sync_bn.
        if self.is_distributed and not getattr(self.config, "sync_bn", False):
            per_rank_batch = max(1, self.config.batch // max(self.world_size, 1))
            has_batchnorm = any(
                isinstance(m, nn.modules.batchnorm._BatchNorm)
                for m in self.model.modules()
            )
            if has_batchnorm and per_rank_batch < 16 and is_main_process():
                logger.warning(
                    "DDP per-rank batch is %d (global batch %d / world_size %d) "
                    "and sync_bn is disabled; BatchNorm running statistics are "
                    "computed per rank on this small shard, which can reduce "
                    "accuracy versus single-GPU. Consider setting sync_bn=True.",
                    per_rank_batch,
                    self.config.batch,
                    self.world_size,
                )

        self._setup_data()
        self._assert_class_balanced_honored()

        # DDP loader invariant: trainers that own their data pipeline must
        # shard like BaseTrainer._setup_data does (issue #484: three trainers
        # built plain DataLoaders and every rank trained the full dataset at
        # the full global batch). Catch that bug class at startup for every
        # current and future family.
        if self.is_distributed:
            self._check_ddp_train_loader()

        self._apply_freeze_config()
        self.optimizer = self._setup_optimizer()
        self._setup_distillation()
        self.lr_scheduler = self.create_scheduler(self._scheduler_steps_per_epoch())

        # resume() may be called before setup() when the optimizer doesn't exist
        # yet. Apply the deferred state now so momentum buffers are restored before
        # _initialize_scheduler_lr() sets the correct LR on top.
        if getattr(self, "_resume_optimizer_state", None) is not None:
            try:
                restore_optimizer_state(self.optimizer, self._resume_optimizer_state)
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
            amp_torch_dtype = torch_amp_dtype(self.config.amp_dtype)
            if (
                amp_torch_dtype == torch.bfloat16
                and torch.cuda.is_available()
                and not torch.cuda.is_bf16_supported()
            ):
                raise RuntimeError(
                    "amp_dtype='bfloat16' requires a CUDA device with BF16 support"
                )
            # FP16 needs loss scaling. BF16's wider exponent range does not,
            # but a disabled scaler keeps the optimizer path shared.
            self.scaler = GradScaler(
                "cuda", enabled=amp_uses_grad_scaler(self.config.amp_dtype)
            )
            if is_main_process():
                logger.info(
                    "Using mixed precision training (AMP, dtype=%s)",
                    self.config.amp_dtype,
                )
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

        average_best = int(getattr(self.config, "average_best", 0) or 0)
        if average_best > 0:
            from .weight_averaging import MetricGatedAverager

            self._weight_averager = MetricGatedAverager(average_best)
            if is_main_process():
                logger.info(
                    "Averaging the %d best checkpoints by the watched metric "
                    "(weights/average.pt at the end of training)",
                    average_best,
                )
            deferred = getattr(self, "_resume_average_pool_path", None)
            if deferred:
                self._restore_average_pool(deferred)
                self._resume_average_pool_path = None

        # Save-dir creation, config dump, and TB writer all live on rank 0.
        # The resolved name (which may include an auto-increment suffix when
        # exist_ok=False and a previous run exists) is broadcast to other
        # ranks so every rank's ``self.save_dir`` agrees and event-level
        # paths emitted by callbacks are consistent.
        if is_main_process():
            self.save_dir = self._get_save_dir()
            self.config.to_yaml(self.save_dir / "train_config.yaml")
            logger.info(f"Saving to: {self.save_dir}")
            logger.info(
                f"Tip: watch this run live in your browser with "
                f"`libreyolo monitor {self.save_dir}`"
            )
        else:
            self.save_dir = Path(self.config.project) / self.config.name

        if self.is_distributed:
            import torch.distributed as _dist

            container = [str(self.save_dir)] if is_main_process() else [None]
            _dist.broadcast_object_list(container, src=0)
            self.save_dir = Path(container[0])

        # Wait for rank 0 to finish dir creation before any rank proceeds.
        barrier()

        # Optional training-step profiler (opt-in via config.profile). Built on
        # the main process; emits the breakdown + Chrome trace into save_dir.
        # Disabled under DDP (rank-0-only syncs, and the profile_then_stop
        # early stop would desync ranks).
        if getattr(self.config, "profile", False):
            if self.is_distributed:
                if is_main_process():
                    logger.warning("profile=True is ignored under distributed training.")
            elif is_main_process():
                from libreyolo.training.profiler import TrainStepProfiler

                profile_warmup = getattr(self.config, "profile_warmup", 5)
                profile_steps = getattr(self.config, "profile_steps", 20)
                accum_steps = self._accum_steps
                if accum_steps > 1:
                    profile_warmup = math.ceil(profile_warmup / accum_steps) * accum_steps
                    profile_steps = math.ceil(profile_steps / accum_steps) * accum_steps
                    logger.info(
                        "profile window rounded to accumulation boundaries "
                        "(warmup=%d, steps=%d, accum=%d)",
                        profile_warmup,
                        profile_steps,
                        accum_steps,
                    )
                self._profiler = TrainStepProfiler(
                    device=self.device,
                    warmup=profile_warmup,
                    active=profile_steps,
                    trace=getattr(self.config, "profile_trace", True),
                    open_report=getattr(self.config, "profile_open", True),
                    save_dir=self.save_dir,
                    logger=logger,
                    meta={
                        "model": self.get_model_tag(),
                        "device": str(self.device),
                        "batch": self.config.batch,
                        "imgsz": self.config.imgsz,
                        "amp": bool(self.config.amp),
                        "amp_dtype": self.config.amp_dtype,
                        "workers": self.config.workers,
                    },
                )

        # CUDA graph training capture (opt-in). Constructed last so capture
        # can only ever see the fully-built model, optimizer and criterion.
        # Unsupported run shapes downgrade to eager with one clear warning
        # instead of failing the run.
        if getattr(self.config, "cuda_graph", False):
            reason = None
            if self.device.type != "cuda":
                reason = "device is not CUDA"
            elif self.is_distributed:
                reason = "distributed training is not supported yet"
            elif self.distiller is not None:
                reason = "distillation runs are not supported"
            if reason is not None:
                if is_main_process():
                    logger.warning(
                        "cuda_graph=True ignored (%s); training runs eager.",
                        reason,
                    )
            else:
                from libreyolo.training.cuda_graph import TrainGraphManager

                self._cuda_graph_manager = TrainGraphManager()

        self._is_setup = True

    def _ddp_find_unused_parameters(self) -> bool:
        """Subclasses override to flip when their forward graph is conditional.

        Default False matches PyTorch's default. rf-detr flips True when a
        segmentation head is present because the sparse branch leaves some
        params un-grad'd on some batches.
        """
        return False

    def _autocast_context(self):
        """Create a CUDA autocast context using the configured dtype.

        With CUDA graph training capture active, autocast's weight-cast
        cache must be disabled: the capture recipe requires cache_enabled
        =False so warm-up, capture and replay all see the same cast ops.
        """
        return autocast(
            "cuda",
            dtype=torch_amp_dtype(getattr(self.config, "amp_dtype", "float16")),
            cache_enabled=getattr(self, "_cuda_graph_manager", None) is None,
        )

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
        # May be stale from a previous profile_then_stop run on this instance;
        # a leftover True would silently truncate this run's first epoch.
        self._stop_training = False
        try:
            self.setup()
            self._maybe_export_check()

            if is_main_process():
                logger.info(f"Starting training for {self.config.epochs} epochs")
                logger.info(f"Model: {self.get_model_tag()}")
                logger.info(f"Batch size: {self.config.batch}")
                logger.info(f"Input size: {self.input_size}")
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

                profile_truncated = bool(getattr(self, "_stop_training", False))
                is_best = (
                    False
                    if profile_truncated
                    else self._update_best_state(epoch, val_metrics)
                )
                # Rank 0 owns validation metrics, so broadcast the patience
                # decision before any rank can enter the next epoch alone.
                epochs_since_best = (
                    (epoch + 1) - self.best_epoch if self.best_epoch else 0
                )
                should_stop = (
                    self.config.patience > 0
                    and self.best_epoch > 0
                    and epochs_since_best >= self.config.patience
                )
                should_stop = self._sync_main_bool(should_stop)
                if should_stop and not profile_truncated:
                    save_final_plots = bool(
                        getattr(self.config, "save_plots", False)
                    ) and not self._is_final_epoch(epoch)
                    if self._maybe_precise_bn(epoch, force=True):
                        refreshed = self._validate_epoch(
                            epoch, save_plots=save_final_plots
                        )
                        if refreshed is not None:
                            val_metrics = refreshed
                            is_best = self._update_best_state(epoch, val_metrics)
                    elif save_final_plots:
                        self._validate_epoch(epoch, save_plots=True)
                    should_stop = self._sync_main_bool(
                        should_stop and not is_best
                    )
                # Write ``last.pt`` every epoch so a crash never costs more than
                # a single epoch. ``best.pt`` (is_best) and periodic
                # ``epoch_N.pt`` (save_period) stay gated inside
                # _save_checkpoint, so those are unaffected. A profile-only run
                # (profile_then_stop) truncated the epoch after the profile
                # window, so no checkpoint is written for it: stamping the
                # partial epoch as complete would make a later resume skip the
                # rest of it.
                if not profile_truncated:
                    self._maybe_offer_average(val_metrics)
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
                # Patience is measured in EPOCHS since the last metric
                # improvement. ``best_epoch`` is the 1-based epoch of the best
                # result, so this stays meaningful even when validation only
                # runs on an interval (unlike counting discrete val events).
                if should_stop:
                    if is_main_process():
                        logger.info(
                            f"Early stopping triggered after {epoch + 1} epochs "
                            f"(patience={self.config.patience}, no improvement for {epochs_since_best} epochs)"
                        )
                    break

                if getattr(self, "_stop_training", False):
                    break

            if getattr(self, "distiller", None) is not None:
                self.distiller.cleanup()

            self._refresh_best_precise_bn_checkpoint()
            self._write_average_checkpoint()
            total_time = time.time() - start_time
            if is_main_process():
                if getattr(self, "_stop_training", False):
                    logger.info(
                        "Profile-only run (profile_then_stop=True): training "
                        f"stopped after the profile window in {total_time:.1f}s; "
                        "the partial epoch was not validated or checkpointed"
                    )
                else:
                    logger.info(
                        f"Training complete in {total_time / 3600:.2f} hours"
                    )

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
        best_checkpoint = weights_dir / "best.pt"
        last_checkpoint = weights_dir / "last.pt"
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
            "best_checkpoint": (
                str(best_checkpoint) if best_checkpoint.exists() else None
            ),
            "last_checkpoint": (
                str(last_checkpoint) if last_checkpoint.exists() else None
            ),
            "average_checkpoint": (
                str(weights_dir / "average.pt")
                if (weights_dir / "average.pt").exists()
                else None
            ),
            "average_metrics": getattr(self, "_average_validation_metrics", None),
        }

    def _event_context(self) -> Dict[str, Any]:
        return {
            "total_epochs": self.config.epochs,
            "model_family": self.get_model_family(),
            "model_size": getattr(self.config, "size", None),
            "task": getattr(getattr(self, "wrapper_model", None), "task", "detect"),
            "save_dir": str(getattr(self, "save_dir", "")),
        }

    def _build_train_start_event(self) -> TrainStartEvent:
        return TrainStartEvent(
            start_epoch=self.start_epoch + 1,
            config=self.config.to_dict(),
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
            task=getattr(getattr(self, "wrapper_model", None), "task", "detect"),
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

    def _sync_main_bool(self, value: bool) -> bool:
        if not self.is_distributed:
            return value
        import torch.distributed as _dist

        flag = torch.tensor(int(value), dtype=torch.int, device=self.device)
        _dist.broadcast(flag, src=0)
        return bool(flag.item())

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

    def _apply_scheduler(self, iters: int) -> float:
        """Step the LR scheduler and optional momentum warmup."""
        lr = self.lr_scheduler.update_lr(iters)
        self._set_optimizer_lr(lr)
        update_momentum = getattr(self.lr_scheduler, "update_momentum", None)
        if callable(update_momentum):
            momentum = update_momentum(iters)
            if momentum is not None:
                for param_group in self.optimizer.param_groups:
                    if "momentum" in param_group:
                        param_group["momentum"] = momentum
        return lr

    def _initialize_scheduler_lr(self) -> None:
        if self.optimizer is None or self.lr_scheduler is None:
            return
        init_iter = getattr(self, "start_epoch", 0) * self._scheduler_steps_per_epoch()
        self._apply_scheduler(init_iter)

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

    def _prof_phase(self, name: str):
        """Profiler phase context manager (no-op when profiling is disabled)."""
        prof = getattr(self, "_profiler", None)
        return prof.phase(name) if prof is not None else contextlib.nullcontext()

    def _train_epoch(
        self, epoch: int
    ) -> Tuple[float, Optional[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
        self.model.train()
        self._enforce_frozen_bn_eval()

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

        # getattr: test doubles may bypass BaseTrainer.__init__, same as _profiler.
        distiller = getattr(self, "distiller", None)
        prof = getattr(self, "_profiler", None)
        loader = prof.wrap_loader(pbar) if prof is not None else pbar
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 5:
                imgs, targets, img_infos, img_ids, polygons = batch
            else:
                imgs, targets, img_infos, img_ids = batch
                polygons = None
            self.current_iter = epoch * len(self.train_loader) + batch_idx

            with self._prof_phase("to_device"):
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            if hasattr(self, "_apply_multi_scale_batch"):
                imgs, targets, polygons = self._apply_multi_scale_batch(
                    imgs,
                    targets,
                    polygons,
                    step=self.current_iter,
                )

            # Teacher forward (no-grad). Under AMP it runs in autocast too, so
            # the frozen teacher doesn't pay full-precision compute each step.
            if distiller is not None:
                if self.scaler is not None:
                    with self._autocast_context():
                        distiller.teacher_forward(imgs)
                else:
                    distiller.teacher_forward(imgs)

            # Forward + backward. Under DDP the loss needs no rescaling:
            # every family's loss is mean/ratio-normalized, so DDP's gradient
            # averaging already composes the per-rank gradients into the
            # single-GPU-equivalent gradient (see scale_loss_for_ddp, #484).
            # The distiller's adapter grads are averaged the same way in
            # _sync_distiller_grads, keeping student and distiller consistent.
            if self.scaler is not None:
                with self._prof_phase("forward"):
                    with self._autocast_context():
                        outputs = self._forward_train(imgs, targets, polygons)
                        total_loss_raw = outputs["total_loss"]
                        if distiller is not None:
                            distill_loss = distiller.compute_loss()
                            total_loss_raw = total_loss_raw + distill_loss
                            self._distill_loss_val = distill_loss.item()
                loss = scale_loss_for_ddp(total_loss_raw)
                self.optimizer.zero_grad()
                with self._prof_phase("backward"):
                    self.scaler.scale(loss).backward()
                    self._sync_distiller_grads()
                    if self._should_clip_gradients():
                        self.scaler.unscale_(self.optimizer)
                        self._clip_gradients()
                with self._prof_phase("optimizer"):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                with self._prof_phase("forward"):
                    outputs = self._forward_train(imgs, targets, polygons)
                    total_loss_raw = outputs["total_loss"]
                    if distiller is not None:
                        distill_loss = distiller.compute_loss()
                        total_loss_raw = total_loss_raw + distill_loss
                        self._distill_loss_val = distill_loss.item()
                loss = scale_loss_for_ddp(total_loss_raw)
                self.optimizer.zero_grad()
                with self._prof_phase("backward"):
                    loss.backward()
                    self._sync_distiller_grads()
                    self._clip_gradients()
                with self._prof_phase("optimizer"):
                    self.optimizer.step()

            if distiller is not None:
                distiller.step()

            # EMA
            if self.ema_model is not None:
                self.ema_model.update(self.model)

            # Logging captures the pre-scale value so single-GPU and DDP
            # report identical magnitudes (single-GPU semantics). ``.item()``
            # already returns a Python float and detaches from autograd.
            loss_val = float(total_loss_raw.item())
            loss_components = self._scalar_mapping(self.get_loss_components(outputs))
            if distiller is not None:
                loss_components["distill"] = self._distill_loss_val
            total_loss += loss_val
            for name, value in loss_components.items():
                loss_component_sums[name] = loss_component_sums.get(name, 0.0) + value

            # total_loss_raw is included so no reference to this step's
            # autograd graph outlives the iteration: a live graph pins
            # AccumulateGrad nodes to the default stream, which invalidates
            # CUDA graph capture on a later batch (and idly retains memory
            # otherwise).
            del outputs, loss, total_loss_raw

            # LR update
            lr = self._apply_scheduler(self.current_iter + 1)
            num_batches += 1

            # Progress bar
            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{lr:.6f}"}
            postfix.update({k: f"{v:.4f}" for k, v in loss_components.items()})
            pbar.set_postfix(postfix)

            if prof is not None:
                prof.step()
                if prof.finished:
                    if getattr(self.config, "profile_then_stop", False):
                        self._stop_training = True
                        break
                    # Window closed: drop the hooks so the rest of the run
                    # pays nothing, and keep training.
                    logger.info(
                        "Profile window complete; training continues "
                        "(profile_then_stop=True stops here instead)"
                    )
                    self._profiler = None
                    prof = None

        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_components = {
            name: value / max(num_batches, 1)
            for name, value in loss_component_sums.items()
        }
        if is_main_process():
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

        self._maybe_precise_bn(epoch)

        # Validation. A profile-only run (profile_then_stop) truncated the
        # epoch, so validating the barely-trained weights would waste time and
        # could poison the best-metric state.
        val_metrics = None
        if not getattr(self, "_stop_training", False) and self._should_validate_epoch(
            epoch
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
        self._enforce_frozen_bn_eval()

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

        # getattr: test doubles may bypass BaseTrainer.__init__, same as _profiler.
        distiller = getattr(self, "distiller", None)

        prof = getattr(self, "_profiler", None)
        loader = prof.wrap_loader(pbar) if prof is not None else pbar
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 5:
                imgs, targets, img_infos, img_ids, polygons = batch
            else:
                imgs, targets, img_infos, img_ids = batch
                polygons = None

            is_opt_step = (batch_idx + 1) % accum == 0 or batch_idx == len(self.train_loader) - 1
            opt_step = epoch * steps_per_epoch + batch_idx // accum
            self.current_iter = opt_step

            with self._prof_phase("to_device"):
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

            # Teacher forward (no-grad). Under AMP it runs in autocast too, so
            # the frozen teacher doesn't pay full-precision compute each step.
            if distiller is not None:
                if self.scaler is not None:
                    with self._autocast_context():
                        distiller.teacher_forward(imgs)
                else:
                    distiller.teacher_forward(imgs)

            # Forward + backward. Gradients accumulate across the window; the
            # optimizer step, clipping, EMA and LR update fire only on the
            # window boundary (``is_opt_step``). The division by the window
            # keeps mean semantics across micro-batches; under DDP no further
            # rescaling is needed (losses are mean-normalized, so DDP's
            # gradient averaging is already correct — see scale_loss_for_ddp).
            if self.scaler is not None:
                with self._prof_phase("forward"):
                    with self._autocast_context():
                        outputs = self._forward_train(imgs, targets, polygons)
                        total_loss_raw = outputs["total_loss"]
                        if distiller is not None:
                            distill_loss = distiller.compute_loss()
                            total_loss_raw = total_loss_raw + distill_loss
                            self._distill_loss_val = distill_loss.item()
                        loss = total_loss_raw / actual_window
                loss = scale_loss_for_ddp(loss)
                with self._prof_phase("backward"):
                    self.scaler.scale(loss).backward()
                if is_opt_step:
                    with self._prof_phase("optimizer"):
                        self._sync_distiller_grads()
                        if self._should_clip_gradients():
                            self.scaler.unscale_(self.optimizer)
                            self._clip_gradients()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
            else:
                with self._prof_phase("forward"):
                    outputs = self._forward_train(imgs, targets, polygons)
                    total_loss_raw = outputs["total_loss"]
                    if distiller is not None:
                        distill_loss = distiller.compute_loss()
                        total_loss_raw = total_loss_raw + distill_loss
                        self._distill_loss_val = distill_loss.item()
                    loss = total_loss_raw / actual_window
                loss = scale_loss_for_ddp(loss)
                with self._prof_phase("backward"):
                    loss.backward()
                if is_opt_step:
                    with self._prof_phase("optimizer"):
                        self._sync_distiller_grads()
                        self._clip_gradients()
                        self.optimizer.step()

            if distiller is not None:
                distiller.step()

            if is_opt_step:
                # EMA
                if self.ema_model is not None:
                    self.ema_model.update(self.model)
                # LR update
                lr = self._apply_scheduler(opt_step + 1)

            # Logging uses the raw pre-scale value (single-GPU semantics).
            loss_val = float(total_loss_raw.detach().item())
            loss_components = self._scalar_mapping(self.get_loss_components(outputs))
            if distiller is not None:
                loss_components["distill"] = self._distill_loss_val
            total_loss += loss_val
            num_batches += 1
            for name, value in loss_components.items():
                loss_component_sums[name] = loss_component_sums.get(name, 0.0) + value

            # See the matching del in _train_epoch: total_loss_raw must not
            # keep the autograd graph alive across iterations.
            del outputs, loss, total_loss_raw

            # Progress bar
            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{lr:.6f}"}
            postfix.update({k: f"{v:.4f}" for k, v in loss_components.items()})
            pbar.set_postfix(postfix)

            if prof is not None:
                prof.step()
                if prof.finished:
                    if getattr(self.config, "profile_then_stop", False):
                        self._stop_training = True
                        break
                    # Window closed: drop the hooks so the rest of the run
                    # pays nothing, and keep training.
                    logger.info(
                        "Profile window complete; training continues "
                        "(profile_then_stop=True stops here instead)"
                    )
                    self._profiler = None
                    prof = None

        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_components = {
            name: value / max(num_batches, 1)
            for name, value in loss_component_sums.items()
        }
        if is_main_process():
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

        self._maybe_precise_bn(epoch)

        # Validation. A profile-only run (profile_then_stop) truncated the
        # epoch, so validating the barely-trained weights would waste time and
        # could poison the best-metric state.
        val_metrics = None
        if not getattr(self, "_stop_training", False) and self._should_validate_epoch(
            epoch
        ):
            val_metrics = self._validate_epoch(epoch)

        return avg_loss, val_metrics, avg_loss_components, self._current_lrs()

    # =========================================================================
    # Validation
    # =========================================================================

    def _should_validate_epoch(self, epoch: int) -> bool:
        scheduled = (
            self.config.eval_interval > 0
            and (epoch + 1) % self.config.eval_interval == 0
        )
        final_plot = (
            bool(getattr(self.config, "save_plots", False))
            and self._is_final_epoch(epoch)
        )
        precise_bn_final = (
            int(getattr(self.config, "precise_bn", 0) or 0) > 0
            and self._is_final_epoch(epoch)
        )
        return scheduled or final_plot or precise_bn_final

    def _is_final_epoch(self, epoch: int) -> bool:
        return (epoch + 1) >= self.config.epochs

    def _validate_epoch(
        self, epoch: int, *, save_plots: bool | None = None
    ) -> Optional[Dict[str, Any]]:
        # First-cut policy: validation runs on rank 0 only. Non-zero ranks
        # barrier-wait so the next epoch's set_epoch fires in lockstep.
        # Rank 0 barriers once at the bottom regardless of outcome.
        if self.is_distributed and not is_main_process():
            barrier()
            return None
        try:
            return self._run_validation(epoch, save_plots=save_plots)
        finally:
            if self.is_distributed:
                barrier()

    def _run_validation(
        self, epoch: int, *, save_plots: bool | None = None
    ) -> Optional[Dict[str, Any]]:
        validation_task = getattr(
            getattr(self, "wrapper_model", None), "task", "detect"
        )
        if validation_task == "classify":
            return self._run_classify_validation(epoch)
        if validation_task == "semantic":
            return self._run_semantic_validation(epoch)
        if validation_task == "depth":
            return self._run_depth_validation(epoch)
        if validation_task == "restore":
            return self._run_restore_validation(epoch)
        try:
            from libreyolo.validation import (
                DetectionValidator,
                OBBValidator,
                PointValidator,
                SegmentationValidator,
                ValidationConfig,
            )
            from libreyolo.validation.loss import ValidationLossMixin

            logger.info(f"Running validation for epoch {epoch + 1}")

            # Only save plots on the final epoch when explicitly requested.
            is_final_epoch = self._is_final_epoch(epoch)
            val_save_plots = (
                bool(save_plots)
                if save_plots is not None
                else bool(getattr(self.config, "save_plots", False)) and is_final_epoch
            )
            val_save_dir = (
                str(self.save_dir / "val") if val_save_plots else None
            )

            val_config = ValidationConfig(
                data=self.config.data,
                single_cls=getattr(self.config, "single_cls", False),
                batch_size=max(1, self.config.batch // max(getattr(self, "world_size", 1), 1)),
                imgsz=self.config.imgsz,
                conf_thres=0.001,
                iou_thres=0.65,
                max_det=getattr(self.config, "max_det", 300),
                eval_max_det=getattr(self.config, "eval_max_det", None),
                faster_coco_eval=getattr(self.config, "faster_coco_eval", False),
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                amp_dtype=getattr(self.config, "amp_dtype", "float16"),
                verbose=False,
                num_workers=self.config.workers,
                save_plots=val_save_plots,
                save_dir=val_save_dir,
                # One knob for both loops: a run that opts into image caching
                # for training gets the same for its (deterministic) validation.
                cache=getattr(self.config, "cache", False),
                # Same principle for graph replay, gated on the inference-side
                # capability: training capture (CudaGraphTrainSpec) and forward
                # capture (SUPPORTS_CUDA_GRAPH) are separate opt-ins, and a
                # family with only the former must validate eagerly rather
                # than fail. Replay is bit-identical, so this is an execution
                # detail, not a protocol change.
                cuda_graph=(
                    bool(getattr(self.config, "cuda_graph", False))
                    and bool(
                        getattr(self.wrapper_model, "SUPPORTS_CUDA_GRAPH", False)
                    )
                ),
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
                task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
                if task == "segment":
                    validator_cls = SegmentationValidator
                elif task == "obb":
                    validator_cls = OBBValidator
                elif task == "point":
                    validator_cls = PointValidator
                else:
                    validator_cls = DetectionValidator
                # One validator instance for the whole run, so the dataset,
                # dataloader workers, pinned buffers and parsed ground truth
                # survive between epochs instead of being rebuilt ~100 times.
                # The per-epoch config is still honored: only save_plots and
                # save_dir ever differ between the configs this loop builds,
                # and neither is baked into the reused state.
                validator = getattr(self, "_epoch_validator", None)
                if validator is None or type(validator) is not validator_cls:
                    # The adapter is built once with the cached validator: the
                    # EMA/eval module object is stable across epochs, and
                    # _init_metrics() re-arms the adapter every run. OBB and
                    # point validators do not take a loss_adapter, so the
                    # mixin check keeps a family that opts in for a task this
                    # branch cannot serve from breaking its own validation.
                    validator_kwargs = (
                        self._validation_loss_kwargs(eval_pytorch_model)
                        if issubclass(validator_cls, ValidationLossMixin)
                        else {}
                    )
                    validator = validator_cls(
                        model=self.wrapper_model,
                        config=val_config,
                        **validator_kwargs,
                    )
                    self._epoch_validator = validator
                else:
                    validator.config = val_config
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            raw_metrics = self._scalar_mapping(results)
            best_key = getattr(self, "best_metric_key", "metrics/mAP50-95")
            if task == "point" and best_key == "metrics/mAP50-95":
                best_key = "fitness"
            best_metric = raw_metrics.get(
                best_key, raw_metrics.get("metrics/mAP50-95", 0.0)
            )
            if task == "point":
                primary_th = getattr(validator, "_primary_threshold", 0.01)
                mAP50 = raw_metrics.get(f"metrics/mAP@{primary_th:.2f}", 0.0)
            else:
                mAP50 = raw_metrics.get(
                    "metrics/mAP50", raw_metrics.get("metrics/mAP50(B)", 0.0)
                )
            metrics = {
                "mAP50": mAP50,
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

    def _run_classify_validation(
        self, epoch: int
    ) -> Optional[Dict[str, Any]]:
        """Validate the classification head (top-1/top-5) on the val split."""
        try:
            from libreyolo.validation import ClassifyValidator, ValidationConfig

            if self.wrapper_model is None:
                logger.error("Validation requires wrapper_model to be provided to trainer")
                return None

            logger.info(f"Running classification validation for epoch {epoch + 1}")
            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=max(1, self.config.batch // max(getattr(self, "world_size", 1), 1)),
                imgsz=self.config.imgsz,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                amp_dtype=self.config.amp_dtype,
                verbose=False,
                num_workers=self.config.workers,
                split="val",
            )

            eval_pytorch_model = (
                self.ema_model.ema if self.ema_model else unwrap_model(self.model)
            )
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model
            try:
                validator = ClassifyValidator(
                    model=self.wrapper_model,
                    config=val_config,
                    **self._validation_loss_kwargs(eval_pytorch_model),
                )

                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            raw_metrics = self._scalar_mapping(results)
            top1 = raw_metrics.get("metrics/accuracy_top1", 0.0)
            top5 = raw_metrics.get("metrics/accuracy_top5", 0.0)
            logger.info("Validation - top1: %.4f, top5: %.4f", top1, top5)
            return {
                "mAP50": top1,
                "mAP50_95": top1,
                "best_metric": top1,
                "best_metric_key": "metrics/accuracy_top1",
                "metrics": raw_metrics,
            }
        except Exception as e:
            logger.error(f"Classification validation failed: {e}")
            import traceback

            logger.debug(f"Validation traceback:\n{traceback.format_exc()}")
            return None

    def _run_semantic_validation(
        self, epoch: int
    ) -> Optional[Dict[str, Any]]:
        """Validate the semantic head (mIoU / pixel accuracy) on the val split."""
        try:
            from libreyolo.validation import SemanticValidator, ValidationConfig

            if self.wrapper_model is None:
                logger.error("Validation requires wrapper_model to be provided to trainer")
                return None

            logger.info(f"Running semantic validation for epoch {epoch + 1}")
            # A family whose reference recipe validates on a canvas different
            # from its train crop (PP-LiteSeg's 75 sizes train on 768x768 and
            # validate on 768x1536) declares it here; everyone else keeps
            # validating at the training imgsz.
            val_imgsz = (
                getattr(self.wrapper_model, "semantic_val_imgsz", None) or self.config.imgsz
            )
            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=max(1, self.config.batch // max(getattr(self, "world_size", 1), 1)),
                imgsz=val_imgsz,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                amp_dtype=self.config.amp_dtype,
                verbose=False,
                num_workers=self.config.workers,
                split="val",
            )

            eval_pytorch_model = (
                self.ema_model.ema if self.ema_model else unwrap_model(self.model)
            )
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model
            try:
                validator = SemanticValidator(
                    model=self.wrapper_model,
                    config=val_config,
                    **self._validation_loss_kwargs(eval_pytorch_model),
                )
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            raw_metrics = self._scalar_mapping(results)
            miou = raw_metrics.get("metrics/mIoU", 0.0)
            accuracy = raw_metrics.get("metrics/pixel_accuracy", 0.0)
            logger.info("Validation - mIoU: %.4f, pixel accuracy: %.4f", miou, accuracy)
            return {
                "mAP50": miou,
                "mAP50_95": miou,
                "best_metric": miou,
                "best_metric_key": "metrics/mIoU",
                "metrics": raw_metrics,
            }
        except Exception as e:
            logger.error(f"Semantic validation failed: {e}")
            import traceback

            logger.debug(f"Validation traceback:\n{traceback.format_exc()}")
            return None

    def _run_depth_validation(
        self, epoch: int
    ) -> Optional[Dict[str, Any]]:
        """Validate the depth head (AbsRel / delta1) on the val split."""
        try:
            from libreyolo.validation import DepthValidator, ValidationConfig

            if self.wrapper_model is None:
                logger.error("Validation requires wrapper_model to be provided to trainer")
                return None

            logger.info(f"Running depth validation for epoch {epoch + 1}")
            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=max(1, self.config.batch // max(getattr(self, "world_size", 1), 1)),
                imgsz=self.config.imgsz,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                amp_dtype=self.config.amp_dtype,
                verbose=False,
                num_workers=self.config.workers,
                split="val",
                allow_download_scripts=self.config.allow_download_scripts,
            )

            eval_pytorch_model = (
                self.ema_model.ema if self.ema_model else unwrap_model(self.model)
            )
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model
            try:
                validator = DepthValidator(model=self.wrapper_model, config=val_config)
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            raw_metrics = self._scalar_mapping(results)
            delta1 = raw_metrics.get("metrics/delta1", 0.0)
            abs_rel = raw_metrics.get("metrics/abs_rel", 0.0)
            logger.info("Validation - delta1: %.4f, AbsRel: %.4f", delta1, abs_rel)
            return {
                "mAP50": delta1,
                "mAP50_95": delta1,
                "best_metric": delta1,
                "best_metric_key": "metrics/delta1",
                "metrics": raw_metrics,
            }
        except Exception as e:
            logger.error(f"Depth validation failed: {e}")
            import traceback

            logger.debug(f"Validation traceback:\n{traceback.format_exc()}")
            return None

    def _run_restore_validation(
        self, epoch: int
    ) -> Optional[Dict[str, Any]]:
        """Validate the restoration model (PSNR / SSIM) on the val split."""
        try:
            from libreyolo.validation import RestoreValidator, ValidationConfig

            if self.wrapper_model is None:
                logger.error("Validation requires wrapper_model to be provided to trainer")
                return None

            logger.info(f"Running restore validation for epoch {epoch + 1}")
            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=max(1, self.config.batch // max(getattr(self, "world_size", 1), 1)),
                imgsz=self.config.imgsz,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                amp_dtype=self.config.amp_dtype,
                verbose=False,
                num_workers=self.config.workers,
                split="val",
                allow_download_scripts=self.config.allow_download_scripts,
            )

            eval_pytorch_model = (
                self.ema_model.ema if self.ema_model else unwrap_model(self.model)
            )
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model
            try:
                validator = RestoreValidator(
                    model=self.wrapper_model,
                    config=val_config,
                    **self._validation_loss_kwargs(eval_pytorch_model),
                )
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            raw_metrics = self._scalar_mapping(results)
            psnr = raw_metrics.get("metrics/PSNR", 0.0)
            ssim = raw_metrics.get("metrics/SSIM", 0.0)
            logger.info("Validation - PSNR: %.4f, SSIM: %.4f", psnr, ssim)
            return {
                "mAP50": psnr,
                "mAP50_95": psnr,
                "best_metric": psnr,
                "best_metric_key": "metrics/PSNR",
                "metrics": raw_metrics,
            }
        except Exception as e:
            logger.error(f"Restore validation failed: {e}")
            import traceback

            logger.debug(f"Validation traceback:\n{traceback.format_exc()}")
            return None

    def _maybe_export_check(self) -> None:
        if not bool(getattr(self.config, "export_check", False)):
            return
        error: Optional[BaseException] = None
        if is_main_process():
            try:
                from .export_check import run_export_parity_check

                wrapper = self.wrapper_model
                if wrapper is None:
                    raise RuntimeError("export_check=True requires a wrapper model")
                logger.info("export_check: exporting ONNX before epoch 1")
                run_export_parity_check(
                    wrapper,
                    out_dir=self.save_dir,
                    imgsz=getattr(self.config, "imgsz", 640),
                )
            except BaseException as exc:
                error = exc
        barrier()
        if self.is_distributed:
            import torch.distributed as dist

            payload = [None if error is None else f"{type(error).__name__}: {error}"]
            dist.broadcast_object_list(payload, src=0)
            if payload[0] is not None:
                if error is not None:
                    raise error
                raise RuntimeError(f"export_check failed on rank 0: {payload[0]}")
        elif error is not None:
            raise error

    def _assert_class_balanced_honored(self) -> None:
        if not bool(getattr(self.config, "class_balanced", False)):
            return
        from torch.utils.data import WeightedRandomSampler

        from ..data.class_balanced import DistributedClassBalancedSampler

        sampler = getattr(getattr(self, "train_loader", None), "sampler", None)
        if isinstance(
            sampler, (WeightedRandomSampler, DistributedClassBalancedSampler)
        ):
            return
        raise ValueError(
            f"class_balanced=True is not supported for {self.get_model_family()}: "
            "this family does not build the train loader through the shared "
            "detection sampler. Omit class_balanced or use a family that does "
            "(e.g. YOLO9 / YOLOX)."
        )

    def _maybe_precise_bn(self, epoch: int, *, force: bool = False) -> bool:
        samples = int(getattr(self.config, "precise_bn", 0) or 0)
        if samples <= 0:
            return False
        if not force and not self._is_final_epoch(epoch):
            return False
        if getattr(self, "_stop_training", False):
            return False
        if getattr(self, "_precise_bn_done", False):
            return False
        from .precise_bn import compute_precise_bn_stats

        updated = 0
        raw_model = unwrap_model(self.model)
        frozen_ids = {id(module) for module in getattr(self, "_frozen_bn_modules", ())}
        excluded_names = {
            name
            for name, module in raw_model.named_modules()
            if id(module) in frozen_ids
        }
        targets = [raw_model]
        ema = getattr(self, "ema_model", None)
        if ema is not None and getattr(ema, "ema", None) is not None:
            targets.append(ema.ema)
        for net in targets:
            updated = max(
                updated,
                compute_precise_bn_stats(
                    net,
                    self.train_loader,
                    samples,
                    device=self.device,
                    excluded_names=excluded_names,
                ),
            )
        self._precise_bn_done = True
        return updated > 0

    def _refresh_best_precise_bn_checkpoint(self) -> bool:
        """Recompute BN statistics for a historical best checkpoint.

        The final epoch path recalibrates the live model before validation and
        checkpointing. If an earlier epoch remains best, rank 0 loads that
        state and broadcasts it before recalibration so no shared filesystem is
        required. Live last-epoch weights are restored afterward.
        """
        samples = int(getattr(self.config, "precise_bn", 0) or 0)
        if samples <= 0 or getattr(self, "_stop_training", False):
            return False

        from .precise_bn import compute_precise_bn_stats

        distributed = bool(getattr(self, "is_distributed", False))
        refresh = int(getattr(self, "best_epoch", 0) or 0) != self.current_epoch + 1
        refresh = self._sync_main_bool(refresh)
        if not refresh:
            return False
        try:
            best_path = self.save_dir / "weights" / "best.pt"
            checkpoint = None
            model_state = None
            checkpoint_error = None
            if not distributed or is_main_process():
                if not best_path.is_file():
                    checkpoint_error = f"best checkpoint not found: {best_path}"
                else:
                    try:
                        checkpoint = load_trusted_torch_file(
                            best_path,
                            map_location="cpu",
                            context="precise BN best checkpoint",
                        )
                    except Exception as exc:
                        checkpoint_error = (
                            f"could not read best checkpoint {best_path}: {exc}"
                        )
                    else:
                        model_state = (
                            checkpoint.get("model")
                            if isinstance(checkpoint, Mapping)
                            else None
                        )
                        if not isinstance(model_state, Mapping):
                            checkpoint_error = (
                                f"best checkpoint has no model state: {best_path}"
                            )

            if distributed:
                import torch.distributed as _dist

                status = [checkpoint_error]
                _dist.broadcast_object_list(status, src=0)
                checkpoint_error = status[0]
            if checkpoint_error is not None:
                if is_main_process():
                    logger.warning("precise_bn: %s", checkpoint_error)
                return False

            raw_model = unwrap_model(self.model)
            ema = getattr(self, "ema_model", None)
            ema_model = getattr(ema, "ema", None) if ema is not None else None
            raw_checkpoint_state = (
                checkpoint.get("train_model", model_state)
                if checkpoint is not None
                else None
            )
            targets = [
                (
                    raw_model,
                    raw_checkpoint_state,
                    "train_model" if ema_model is not None else "model",
                )
            ]
            if ema_model is not None:
                targets.append(
                    (
                        ema_model,
                        checkpoint.get("ema", model_state)
                        if checkpoint is not None
                        else None,
                        "ema",
                    )
                )

            frozen_ids = {
                id(module) for module in getattr(self, "_frozen_bn_modules", ())
            }
            excluded_names = {
                name
                for name, module in raw_model.named_modules()
                if id(module) in frozen_ids
            }
            live_states = [
                {
                    key: value.detach().to("cpu").clone()
                    for key, value in target.state_dict().items()
                }
                for target, _state, _key in targets
            ]
            refreshed_states: dict[str, dict[str, torch.Tensor]] = {}
            updated = 0
            try:
                load_error = None
                if not distributed or is_main_process():
                    try:
                        for target, state, _key in targets:
                            target.load_state_dict(state, strict=True)
                    except Exception as exc:
                        load_error = f"could not load historical best state: {exc}"
                if distributed:
                    status = [load_error]
                    _dist.broadcast_object_list(status, src=0)
                    load_error = status[0]
                if load_error is not None:
                    raise RuntimeError(f"precise_bn: {load_error}")

                if distributed:
                    for target, _state, _key in targets:
                        for value in target.state_dict().values():
                            _dist.broadcast(value, src=0)

                for target, state, key in targets:
                    count = compute_precise_bn_stats(
                        target,
                        self.train_loader,
                        samples,
                        device=self.device,
                        excluded_names=excluded_names,
                    )
                    updated = max(updated, count)
                    if is_main_process():
                        refreshed_states[key] = {
                            name: value.detach().to("cpu").clone()
                            for name, value in target.state_dict().items()
                        }
            finally:
                for (target, _state, _key), live_state in zip(
                    targets, live_states, strict=True
                ):
                    target.load_state_dict(live_state, strict=True)

            if updated <= 0:
                return False
            if is_main_process():
                assert checkpoint is not None
                raw_state = refreshed_states[targets[0][2]]
                selected_state = (
                    refreshed_states["ema"] if ema_model is not None else raw_state
                )
                checkpoint["model"] = selected_state
                if ema_model is not None:
                    checkpoint["train_model"] = raw_state
                    checkpoint["ema"] = selected_state
                validate_checkpoint_metadata(checkpoint, strict=True)
                torch.save(checkpoint, best_path)
                logger.info(
                    "precise_bn: refreshed historical best checkpoint: %s",
                    best_path,
                )
            return True
        finally:
            if distributed:
                barrier()

    def _maybe_offer_average(self, val_metrics: Optional[Dict[str, Any]]) -> None:
        averager = getattr(self, "_weight_averager", None)
        if averager is None or not val_metrics or not is_main_process():
            return
        metric = self._best_metric_value(val_metrics)
        source = self.ema_model.ema if self.ema_model else unwrap_model(self.model)
        if averager.consider(source, metric):
            logger.info(
                "average_best: admitted snapshot (metric=%.5f, pool=%d/%d)",
                metric,
                averager.size,
                averager.n,
            )
            self._persist_average_pool()

    def _write_average_checkpoint(self) -> Optional[Path]:
        averager = getattr(self, "_weight_averager", None)
        if averager is None:
            return None
        distributed = bool(getattr(self, "is_distributed", False))
        if not is_main_process():
            if distributed:
                barrier()
            return None
        try:
            averaged = averager.average_state_dict()
            if averaged is None:
                return None

            logger.info("average_best: validating averaged weights")
            average_metrics = self._validate_average_state(averaged)
            self._average_validation_metrics = average_metrics
            if average_metrics is None:
                logger.warning(
                    "average_best: averaged weights were written without validation "
                    "metrics because final validation did not complete"
                )

            weights_dir = self.save_dir / "weights"
            weights_dir.mkdir(exist_ok=True)
            path = weights_dir / "average.pt"
            names = (
                self.wrapper_model.names
                if self.wrapper_model is not None
                and hasattr(self.wrapper_model, "names")
                else build_class_names(
                    int(getattr(self, "num_classes", self.config.num_classes))
                )
            )
            checkpoint_imgsz = getattr(self.config, "imgsz", 640)
            extra_checkpoint_meta = {}
            if isinstance(checkpoint_imgsz, (list, tuple)):
                cp_h, cp_w = int(checkpoint_imgsz[0]), int(checkpoint_imgsz[1])
                checkpoint_imgsz = max(cp_h, cp_w)
                if cp_h != cp_w:
                    extra_checkpoint_meta["imgsz_h"] = cp_h
                    extra_checkpoint_meta["imgsz_w"] = cp_w
            extra_checkpoint_meta.update(self._checkpoint_extra_metadata())
            average_metric_key = (
                average_metrics.get("best_metric_key") if average_metrics else None
            )
            average_metric_value = (
                self._best_metric_value(average_metrics) if average_metrics else None
            )
            checkpoint = wrap_libreyolo_checkpoint(
                averaged,
                model_family=self.get_model_family(),
                size=self.config.size,
                task=getattr(getattr(self, "wrapper_model", None), "task", "detect"),
                nc=int(getattr(self, "num_classes", self.config.num_classes)),
                names=names,
                imgsz=int(checkpoint_imgsz),
                epoch=self.current_epoch,
                best_mAP50_95=self.best_mAP50_95,
                best_mAP50=self.best_mAP50,
                best_metric_key=getattr(self, "best_metric_key", "metrics/mAP50-95"),
                best_metric_value=self.best_mAP50_95,
                best_epoch=self.best_epoch,
                is_ema_weights=self.ema_model is not None,
                averaged_snapshot_count=averager.size,
                average_metric_key=average_metric_key,
                average_metric_value=average_metric_value,
                **extra_checkpoint_meta,
            )
            torch.save(checkpoint, path)
            logger.info(
                "average_best: wrote %s from %d snapshots (source metrics=%s)",
                path,
                averager.size,
                averager.metrics(),
            )
            return path
        finally:
            if distributed:
                barrier()

    def _validate_average_state(
        self, averaged: Mapping[str, torch.Tensor]
    ) -> Optional[Dict[str, Any]]:
        """Validate averaged weights once without changing the live train state."""
        source = self.ema_model.ema if self.ema_model else unwrap_model(self.model)
        live_state = {
            key: value.detach().to("cpu").clone()
            for key, value in source.state_dict().items()
        }
        try:
            source.load_state_dict(averaged, strict=True)
            return self._run_validation(self.current_epoch, save_plots=False)
        finally:
            source.load_state_dict(live_state, strict=True)

    def _average_pool_path(self, directory: Path | None = None) -> Path:
        root = directory if directory is not None else (self.save_dir / "weights")
        return Path(root) / "average_pool.pt"

    def _persist_average_pool(self) -> None:
        averager = getattr(self, "_weight_averager", None)
        if averager is None or averager.size == 0 or not is_main_process():
            return
        save_dir = getattr(self, "save_dir", None)
        if save_dir is None:
            return
        averager.save(self._average_pool_path())

    def _restore_average_pool(self, checkpoint_path: str | Path) -> None:
        averager = getattr(self, "_weight_averager", None)
        if averager is None:
            self._resume_average_pool_path = str(checkpoint_path)
            return
        checkpoint_path = Path(checkpoint_path)
        sidecar = checkpoint_path.parent / "average_pool.pt"
        if sidecar.is_file():
            try:
                loaded = averager.load(sidecar)
            except Exception as exc:
                logger.warning(
                    "Could not restore average_best pool from %s: %s", sidecar, exc
                )
            else:
                if loaded and is_main_process():
                    logger.info(
                        "average_best: restored %d snapshot(s) from %s",
                        loaded,
                        sidecar,
                    )
                return
        # last.pt (or any non-best resume) is not the historical best. Tagging
        # those live weights with best_mAP50_95 would let them block later
        # snapshots while the real best stays out of average.pt.
        seed = self._average_seed_snapshot(checkpoint_path)
        if seed is None:
            if is_main_process():
                logger.info(
                    "average_best: starting with an empty pool; no average_pool.pt "
                    "next to %s and the resumed weights are not the recorded best",
                    checkpoint_path,
                )
            return
        seed_state, metric = seed
        if averager.consider_state(seed_state, metric) and is_main_process():
            logger.info(
                "average_best: seeded pool from the recorded best weights "
                "(metric=%.5f); no average_pool.pt next to %s",
                metric,
                checkpoint_path,
            )

    def _average_seed_snapshot(
        self, checkpoint_path: Path
    ) -> Optional[Tuple[Dict[str, torch.Tensor], float]]:
        """State and metric that were recorded together, or None."""
        resumed_metric = float(getattr(self, "best_mAP50_95", 0.0) or 0.0)
        if checkpoint_path.name.lower() == "best.pt":
            return self._live_average_source_state(), resumed_metric
        sibling = checkpoint_path.parent / "best.pt"
        if sibling.is_file():
            snapshot = self._snapshot_from_checkpoint(sibling)
            if snapshot is not None:
                return snapshot
        resumed_epoch = int(getattr(self, "start_epoch", 1))
        if resumed_epoch == int(getattr(self, "best_epoch", -1)):
            return self._live_average_source_state(), resumed_metric
        return None

    def _live_average_source_state(self) -> Dict[str, torch.Tensor]:
        source = self.ema_model.ema if self.ema_model else unwrap_model(self.model)
        return source.state_dict()

    def _snapshot_from_checkpoint(
        self, path: Path
    ) -> Optional[Tuple[Dict[str, torch.Tensor], float]]:
        try:
            blob = load_trusted_torch_file(
                path, map_location="cpu", context="average_best seed checkpoint"
            )
        except Exception as exc:
            logger.warning("Could not read %s to seed average_best: %s", path, exc)
            return None
        state = blob.get("model") if isinstance(blob, Mapping) else None
        if not isinstance(state, Mapping):
            logger.warning(
                "Checkpoint %s has no model state to seed average_best", path
            )
            return None
        metric = None
        for key in ("best_metric_value", "best_metric", "best_mAP50_95"):
            try:
                candidate = float(blob[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(candidate):
                metric = candidate
                break
        if metric is None:
            logger.warning(
                "Checkpoint %s has no finite best metric to seed average_best", path
            )
            return None
        return state, metric

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
            else build_class_names(int(getattr(self, "num_classes", self.config.num_classes)))
        )
        checkpoint_nc = int(getattr(self, "num_classes", self.config.num_classes))
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

        # Handle rectangular imgsz: checkpoint schema stores a scalar imgsz
        # (legacy), while imgsz_h/imgsz_w store the actual dimensions.
        if isinstance(checkpoint_imgsz, (list, tuple)):
            cp_h, cp_w = int(checkpoint_imgsz[0]), int(checkpoint_imgsz[1])
            cp_imgsz_scalar = max(cp_h, cp_w)
        else:
            cp_imgsz_scalar = int(checkpoint_imgsz)
            cp_h = cp_w = cp_imgsz_scalar

        extra_checkpoint_meta = {}
        if cp_h != cp_w:
            extra_checkpoint_meta["imgsz_h"] = cp_h
            extra_checkpoint_meta["imgsz_w"] = cp_w

        checkpoint = wrap_libreyolo_checkpoint(
            model_to_save.state_dict(),
            model_family=self.get_model_family(),
            size=self.config.size,
            task=getattr(getattr(self, "wrapper_model", None), "task", "detect"),
            nc=checkpoint_nc,
            names=names,
            imgsz=cp_imgsz_scalar,
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
            **extra_checkpoint_meta,
        )
        checkpoint.update(self._checkpoint_extra_metadata())
        quant_manifest = getattr(self.wrapper_model, "_quant_manifest", None)
        if quant_manifest:
            # QAT/QAD checkpoints must be self-describing so LibreYOLO(path)
            # rebuilds the quantized structure before loading scales.
            checkpoint["quant"] = dict(quant_manifest)
        checkpoint["best_metric"] = self.best_mAP50_95
        checkpoint["best_metric_name"] = checkpoint["best_metric_key"]
        if self.ema_model is not None:
            checkpoint["train_model"] = raw_model.state_dict()
            checkpoint["ema"] = self.ema_model.ema.state_dict()
            checkpoint["ema_updates"] = self.ema_model.updates
        if getattr(self, "distiller", None) is not None:
            # Adapter/generator weights live outside the student model; persist
            # them so resume doesn't restart the distillation projectors cold.
            checkpoint["distiller"] = self.distiller.loss_modules.state_dict()

        # AMP + RNG state so ``resume()`` continues bit-for-bit instead of
        # re-warming the GradScaler from 65536 and reseeding the RNGs. All keys
        # are optional; older checkpoints without them still load fine.
        if self.scaler is not None:
            try:
                checkpoint["scaler"] = self.scaler.state_dict()
            except Exception:
                logger.warning("Could not capture GradScaler state for checkpoint")
        rng_state: Dict[str, Any] = {"torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        try:
            import numpy as _np

            # Store numpy's MT19937 state in a ``weights_only=True``-safe form
            # (a torch tensor + primitives). The raw ``get_state()`` tuple holds
            # a numpy ndarray, which the repo's safe checkpoint loader rejects.
            _keys, _pos, _has_gauss, _cached = _np.random.get_state()[1:]
            rng_state["numpy"] = {
                "keys": torch.from_numpy(_keys.astype("int64", copy=True)),
                "pos": int(_pos),
                "has_gauss": int(_has_gauss),
                "cached_gaussian": float(_cached),
            }
        except Exception:
            pass
        checkpoint["rng_state"] = rng_state

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

        if self.config.save_period > 0 and (epoch + 1) % self.config.save_period == 0:
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
                    restore_optimizer_state(self.optimizer, checkpoint["optimizer"])
                    logger.info("Optimizer state restored")
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")
            else:
                # setup() hasn't run yet — defer until the optimizer exists.
                self._resume_optimizer_state = checkpoint["optimizer"]
                logger.info("Optimizer state deferred until after setup()")

        if "distiller" in checkpoint:
            if self.distiller is not None:
                try:
                    self.distiller.loss_modules.load_state_dict(checkpoint["distiller"])
                    logger.info("Distiller adapter state restored")
                except Exception as e:
                    logger.warning(f"Could not load distiller state: {e}")
            else:
                # setup() hasn't run yet — defer until the distiller exists.
                self._resume_distiller_state = checkpoint["distiller"]
                logger.info("Distiller state deferred until after setup()")

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

        # Restore AMP + RNG state when present (backward-compatible: pre-v1.3
        # checkpoints lack these keys and simply skip this block).
        if "scaler" in checkpoint and self.scaler is not None:
            try:
                self.scaler.load_state_dict(checkpoint["scaler"])
                logger.info("GradScaler state restored")
            except Exception as e:
                logger.warning(f"Could not load GradScaler state: {e}")

        rng_state = checkpoint.get("rng_state")
        if rng_state:
            try:
                torch_state = rng_state.get("torch")
                if torch_state is not None:
                    # map_location may have moved the byte tensor to GPU; the
                    # RNG setters require CPU ByteTensors.
                    torch.set_rng_state(torch_state.cpu())
                cuda_state = rng_state.get("cuda")
                if cuda_state is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all([s.cpu() for s in cuda_state])
                np_state = rng_state.get("numpy")
                if np_state is not None:
                    import numpy as _np

                    _np.random.set_state(
                        (
                            "MT19937",
                            np_state["keys"].cpu().numpy().astype("uint32"),
                            int(np_state["pos"]),
                            int(np_state["has_gauss"]),
                            float(np_state["cached_gaussian"]),
                        )
                    )
                logger.info("RNG state restored")
            except Exception as e:
                logger.warning(f"Could not restore RNG state: {e}")

        self.patience_counter = 0
        self._restore_average_pool(checkpoint_path)
        logger.info(
            f"Resumed from epoch {self.start_epoch} "
            f"(will train to epoch {self.config.epochs})"
        )

        # setup() may have already run (immediate path: setup → resume). Re-sync
        # the LR now that start_epoch is known — _initialize_scheduler_lr() is
        # idempotent and fast-forwards to the correct schedule position.
        if self.lr_scheduler is not None and self.train_loader is not None:
            self._initialize_scheduler_lr()
