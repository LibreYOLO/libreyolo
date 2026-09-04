"""Training configuration dataclasses for LibreYOLO."""

import logging
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional, Tuple, Union
import yaml

from libreyolo.utils.amp import normalize_amp_dtype
from libreyolo.utils.image_size import normalize_imgsz

logger = logging.getLogger(__name__)


def load_train_cfg(path) -> dict:
    """Load a training-config yaml as a dict suitable for ``model.train(**out)``.

    Args:
        path: Path to a yaml file containing training parameters.

    Returns:
        Dict of training kwargs parsed from the yaml.

    Raises:
        FileNotFoundError: If the yaml file does not exist.
        ValueError: If the yaml content is not a mapping.
    """
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Training cfg yaml not found: {yaml_path}")
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Training cfg {yaml_path} must be a yaml mapping, "
            f"got {type(raw).__name__}."
        )
    return raw


@dataclass(kw_only=True)
class TrainConfig:
    """Base training configuration. Subclasses override defaults per model family."""

    # Model
    size: str = "s"
    num_classes: int = 80

    # Data
    data: Optional[str] = None
    data_dir: Optional[str] = None
    imgsz: Union[int, Tuple[int, int], List[int], str] = 640
    # Train every detection label as class 0 without changing source annotations.
    # Supported by G0/G1 detection families only; shared API/CLI gates reject
    # unsupported families and tasks before a trainer is built.
    single_cls: bool = False

    # Training
    epochs: int = 300
    # Global batch size. Under multi-GPU DDP the per-rank batch is
    # ``batch // world_size``.
    # Set to -1 to enable automatic selection: the trainer probes GPU memory
    # at small batch sizes, fits a linear model, and picks the largest batch
    # that fits within 70 % of total VRAM.
    batch: int = 16
    # Single device or multi-device spec. Accepts:
    #   - "auto" / "" → auto-pick (cuda → mps → cpu)
    #   - "cpu", "mps", "0", "cuda:0", 0 → single device
    #   - [0, 1] or "0,1" → automatic local multi-GPU DDP (torchrun also works)
    device: Union[str, int, List[int]] = "auto"
    # SyncBatchNorm across ranks under DDP. Off here; BatchNorm-heavy CNN
    # families (e.g. yolo9) override to True so BN statistics are computed
    # across the global batch instead of each rank's small shard. No-op when
    # not distributed.
    sync_bn: bool = False

    # Optimizer
    optimizer: str = "sgd"
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    nesterov: bool = True

    # Scheduler
    scheduler: str = "yoloxwarmcos"
    warmup_epochs: int = 5
    warmup_lr_start: float = 0.0
    no_aug_epochs: int = 15
    min_lr_ratio: float = 0.05

    # Augmentation
    mosaic_prob: float = 1.0
    mixup_prob: float = 1.0
    hsv_prob: float = 1.0
    flip_prob: float = 0.5
    degrees: float = 10.0
    translate: float = 0.1
    mosaic_scale: Tuple[float, float] = (0.1, 2.0)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 2.0
    # Projective (perspective) warp magnitude, following the de-facto YOLO
    # knob. The two projective terms are sampled in [-perspective, +perspective]
    # (~0.0005 is a typical scale). Default 0.0 keeps the pure-affine warp.
    perspective: float = 0.0
    # Vertical-flip probability (top-to-bottom). Off by default; useful for
    # datasets without a fixed up/down orientation (e.g. aerial imagery).
    flipud: float = 0.0

    # Classification augmentation pack. These drive the classification
    # ImageFolder pipeline only (detection families ignore them) and all
    # default off, so existing training behavior is unchanged unless requested.
    #   - auto_augment: one of "randaugment", "autoaugment", "augmix" or None.
    #   - erasing: RandomErasing probability, 0 <= erasing < 1.
    #   - mixup / cutmix: per-batch probability of applying the MixUp / CutMix
    #     op (soft labels). At most one op runs per batch: MixUp is applied with
    #     probability ``mixup``, otherwise CutMix with probability ``cutmix``, so
    #     the two are additive and should sum to at most 1.
    # Note: on the CLI, ``--mixup`` is the detection ``mixup_prob`` alias; the
    # classification ``mixup`` knob is Python-API only (model.train(mixup=...)).
    auto_augment: Optional[str] = None
    erasing: float = 0.0
    mixup: float = 0.0
    cutmix: float = 0.0

    # Training features
    ema: bool = True
    ema_decay: float = 0.9998
    amp: bool = True
    # CUDA autocast dtype when AMP is enabled. The explicit default preserves
    # historical amp=True behavior while allowing reproducible BF16 training.
    amp_dtype: str = "float16"
    # Capture the network's training forward/backward into CUDA graphs to
    # cut kernel-launch overhead on launch-bound (small-model) runs. Opt-in
    # and single-GPU only; families without capture support, distributed
    # runs and distillation runs fall back to eager training with a warning.
    # Most supported families reproduce eager numerics exactly; documented
    # exceptions use family-specific parity tolerances. Batches whose shape
    # differs from the captured shape (multi-scale, last partial batch) run
    # eager. See docs/training_cuda_graphs.md.
    cuda_graph: bool = False
    # Layer freezing. An int freezes the first N family-defined freeze groups;
    # a list freezes explicit group indices or module-name selectors; a string
    # freezes matching module/parameter names.
    freeze: Optional[Union[int, str, List[Union[int, str]]]] = None
    # Parameter-efficient fine-tuning. ``lora=True`` injects LoRA adapters into
    # the transformer components of supported families (RF-DETR: DINOv2
    # backbone attention; D-FINE/DEIM: encoder/decoder Linears with the CNN
    # backbone frozen) and trains only the adapters plus the parts that must
    # stay dense (heads, projections), for low-VRAM fine-tuning on a custom
    # dataset. Requires the optional ``peft`` dependency
    # (``pip install "libreyolo[lora]"``). Families that do not support LoRA
    # raise a clear error rather than silently ignoring the flag.
    lora: bool = False
    # Nominal (effective) batch size for gradient accumulation. When set, the
    # trainer accumulates ``round(nbs / batch)`` micro-batches per optimizer
    # step so the effective batch size is ``nbs``.
    # Left as None (the default), gradient accumulation is disabled and
    # training is unchanged.
    nbs: Optional[int] = None

    # Knowledge distillation. ``distill_model`` is a teacher-checkpoint path,
    # or a foundation-teacher id (e.g. ``"dinov2"``); setting it turns
    # distillation on. ``dis`` is the global distillation loss weight; left as
    # None it falls back to the selected loss type's published default (MGD:
    # 2e-5, CWD: 1.0, feat_mse: 1.0). ``distill_loss_type`` picks the feature
    # loss ("mgd" or "cwd") for detector teachers; a foundation teacher always
    # uses "feat_mse" on a single backbone stage. ``distill_mask_ratio`` (MGD)
    # and ``distill_tau`` (CWD) are the per-loss hyper-parameters;
    # ``distill_normalize`` L2-normalizes features before the feat_mse loss.
    # Families without a ``get_distill_config()`` (or, for foundation teachers,
    # ``get_backbone_distill_config()``) raise a clear error at setup.
    distill_model: Optional[str] = None
    dis: Optional[float] = None
    distill_loss_type: str = "mgd"
    distill_mask_ratio: float = 0.65
    distill_tau: float = 1.0
    distill_normalize: bool = False

    # Checkpointing / output
    project: str = "runs/train"
    name: str = "exp"
    exist_ok: bool = False
    save_period: int = 10
    eval_interval: int = 10
    # Prediction/NMS cap used by validation during training.
    max_det: int = 300
    # Optional COCO evaluator cap. None preserves pycocotools' historical
    # maxDets=[1, 10, 100] behavior independently of the prediction cap.
    eval_max_det: Optional[int] = None
    # Use the faster-coco-eval C++ backend for in-training COCO validation
    # metrics (bbox/segm). On by default; falls back to pycocotools with a
    # warning if the faster-coco-eval package is not installed.
    faster_coco_eval: bool = True
    save_plots: bool = False
    # Compute the family's training objective on validation batches and emit
    # metrics/loss plus its per-component values. Off by default because target
    # assignment adds validation time and memory use. Families that do not
    # implement it reject ``val_loss=True`` in
    # ``BaseTrainer.validate_validation_loss_config``.
    val_loss: bool = False

    # System
    workers: int = 4
    # Image caching to speed dataloading across epochs. Accepts False (off),
    # True/'ram' (cached images in RAM), or 'disk' (cached images as .npy
    # beside each source image). Families whose transform consumes the
    # dataset's deterministic resize cache the *resized* image (skipping decode
    # and resize, ~an order of magnitude smaller than caching the decode);
    # families that opt into wants_unresized_image cache the full-resolution
    # decode. Cached reads are byte-identical to fresh ones either way. The
    # flag also enables caching in the per-epoch validation loop. 'disk' is
    # the safest choice with dataloader workers; default is off.
    cache: Union[bool, str] = False
    # Opt-in epoch-length floor for tiny datasets. When min_samples > 0 and
    # the training dataset has fewer images, each epoch draws min_samples
    # samples with replacement instead of one short pass over the dataset, so
    # per-epoch overhead (dataloader spin-up, validation, checkpointing) stops
    # dominating wall-clock time on ~100-image datasets. Under DDP the draw is
    # sharded across ranks (rounded up to a multiple of world_size) and
    # reshuffled per epoch via set_epoch. Dataloader workers are clamped to
    # the dataset length while the floor is active. 0 (the default) is off:
    # behavior is identical to before the knob existed. Only families that
    # build their train loader through the shared create_dataloader honor it.
    min_samples: int = 0
    # Opt-in LVIS-style repeat-factor sampling. Rare classes are oversampled
    # so long-tailed sets (RF100-style) see them every epoch. Off by default:
    # the historical uniform / DistributedSampler is unchanged. Honored by
    # families that build the train loader through create_dataloader.
    class_balanced: bool = False
    # Rolling uniform average of the N best checkpoints ranked by the
    # watched validation metric, written to weights/average.pt at the end
    # of training. 0 (default) is off: best.pt / last.pt are unchanged.
    average_best: int = 0
    # Export the model to ONNX before epoch 1 and fail the run if export
    # breaks. Numeric compare runs only when torch and ONNX layouts match.
    export_check: bool = False
    # Recompute BatchNorm running stats from this many train images after
    # the last epoch (before its validation), and refresh a historical
    # best checkpoint if it remains selected. 0 (default) is off. No-op on
    # LayerNorm-only families.
    precise_bn: int = 0
    patience: int = 50
    resume: bool = False
    log_interval: int = 10
    seed: int = 0
    allow_download_scripts: bool = False
    # Profiling. When ``profile`` is True the trainer profiles a short window of
    # real training steps (``profile_warmup`` discarded, then ``profile_steps``
    # measured), prints a per-phase breakdown + GPU-idle verdict, writes a Chrome
    # trace (open at https://ui.perfetto.dev), then drops the hooks and KEEPS
    # TRAINING. ``profile_then_stop`` instead stops the run right after the
    # window (benchmark mode, what ``libreyolo profile run`` uses); the partial
    # epoch is then neither validated nor checkpointed, so it cannot poison a
    # later resume. Zero overhead when off. Ignored under distributed training.
    profile: bool = False
    profile_then_stop: bool = False
    profile_warmup: int = 5
    profile_steps: int = 20
    profile_trace: bool = True
    profile_open: bool = True

    def __post_init__(self):
        self.amp_dtype = normalize_amp_dtype(self.amp_dtype)
        self.imgsz = normalize_imgsz(
            self.imgsz,
            name="imgsz",
            allow_string=True,
        )
        if self.max_det < 1:
            raise ValueError(f"max_det must be >= 1, got {self.max_det}")
        if self.eval_max_det is not None:
            self.eval_max_det = int(self.eval_max_det)
            if self.eval_max_det < 1:
                raise ValueError(f"eval_max_det must be >= 1, got {self.eval_max_det}")
        self.min_samples = int(self.min_samples)
        if self.min_samples < 0:
            raise ValueError(f"min_samples must be >= 0, got {self.min_samples}")
        self.average_best = int(self.average_best)
        if self.average_best < 0:
            raise ValueError(f"average_best must be >= 0, got {self.average_best}")
        self.precise_bn = int(self.precise_bn)
        if self.precise_bn < 0:
            raise ValueError(f"precise_bn must be >= 0, got {self.precise_bn}")
        self.single_cls = bool(self.single_cls)
        self.class_balanced = bool(self.class_balanced)
        self.export_check = bool(self.export_check)

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Construct config, warning on unknown keys."""
        valid = {f.name for f in fields(cls)}
        unknown = set(kwargs) - valid
        if unknown:
            warnings.warn(
                f"Unknown training config keys (ignored): {sorted(unknown)}",
                stacklevel=2,
            )
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to dict with tuples converted to lists for YAML/checkpoint."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def to_yaml(self, path) -> None:
        """Serialize config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


@dataclass(kw_only=True)
class YOLOXConfig(TrainConfig):
    """YOLOX-specific training defaults."""

    # BatchNorm-heavy pure CNN: sync BN stats across ranks under DDP (same
    # rationale as :class:`YOLO9Config`, issue #484). No-op outside DDP.
    sync_bn: bool = True
    momentum: float = 0.9
    warmup_epochs: int = 5
    warmup_lr_start: float = 0.0
    no_aug_epochs: int = 15
    min_lr_ratio: float = 0.05
    degrees: float = 10.0
    shear: float = 2.0
    mosaic_scale: Tuple[float, float] = (0.1, 2.0)
    mixup_prob: float = 1.0
    ema_decay: float = 0.9998
    name: str = "exp"


@dataclass(kw_only=True)
class YOLOv7Config(YOLOXConfig):
    """YOLOv7 training defaults.

    v7 is anchor-based but trains through the YOLOX-style pipeline (SimOTA
    assignment + mosaic/mixup), so this subclasses :class:`YOLOXConfig` and
    overrides only the real differences: v5/v7-lineage momentum, a shorter
    warmup, and slower EMA. ``sync_bn=True`` is inherited from
    :class:`YOLOXConfig` (v7 is a BatchNorm-heavy pure CNN, same rationale
    as :class:`YOLO9Config`, issue #484).

    Note: unlike YOLOX, the final no-aug epochs run without an L1 refinement
    stage — the v7 SimOTA loss has no raw-offset L1 branch.
    """

    # v7 ships a single size; TrainConfig's "s" default doesn't exist here.
    size: str = "b"
    momentum: float = 0.937
    warmup_epochs: int = 3
    ema_decay: float = 0.9999


@dataclass(kw_only=True)
class YOLO9Config(TrainConfig):
    """YOLOv9-specific training defaults."""

    momentum: float = 0.937
    scheduler: str = "linear"
    warmup_epochs: int = 3
    warmup_lr_start: float = 0.0001
    no_aug_epochs: int = 15
    min_lr_ratio: float = 0.01
    degrees: float = 0.0
    shear: float = 0.0
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    mixup_prob: float = 0.0
    ema_decay: float = 0.9999
    name: str = "yolo9_exp"
    workers: int = 8
    mask_downsample_ratio: int = 4
    # YOLO9 is BatchNorm-heavy. Under multi-GPU DDP the per-rank batch is
    # ``batch // world_size``; without SyncBatchNorm each rank's BN running
    # statistics track only its own small shard, which measurably degrades the
    # converged model versus single-GPU (issue #484). Sync BN across ranks.
    sync_bn: bool = True
    # Per-image ground-truth cap in the train transforms. Dense datasets
    # (e.g. aerial imagery) exceed the historical 100-box default; boxes
    # beyond the cap are silently dropped. 300 matches the MTL/YOLO-NAS
    # recipe and is a training-only change (old checkpoints still load).
    max_labels: int = 300
    # PGI auxiliary-head loss weight. 0 disables the branch. Training-only;
    # inference stays single-head. Resume of a checkpoint without ``aux.*``
    # weights keeps the single-head graph.
    aux_weight: float = 0.25
    # SGD momentum at the start of warmup (MTL LinearL: 0.8 → 0.937).
    warmup_momentum: float = 0.8
    # Letterbox pad for new training. ``None`` inherits the loaded
    # checkpoint stamp, or top-left when the checkpoint is unmarked.
    letterbox_pad: Optional[str] = None
    # Copy-paste instance augmentation (segmentation task only). ``copy_paste``
    # is the per-sample probability (0 disables it); ``copy_paste_mode`` selects
    # the source: "flip" reuses the same sample mirrored, "mixup" pulls a second
    # random sample.
    copy_paste: float = 0.0
    copy_paste_mode: str = "flip"
    # Probability of a random k*90-degree rotation for oriented-box (OBB)
    # training. Off by default; only applied on the OBB path (samples carrying
    # angle targets) and ignored for axis-aligned detection.
    rot90: float = 0.0


@dataclass(kw_only=True)
class YOLO9PoseConfig(YOLO9Config):
    """YOLO9 pose-estimation training defaults."""

    num_classes: int = 1
    num_keypoints: int = 17
    keypoint_dim: int = 3
    oks_sigmas: Optional[List[float]] = None
    pose_weight: float = 12.0
    pose_l1_weight: float = 2.0
    pose_vis_weight: float = 1.0
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    flip_prob: float = 0.5
    hsv_prob: float = 1.0
    affine_prob: float = 0.5
    pose_scale: Tuple[float, float] = (0.75, 1.25)
    pin_memory: bool = False
    prefetch_factor: int = 1
    persistent_workers: bool = True
    decode_scale: int = 1
    eval_interval: int = 1
    name: str = "yolo9_pose_exp"


# Upstream's custom fine-tune configs pin the multi-scale collate's
# base_size_repeat per size (Peterande/D-FINE, configs/dfine/custom/*.yml):
# N disables multi-scale outright (`base_size_repeat: ~`) and smaller models
# get more scale variety. Verified against upstream 2026-08; see issue #675.
DFINE_BASE_SIZE_REPEAT: dict = {"n": None, "s": 20, "m": 6, "l": 4, "x": 3}


def resolve_dfine_base_size_repeat(size, override=None):
    """The multi-scale repeat the D-FINE trainer should hand its collate.

    An explicit ``override`` (``DFINEConfig.base_size_repeat``) wins; otherwise
    the upstream per-size default applies. ``None`` means the collate keeps
    every batch at ``base_size`` (upstream's ``~`` for the N size). Unknown
    sizes fall back to 3, the value that used to be hardcoded for everyone.
    """
    if override is not None:
        return int(override)
    return DFINE_BASE_SIZE_REPEAT.get(str(size).lower(), 3)


@dataclass(kw_only=True)
class DFINEConfig(TrainConfig):
    """D-FINE-specific training defaults.

    Training is a v1 cut: AdamW with no-wd on norms/biases, flat LR with
    warmup + cosine tail, hflip-only aug, no mosaic/mixup. AMP on by
    default: upstream's official training commands all pass ``--use-amp``,
    and the decoder's ±65504 clamp is the FP16-range guard that makes that
    safe (it is not evidence that FP32 is required).
    """

    optimizer: str = "adamw"
    lr0: float = 2e-4
    weight_decay: float = 1e-4

    scheduler: str = "flat_cosine"
    warmup_epochs: int = 2
    warmup_lr_start: float = 1e-6
    no_aug_epochs: int = 4
    min_lr_ratio: float = 0.05

    # No mosaic / no mixup / no color or geometric aug for v1.
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    hsv_prob: float = 0.0
    flip_prob: float = 0.5
    degrees: float = 0.0
    translate: float = 0.0
    shear: float = 0.0

    ema: bool = True
    ema_decay: float = 0.9999
    ema_restart_decay: float = 0.9999

    # D-FINE-specific training knobs (paper-faithful fine-tune defaults).
    backbone_lr_mult: float = 0.5  # upstream's fine-tune recipe uses 0.5×
    clip_max_norm: float = 0.1  # upstream default; 0 disables clipping
    multi_scale: bool = True  # per-batch random resize via DFINEMultiScaleCollate
    # How often the base size appears among the multi-scale collate's choices.
    # Unset (None) resolves to the upstream per-size default via
    # resolve_dfine_base_size_repeat: n disables multi-scale, s 20, m 6, l 4,
    # x 3. Set an int to override for every size.
    base_size_repeat: Optional[int] = None
    aug_stop_epoch_ratio: float = 0.85  # disable strong augs at epoch * ratio
    crop_resize_prob: float = 0.0

    # D-FINE-seg mask supervision (only used when task='segment').
    mask_bce_loss_weight: float = 1.0
    mask_dice_loss_weight: float = 1.0
    mask_match_cost: float = 1.0
    mask_dice_match_cost: float = 1.0

    amp: bool = True
    epochs: int = 132
    name: str = "dfine_exp"


@dataclass(kw_only=True)
class DOMEDETRConfig(DFINEConfig):
    """Dome-DETR fine-tuning defaults.

    Inherits D-FINE's recipe (that is what upstream builds on) and changes only
    what Dome-DETR's own configs change:

    - ``imgsz=800``: every shipped Dome-DETR config evaluates at 800x800.
    - ``lr0``/``weight_decay``/``backbone_lr_mult`` from
      ``configs/dome/Dome-*-*.yml``: backbone at 2e-5 against a 2e-4 base is a
      0.1x multiplier, tighter than D-FINE's 0.5x.
    - ``multi_scale=False``: MWAS requires the stride-8 map to divide evenly by
      the window size, so a random per-batch resize would break the forward.

    Upstream trains 160 epochs with ``MultiStepLR(milestones=[80, 120],
    gamma=0.8)``. That is a from-scratch schedule; these defaults target
    fine-tuning and keep D-FINE's flat-cosine, so reproducing the paper's
    numbers needs the upstream schedule, not this config.
    """

    imgsz: int = 800
    lr0: float = 2e-4
    weight_decay: float = 6.5e-5
    backbone_lr_mult: float = 0.1
    multi_scale: bool = False
    base_size_repeat: Optional[int] = None

    # DeFE supervision weights. Note upstream's DomeCriterion *code* default
    # for defe_density_map_weight is 4, but every shipped config in
    # configs/dome/ overrides it to 1, so 1 is what the released
    # checkpoints were trained with and what reproduces their loss values.
    defe_density_map_weight: float = 1.0
    density_recall_penalty: float = 0.3
    defe_reg_loss_weight: float = 1.0

    # Upstream's dist_train.sh trains WITHOUT --use-amp (unlike D-FINE), so
    # Dome-DETR pins fp32 rather than inheriting D-FINE's AMP default.
    amp: bool = False

    epochs: int = 160
    name: str = "domedetr_exp"


@dataclass(kw_only=True)
class DEIMConfig(TrainConfig):
    """DEIM-D-FINE fine-tuning defaults.

    DEIM keeps the D-FINE HGNetv2 architecture and replaces the classification
    objective with MAL from the Dense O2O recipe. These defaults are for
    practical LibreYOLO fine-tuning, not reproducing DEIM's full COCO training
    recipe. The upstream Mosaic/MixUp schedule is intentionally left for the
    shared augmentation refactor.
    """

    optimizer: str = "adamw"
    # Fine-tune defaults, per the class docstring. Upstream's published COCO
    # recipe uses lr0=4e-4 with min_lr_ratio=0.5 (its lr_gamma) at total batch
    # 32 over 132 epochs; at practical fine-tune batches (8-16) on small
    # datasets that keeps the whole run between 4e-4 and 2e-4 and measurably
    # degrades transfer (aquarium/bccd bench, 2026-07). Pass the recipe values
    # explicitly to reproduce upstream COCO training.
    lr0: float = 1e-4
    weight_decay: float = 1e-4

    scheduler: str = "flat_cosine"
    warmup_epochs: int = 2
    warmup_lr_start: float = 1e-6
    no_aug_epochs: int = 12
    min_lr_ratio: float = 0.05

    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    hsv_prob: float = 0.0
    flip_prob: float = 0.5
    degrees: float = 10.0
    translate: float = 0.1
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 0.0

    ema: bool = True
    ema_decay: float = 0.9999
    ema_restart_decay: float = 0.9999

    backbone_lr_mult: Optional[float] = None
    clip_max_norm: float = 0.1
    multi_scale: bool = True
    aug_stop_epoch_ratio: float = 0.91

    # Upstream's official training commands pass --use-amp, same as D-FINE.
    amp: bool = True
    epochs: int = 132
    name: str = "deim_exp"


@dataclass(kw_only=True)
class RTDETRv4Config(DEIMConfig):
    """RT-DETRv4 fine-tuning defaults."""

    lr0: float = 5e-4
    weight_decay: float = 1.25e-4
    epochs: int = 58
    name: str = "rtdetrv4_exp"


DEIMV2_SIZE_DEFAULTS = {
    # Released DEIMv2 COCO recipes, flattened from /configs/deimv2/*.yml in
    # Intellindust-AI-Lab/DEIMv2. The tiny HGNetv2 models intentionally omit
    # local FGL/DDF loss and disable GO-union matching.
    "atto": {
        "imgsz": 320,
        "epochs": 500,
        "batch": 128,
        "lr0": 2e-3,
        "weight_decay": 1e-4,
        "warmup_iters": 4000,
        "flat_epochs": 250,
        "no_aug_epochs": 32,
        "min_lr_ratio": 0.5,
        "backbone_lr_mult": 0.5,
        "base_size_repeat": None,
        "sanitize_min_size": 12,
        "aug_stop_epoch_ratio": 468 / 500,
        "losses": ("mal", "boxes"),
        "use_uni_set": False,
        "change_matcher": True,
        "iou_order_alpha": 4.0,
        "matcher_change_epoch": 450,
    },
    "femto": {
        "imgsz": 416,
        "epochs": 500,
        "batch": 128,
        "lr0": 1.6e-3,
        "weight_decay": 1e-4,
        "warmup_iters": 4000,
        "flat_epochs": 250,
        "no_aug_epochs": 32,
        "min_lr_ratio": 0.5,
        "backbone_lr_mult": 0.5,
        "base_size_repeat": None,
        "sanitize_min_size": 10,
        "aug_stop_epoch_ratio": 468 / 500,
        "losses": ("mal", "boxes"),
        "use_uni_set": False,
        "change_matcher": True,
        "iou_order_alpha": 4.0,
        "matcher_change_epoch": 450,
    },
    "pico": {
        "imgsz": 640,
        "epochs": 500,
        "batch": 128,
        "lr0": 1.6e-3,
        "weight_decay": 1e-4,
        "warmup_iters": 4000,
        "flat_epochs": 250,
        "no_aug_epochs": 32,
        "min_lr_ratio": 0.5,
        "backbone_lr_mult": 0.5,
        "base_size_repeat": None,
        "sanitize_min_size": 8,
        "aug_stop_epoch_ratio": 468 / 500,
        "losses": ("mal", "boxes"),
        "use_uni_set": False,
        "change_matcher": True,
        "iou_order_alpha": 4.0,
        "matcher_change_epoch": 450,
    },
    "n": {
        "imgsz": 640,
        "epochs": 160,
        "batch": 32,
        "lr0": 8e-4,
        "weight_decay": 1e-4,
        "warmup_iters": 2000,
        # Epoch-scale, like every other size (flat ~= 0.49*epochs). The prior
        # 7800 was the iteration count (160 epochs * ~49 it/ep) mis-placed here.
        "flat_epochs": 78,
        "no_aug_epochs": 12,
        "min_lr_ratio": 1.0,
        "backbone_lr_mult": 0.5,
        "base_size_repeat": None,
        "sanitize_min_size": 1,
        "aug_stop_epoch_ratio": 148 / 160,
        "losses": ("mal", "boxes", "local"),
        "use_uni_set": True,
        "change_matcher": True,
        "iou_order_alpha": 4.0,
        "matcher_change_epoch": 136,
    },
    "s": {
        "imgsz": 640,
        "epochs": 132,
        "batch": 32,
        "lr0": 5e-4,
        "weight_decay": 1e-4,
        "warmup_iters": 2000,
        "flat_epochs": 64,
        "no_aug_epochs": 12,
        "min_lr_ratio": 0.5,
        "backbone_lr_mult": 0.05,
        "base_size_repeat": 20,
        "sanitize_min_size": 1,
        "aug_stop_epoch_ratio": 120 / 132,
        "losses": ("mal", "boxes", "local"),
        "use_uni_set": True,
        "change_matcher": True,
        "iou_order_alpha": 4.0,
        "matcher_change_epoch": 100,
    },
    "m": {
        "imgsz": 640,
        "epochs": 102,
        "batch": 32,
        "lr0": 5e-4,
        "weight_decay": 1e-4,
        "warmup_iters": 2000,
        "flat_epochs": 49,
        "no_aug_epochs": 12,
        "min_lr_ratio": 0.5,
        "backbone_lr_mult": 0.05,
        "base_size_repeat": 6,
        "sanitize_min_size": 1,
        "aug_stop_epoch_ratio": 90 / 102,
        "losses": ("mal", "boxes", "local"),
        "use_uni_set": True,
        "change_matcher": True,
        "iou_order_alpha": 4.0,
        "matcher_change_epoch": 80,
    },
    "l": {
        "imgsz": 640,
        "epochs": 68,
        "batch": 32,
        "lr0": 5e-4,
        "weight_decay": 1.25e-4,
        "warmup_iters": 2000,
        "flat_epochs": 34,
        "no_aug_epochs": 8,
        "min_lr_ratio": 0.5,
        "backbone_lr_mult": 0.025,
        "base_size_repeat": 3,
        "sanitize_min_size": 1,
        "aug_stop_epoch_ratio": 60 / 68,
        "losses": ("mal", "boxes", "local"),
        "use_uni_set": True,
        "change_matcher": True,
        "iou_order_alpha": 4.0,
        "matcher_change_epoch": 50,
    },
    "x": {
        "imgsz": 640,
        "epochs": 58,
        "batch": 32,
        "lr0": 5e-4,
        "weight_decay": 1.25e-4,
        "warmup_iters": 2000,
        "flat_epochs": 29,
        "no_aug_epochs": 8,
        "min_lr_ratio": 0.5,
        "backbone_lr_mult": 0.02,
        "base_size_repeat": 3,
        "sanitize_min_size": 1,
        "aug_stop_epoch_ratio": 50 / 58,
        "losses": ("mal", "boxes", "local"),
        "use_uni_set": True,
        "change_matcher": True,
        "iou_order_alpha": 4.0,
        "matcher_change_epoch": 45,
    },
}


@dataclass(kw_only=True)
class DEIMv2Config(TrainConfig):
    """DEIMv2 fine-tuning defaults.

    DEIMv2 keeps DEIM's Dense O2O training contract but mixes HGNetv2 tiny
    backbones with DINOv3/STAs larger backbones. Size-specific recipes are
    applied by ``DEIMv2Trainer`` from ``DEIMV2_SIZE_DEFAULTS`` so direct Python
    calls can default to the upstream COCO YAML values for each released size.
    Mosaic/MixUp/CopyBlend are still intentionally omitted from the native
    trainer; the train transform follows LibreYOLO's existing DEIM fine-tune
    path with photometric/zoom/crop/hflip plus epoch-aware multi-scale collate.
    """

    optimizer: str = "adamw"
    lr0: float = 5e-4
    weight_decay: float = 1e-4

    scheduler: str = "flat_cosine"
    warmup_epochs: int = 2
    warmup_lr_start: float = 0.0
    no_aug_epochs: int = 12
    min_lr_ratio: float = 0.5

    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    hsv_prob: float = 0.0
    flip_prob: float = 0.5
    degrees: float = 10.0
    translate: float = 0.1
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 0.0

    ema: bool = True
    ema_decay: float = 0.9999
    ema_restart_decay: float = 0.9999

    backbone_lr_mult: Optional[float] = None
    clip_max_norm: float = 0.1
    multi_scale: bool = True
    aug_stop_epoch_ratio: Optional[float] = None
    base_size_repeat: Optional[int] = None
    sanitize_min_size: int = 1

    warmup_iters: Optional[int] = None
    flat_epochs: Optional[int] = None
    change_matcher: Optional[bool] = None
    iou_order_alpha: Optional[float] = None
    matcher_change_epoch: Optional[int] = None
    use_uni_set: Optional[bool] = None
    losses: Optional[Tuple[str, ...]] = None
    reg_max: int = 32

    amp: bool = True
    epochs: int = 132
    batch: int = 32
    name: str = "deimv2_exp"


TINYFORMER_SIZE_DEFAULTS = {
    # Released TinyFormer PBM COCO recipes, flattened from
    # configs/tinyformer/tinyformer_dinov3_*_coco_pbm.yml in
    # mmpmmpmmpjosh/TinyFormer. s/m/l/x reuse DEIMv2's DINO-size recipes
    # verbatim; xl is TinyFormer's ViT-B addition (halved base LR).
    "s": dict(DEIMV2_SIZE_DEFAULTS["s"]),
    "m": dict(DEIMV2_SIZE_DEFAULTS["m"]),
    "l": dict(DEIMV2_SIZE_DEFAULTS["l"]),
    "x": dict(DEIMV2_SIZE_DEFAULTS["x"]),
    "xl": {
        "imgsz": 640,
        "epochs": 58,
        "batch": 32,
        "lr0": 2.5e-4,
        "weight_decay": 1.25e-4,
        "warmup_iters": 2000,
        "flat_epochs": 29,
        "no_aug_epochs": 8,
        "min_lr_ratio": 0.5,
        "backbone_lr_mult": 0.02,
        "base_size_repeat": 3,
        "sanitize_min_size": 1,
        "aug_stop_epoch_ratio": 50 / 58,
        "losses": ("mal", "boxes", "local"),
        "use_uni_set": True,
        "change_matcher": True,
        "iou_order_alpha": 4.0,
        "matcher_change_epoch": 45,
    },
}


@dataclass(kw_only=True)
class TinyFormerConfig(DEIMv2Config):
    """TinyFormer fine-tuning defaults.

    TinyFormer keeps DEIMv2's Dense O2O training contract and DINOv3 optimizer
    grouping; every size runs a DINO-lineage tower, so ImageNet normalisation
    is unconditional. Size-specific recipes are applied by
    ``TinyFormerTrainer`` from ``TINYFORMER_SIZE_DEFAULTS``.
    """

    name: str = "tinyformer_exp"


@dataclass(kw_only=True)
class ECConfig(TrainConfig):
    """EC-specific training defaults.

    Fine-tune defaults keep the optimizer/scheduler/loss shape from
    EdgeCrafter's published recipe (S/M):
    AdamW with backbone-LR multiplier 0.05 (≈2.5e-5 vs head 5e-4), no-decay
    on norms/biases, FlatCosine schedule with quadratic warmup, EMA 0.9999,
    Loss = MAL + L1 + GIoU + FGL + DDF.

    The current LibreYOLO detection trainer uses a per-image D-FINE-style
    pass-through transform with ImageNet normalization. Mosaic/MixUp and the
    strong color/geometric knobs are disabled here so the public config matches
    the effective training path.

    Full real-data fine-tune convergence has not yet been validated.
    """

    optimizer: str = "adamw"
    lr0: float = 5e-4
    weight_decay: float = 1e-4

    scheduler: str = "flat_cosine"
    warmup_epochs: int = 2
    warmup_lr_start: float = 1e-6
    no_aug_epochs: int = 4
    min_lr_ratio: float = 0.5  # EC's lr_gamma in upstream

    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    hsv_prob: float = 0.0
    flip_prob: float = 0.5
    degrees: float = 0.0
    translate: float = 0.0
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 0.0

    ema: bool = True
    ema_decay: float = 0.9999
    ema_restart_decay: float = 0.9999

    # EC-specific knobs.
    backbone_lr_mult: float = 0.05  # 2.5e-5 / 5e-4 ≈ 0.05 for S/M; L/X use 0.01
    clip_max_norm: float = 0.1
    multi_scale: bool = (
        False  # upstream uses fixed 640; multi-scale not in their config
    )
    aug_stop_epoch_ratio: float = 0.97  # stop_epoch=72 with epochs=74 → 72/74

    amp: bool = True
    epochs: int = 74
    name: str = "ec_exp"


@dataclass(kw_only=True)
class ECSegConfig(ECConfig):
    """EC segmentation fine-tune defaults.

    Inherits the EC detect recipe and adds the instance-mask loss knobs. The
    seg data path reuses RF-DETR's square-resize + polygon-rasterization
    transform (no mosaic/mixup), so the mosaic probabilities are forced off.
    Masks are rasterized at ``imgsz`` and the head emits them at
    ``imgsz / mask_downsample_ratio``; point sampling reconciles the two.
    """

    # No mosaic/mixup on the seg path (RFDETRSegPassThroughDataset is a
    # per-sample passthrough — these are ignored, set to 0 for clarity).
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    crop_resize_prob: float = 0.0

    # Mask loss.
    mask_ce_loss_weight: float = 5.0
    mask_dice_loss_weight: float = 5.0
    mask_point_sample_ratio: int = 16
    mask_downsample_ratio: int = 4

    @classmethod
    def from_kwargs(cls, **kwargs):
        cfg = super().from_kwargs(**kwargs)
        size = str(cfg.size).lower()
        if size in {"l", "x"}:
            if "backbone_lr_mult" not in kwargs:
                cfg.backbone_lr_mult = 0.005
            if "weight_decay" not in kwargs:
                cfg.weight_decay = 1.25e-4
        return cfg

    name: str = "ec_seg_exp"


@dataclass(kw_only=True)
class ECPoseConfig(ECConfig):
    """EC (DETR-style) pose fine-tune defaults.

    EdgeCrafter's ECPose is a DETRPose-style keypoint transformer (Hungarian
    matching + OKS). This config carries the keypoint count / sigmas and the
    classification / keypoint-L1 / OKS loss weights. The pose data path owns
    its loader (YOLOPoseDataset + keypoint-aware transforms), so detection-style
    mosaic/multi-scale settings do not apply.
    """

    num_classes: int = 1  # user-facing single class ("person")
    num_keypoints: int = 17
    keypoint_dim: int = 3
    oks_sigmas: Optional[List[float]] = None
    flip_idx: Optional[List[int]] = None

    # Loss weights — DETRPose released recipe (loss_vfl/loss_keypoints/loss_oks).
    cls_loss_weight: float = 2.0
    keypoint_l1_loss_weight: float = 10.0
    oks_loss_weight: float = 4.0

    # Contrastive denoising (DETRPose: dn_number=20, label_noise_ratio=0.5).
    dn_number: int = 20
    label_noise_ratio: float = 0.5

    # Keypoint-aware augmentation (matches the YOLO-pose transform knobs).
    hsv_prob: float = 0.5
    flip_prob: float = 0.5
    brightness_contrast_prob: float = 0.5
    affine_prob: float = 0.75
    degrees: float = 5.0
    translate: float = 0.1
    pose_scale: Tuple[float, float] = (0.75, 1.5)
    affine_interpolation: str = "linear"

    pin_memory: bool = False
    prefetch_factor: int = 1
    persistent_workers: bool = True
    decode_scale: int = 1

    eval_interval: int = 5
    name: str = "ec_pose_exp"


@dataclass(kw_only=True)
class YOLONASConfig(TrainConfig):
    """YOLO-NAS-specific training defaults."""

    # BatchNorm-heavy pure CNN: sync BN stats across ranks under DDP (same
    # rationale as :class:`YOLO9Config`, issue #484). No-op outside DDP.
    sync_bn: bool = True
    optimizer: str = "adamw"
    lr0: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 1e-5
    scheduler: str = "cos"
    warmup_epochs: int = 1
    warmup_lr_start: float = 1e-6
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.1
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.5
    hsv_prob: float = 0.5
    flip_prob: float = 0.5
    degrees: float = 0.0
    translate: float = 0.25
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 0.0
    ema_decay: float = 0.9997
    # Upstream's coco2017_yolo_nas_s.yaml trains with mixed_precision: True.
    amp: bool = True
    name: str = "yolonas_exp"


@dataclass(kw_only=True)
class PPYOLOEConfig(TrainConfig):
    """PP-YOLOE training defaults.

    Optimizer, weight decay, warmup shape, cosine floor, EMA decay and the
    augmentation probabilities follow the released
    ``coco2017_ppyoloe_train_params`` recipe. Two deliberate departures, both
    because ``train()`` is a fine-tune entry point rather than a 500-epoch
    from-scratch run:

    - ``lr0`` defaults well below the source's from-scratch 2e-3 / 1e-3.
      ``libreyolo.models.ppyoloe.trainer.SOURCE_RECIPE_LR0`` records the
      per-size recipe values; pass ``lr0=`` to reproduce them.
    - ``static_assigner_epochs=None`` scales the source's epoch-150-of-500
      ATSS to TaskAligned switch to the requested budget. Set an integer to
      pin it.
    """

    # BatchNorm-heavy pure CNN, same rationale as YOLO-NAS / YOLO9.
    sync_bn: bool = True
    optimizer: str = "adamw"
    lr0: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    scheduler: str = "cos"
    warmup_epochs: int = 1
    warmup_lr_start: float = 1e-6
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.1
    # Source recipe has no mosaic; affine + mixup + flip + HSV + rot90 + BGR swap.
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.5
    hsv_prob: float = 0.5
    flip_prob: float = 0.5
    rot90_prob: float = 0.5
    rgb2bgr_prob: float = 0.25
    degrees: float = 0.0
    translate: float = 0.25
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 0.0
    ema_decay: float = 0.9997
    amp: bool = False
    static_assigner_epochs: Optional[int] = None
    name: str = "ppyoloe_exp"


@dataclass(kw_only=True)
class YOLONASPoseConfig(YOLONASConfig):
    """YOLO-NAS pose-estimation training defaults.

    Mirrors the SuperGradients ``coco2017_yolo_nas_pose`` recipe where it
    applies to a single-GPU fine-tune: AdamW, low weight decay, cosine LR.
    ``num_keypoints`` is resolved from the dataset ``kpt_shape`` by
    ``LibreYOLONAS.train()``. ``oks_sigmas`` may be overridden per dataset;
    when ``None`` the trainer uses the COCO-17 sigmas (17 keypoints) or
    ``1 / num_keypoints`` otherwise.

    best.pt is selected by pose AP when validation is available, so
    ``eval_interval`` defaults to every epoch.
    """

    num_classes: int = 1
    num_keypoints: int = 17
    keypoint_dim: int = 3
    oks_sigmas: Optional[List[float]] = None
    classification_loss_type: str = "focal"
    regression_iou_loss_type: str = "ciou"
    classification_loss_weight: float = 1.0
    iou_loss_weight: float = 2.5
    dfl_loss_weight: float = 0.01
    pose_cls_loss_weight: float = 1.0
    pose_reg_loss_weight: float = 34.0
    pose_classification_loss_type: str = "focal"
    bbox_assigner_topk: int = 13
    bbox_assigned_alpha: float = 1.0
    bbox_assigned_beta: float = 6.0
    assigner_multiply_by_pose_oks: bool = True
    rescale_pose_loss_with_assigned_score: bool = True
    brightness_contrast_prob: float = 0.5
    affine_prob: float = 0.75
    pose_scale: Tuple[float, float] = (0.75, 1.5)
    affine_interpolation: str = "linear"
    pin_memory: bool = False
    prefetch_factor: int = 1
    persistent_workers: bool = True
    decode_scale: int = 1

    lr0: float = 2e-3
    weight_decay: float = 1e-6
    warmup_epochs: int = 10
    min_lr_ratio: float = 0.05
    epochs: int = 1000
    ema_decay: float = 0.997
    amp: bool = True
    eval_interval: int = 1
    name: str = "yolonas_pose_exp"


@dataclass(kw_only=True)
class YOLONASOBBConfig(YOLONASConfig):
    """YOLO-NAS-R (oriented boxes) training defaults.

    Loss weights, assigner top-k, optimizer and schedule follow upstream's
    DOTA recipe at the pinned YOLO-NAS-R commit
    (``recipes/training_hyperparams/default_yolo_nas_r_train_params.yaml``):
    AdamW at 5e-5 with weight decay 3.5e-6, cosine to 0.1 of the initial LR,
    EMA 0.9997, no AMP, assigner top-k 12, and loss weights
    ``classification 2.5 / iou 2.0 / dfl 0.5``.

    Augmentation is flips plus HSV only -- see
    :class:`libreyolo.data.augment.yolonas.YOLONASOBBTrainTransform` for why
    mosaic, mixup and affine are not offered for rotated boxes.
    """

    classification_loss_weight: float = 2.5
    iou_loss_weight: float = 2.0
    dfl_loss_weight: float = 0.5
    bbox_assigner_topk: int = 12
    bbox_assigned_alpha: float = 1.0
    bbox_assigned_beta: float = 6.0
    use_varifocal_loss: bool = True
    max_labels: int = 300

    lr0: float = 5e-5
    weight_decay: float = 3.5e-6
    warmup_epochs: int = 1
    min_lr_ratio: float = 0.1
    epochs: int = 100
    ema_decay: float = 0.9997
    amp: bool = False
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    eval_interval: int = 1
    name: str = "yolonas_obb_exp"


@dataclass(kw_only=True)
class PICODETConfig(TrainConfig):
    """PICODET-specific training defaults.

    Bo's recipe (configs/picodet/picodet_s_320_coco.py):
    - SGD, lr 0.4 (4 GPUs * 0.1) momentum 0.9 weight_decay 4e-5
    - Cosine schedule, 300 epochs, linear warmup 300 iters @ ratio 0.1
    - Pipeline: MinIoURandomCrop -> multiscale Resize -> RandomFlip ->
      PhotoMetricDistortion -> Normalize(ImageNet) -> Pad

    LibreYOLO v1 cut: SGD + cosine + hflip + ImageNet normalise. Multi-scale
    resize and PhotoMetricDistortion are deferred to a follow-up commit
    (skill §6: aim for fine-tune parity, not paper parity).

    lr0 note (issue #566): Bo's 0.4 is the total LR at total batch 512
    (4 GPUs x 128 samples), i.e. ~7.8e-4 per image. Copying 0.1 unscaled at
    the default batch 16 is ~8x that per image and demonstrably destroys a
    COCO-pretrained model within a few epochs (coco128 fine-tune: 0.40 ->
    0.14 mAP at 0.1 vs 0.40 -> 0.49 at 0.01). 0.01 matches the upstream
    per-image rate at the default batch.
    """

    # BatchNorm-heavy pure CNN: sync BN stats across ranks under DDP (same
    # rationale as :class:`YOLO9Config`, issue #484). No-op outside DDP.
    sync_bn: bool = True
    optimizer: str = "sgd"
    lr0: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 4e-5

    scheduler: str = "cos"
    warmup_epochs: int = 1
    warmup_lr_start: float = 0.001
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.0

    # No mosaic/mixup; PICODET doesn't use them.
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    hsv_prob: float = 0.0
    flip_prob: float = 0.5
    degrees: float = 0.0
    shear: float = 0.0
    translate: float = 0.0

    ema_decay: float = 0.9998
    epochs: int = 300
    amp: bool = True
    name: str = "picodet_exp"


@dataclass
class RTMDetConfig(TrainConfig):
    """RTMDet training defaults.

    Upstream recipe (mmdetection/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py):
    - AdamW, lr 0.004 (8 GPUs * 32 batch), weight_decay 0.05
    - Linear warmup 1000 iters from start_factor 1e-5
    - Cosine annealing from epoch 150 to 300, eta_min = 5% of base_lr
    - 300 epochs, last 20 epochs ('stage 2') turn off Mosaic + MixUp
    - Mean=[103.53, 116.28, 123.675] / Std=[57.375, 57.12, 58.395] in BGR
    - paramwise: norm_decay_mult=0, bias_decay_mult=0
    - Cached Mosaic (img_scale=640, max_cached=40) + Cached MixUp (max_cached=20)
    - DynamicSoftLabelAssigner (topk=13)
    - QualityFocalLoss (beta=2.0, weight=1.0) + GIoULoss (weight=2.0)

    Detection training is implemented. Small-dataset fine-tune convergence,
    from-scratch paper parity, multi-GPU behavior, cached Mosaic/MixUp
    throughput, and the strict upstream two-stage pipeline switch have not yet
    been validated.
    """

    # BatchNorm-heavy pure CNN: sync BN stats across ranks under DDP (same
    # rationale as :class:`YOLO9Config`, issue #484). No-op outside DDP.
    sync_bn: bool = True
    optimizer: str = "adamw"
    lr0: float = 0.004
    momentum: float = 0.9  # unused for adamw; kept for TrainConfig compatibility
    weight_decay: float = 0.05

    scheduler: str = "cos"
    warmup_epochs: int = 1  # ~1000 iters at batch 32 / 8GPUs equates to roughly 1 epoch
    warmup_lr_start: float = 4e-8  # 1e-5 * 0.004
    no_aug_epochs: int = 20  # stage-2 epochs without Mosaic+MixUp
    min_lr_ratio: float = 0.05

    mosaic_prob: float = 1.0
    mixup_prob: float = 1.0
    hsv_prob: float = 1.0
    flip_prob: float = 0.5
    degrees: float = 0.0
    shear: float = 0.0
    translate: float = 0.0

    ema_decay: float = 0.9998
    epochs: int = 300
    amp: bool = True
    name: str = "rtmdet_exp"


@dataclass(kw_only=True)
class SegformerConfig(TrainConfig):
    """SegFormer training defaults — the paper / mmsegmentation ADE20K recipe.

    Used both to fine-tune the pretrained (non-commercial) ADE20K checkpoints on
    a new dataset and to train from scratch for unrestricted use; see the family
    NOTICE for the weight licensing.
    Defaults follow SegFormer's ADE20K config: AdamW, backbone base LR 6e-5 with
    the decode head at 10x (SegformerTrainer applies the lr_mult), LayerNorm and
    the Mix-FFN positional conv at weight_decay=0, linear (poly-like) decay, and
    scale-jitter 0.5..2.0 (LibreSegformer.semantic_scale_jitter). Convergence for
    the larger sizes (b3-b5) is unvalidated — see docs/nomenclature.md.
    """

    optimizer: str = "adamw"
    lr0: float = 6e-5
    weight_decay: float = 0.01
    # Decode-head LR multiplier over the backbone base LR (mmseg SegFormer uses
    # 10x). Set to 1.0 for a uniform LR (e.g. to ablate the backbone/head split).
    head_lr_mult: float = 10.0

    scheduler: str = "linear"
    warmup_epochs: int = 5
    warmup_lr_start: float = 1e-6
    min_lr_ratio: float = 0.0

    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    flip_prob: float = 0.5
    degrees: float = 0.0
    translate: float = 0.0
    shear: float = 0.0
    # NOTE: photometric jitter is deliberately NOT declared here. The semantic
    # pipeline builds SemanticDataset directly and never reads config.hsv_prob,
    # so a value here would be silently ignored (it was: the recipe said 0.0
    # while training ran at the dataset's 0.5). The live knob is
    # LibreSegformer.semantic_hsv_prob = 0.0, per the reference recipe.

    ema: bool = True
    ema_decay: float = 0.999
    amp: bool = True

    imgsz: int = 512
    epochs: int = 160
    batch: int = 8
    eval_interval: int = 1

    name: str = "segformer_exp"


@dataclass(kw_only=True)
class PPLiteSegConfig(TrainConfig):
    """PP-LiteSeg training defaults — the source Cityscapes recipe.

    SGD (momentum 0.9, weight decay 5e-4) with the STDC backbone at ``lr0`` and
    every other parameter at ``lr0 * head_lr_mult``; polynomial decay (power
    0.9) after 10 warmup epochs; EMA at 0.9999; mixed precision off, as the
    released recipe specifies. ``lr0=0.01`` is the source value for an
    *effective* batch of 32 (8 per GPU across 4 GPUs) — scale it if you train
    at a different effective batch.

    ``imgsz`` here is the train crop, which is not the validation canvas for
    the 75 sizes: ``LibrePPLiteSeg.train()`` fills it from the size's recipe
    and validation runs at ``semantic_val_imgsz``.
    """

    optimizer: str = "sgd"
    lr0: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    # Non-backbone LR multiplier (source `multiply_head_lr: 10.`).
    head_lr_mult: float = 10.0
    zero_weight_decay_on_bias_and_bn: bool = True

    scheduler: str = "poly"
    poly_power: float = 0.9
    warmup_epochs: int = 10
    warmup_lr_start: float = 0.0
    min_lr_ratio: float = 0.0

    # Edge-attention kernel of the compound loss (source `edge_kernel: 5`).
    edge_kernel: int = 5

    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    flip_prob: float = 0.5
    degrees: float = 0.0
    translate: float = 0.0
    shear: float = 0.0
    # Photometric jitter is family-local (LibrePPLiteSeg.semantic_photometric);
    # SemanticDataset never reads config.hsv_prob, so declaring it here would
    # be a knob that silently does nothing.

    ema: bool = True
    ema_decay: float = 0.9999
    # The released recipe trains in full precision; opting into AMP changes it.
    amp: bool = False

    imgsz: Union[int, Tuple[int, int], List[int], str] = (512, 1024)
    epochs: int = 800
    batch: int = 8
    eval_interval: int = 1

    name: str = "ppliteseg_exp"


@dataclass(kw_only=True)
class UNetConfig(TrainConfig):
    """U-Net training defaults — the mmseg Cityscapes UNet-S5-D16 recipe.

    SGD (momentum 0.9, weight decay 5e-4) with polynomial decay (power 0.9)
    after a short warmup; EMA on; mixed precision off, matching the released
    full-precision Cityscapes schedule. ``lr0=0.01`` is the source value for
    an effective batch of 16 (4 per GPU across 4 GPUs in the 4x4 config) —
    scale it if you train at a different effective batch.

    ``imgsz`` is the train crop (512x1024), sampled with the family's
    ``rescale_crop`` recipe (source rescale 0.5..2.0, random crop, cat_max_ratio
    0.75); validation runs whole frames at ``semantic_val_imgsz`` (1024x2048).
    Auxiliary CE is weighted 0.4 via ``aux_weight``.
    """

    # BatchNorm-heavy CNN: sync BN stats across ranks under DDP.
    sync_bn: bool = True
    optimizer: str = "sgd"
    lr0: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = False
    zero_weight_decay_on_bias_and_bn: bool = False

    scheduler: str = "poly"
    poly_power: float = 0.9
    warmup_epochs: int = 2
    warmup_lr_start: float = 0.0
    min_lr_ratio: float = 0.0

    aux_weight: float = 0.4

    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    flip_prob: float = 0.5
    degrees: float = 0.0
    translate: float = 0.0
    shear: float = 0.0

    ema: bool = True
    ema_decay: float = 0.999
    amp: bool = False

    imgsz: Union[int, Tuple[int, int], List[int], str] = (512, 1024)
    epochs: int = 160
    batch: int = 4
    eval_interval: int = 1

    name: str = "unet_exp"


@dataclass(kw_only=True)
class LingBotVisionConfig(TrainConfig):
    """LingBot-Vision semantic training defaults — the report's linear probe.

    The backbone stays frozen and only the 1x1 dense head trains (AdamW,
    cosine decay), matching the upstream evaluation protocol that produced the
    LibreYOLO-hosted weights. Set ``freeze_backbone=False`` for a full
    fine-tune (expect to lower ``lr0`` accordingly).
    """

    optimizer: str = "adamw"
    lr0: float = 1e-3
    weight_decay: float = 1e-4
    # Freeze the ViT backbone and train the head only (the linear probe).
    freeze_backbone: bool = True

    scheduler: str = "cosine"
    warmup_epochs: int = 1
    warmup_lr_start: float = 1e-6
    min_lr_ratio: float = 0.0

    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    flip_prob: float = 0.5
    degrees: float = 0.0
    translate: float = 0.0
    shear: float = 0.0
    # NOTE: photometric jitter is deliberately NOT declared here; the live knob
    # is LibreLingBotVision.semantic_hsv_prob = 0.0 (see SegformerConfig).

    ema: bool = True
    ema_decay: float = 0.999
    amp: bool = True

    imgsz: int = 512
    epochs: int = 20
    batch: int = 16
    eval_interval: int = 1

    name: str = "lingbotvision_exp"


@dataclass(kw_only=True)
class FOMOConfig(TrainConfig):
    """FOMO point-localizer training defaults."""

    # BatchNorm-heavy pure CNN: sync BN stats across ranks under DDP (same
    # rationale as :class:`YOLO9Config`, issue #484). No-op outside DDP.
    sync_bn: bool = True
    optimizer: str = "adam"
    lr0: float = 3e-4
    weight_decay: float = 0.0

    fg_weight: float = 100.0

    scheduler: str = "cos"
    warmup_epochs: int = 0
    warmup_lr_start: float = 0.0
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.05

    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    hsv_prob: float = 0.0
    flip_prob: float = 0.0
    degrees: float = 0.0
    translate: float = 0.0
    shear: float = 0.0

    ema: bool = False
    amp: bool = False

    epochs: int = 40
    batch: int = 32
    eval_interval: int = 1

    conf_thresholds: Tuple[float, ...] = (0.25, 0.35, 0.50, 0.65, 0.80, 0.90)
    nms_radii: Tuple[int, ...] = (1, 2)
    distance_tolerance: float = 1.5

    name: str = "fomo_exp"
