"""LibreYOLO9 inference and training wrapper."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from libreyolo.training.ddp_spawn import ddp_aware
from PIL import Image

from ...training.callbacks import TrainCallbacks
from ..base import BaseModel
from ...training.config import YOLO9Config
from ...tasks import normalize_task
from ...utils.image_loader import ImageInput
from ...utils.serialization import (
    REQUIRED_CHECKPOINT_METADATA_KEYS,
    load_untrusted_torch_file,
    validate_checkpoint_metadata,
)
from .nn import LibreYOLO9Model
from ...postprocess.yolo9 import postprocess
from .utils import preprocess_image
from ...validation.preprocessors import YOLO9ValPreprocessor

# Single source of truth for training defaults
_TRAIN_DEFAULTS = YOLO9Config()
logger = logging.getLogger(__name__)


def _is_yolo9_aux_key(key: str) -> bool:
    """True for PGI tensors (``aux.*`` neck or ``aux_head.*``)."""
    return str(key).startswith("aux.") or str(key).startswith("aux_head.")


def resolve_aux_weight(aux_weight) -> float:
    """Canonical PGI weight. ``None`` is the recipe default; ``0`` stays off."""
    if aux_weight is None:
        return 0.25
    return float(aux_weight)


class LibreYOLO9(BaseModel):
    """YOLOv9 model for object detection.

    Args:
        model_path: Path to weights, pre-loaded state_dict, or None for fresh model.
        size: Model size variant ("t", "s", "m", "c").
        reg_max: Regression max value for DFL (default: 16).
        nb_classes: Number of classes (default: 80 for COCO).
        device: Device for inference.

    Example::

        >>> model = LibreYOLO9(model_path="path/to/weights.pt", size="s")
        >>> detections = model(image=image_path, save=True)
    """

    # Class-level metadata
    FAMILY = "yolo9"
    FILENAME_PREFIX = "LibreYOLO9"
    INPUT_SIZES = {"t": 640, "s": 640, "m": 640, "c": 640}
    SUPPORTED_TASKS = ("detect",)
    TASK_INPUT_SIZES = {
        "detect": INPUT_SIZES,
    }
    TRAIN_CONFIG = YOLO9Config
    val_preprocessor_class = YOLO9ValPreprocessor
    # The detection forward is pure tensor work with no host sync, so it
    # captures and replays bit-identically (tests/unit/test_cuda_graph.py).
    SUPPORTS_CUDA_GRAPH = True
    # Additional checkpoint model_family values accepted as transfer-learning
    # sources (subclass hook; e.g. yolo9_p2 accepts base yolo9 checkpoints).
    TRANSFER_COMPATIBLE_FAMILIES: tuple = ()

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys_lower = [k.lower() for k in weights_dict]
        # Explicitly exclude E2E checkpoints so LibreYOLO9E2E.can_load wins first.
        if any("one2one_cv2" in k or "one2one_cv3" in k for k in keys_lower):
            return False
        # Explicitly exclude P2 checkpoints so LibreYOLO9P2.can_load wins first.
        if any(
            k.startswith("neck.elan_up3") or k.startswith("neck.elan_down0")
            for k in weights_dict
        ):
            return False
        return any(
            "repncspelan" in k or "adown" in k or "sppelan" in k for k in keys_lower
        ) or any("backbone.elan" in k or "neck.elan" in k for k in weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        key = "backbone.conv0.conv.weight"
        if key not in weights_dict:
            return None
        first_channel = weights_dict[key].shape[0]
        if first_channel == 16:
            return "t"
        if first_channel == 64:
            return "c"
        if first_channel == 32:
            secondary_key = "backbone.elan1.cv1.conv.weight"
            if secondary_key in weights_dict:
                mid_channel = weights_dict[secondary_key].shape[0]
                if mid_channel == 64:
                    return "s"
                elif mid_channel == 128:
                    return "m"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        if "head.linear.weight" in weights_dict:
            return int(weights_dict["head.linear.weight"].shape[0])
        if "head.predict.weight" in weights_dict:
            return int(weights_dict["head.predict.weight"].shape[0])
        for key, tensor in weights_dict.items():
            if re.match(r"head\.cv3\.\d+\.2\.weight", key):
                return tensor.shape[0]
        return None

    @classmethod
    def detect_num_keypoints(cls, weights_dict: dict) -> Optional[int]:
        for key, tensor in weights_dict.items():
            if re.match(r"head\.cv4\.\d+\.2\.weight", key):
                channels = int(tensor.shape[0])
                if channels % 3 == 0:
                    return channels // 3
        return None

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        """Remap upstream numbered-index YOLO9 layouts to native semantic keys.

        Claims only the upstream numbered layout; bare native-keyed dicts keep
        going through the factory's legacy path unchanged.
        """
        from .convert import convert_state_dict, infer_config, is_upstream_state_dict

        if not is_upstream_state_dict(state_dict):
            return None
        config = infer_config(state_dict)
        if config is None:
            return None
        converted, _stats = convert_state_dict(state_dict, config)
        return converted

    @classmethod
    def detect_checkpoint_task(cls, weights_dict: dict) -> Optional[str]:
        """Infer YOLO9 task from task-specific head branches."""
        return None

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        model_path,
        size: str,
        reg_max: int = 16,
        nb_classes: int = 80,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ):
        self.reg_max = reg_max
        # Unmarked checkpoints (LibreYOLO <=1.5) used top-left letterbox.
        # Official MTL conversions stamp ``center``. Never flip unmarked files.
        from ...preprocess.letterbox import DEFAULT_LETTERBOX_PAD

        self.letterbox_pad = DEFAULT_LETTERBOX_PAD
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task,
            **kwargs,
        )
        if isinstance(model_path, str):
            self._load_weights(model_path)

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return LibreYOLO9Model(
            config=self.size,
            reg_max=self.reg_max,
            nb_classes=self.nb_classes,
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone_conv0": self.model.backbone.conv0,
            "backbone_conv1": self.model.backbone.conv1,
            "backbone_elan1": self.model.backbone.elan1,
            "backbone_down2": self.model.backbone.down2,
            "backbone_elan2": self.model.backbone.elan2,
            "backbone_down3": self.model.backbone.down3,
            "backbone_elan3": self.model.backbone.elan3,
            "backbone_down4": self.model.backbone.down4,
            "backbone_elan4": self.model.backbone.elan4,
            "backbone_spp": self.model.backbone.spp,
            "neck_elan_up1": self.model.neck.elan_up1,
            "neck_elan_up2": self.model.neck.elan_up2,
            "neck_elan_down1": self.model.neck.elan_down1,
            "neck_elan_down2": self.model.neck.elan_down2,
        }

    def _strict_loading(self) -> bool:
        return False

    def _validate_loaded_state_dict_for_task(
        self,
        state_dict: dict,
        checkpoint: dict | None = None,
    ) -> None:
        if isinstance(checkpoint, dict) and "letterbox_pad" in checkpoint:
            from ...preprocess.letterbox import normalize_letterbox_pad

            self.letterbox_pad = normalize_letterbox_pad(checkpoint.get("letterbox_pad"))

    def _filter_incoming_state_dict(
        self,
        state_dict: dict,
        *,
        loaded: dict | None = None,
        checkpoint_task: str | None = None,
    ) -> dict:
        """Drop training-only ``aux.*`` keys when the live model is single-head.

        Official conversions keep PGI weights so fine-tunes can load them.
        Inference and old unmarked checkpoints stay on the main head only.
        """
        has_aux = getattr(self.model, "aux", None) is not None
        if has_aux:
            return state_dict
        return {
            k: v
            for k, v in state_dict.items()
            if not _is_yolo9_aux_key(k)
        }

    def _save_extra_metadata(self) -> dict:
        from ...preprocess.letterbox import normalize_letterbox_pad

        return {"letterbox_pad": normalize_letterbox_pad(self.letterbox_pad)}

    def _prepare_state_dict(
        self,
        state_dict: dict,
    ) -> dict:
        """Remap legacy 'detect.*' keys to 'head.*' for backward compatibility."""
        remapped = {}
        for key, value in state_dict.items():
            new_key = (
                key.replace("detect.", "head.", 1) if key.startswith("detect.") else key
            )
            remapped[new_key] = value
        return remapped

    def _rebuild_detect_class_layers(self, detect, new_nc: int) -> None:
        detect.nc = new_nc
        detect.no = new_nc + detect.reg_max * 4
        for seq in detect.cv3:
            old_final = seq[-1]
            in_channels = old_final.weight.shape[1]
            seq[-1] = nn.Conv2d(in_channels, new_nc, 1)
        detect._init_bias()
        detect._loss_fn = None
        detect.to(next(self.model.parameters()).device)

    def _rebuild_for_new_classes(self, new_nc: int):
        """Replace only the final classification layers for a new class count."""
        self.nb_classes = new_nc
        self.model.nc = new_nc
        self._rebuild_detect_class_layers(self.model.head, new_nc)
        aux_head = getattr(self.model, "aux_head", None)
        if aux_head is not None:
            self._rebuild_detect_class_layers(aux_head, new_nc)

    def _rebuild_for_checkpoint_classes(self, new_nc: int, state_dict: dict):
        """Match YOLO9 checkpoints with either COCO-width or scratch class towers."""
        hidden_key = "head.cv3.0.0.conv.weight"
        checkpoint_hidden = (
            int(state_dict[hidden_key].shape[0]) if hidden_key in state_dict else None
        )
        current_hidden = None
        current_state = self.model.state_dict()
        if hidden_key in current_state:
            current_hidden = int(current_state[hidden_key].shape[0])

        if checkpoint_hidden is not None and current_hidden != checkpoint_hidden:
            self.nb_classes = new_nc
            self.names = {i: f"class_{i}" for i in range(new_nc)}
            self.model = self._init_model()
            self.model.to(self.device)
            # Transfer-trained checkpoints keep the source checkpoint's tower
            # width, which a fresh build at new_nc may not reproduce (e.g.
            # yolo9_p2 transfer-initialized from stock yolo9 towers).
            self._align_class_towers_for_transfer(state_dict)
            return

        self._rebuild_for_new_classes(new_nc)

    def _restore_after_training(self, results: dict) -> None:
        """Reload the saved checkpoint and leave the model ready for inference."""
        checkpoint = None
        for key in ("best_checkpoint", "last_checkpoint"):
            path = results.get(key)
            if path and Path(path).exists():
                checkpoint = str(path)
                break

        if checkpoint is not None:
            self.model_path = checkpoint
            self._load_weights(checkpoint)

        self.model.to(self.device).eval()

    def _align_class_towers_for_transfer(self, state_dict: dict) -> None:
        """Match COCO-width class towers before partial transfer loading."""
        hidden_key = "head.cv3.0.0.conv.weight"
        if hidden_key not in state_dict:
            return

        checkpoint_hidden = int(state_dict[hidden_key].shape[0])
        head = self.model.head
        current_state = self.model.state_dict()
        if hidden_key not in current_state:
            return
        current_hidden = int(current_state[hidden_key].shape[0])
        if current_hidden == checkpoint_hidden:
            return

        channels = [int(seq[0].conv.weight.shape[1]) for seq in head.cv3]
        head.cv3 = head._build_class_towers(
            channels,
            checkpoint_hidden,
            self.nb_classes,
        )
        head._class_hidden_channels = checkpoint_hidden
        head.nc = self.nb_classes
        head.no = self.nb_classes + head.reg_max * 4
        head._init_bias()
        head._loss_fn = None
        head.to(next(self.model.parameters()).device)

    def _load_transfer_weights(self, weights: str | Path) -> dict[str, int]:
        """Partially load same-family weights for training initialization."""
        path = Path(self._resolve_weights_path(str(weights)))
        if not path.exists():
            from ...utils.download import download_weights

            download_weights(str(path), self.size)

        if not path.exists():
            raise FileNotFoundError(f"Transfer weights not found at {weights}")

        loaded = load_untrusted_torch_file(
            str(path),
            map_location="cpu",
            context="transfer weights",
        )
        if isinstance(loaded, dict):
            metadata_keys = set(REQUIRED_CHECKPOINT_METADATA_KEYS) - {"model"}
            if metadata_keys & set(loaded):
                metadata_errors = validate_checkpoint_metadata(loaded, strict=False)
                if metadata_errors:
                    raise RuntimeError(
                        "Transfer checkpoint metadata is incomplete: "
                        + "; ".join(metadata_errors)
                    )
            if "letterbox_pad" in loaded:
                from ...preprocess.letterbox import normalize_letterbox_pad

                self.letterbox_pad = normalize_letterbox_pad(loaded.get("letterbox_pad"))

            ckpt_family = loaded.get("model_family", "")
            allowed_families = {
                self._get_model_name(),
                *getattr(self, "TRANSFER_COMPATIBLE_FAMILIES", ()),
            }
            if ckpt_family and ckpt_family not in allowed_families:
                raise RuntimeError(
                    f"Transfer checkpoint model_family='{ckpt_family}' does not "
                    f"match '{self._get_model_name()}'."
                )

            ckpt_task = loaded.get("task")
            if ckpt_task is not None:
                normalized_ckpt_task = normalize_task(ckpt_task)
                if normalized_ckpt_task != self.task and normalized_ckpt_task != "detect":
                    raise RuntimeError(
                        f"Transfer checkpoint task='{normalized_ckpt_task}' is "
                        f"not compatible with task='{self.task}'."
                    )

            if "model" in loaded:
                state_dict = loaded["model"]
            elif "state_dict" in loaded:
                state_dict = loaded["state_dict"]
            else:
                state_dict = loaded
        else:
            state_dict = loaded

        state_dict = self._prepare_state_dict(self._strip_ddp_prefix(state_dict))
        total_tensors = len(state_dict)
        self._align_class_towers_for_transfer(state_dict)

        current = self.model.state_dict()
        matched = {
            key: value
            for key, value in state_dict.items()
            if key in current and current[key].shape == value.shape
        }
        current.update(matched)
        self.model.load_state_dict(current, strict=True)
        self.model.to(self.device)
        return {
            "loaded": len(matched),
            "skipped": max(total_tensors - len(matched), 0),
        }

    def _default_transfer_weights_name(self) -> str:
        """Return the matching detect checkpoint filename for transfer learning."""
        return f"{self.FILENAME_PREFIX}{self.size}{self.WEIGHT_EXT}"

    def _trainer_class(self):
        """Trainer class used by ``train()`` (subclass hook)."""
        from .trainer import YOLO9Trainer

        return YOLO9Trainer

    def _extract_checkpoint_state(self, source: str | Path | dict | None) -> dict:
        """Return the weight dict from a path or already-loaded checkpoint."""
        if source is None:
            return {}
        if isinstance(source, dict):
            loaded = source
        else:
            path = Path(source)
            if not path.exists():
                return {}
            loaded = load_untrusted_torch_file(
                str(path),
                map_location="cpu",
                context="yolo9 aux probe",
            )
        if not isinstance(loaded, dict):
            return {}
        if "model" in loaded and isinstance(loaded["model"], dict):
            state = loaded["model"]
        elif "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            state = loaded["state_dict"]
        else:
            state = loaded
        return self._prepare_state_dict(self._strip_ddp_prefix(state))

    def _maybe_enable_aux_from_path(
        self, source: str | Path | dict | None, aux_weight: float | None = None
    ) -> int:
        """Attach PGI if *source* carries aux tensors. Returns loaded aux count."""
        weight = resolve_aux_weight(aux_weight)
        if weight <= 0 or type(self.model).__name__ != "LibreYOLO9Model":
            return 0
        state = self._extract_checkpoint_state(source)
        if not any(_is_yolo9_aux_key(key) for key in state):
            return 0
        self.model.enable_aux(weight=weight)
        return self._load_aux_tensors(state)

    def _reload_aux_from_path(self, source: str | Path | dict | None) -> int:
        """Load aux tensors into an already-attached PGI branch."""
        if getattr(self.model, "aux", None) is None:
            return 0
        return self._load_aux_tensors(self._extract_checkpoint_state(source))

    def _load_aux_tensors(self, state_dict: dict) -> int:
        if not state_dict:
            return 0
        current = self.model.state_dict()
        matched = {
            key: value
            for key, value in state_dict.items()
            if _is_yolo9_aux_key(key)
            and key in current
            and current[key].shape == value.shape
        }
        if not matched:
            return 0
        current.update(matched)
        self.model.load_state_dict(current, strict=True)
        self.model.to(self.device)
        return len(matched)

    # =========================================================================
    # Inference pipeline
    # =========================================================================

    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy

        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_size = (
            input_size if input_size is not None else self._get_input_size()
        )
        tensor, img, size = preprocess_image(
            image,
            input_size=effective_size,
            color_format=color_format,
            letterbox_pad=self.letterbox_pad,
        )
        return tensor, img, size, 1.0

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        actual_input_size = kwargs.get("input_size", self._get_input_size())
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            max_det=max_det,
            letterbox=kwargs.get("letterbox", True),
            letterbox_pad=self.letterbox_pad,
        )

    def _get_val_preprocessor(self, img_size: int | None = None):
        if img_size is None:
            img_size = self._get_input_size()
        return self.val_preprocessor_class(
            img_size=(img_size, img_size),
            letterbox_pad=self.letterbox_pad,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def get_distill_config(self) -> Dict:
        """Return distillation config derived from this model's architecture.

        The tap points are the three neck FPN outputs (P3, P4, P5) and
        channel dimensions come from ``YOLO9_CONFIGS[size]["head_channels"]``.
        """
        from .nn import YOLO9_CONFIGS

        cfg = YOLO9_CONFIGS[self.size]
        return {
            "tap_points": ["neck.elan_up2", "neck.elan_down1", "neck.elan_down2"],
            "channels": list(cfg["head_channels"]),
            "strides": [8, 16, 32],
        }

    def get_backbone_distill_config(self) -> Dict:
        """Single-scale backbone config for foundation-teacher distillation.

        Used when the teacher is a semantic ViT (e.g. DINOv2) rather than a
        detector: supervision lands on the stride-16 backbone stage (P4,
        ``backbone.elan3``) instead of the detection-specific neck. The teacher
        is single-scale, so this returns one tap point.
        """
        from .nn import YOLO9_CONFIGS

        cfg = YOLO9_CONFIGS[self.size]
        p4_channels = cfg["stages"][1][1]  # elan3 (P4) output width
        return {
            "tap_points": ["backbone.elan3"],
            "channels": [p4_channels],
            "strides": [16],
        }

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = _TRAIN_DEFAULTS.epochs,
        batch: int = _TRAIN_DEFAULTS.batch,
        imgsz: Union[int, Tuple[int, int]] = _TRAIN_DEFAULTS.imgsz,
        lr0: float = _TRAIN_DEFAULTS.lr0,
        optimizer: str = _TRAIN_DEFAULTS.optimizer,
        device: str = "",
        workers: int = _TRAIN_DEFAULTS.workers,
        seed: int = _TRAIN_DEFAULTS.seed,
        project: str = _TRAIN_DEFAULTS.project,
        name: str = _TRAIN_DEFAULTS.name,
        exist_ok: bool = _TRAIN_DEFAULTS.exist_ok,
        resume: bool = _TRAIN_DEFAULTS.resume,
        amp: bool = _TRAIN_DEFAULTS.amp,
        amp_dtype: str = _TRAIN_DEFAULTS.amp_dtype,
        patience: int = _TRAIN_DEFAULTS.patience,
        best_metric: Optional[str] = _TRAIN_DEFAULTS.best_metric,
        allow_download_scripts: bool = False,
        pretrained: bool | str | Path | None = None,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Train the YOLOv9 model on a dataset.

        Args:
            data: Path to data.yaml file (required).
            epochs: Number of epochs to train.
            batch: Batch size.
            imgsz: Input image size. Accepts an int (square) or (height, width) tuple.
            lr0: Initial learning rate.
            optimizer: Optimizer name ('SGD', 'Adam', 'AdamW').
            device: Device to train on ('' = auto-detect).
            workers: Number of dataloader workers.
            seed: Random seed for reproducibility.
            project: Root directory for training runs.
            name: Experiment name.
            exist_ok: If True, overwrite existing experiment directory.
            resume: If True, resume training from checkpoint.
            amp: Enable automatic mixed precision training.
            amp_dtype: CUDA AMP dtype, ``float16`` or ``bfloat16``.
            patience: Early stopping patience.
            best_metric: Metric that picks best.pt and feeds early stopping:
                ``map50-95`` (default), ``map50``, ``map75`` or ``f1``.
            pretrained: Optional training initialization weights. Use True to
                load the matching LibreYOLO9 detect checkpoint for transfer
                learning, or pass a checkpoint path/name.
            callbacks: Optional training callback or iterable of callbacks.
            loggers: Optional built-in experiment loggers: a registered name,
                a configured logger instance, or an iterable mixing both.

        Returns:
            Training results dict with final_loss, best_mAP50, best_mAP50_95, etc.
        """
        from libreyolo.data import load_data_config

        # Seed before adapting the class head. Rebuilding for a dataset with a
        # different class count initializes new parameters, so seeding later
        # would make identical seed= runs start from different heads.
        if seed >= 0:
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if str(device).lower() not in ("cpu", "mps") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        try:
            data_config = load_data_config(
                data,
                autodownload=True,
                allow_scripts=allow_download_scripts,
                single_cls=bool(kwargs.get("single_cls", False)),
            )
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")

        # If no nc in data.yaml, infer it by counting.
        if yaml_nc is None and yaml_names is not None:
            yaml_nc = len(yaml_names)
        if yaml_nc is not None:
            yaml_nc = int(yaml_nc)

        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)

        # Apply custom class names from data config
        if yaml_names is not None:
            if isinstance(yaml_names, list):
                yaml_names = {i: n for i, n in enumerate(yaml_names)}
            self.names = self._sanitize_names(yaml_names, self.nb_classes)

        if resume and pretrained:
            raise ValueError("pretrained transfer cannot be combined with resume=True.")

        # PGI aux is training-only. Attach before trainer.setup() so the
        # optimizer / EMA / DDP see the extra parameters. Resume of a
        # single-head 1.5 checkpoint stays single-head.
        aux_weight = resolve_aux_weight(kwargs.get("aux_weight", _TRAIN_DEFAULTS.aux_weight))
        if type(self.model).__name__ == "LibreYOLO9Model":
            if resume:
                self._maybe_enable_aux_from_path(self.model_path, aux_weight)
            elif aux_weight > 0:
                self.model.enable_aux(weight=aux_weight)
                # Inference load stripped aux.* from official converts; put
                # those PGI tensors back now that the branch exists.
                self._reload_aux_from_path(self.model_path)

        if pretrained:
            transfer_weights: str | Path
            if pretrained is True:
                transfer_weights = self._default_transfer_weights_name()
            else:
                transfer_weights = pretrained
            stats = self._load_transfer_weights(transfer_weights)
            logger.info(
                "Loaded %d transfer tensors from %s; skipped %d incompatible tensors.",
                stats["loaded"],
                transfer_weights,
                stats["skipped"],
            )

        trainer_kwargs = dict(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            optimizer=optimizer.lower(),
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            amp_dtype=amp_dtype,
            patience=patience,
            best_metric=best_metric,
            allow_download_scripts=allow_download_scripts,
            callbacks=callbacks,
            loggers=loggers,
            **kwargs,
        )
        trainer = self._trainer_class()(**trainer_kwargs)

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreYOLO9('path/to/last.pt', size='t'); model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()

        self._restore_after_training(results)

        return results
