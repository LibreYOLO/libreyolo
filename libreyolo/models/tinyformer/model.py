"""LibreTinyFormer — BaseModel wrapper for the TinyFormer detection family."""

from __future__ import annotations

import re
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from libreyolo.training.ddp_spawn import ddp_aware

from ...training.callbacks import TrainCallbacks
from ...training.config import TinyFormerConfig
from ...utils.image_loader import ImageInput
from ...utils.serialization import load_untrusted_torch_file
from ...validation.preprocessors import DEIMv2DINOValPreprocessor
from ..base import BaseModel
from .nn import SIZE_CONFIGS, LibreTinyFormerModel, normalize_size
from .utils import (
    postprocess,
    preprocess_image,
    preprocess_numpy,
    unwrap_deim_checkpoint,
)


class LibreTinyFormer(BaseModel):
    """LibreYOLO wrapper for TinyFormer (arXiv:2605.25046).

    TinyFormer is a DEIMv2-derived YOLO-DETR hybrid specialised for tiny
    objects: a Spatial Semantic Adapter (SSA) injects stride-4/8 conv detail
    into the ViT token features, and the Parallel Bi-fusion Module (PBM) neck
    fuses a stride-4 level inside a 4-scale FPN/PAN while the decoder keeps 3
    levels. Every released size is the PBM variant.

    License split (mirrors DEIMv2): the family code is Apache-2.0, but the
    l/x/xl sizes run on vendored DINOv3 towers whose *weights* carry Meta's
    DINOv3 License Agreement; s/m use the DEIMv2-distilled ViT-Tiny towers,
    which are themselves DINOv3-distilled, so the hosted weight repos ship the
    Apache-2.0 + DINOv3 dual license for every size (DEIMv2-s precedent).
    See ``libreyolo/models/deimv2/engine/backbone/dinov3/LICENSE.md``.
    """

    FAMILY = "tinyformer"
    SQUARE_IMGSZ_CALLS = frozenset({"predict", "val"})
    FILENAME_PREFIX = "LibreTinyFormer"
    INPUT_SIZES = {size: int(cfg["input_size"]) for size, cfg in SIZE_CONFIGS.items()}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = TinyFormerConfig
    val_preprocessor_class = DEIMv2DINOValPreprocessor
    TTA_FIXED_SIZE = True  # resizes to a fixed square; multi-scale TTA is a no-op
    IMGSZ_DIVISOR = 32
    # COCO-default weights have no suffix; these dataset-variant suffixes are
    # the official VisDrone finetunes and Objects365->COCO checkpoints.
    WEIGHT_VARIANTS = ("visdrone", "obj2coco")

    @classmethod
    def _validate_imgsz(cls, imgsz: int, *, context: str = "TinyFormer imgsz") -> int:
        imgsz = int(imgsz)
        if imgsz <= 0 or imgsz % cls.IMGSZ_DIVISOR:
            raise ValueError(
                f"{context} must be a positive multiple of "
                f"{cls.IMGSZ_DIVISOR}, got {imgsz}."
            )
        return imgsz

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # The SSA stem + stride-4 projection pair exists only in TinyFormer.
        # DEIMv2 checkpoints carry backbone.sta.* instead; requiring both keys
        # keeps this strict against every DEIM-lineage sibling. Registered
        # before LibreDEIMv2, whose can_load also matches the shared
        # backbone.dinov3.* keys.
        has_sda = any(k.startswith("backbone.sda.") for k in weights_dict)
        has_c1 = any(k.startswith("backbone.proj_c1.") for k in weights_dict)
        return has_sda and has_c1

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        lower = filename.lower()
        # "xl" before the single-char codes (multi-char size precedence).
        m = re.search(r"libretinyformer(xl|[smlx])", lower)
        if m:
            return normalize_size(m.group(1))
        m = re.search(r"tinyformer[-_](xl|[smlx])[-_]", lower)
        if m:
            return normalize_size(m.group(1))
        return None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        key = "decoder.dec_score_head.0.weight"
        if key not in weights_dict:
            return None
        hidden = int(weights_dict[key].shape[1])
        if hidden == 192:
            return "s"
        if hidden == 224:
            return "l"
        if hidden == 256:
            # m / x / xl all decode at 256 wide. XL is the only size on the
            # 768-dim ViT-B tower; x has 6 decoder layers vs m's 4.
            cls_token = weights_dict.get("backbone.dinov3.cls_token")
            if cls_token is not None and int(cls_token.shape[-1]) == 768:
                return "xl"
            n_heads = sum(
                1
                for k in weights_dict
                if re.match(r"decoder\.dec_score_head\.\d+\.weight$", k)
            )
            return "x" if n_heads >= 6 else "m"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "decoder.dec_score_head.0.bias"
        if key in weights_dict:
            return int(weights_dict[key].shape[0])
        return None

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        size = normalize_size(size)
        checkpoint = model_path if isinstance(model_path, dict) else None
        pending_state_dict = None
        if isinstance(model_path, dict):
            pending_state_dict = self._strip_ddp_prefix(
                unwrap_deim_checkpoint(model_path)
            )
            model_path = None
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )
        if checkpoint is not None:
            self._cache_checkpoint_train_config(checkpoint)
        if pending_state_dict is not None:
            self._load_state_dict_checked(pending_state_dict)
            self.model.eval()
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        return LibreTinyFormerModel(config=self.size, nb_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "backbone_sda": self.model.backbone.sda,
            "encoder": self.model.encoder,
            "decoder": self.model.decoder,
            "dec_bbox_head": self.model.decoder.dec_bbox_head,
            "dec_score_head": self.model.decoder.dec_score_head,
        }

    def _get_preprocess_numpy(self):
        return preprocess_numpy

    def _get_val_preprocessor(self, img_size: int | None = None):
        if img_size is None:
            img_size = self._get_input_size()
        img_size = self._validate_imgsz(img_size, context="TinyFormer validation imgsz")
        return DEIMv2DINOValPreprocessor(img_size=(img_size, img_size))

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        effective_size = input_size if input_size is not None else self.input_size
        effective_size = self._validate_imgsz(
            effective_size, context="TinyFormer prediction imgsz"
        )
        return preprocess_image(
            image,
            input_size=effective_size,
            color_format=color_format,
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
        )

    def _strict_loading(self) -> bool:
        return False

    def _load_state_dict_checked(self, state_dict: dict) -> None:
        missing, unexpected = self.model.load_state_dict(
            state_dict, strict=self._strict_loading()
        )
        if unexpected:
            preview = sorted(unexpected)[:10]
            raise RuntimeError(
                f"Unexpected keys when loading TinyFormer weights: {preview}"
                + (f" (+{len(unexpected) - 10} more)" if len(unexpected) > 10 else "")
            )
        # decoder.up / decoder.reg_scale are non-persistent DEIM buffers the
        # released checkpoints never carry.
        ignored_missing = {"decoder.up", "decoder.reg_scale"}
        unresolved_missing = sorted(set(missing) - ignored_missing)
        if unresolved_missing:
            raise RuntimeError(
                f"Missing keys when loading TinyFormer weights: "
                f"{unresolved_missing[:10]}"
                + (
                    f" (+{len(unresolved_missing) - 10} more)"
                    if len(unresolved_missing) > 10
                    else ""
                )
            )

    def _load_weights(self, model_path: str):
        model_path = self._resolve_weights_path(model_path)
        path = Path(model_path)
        download_error = None
        if not path.exists():
            from ...utils.download import download_weights

            try:
                download_weights(model_path, self.size)
            except Exception as exc:
                download_error = exc
        if not path.exists():
            if download_error is not None:
                raise FileNotFoundError(
                    f"TinyFormer weights file not found: {model_path}\n"
                    f"Auto-download failed: {download_error}"
                ) from download_error
            raise FileNotFoundError(f"TinyFormer weights file not found: {model_path}")

        try:
            loaded = load_untrusted_torch_file(
                model_path,
                map_location="cpu",
                context="TinyFormer model weights",
            )
            self._cache_checkpoint_train_config(loaded)
            state_dict = self._strip_ddp_prefix(unwrap_deim_checkpoint(loaded))

            if isinstance(loaded, dict):
                ckpt_family = loaded.get("model_family", "")
                if ckpt_family and ckpt_family != self.FAMILY:
                    raise RuntimeError(
                        f"Checkpoint was trained with model_family='{ckpt_family}' "
                        f"but is being loaded into '{self.FAMILY}'."
                    )
                ckpt_nc = loaded.get("nc")
                if ckpt_nc is not None and ckpt_nc != self.nb_classes:
                    self._rebuild_for_new_classes(int(ckpt_nc))
                ckpt_names = loaded.get("names")
                effective_nc = int(ckpt_nc) if ckpt_nc is not None else self.nb_classes
                if ckpt_names is not None:
                    self.names = self._sanitize_names(ckpt_names, effective_nc)

            self._load_state_dict_checked(state_dict)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TinyFormer weights from {model_path}: {e}"
            ) from e

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: Optional[int] = None,
        batch: Optional[int] = None,
        imgsz: Optional[int] = None,
        lr0: Optional[float] = None,
        device: str = "",
        workers: Optional[int] = None,
        seed: int = 0,
        project: str = "runs/train",
        name: Optional[str] = None,
        exist_ok: bool = False,
        resume: bool = False,
        amp: Optional[bool] = None,
        patience: int = 50,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Fine-tune TinyFormer on a YOLO-format dataset config.

        Args:
            data: Path to the dataset YAML file.
            epochs: Number of epochs to train (None uses the family default).
            batch: Batch size (None uses the family default).
            imgsz: Input image size (None uses the family default).
            lr0: Initial learning rate (None uses the family default).
            device: Device to train on ('' = auto-detect).
            workers: Number of dataloader workers (None uses the family default).
            seed: Random seed for reproducibility.
            project: Root directory for training runs.
            name: Experiment name (None uses the family default).
            exist_ok: If True, overwrite existing experiment directory.
            resume: If True, resume training from the loaded checkpoint.
            amp: Enable automatic mixed precision training (None uses the
                family default).
            patience: Early stopping patience.
            callbacks: Optional training callback or iterable of callbacks.
            loggers: Optional built-in experiment loggers: a registered name,
                a configured logger instance, or an iterable mixing both.
        """
        from libreyolo.data import load_data_config

        from .trainer import TinyFormerTrainer

        kwargs.pop("pretrained", None)
        if imgsz is not None:
            imgsz = self._validate_imgsz(imgsz, context="TinyFormer training imgsz")

        try:
            data_config = load_data_config(
                data,
                autodownload=True,
                single_cls=bool(kwargs.get("single_cls", False)),
            )
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")
        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)
        if yaml_names is not None:
            if isinstance(yaml_names, list):
                yaml_names = {i: n for i, n in enumerate(yaml_names)}
            self.names = self._sanitize_names(yaml_names, self.nb_classes)

        if seed >= 0:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if str(device).lower() not in ("cpu", "mps") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer_kwargs = {
            "model": self.model,
            "wrapper_model": self,
            "size": self.size,
            "num_classes": self.nb_classes,
            "data": data,
            "device": device if device else "auto",
            "seed": seed,
            "project": project,
            "exist_ok": exist_ok,
            "resume": resume,
            "patience": patience,
            "callbacks": callbacks,
            "loggers": loggers,
            **kwargs,
        }
        optional = {
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "lr0": lr0,
            "workers": workers,
            "name": name,
            "amp": amp,
        }
        trainer_kwargs.update({k: v for k, v in optional.items() if v is not None})

        trainer = TinyFormerTrainer(**trainer_kwargs)

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreTinyFormer('path/to/last.pt'); "
                    "model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))
            return trainer.train()

        results = trainer.train()

        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self.model_path = best_ckpt
            self._load_weights(best_ckpt)

        self.model.to(self.device)

        return results
