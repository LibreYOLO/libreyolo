"""LibreDEIMv2 — BaseModel wrapper for the DEIMv2 detection family."""

from __future__ import annotations

from functools import partial
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from libreyolo.training.ddp_spawn import ddp_aware

from ...training.callbacks import TrainCallbacks
from ...training.config import DEIMv2Config
from ...utils.image_loader import ImageInput
from ...utils.serialization import load_untrusted_torch_file
from ...validation.preprocessors import (
    DEIMv2DINOValPreprocessor,
    DEIMv2ValPreprocessor,
)
from ..base import BaseModel
from .nn import DINO_SIZES, SIZE_CONFIGS, LibreDEIMv2Model, normalize_size
from .utils import (
    postprocess,
    preprocess_image,
    preprocess_numpy,
    unwrap_deim_checkpoint,
)


class LibreDEIMv2(BaseModel):
    """LibreYOLO wrapper for DEIMv2.

    The released DEIMv2 family has mixed backbones:
    HGNetv2 for atto/femto/pico/n and DINOv3/ViT-derived backbones for s/m/l/x.
    """

    FAMILY = "deimv2"
    SQUARE_IMGSZ_CALLS = frozenset({"predict", "val"})
    FILENAME_PREFIX = "LibreDEIMv2"
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    INPUT_SIZES = {size: int(cfg["input_size"]) for size, cfg in SIZE_CONFIGS.items()}
    TRAIN_CONFIG = DEIMv2Config
    val_preprocessor_class = DEIMv2ValPreprocessor
    TTA_FIXED_SIZE = True  # resizes to a fixed square; multi-scale TTA is a no-op
    IMGSZ_DIVISOR = 32
    # Same graph-safety argument as D-FINE (DEIMv2 shares the DEIM/D-FINE
    # decoder line): fixed eval_spatial_size, lazily built anchors settle in
    # capture warmup, static top-k. Captures and replays bit-identically
    # (tests/unit/test_cuda_graph_detr_families.py).
    SUPPORTS_CUDA_GRAPH = True

    @classmethod
    def _validate_imgsz(cls, imgsz: int, *, context: str = "DEIMv2 imgsz") -> int:
        imgsz = int(imgsz)
        if imgsz <= 0 or imgsz % cls.IMGSZ_DIVISOR:
            raise ValueError(
                f"{context} must be a positive multiple of "
                f"{cls.IMGSZ_DIVISOR}, got {imgsz}."
            )
        return imgsz

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # TinyFormer (a DEIMv2 derivative) shares the backbone.dinov3.* keys
        # but carries its own SSA markers (backbone.sda./backbone.proj_c1.);
        # routing between the two families is enforced here in both directions
        # (LibreTinyFormer.can_load requires those markers), not by registry
        # order — importing models.tinyformer pulls in models.deimv2 first, so
        # DEIMv2 registers earlier regardless of import position.
        if any(k.startswith("backbone.sda.") for k in weights_dict):
            return False
        return any(
            "swish_ffn" in k
            or k.startswith("backbone.dinov3.")
            or k.startswith("backbone.sta.")
            for k in weights_dict
        )

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        lower = filename.lower()
        m = re.search(r"libredeimv2(atto|femto|pico|[nsmlx])", lower)
        if m:
            return normalize_size(m.group(1))
        m = re.search(r"deimv2_hgnetv2_(atto|femto|pico|n)", lower)
        if m:
            return normalize_size(m.group(1))
        m = re.search(r"deimv2_dinov3_([smlx])", lower)
        if m:
            return normalize_size(m.group(1))
        return None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        key = "decoder.dec_score_head.0.weight"
        if key not in weights_dict:
            return None
        hidden = int(weights_dict[key].shape[1])
        if hidden == 64:
            return "atto"
        if hidden == 96:
            return "femto"
        if hidden == 112:
            return "pico"
        if hidden == 128:
            return "n"
        if hidden == 192:
            return "s"
        if hidden == 224:
            return "l"
        if hidden == 256:
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
            pending_state_dict = self._prepare_state_dict(
                unwrap_deim_checkpoint(model_path), size
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
        return LibreDEIMv2Model(config=self.size, nb_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        layers = {
            "backbone": self.model.backbone,
            "encoder": self.model.encoder,
            "decoder": self.model.decoder,
            "dec_bbox_head": self.model.decoder.dec_bbox_head,
            "dec_score_head": self.model.decoder.dec_score_head,
        }
        if hasattr(self.model.backbone, "sta"):
            layers["backbone_sta"] = self.model.backbone.sta
        return layers

    def _get_preprocess_numpy(self):
        return partial(preprocess_numpy, imagenet_norm=self.size in DINO_SIZES)

    def _get_val_preprocessor(self, img_size: int | None = None):
        if img_size is None:
            img_size = self._get_input_size()
        img_size = self._validate_imgsz(img_size, context="DEIMv2 validation imgsz")
        cls = (
            DEIMv2DINOValPreprocessor
            if self.size in DINO_SIZES
            else DEIMv2ValPreprocessor
        )
        return cls(img_size=(img_size, img_size))

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        effective_size = input_size if input_size is not None else self.input_size
        effective_size = self._validate_imgsz(
            effective_size, context="DEIMv2 prediction imgsz"
        )
        return preprocess_image(
            image,
            input_size=effective_size,
            color_format=color_format,
            imagenet_norm=self.size in DINO_SIZES,
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
        """Fine-tune DEIMv2 on a YOLO-format dataset config.

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

        from .trainer import DEIMv2Trainer

        kwargs.pop("pretrained", None)
        if imgsz is not None:
            imgsz = self._validate_imgsz(imgsz, context="DEIMv2 training imgsz")

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

        trainer = DEIMv2Trainer(**trainer_kwargs)

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreDEIMv2('path/to/last.pt'); "
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

    @staticmethod
    def _expand_shared_head_aliases(state_dict: dict, size: str) -> dict:
        """Restore safetensors-dropped aliases for shared DEIMv2 heads.

        HF safetensors does not preserve duplicate state-dict entries for
        shared modules. Upstream Atto/Femto/Pico share the bbox head across
        decoder layers, so only ``dec_bbox_head.0`` is serialized even though
        PyTorch's ModuleList expects one key namespace per layer.
        """
        cfg = SIZE_CONFIGS[size]["decoder"]
        state_dict = dict(state_dict)

        def expand_head(head_name: str, shared: bool) -> None:
            if not shared:
                return
            num_layers = int(cfg["num_layers"])
            prefix0 = f"decoder.{head_name}.0."
            aliases = [
                (key, value)
                for key, value in state_dict.items()
                if key.startswith(prefix0)
            ]
            for layer_idx in range(1, num_layers):
                for key, value in aliases:
                    alias_key = key.replace(
                        f"decoder.{head_name}.0.",
                        f"decoder.{head_name}.{layer_idx}.",
                        1,
                    )
                    state_dict.setdefault(alias_key, value)

        expand_head("dec_bbox_head", bool(cfg.get("share_bbox_head", False)))
        expand_head("dec_score_head", bool(cfg.get("share_score_head", False)))
        return state_dict

    @classmethod
    def _prepare_state_dict(cls, state_dict: dict, size: str) -> dict:
        return cls._expand_shared_head_aliases(
            cls._strip_ddp_prefix(dict(state_dict)), size
        )

    def _load_state_dict_checked(self, state_dict: dict) -> None:
        missing, unexpected = self.model.load_state_dict(
            state_dict, strict=self._strict_loading()
        )
        if unexpected:
            preview = sorted(unexpected)[:10]
            raise RuntimeError(
                f"Unexpected keys when loading DEIMv2 weights: {preview}"
                + (f" (+{len(unexpected) - 10} more)" if len(unexpected) > 10 else "")
            )

        ignored_missing = {"decoder.up", "decoder.reg_scale"}
        unresolved_missing = sorted(set(missing) - ignored_missing)
        if unresolved_missing:
            raise RuntimeError(
                f"Missing keys when loading DEIMv2 weights: {unresolved_missing[:10]}"
                + (
                    f" (+{len(unresolved_missing) - 10} more)"
                    if len(unresolved_missing) > 10
                    else ""
                )
            )

    def _load_safetensors_weights(self, model_path: str) -> None:
        try:
            from safetensors.torch import load_model as load_safetensors_model
        except ImportError as e:
            raise ImportError(
                "Loading DEIMv2 safetensors requires safetensors. "
                "Install with: pip install safetensors"
            ) from e

        missing, unexpected = load_safetensors_model(
            self.model,
            model_path,
            strict=True,
            device="cpu",
        )
        if missing or unexpected:
            raise RuntimeError(
                "Failed to load DEIMv2 safetensors exactly: "
                f"missing={sorted(missing)[:10]}, unexpected={sorted(unexpected)[:10]}"
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
                    f"DEIMv2 weights file not found: {model_path}\n"
                    f"Auto-download failed: {download_error}"
                ) from download_error
            raise FileNotFoundError(f"DEIMv2 weights file not found: {model_path}")

        try:
            if Path(model_path).suffix == ".safetensors":
                self._cache_checkpoint_train_config({})
                self._load_safetensors_weights(model_path)
                return

            loaded = load_untrusted_torch_file(
                model_path,
                map_location="cpu",
                context="DEIMv2 model weights",
            )
            self._cache_checkpoint_train_config(loaded)
            state_dict = unwrap_deim_checkpoint(loaded)
            state_dict = self._prepare_state_dict(state_dict, self.size)

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

            # Replay LoRA injection for adapter checkpoints. A model trained
            # with lora=True saves its transformer Linears under peft keys
            # (``.base_layer.``/``lora_A``/``lora_B``); rebuild the adapted
            # graph before loading so those keys line up.
            from ...training.lora import (
                apply_lora_to_deimv2,
                module_has_lora,
                state_dict_has_lora,
            )

            if state_dict_has_lora(state_dict) and not module_has_lora(self.model):
                apply_lora_to_deimv2(self.model)

            self._load_state_dict_checked(state_dict)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DEIMv2 weights from {model_path}: {e}"
            ) from e
