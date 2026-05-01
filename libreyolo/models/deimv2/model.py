"""LibreDEIMv2 — BaseModel wrapper for the DEIMv2 detection family."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput
from ...utils.serialization import load_untrusted_torch_file
from ...validation.preprocessors import DEIMValPreprocessor, ECDetValPreprocessor
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
    FILENAME_PREFIX = "LibreDEIMv2"
    INPUT_SIZES = {size: int(cfg["input_size"]) for size, cfg in SIZE_CONFIGS.items()}
    TRAIN_CONFIG = None
    val_preprocessor_class = DEIMValPreprocessor

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return any(
            "swish_ffn" in k
            or k.startswith("backbone.dinov3.")
            or k.startswith("backbone.sta.")
            for k in weights_dict
        )

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        lower = filename.lower()
        m = re.search(
            r"(?:libredeimv2|deimv2_(?:hgnetv2|dinov3)_)(atto|femto|pico|[nsmlx])",
            lower,
        )
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

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        size = cls.detect_size_from_filename(filename)
        if size is None:
            return None
        token = {"atto": "Atto", "femto": "Femto", "pico": "Pico"}.get(
            size, size.upper()
        )
        name = f"{cls.FILENAME_PREFIX}{token}"
        return f"https://huggingface.co/LibreYOLO/{name}/resolve/main/{name}.pt"

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        size = normalize_size(size)
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

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _get_val_preprocessor(self, img_size: int | None = None):
        if img_size is None:
            img_size = self._get_input_size()
        cls = ECDetValPreprocessor if self.size in DINO_SIZES else DEIMValPreprocessor
        return cls(img_size=(img_size, img_size))

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        effective_size = input_size if input_size is not None else self.input_size
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
            raise RuntimeError(
                f"Unexpected keys when loading DEIMv2 weights: {sorted(unexpected)[:10]}"
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
        if not Path(model_path).exists():
            raise FileNotFoundError(f"DEIMv2 weights file not found: {model_path}")

        try:
            if Path(model_path).suffix == ".safetensors":
                self._load_safetensors_weights(model_path)
                return

            loaded = load_untrusted_torch_file(
                model_path,
                map_location="cpu",
                context="DEIMv2 model weights",
            )
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

            self._load_state_dict_checked(state_dict)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DEIMv2 weights from {model_path}: {e}"
            ) from e
