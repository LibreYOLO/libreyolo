"""LibreYOLO wrapper for U-Net semantic segmentation.

U-Net (Ronneberger, Fischer, Brox, MICCAI 2015) is the encoder-decoder that
made dense biomedical segmentation practical on small labeled sets. This family
ships the OpenMMLab UNet-S5-D16 + FCN-head graph used by the official
Cityscapes checkpoint: same-padded 2D, five encoder stages, downsample 16,
not the 2015 Caffe valid-convolution graph. Inference runs whole frames at the
upstream evaluation canvas (1024x2048); ``weights/parity_unet.py`` proves the
graph bit-identical to the pinned mmseg implementation.

Licensing: the architecture is Apache-2.0 (open-mmlab/mmsegmentation). The
released Cityscapes checkpoint is redistributable but NON-COMMERCIAL under
Cityscapes dataset terms, the same hosting path as PP-LiteSeg. A fine-tune
started from it inherits that term; train from scratch on your own data for
weights free of it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image
from torch import nn

from ...postprocess.unet import postprocess as _postprocess
from ...postprocess.unet import resize_logits
from ...tasks import normalize_task
from ...training.callbacks import TrainCallbacks
from ...training.config import UNetConfig
from ...training.ddp_spawn import ddp_aware
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import load_trusted_torch_file
from ..base.model import BaseModel
from .convert import convert_upstream_unet_state_dict, is_upstream_state_dict
from .nn import SIZE_CONFIGS, STRIDE, LibreUNetNet
from .utils import _input_size_hw, preprocess_numpy

logger = logging.getLogger(__name__)

CITYSCAPES_NAMES: dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

CITYSCAPES_LICENSE_URL = "https://www.cityscapes-dataset.com/license/"
WEIGHT_LICENSE = "Cityscapes dataset terms, non-commercial"

_UNIQUE_KEYS = (
    "backbone.encoder.4.1.convs.1.conv.weight",
    "backbone.decoder.3.upsample.interp_upsample.1.conv.weight",
    "decode_head.convs.0.conv.weight",
    "decode_head.conv_seg.weight",
    "auxiliary_head.convs.0.conv.weight",
)


class LibreUNet(BaseModel):
    """UNet-S5-D16 family for 19-class Cityscapes-style semantic segmentation."""

    FAMILY: ClassVar[str] = "unet"
    FILENAME_PREFIX: ClassVar[str] = "LibreUNet"
    WEIGHT_EXT: ClassVar[str] = ".pt"
    SUPPORTED_TASKS: ClassVar[tuple[str, ...]] = ("semantic",)
    DEFAULT_TASK: ClassVar[str] = "semantic"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    INPUT_SIZES: ClassVar[dict[str, tuple[int, int]]] = {
        size: config["imgsz"] for size, config in SIZE_CONFIGS.items()
    }
    TRAIN_CONFIG: ClassVar[type[UNetConfig]] = UNetConfig

    # Training samples the mmseg Cityscapes recipe: rescale the source frame by
    # a factor in ``rescale_range``, then random-crop ``train_crop`` with
    # cat_max_ratio 0.75. Validation and inference direct-resize the whole
    # frame to ``imgsz`` (identity on Cityscapes' 1024x2048), matching the
    # upstream ``mode='whole'`` test path. Photometric jitter is the shared HSV
    # jitter, not mmseg's PhotoMetricDistortion; see NOTICE.
    semantic_resize_mode: ClassVar[str] = "rescale_crop"
    semantic_imgsz_divisor: ClassVar[int] = STRIDE

    @classmethod
    def _strip_module_prefix(cls, weights_dict: dict) -> dict:
        if any(str(key).startswith("module.") for key in weights_dict):
            return {
                (str(key).removeprefix("module.")): value
                for key, value in weights_dict.items()
            }
        return weights_dict

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys = set(cls._strip_module_prefix(weights_dict))
        return all(token in keys for token in _UNIQUE_KEYS)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> str | None:
        state = cls._strip_module_prefix(weights_dict)
        stem = state.get("backbone.encoder.0.0.convs.0.conv.weight")
        deepest = state.get("backbone.encoder.4.1.convs.1.conv.weight")
        if stem is None or deepest is None:
            return None
        if int(stem.shape[0]) == 64 and int(deepest.shape[0]) == 1024:
            return "s"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> int | None:
        state = cls._strip_module_prefix(weights_dict)
        head = state.get("decode_head.conv_seg.weight")
        if head is None or getattr(head, "ndim", 0) < 1:
            return None
        nc = int(head.shape[0])
        aux = state.get("auxiliary_head.conv_seg.weight")
        if aux is not None and int(aux.shape[0]) != nc:
            raise RuntimeError(
                "U-Net checkpoint is inconsistent: main head predicts "
                f"{nc} classes but auxiliary head predicts {int(aux.shape[0])}."
            )
        return nc

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> dict | None:
        return convert_upstream_unet_state_dict(state_dict)

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> str | None:
        del url
        return (
            f"{Path(filename).name} is a converted U-Net checkpoint trained on "
            "Cityscapes. The Cityscapes license restricts the dataset and its "
            "derivatives, including this checkpoint, to NON-COMMERCIAL use "
            f"({CITYSCAPES_LICENSE_URL}). The restriction applies to this "
            "pretrained checkpoint, not to LibreYOLO's MIT code or the U-Net "
            "architecture. A fine-tune started from this checkpoint inherits "
            "the restriction; train from scratch on your own data for weights "
            "without that term."
        )

    def __init__(
        self,
        model_path=None,
        size: str = "s",
        nb_classes: int = 19,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else "semantic"
        if resolved_task != "semantic":
            raise ValueError(f"LibreUNet supports only task='semantic'; got {task!r}.")
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )
        self.weight_license: str | None = None
        self.weight_license_url: str | None = None
        self.weight_dataset: str | None = None
        self.weight_commercial_use: bool | None = None
        self.model.eval()
        if self.model_path is not None:
            self._load_weights(str(self.model_path))
        elif self.nb_classes == len(CITYSCAPES_NAMES):
            self.names = dict(CITYSCAPES_NAMES)

    def _init_model(self) -> nn.Module:
        return LibreUNetNet(size=self.size, num_classes=self.nb_classes)

    @property
    def semantic_scale_jitter(self) -> tuple[float, float]:
        return tuple(SIZE_CONFIGS[self.size]["rescale_range"])

    @property
    def semantic_train_imgsz(self) -> tuple[int, int]:
        """Source train crop (512x1024)."""
        return tuple(SIZE_CONFIGS[self.size]["train_crop"])

    @property
    def semantic_val_imgsz(self) -> tuple[int, int]:
        """Whole-frame evaluation canvas (1024x2048), not the train crop."""
        return tuple(SIZE_CONFIGS[self.size]["imgsz"])

    def _rebuild_for_new_size(self, new_size: str) -> None:
        if new_size not in SIZE_CONFIGS:
            raise ValueError(
                f"Unknown U-Net size {new_size!r}; expected one of {tuple(SIZE_CONFIGS)}"
            )
        self.size = new_size
        self.input_size = self.INPUT_SIZES[new_size]
        self.model = self._init_model()
        self.model.to(self.device)

    def _rebuild_for_new_classes(self, new_nb_classes: int) -> None:
        self.model.replace_num_classes(int(new_nb_classes))
        self.nb_classes = int(new_nb_classes)
        self.names = {i: f"class_{i}" for i in range(int(new_nb_classes))}
        self.model.to(self.device)

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "decode_head": self.model.decode_head,
            "auxiliary_head": self.model.auxiliary_head,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: int | tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, Image.Image, tuple[int, int], float]:
        effective_res = input_size if input_size is not None else self._get_input_size()
        input_h, input_w = _input_size_hw(effective_res)
        if input_h % STRIDE or input_w % STRIDE:
            raise ValueError(
                f"LibreUNet imgsz={effective_res} must have both sides divisible "
                f"by {STRIDE} (encoder stride product)."
            )
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), (input_h, input_w))
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess_semantic_logits(
        self,
        output: Any,
        original_size: tuple[int, int],
        ratio: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        del ratio, kwargs
        return resize_logits(output, original_size)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> dict:
        return _postprocess(
            output,
            conf_thres,
            iou_thres,
            original_size,
            max_det=max_det,
            **kwargs,
        )

    def _strict_loading(self) -> bool:
        return True

    def _load_weights(self, model_path: str | dict[str, Any]) -> None:
        if isinstance(model_path, str):
            if not Path(model_path).exists():
                from ...utils.download import download_weights

                download_weights(model_path, self.size)
            loaded = load_trusted_torch_file(
                model_path, map_location="cpu", context="U-Net semantic weights"
            )
        else:
            loaded = model_path

        if not isinstance(loaded, dict):
            raise TypeError("LibreUNet checkpoints must be dictionaries")

        ckpt_family = loaded.get("model_family")
        if isinstance(ckpt_family, str) and ckpt_family and ckpt_family != self.FAMILY:
            raise RuntimeError(
                f"Checkpoint was trained with model_family='{ckpt_family}' "
                f"but is being loaded into '{self.FAMILY}'."
            )

        ckpt_task = loaded.get("task")
        if isinstance(ckpt_task, str) and normalize_task(ckpt_task) != "semantic":
            raise RuntimeError(
                f"Checkpoint was trained for task={normalize_task(ckpt_task)!r}, "
                "but LibreUNet is semantic-only."
            )

        if isinstance(loaded.get("model"), dict):
            state = loaded["model"]
        elif isinstance(loaded.get("state_dict"), dict):
            state = loaded["state_dict"]
        else:
            state = loaded
        state = self._strip_module_prefix(state)
        if is_upstream_state_dict(state) and "model_family" not in loaded:
            converted = convert_upstream_unet_state_dict(state)
            if converted is not None:
                state = converted

        ckpt_size = loaded.get("size") or self.detect_size(state)
        if ckpt_size is not None and str(ckpt_size) != self.size:
            self._rebuild_for_new_size(str(ckpt_size))

        ckpt_nc = loaded.get("nc") or self.detect_nb_classes(state)
        if ckpt_nc is not None and int(ckpt_nc) != self.nb_classes:
            self._rebuild_for_new_classes(int(ckpt_nc))

        if not self.can_load(state):
            raise RuntimeError("Checkpoint does not look like a U-Net semantic model.")
        self.model.load_state_dict(state, strict=True)

        ckpt_names = loaded.get("names")
        if ckpt_names is not None:
            self.names = self._sanitize_names(ckpt_names, self.nb_classes)
        elif self.nb_classes == len(CITYSCAPES_NAMES):
            self.names = dict(CITYSCAPES_NAMES)
        self.weight_license = loaded.get("weight_license")
        self.weight_license_url = loaded.get("weight_license_url")
        self.weight_dataset = loaded.get("weight_dataset")
        commercial = loaded.get("weight_commercial_use")
        self.weight_commercial_use = None if commercial is None else bool(commercial)
        self.model.to(self.device).eval()

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = 160,
        batch: int = 4,
        imgsz: int | tuple[int, int] | None = None,
        lr0: float | None = None,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "unet_exp",
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = False,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Train U-Net with the mmseg Cityscapes-style CE + auxiliary recipe."""
        from .trainer import UNetTrainer

        train_imgsz = imgsz if imgsz is not None else self.semantic_train_imgsz
        train_h, train_w = _input_size_hw(train_imgsz)
        if train_h % STRIDE or train_w % STRIDE:
            raise ValueError(
                f"U-Net training imgsz={train_imgsz!r} must have both sides "
                f"divisible by {STRIDE}."
            )

        train_kwargs = dict(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=(train_h, train_w),
            size=self.size,
            num_classes=self.nb_classes,
            device=device,
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            **kwargs,
        )
        if lr0 is not None:
            train_kwargs["lr0"] = lr0

        trainer = UNetTrainer(
            model=self.model,
            wrapper_model=self,
            callbacks=callbacks,
            loggers=loggers,
            **train_kwargs,
        )
        result = trainer.train()
        self._restore_after_training(result)
        return result

    def _restore_after_training(self, result: dict) -> None:
        checkpoint = None
        for key in ("best_checkpoint", "last_checkpoint"):
            path = result.get(key)
            if path:
                checkpoint = path
                break
        if checkpoint:
            self._load_weights(str(checkpoint))


__all__ = ["CITYSCAPES_NAMES", "WEIGHT_LICENSE", "LibreUNet"]
