"""LibrePicoDet: BaseModel subclass wiring PicoDet into the LibreYOLO factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...training.config import PicoDetConfig
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import PicoDetValPreprocessor
from ..base import BaseModel
from .nn import LibrePicoDetModel
from .utils import postprocess as _picodet_postprocess
from .utils import preprocess_image as _picodet_preprocess


_TRAIN_DEFAULTS = PicoDetConfig()


class LibrePicoDet(BaseModel):
    """PicoDet object detector (s/m/l).

    Inference parity vs Bo396543018/Picodet_Pytorch upstream
    (Apache-2.0, the source of our weights):

    +------+--------------+-----------------+
    | size | ours mAP\\* | upstream mAP    |
    +======+==============+=================+
    | s    | 26.85        | 26.9            |
    +------+--------------+-----------------+
    | m    | 34.23        | 34.2            |
    +------+--------------+-----------------+
    | l    | 40.39        | 40.4            |
    +------+--------------+-----------------+

    \\* COCO val2017 mAP50-95, conf=0.025, iou=0.6, max_per_img=100.
    Forward outputs are bit-equivalent (max_abs_diff ~1e-5) on the same
    input.

    Examples::

        >>> model = LibreYOLO("LibrePicoDets.pt")
        >>> dets = model(image="image.jpg")

        >>> # Inference + train + exports all wired:
        >>> model = LibrePicoDet(size="s")
        >>> model.train(data="coco128.yaml", epochs=10)
    """

    FAMILY = "picodet"
    FILENAME_PREFIX = "LibrePicoDet"
    INPUT_SIZES = {"s": 320, "m": 416, "l": 640}
    TRAIN_CONFIG = PicoDetConfig
    val_preprocessor_class = PicoDetValPreprocessor

    # ---- registry --------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # Tokens unique to PicoDet: shared GFL head + ESNet block list.
        # Avoids matching YOLOX (``head.stems``), YOLOv9, DETR families, etc.
        has_gfl = any("head.gfl_cls" in k for k in weights_dict)
        has_esnet = any("backbone.blocks" in k for k in weights_dict)
        return has_gfl and has_esnet

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        # Distinguish via the first stage's output channels: s=24->96, m=24->128, l=24->160.
        # ``backbone.blocks.0`` is an ESBlockDS; its ``conv_pw_2.conv`` has
        # ``out_channels = mid_channels // 2`` and ``in_channels=24`` (stem out).
        # The unambiguous tell is the neck transformer: ``neck.trans.0.conv.weight``
        # has shape (neck_ch, backbone_c3_ch, 1, 1).
        key = "neck.trans.0.conv.weight"
        if key not in weights_dict:
            return None
        neck_ch = weights_dict[key].shape[0]
        return {96: "s", 128: "m", 160: "l"}.get(int(neck_ch))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "head.gfl_cls.0.weight"
        if key not in weights_dict:
            return None
        # Shared cls/reg head: out_channels = num_classes + 4 * (reg_max + 1).
        # PicoDet uses reg_max=7 -> 32 reg channels. Subtract.
        out_ch = int(weights_dict[key].shape[0])
        nc = out_ch - 32
        return nc if nc > 0 else None

    # ---- init ------------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "s",
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        return LibrePicoDetModel(size=self.size, nb_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone_conv1": self.model.backbone.conv1,
            "backbone_blocks": self.model.backbone.blocks,
            "neck": self.model.neck,
            "head": self.model.head,
        }

    def _strict_loading(self) -> bool:
        # Converted Paddle/Bo checkpoints may carry init_cfg state, EMA buffers,
        # or auxiliary keys we drop. Strict loading would refuse them.
        return False

    # ---- inference -------------------------------------------------------

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
        eff = input_size if input_size is not None else self.input_size
        return _picodet_preprocess(image, input_size=eff, color_format=color_format)

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 100,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        actual_input_size = kwargs.get("input_size", self.input_size)
        return _picodet_postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            max_det=max_det,
        )

    # ---- training --------------------------------------------------------

    def train(
        self,
        data: str,
        *,
        epochs: int = _TRAIN_DEFAULTS.epochs,
        batch: int = _TRAIN_DEFAULTS.batch,
        imgsz: int | None = None,
        lr0: float = _TRAIN_DEFAULTS.lr0,
        optimizer: str = _TRAIN_DEFAULTS.optimizer,
        device: str = "",
        workers: int = _TRAIN_DEFAULTS.workers,
        seed: int = _TRAIN_DEFAULTS.seed,
        project: str = _TRAIN_DEFAULTS.project,
        name: str = _TRAIN_DEFAULTS.name,
        exist_ok: bool = _TRAIN_DEFAULTS.exist_ok,
        pretrained: bool = True,
        resume: bool = _TRAIN_DEFAULTS.resume,
        amp: bool = _TRAIN_DEFAULTS.amp,
        patience: int = _TRAIN_DEFAULTS.patience,
        allow_download_scripts: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Train the PicoDet model on a dataset.

        v1 cut: SGD + cosine LR + hflip + ImageNet normalisation. Bo's full
        upstream pipeline (multiscale resize, MinIoURandomCrop,
        PhotoMetricDistortion) is a known recipe gap and lands in a
        follow-up commit per skill §6's "fine-tune parity, not paper parity"
        guidance.
        """
        from pathlib import Path

        from libreyolo.data import load_data_config

        from .trainer import PicoDetTrainer

        if imgsz is None:
            imgsz = self.input_size

        try:
            data_config = load_data_config(
                data, autodownload=True, allow_scripts=allow_download_scripts,
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

            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = PicoDetTrainer(
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
            patience=patience,
            allow_download_scripts=allow_download_scripts,
            **kwargs,
        )

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibrePicoDet('path/to/last.pt'); model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()
        if Path(results["best_checkpoint"]).exists():
            self._load_weights(results["best_checkpoint"])
        return results
