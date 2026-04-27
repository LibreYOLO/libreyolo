"""LibreYOLO YOLO-NAS wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..base import BaseModel
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import YOLONASValPreprocessor
from .nn import LibreYOLONASModel
from .utils import (
    postprocess,
    preprocess_image,
    unwrap_yolonas_checkpoint,
)


class LibreYOLONAS(BaseModel):
    FAMILY = "yolonas"
    FILENAME_PREFIX = "LibreYOLONAS"
    INPUT_SIZES = {"s": 640, "m": 640, "l": 640}
    val_preprocessor_class = YOLONASValPreprocessor

    _REQUIRED_SIGNATURE_KEYS = (
        "backbone.stem.conv.branch_3x3.conv.weight",
        "backbone.stem.conv.branch_1x1.weight",
        "backbone.stem.conv.rbr_reparam.weight",
        "heads.head1.cls_pred.weight",
        "heads.head1.reg_pred.weight",
    )
    _SIZE_FROM_HEAD_WIDTH = {64: "s", 96: "m", 128: "l"}
    _NUM_CLASSES_KEY = "heads.head1.cls_pred.weight"

    _DECI_CDN_BASE = "https://d2gjn4b69gu75n.cloudfront.net/models"

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return all(key in weights_dict for key in cls._REQUIRED_SIGNATURE_KEYS)

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        # YOLO-NAS weights are under Deci's proprietary license — LibreYOLO
        # links to Deci's public CDN instead of mirroring on its own HF org.
        size = cls.detect_size_from_filename(filename)
        if size is None:
            return None
        return f"{cls._DECI_CDN_BASE}/yolo_nas_{size}_coco.pth"

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        tensor = weights_dict.get(cls._NUM_CLASSES_KEY)
        if tensor is None or tensor.ndim < 2:
            return None
        return cls._SIZE_FROM_HEAD_WIDTH.get(tensor.shape[1])

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        tensor = weights_dict.get(cls._NUM_CLASSES_KEY)
        if tensor is None or tensor.ndim == 0:
            return None
        return int(tensor.shape[0])

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        reg_max: int = 16,
        **kwargs,
    ):
        self.reg_max = reg_max
        if isinstance(model_path, dict):
            model_path = unwrap_yolonas_checkpoint(model_path)
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
        return LibreYOLONASModel(
            config=self.size,
            nb_classes=self.nb_classes,
            reg_max=self.reg_max,
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone_stem": self.model.backbone.stem,
            "backbone_stage1": self.model.backbone.stage1,
            "backbone_stage2": self.model.backbone.stage2,
            "backbone_stage3": self.model.backbone.stage3,
            "backbone_stage4": self.model.backbone.stage4,
            "backbone_context_module": self.model.backbone.context_module,
            "neck1": self.model.neck.neck1,
            "neck2": self.model.neck.neck2,
            "neck3": self.model.neck.neck3,
            "neck4": self.model.neck.neck4,
            "heads": self.model.heads,
        }

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        self.nb_classes = new_nb_classes
        self.model.nc = new_nb_classes
        self.model.heads.replace_num_classes(new_nb_classes)
        self.model.to(self.device)

    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy

        return preprocess_numpy

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
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            if len(output) == 2 and isinstance(output[0], tuple):
                boxes, scores = output[0]
                return {
                    "boxes": boxes,
                    "scores": scores,
                    "raw_predictions": output[1],
                }
            if len(output) == 2 and all(isinstance(x, torch.Tensor) for x in output):
                boxes, scores = output
                return {"boxes": boxes, "scores": scores}
        return output

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        actual_input_size = kwargs.get("input_size", self.input_size)
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            max_det=max_det,
            letterbox=kwargs.get("letterbox", True),
        )

    def _strict_loading(self) -> bool:
        return False

    def _load_weights(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        try:
            loaded = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = unwrap_yolonas_checkpoint(loaded)
            state_dict = self._strip_ddp_prefix(dict(state_dict))
            state_dict = self._prepare_state_dict(state_dict)

            if isinstance(loaded, dict):
                ckpt_family = loaded.get("model_family", "")
                own_family = self._get_model_name()
                if ckpt_family and ckpt_family != own_family:
                    raise RuntimeError(
                        f"Checkpoint was trained with model_family='{ckpt_family}' "
                        f"but is being loaded into '{own_family}'. "
                        f"Use the correct model class for this checkpoint."
                    )

                ckpt_nc = loaded.get("nc")
                if ckpt_nc is not None and ckpt_nc != self.nb_classes:
                    self._rebuild_for_new_classes(int(ckpt_nc))

                ckpt_names = loaded.get("names")
                effective_nc = int(ckpt_nc) if ckpt_nc is not None else self.nb_classes
                if ckpt_names is not None:
                    self.names = self._sanitize_names(ckpt_names, effective_nc)

            self.model.load_state_dict(state_dict, strict=self._strict_loading())
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO-NAS weights from {model_path}: {e}"
            ) from e

    def train(
        self,
        data: str,
        *,
        epochs: int = 300,
        batch: int = 16,
        imgsz: int = 640,
        lr0: float = 5e-4,
        optimizer: str = "AdamW",
        device: str = "",
        workers: int = 8,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "yolonas_exp",
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = False,
        patience: int = 50,
        **kwargs,
    ) -> dict:
        from libreyolo.data import load_data_config

        from .trainer import YOLONASTrainer

        try:
            data_config = load_data_config(data, autodownload=True)
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)

        if seed >= 0:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = YOLONASTrainer(
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
            **kwargs,
        )

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreYOLONAS('path/to/last.pt'); model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))
            return trainer.train()

        results = trainer.train()

        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self.model_path = best_ckpt
            self._load_weights(best_ckpt)

        return results
