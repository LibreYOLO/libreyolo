"""LibreDAMOYOLO: BaseModel subclass wiring DAMO-YOLO into the LibreYOLO factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ...training.config import DAMOYOLOConfig
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import DAMOYOLOValPreprocessor
from ..base import BaseModel
from .builder import build_damoyolo
from .structures import SIZES
from .utils import postprocess_predictions


_TRAIN_DEFAULTS = DAMOYOLOConfig()


class LibreDAMOYOLO(BaseModel):
    """DAMO-YOLO object detector (port of github.com/tinyvision/DAMO-YOLO).

    Examples::

        >>> model = LibreYOLO("LibreDAMOYOLOt.pt")
        >>> dets = model(image="image.jpg")

        >>> model = LibreDAMOYOLO(size="t")
        >>> model.train(data="coco128.yaml", epochs=10, allow_experimental=True)
    """

    FAMILY = "damoyolo"
    FILENAME_PREFIX = "LibreDAMOYOLO"
    INPUT_SIZES = {"t": 640}
    TRAIN_CONFIG = DAMOYOLOConfig
    val_preprocessor_class = DAMOYOLOValPreprocessor

    # ---- registry --------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # Tokens unique to DAMO-YOLO: TinyNAS backbone + GiraffeNeckV2 +
        # GFL ZeroHead. The ``head.gfl_cls.0`` head pattern matches PicoDet
        # too, so we additionally require ``neck.merge_3`` (GiraffeNeckV2's
        # signature node) and ``backbone.block_list`` (TinyNAS).
        has_giraffe = any(k.startswith("neck.merge_3.") for k in weights_dict)
        has_tinynas = any(k.startswith("backbone.block_list.") for k in weights_dict)
        has_gfl = any("head.gfl_cls" in k or "head.gfl_reg" in k for k in weights_dict)
        return has_giraffe and has_tinynas and has_gfl

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        # The neck's merge_3 input dim distinguishes T (96+384=480 → out 384)
        # from S/M/L (different in/out channels). For now only T is wired.
        key = "head.gfl_cls.0.weight"
        if key not in weights_dict:
            return None
        in_ch = int(weights_dict[key].shape[1])
        # T config: head input channel for stride 8 is 64.
        return {64: "t"}.get(in_ch)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # Account for legacy=True: cls_out = num_classes + 1.
        key = "head.gfl_cls.0.weight"
        if key not in weights_dict:
            return None
        out_ch = int(weights_dict[key].shape[0])
        # Try legacy (out_ch - 1) first, then non-legacy.
        for guess in (out_ch - 1, out_ch):
            if guess > 0:
                return guess
        return None

    # ---- init ------------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "t",
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
        # Switch RepConv branches to deploy mode for inference. Doing this
        # eagerly is fine for inference and ONNX export; training builds a
        # fresh model instance via ``_init_model`` on the next ``train()``.
        if self.model is not None:
            self.model.switch_to_deploy()

    def _init_model(self) -> nn.Module:
        if self.size not in SIZES:
            raise ValueError(f"DAMO-YOLO size {self.size!r} not yet ported. Available: {sorted(SIZES)}")
        return build_damoyolo(size=self.size, num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "neck": self.model.neck,
            "head": self.model.head,
        }

    def _strict_loading(self) -> bool:
        # Strict load works for upstream checkpoints; loosen for converted
        # ones that may include extra keys.
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
        """Load + resize an image to model-input space.

        Output tensor is float32 (3, H, W) in RGB, range [0, 255], stretched
        (no keep-ratio) — matching upstream's inference path.
        """
        eff = input_size if input_size is not None else self.input_size

        # Load via PIL (RGB) — mirrors upstream's ``Image.open(...).convert("RGB")``.
        if isinstance(image, (str, Path)):
            pil = Image.open(str(image)).convert("RGB")
            arr = np.asarray(pil)
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
            arr = np.asarray(pil)
        elif isinstance(image, np.ndarray):
            # Heuristic: if 3-channel, assume BGR (cv2 default) and flip.
            arr = image[:, :, ::-1].copy() if image.ndim == 3 and color_format != "rgb" else image.copy()
            pil = Image.fromarray(arr)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        orig_h, orig_w = arr.shape[:2]
        resized = cv2.resize(arr, (eff, eff), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        chw = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
        tensor = torch.from_numpy(chw).unsqueeze(0)  # (1, 3, H, W)
        return tensor, pil, (orig_w, orig_h), 1.0

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
        cls_scores, boxes = output
        preds = postprocess_predictions(
            cls_scores,
            boxes,
            orig_sizes=[original_size],
            input_size=(actual_input_size, actual_input_size),
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
        )[0]
        preds["num_detections"] = int(preds["boxes"].shape[0])
        return preds

    # ---- training --------------------------------------------------------

    def train(
        self,
        data: str,
        *,
        allow_experimental: bool = False,
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
        """Fine-tune DAMO-YOLO on a YOLO-format dataset.

        **EXPERIMENTAL.** Loss + assigner port (GFL + AlignOTA) is bit-faithful
        to upstream and gradients flow through every parameter, but the
        full training schedule (300 epochs + SADA box-level autoaug + EMA
        decay 0.9998) hasn't been validated against upstream's COCO mAP.
        Smoke-tested on a synthetic batch (loss decreases) and intended for
        small-dataset fine-tunes.

        Pass ``allow_experimental=True`` to acknowledge.
        """
        if not allow_experimental:
            raise RuntimeError(
                "DAMO-YOLO training is experimental. Loss + assigner are "
                "bit-faithful to upstream and gradients flow correctly, but "
                "full-COCO training has not been validated. Pass "
                "allow_experimental=True to proceed.\n"
                "Validated: inference parity with upstream, ONNX export, "
                "single-batch loss-decreases gradient flow.\n"
                "Not validated: small-dataset fine-tune convergence, "
                "multi-GPU, SADA augmentation."
            )
        from pathlib import Path

        from libreyolo.data import load_data_config

        from .trainer import DAMOYOLOTrainer

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

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = DAMOYOLOTrainer(
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
                    "model = LibreDAMOYOLO('path/to/last.pt'); model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()
        if Path(results["best_checkpoint"]).exists():
            self._load_weights(results["best_checkpoint"])
        return results
