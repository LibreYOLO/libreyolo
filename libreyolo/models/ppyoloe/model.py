"""LibrePPYOLOE: BaseModel subclass wiring PP-YOLOE into the LibreYOLO factory."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...training.callbacks import TrainCallbacks
from ...training.config import PPYOLOEConfig
from ...training.ddp_spawn import ddp_aware
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import PPYOLOEValPreprocessor
from ..base import BaseModel
from .convert import (
    convert_upstream,
    detect_nb_classes_from_state,
    detect_size_from_state,
    is_ppyoloe_state_dict,
    is_upstream_state_dict,
    unwrap_ppyoloe_checkpoint,
)
from .nn import LibrePPYOLOEModel
from .utils import preprocess_image as _ppyoloe_preprocess

_TRAIN_DEFAULTS = PPYOLOEConfig()

_NATIVE_FILENAME_RE = re.compile(r"ppyoloe_([smlx])_coco", re.IGNORECASE)


class LibrePPYOLOE(BaseModel):
    """PP-YOLOE anchor-free detector (s/m/l/x).

    CSPResNet backbone, CSP-PAN neck, Efficient Task-aligned head with
    Efficient Squeeze-and-Excitation attention. The head emits per-class
    logits and four discrete distance distributions; there is no objectness
    output, so class probability is the detection confidence.

    Pretrained COCO weights stay hosted by the source provider. LibreYOLO
    links to that CDN rather than mirroring the files (see
    ``get_download_url``), verifies a pinned digest before the pickle is
    deserialized, and loads it through the repository's restricted loader.

    Examples::

        >>> model = LibreYOLO("LibrePPYOLOEs.pt")
        >>> results = model.predict("image.jpg")

        >>> model = LibrePPYOLOE(size="s")
        >>> model.train(data="coco128.yaml", epochs=10)
    """

    FAMILY = "ppyoloe"
    FILENAME_PREFIX = "LibrePPYOLOE"
    INPUT_SIZES = {"s": 640, "m": 640, "l": 640, "x": 640}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = PPYOLOEConfig
    RESUME_RESTORES_TRAIN_ARGS = True
    val_preprocessor_class = PPYOLOEValPreprocessor

    # Forward is pure tensor work with no host sync (anchors are rebuilt from
    # feature shapes each call, never cached across batch sizes).
    SUPPORTS_CUDA_GRAPH = True

    _CDN_BASE = "https://d2gjn4b69gu75n.cloudfront.net/models"

    # Pinned on 2026-08-10 against the four released files. A changed digest is
    # a stop condition, not a warning: the pickle is third-party and is only
    # deserialized after this matches.
    _CHECKPOINT_SHA256 = {
        "ppyoloe_s_coco.pth": "f58a1a44bdaf66f80180346cf4548be3066fd30e742f329ee1b3a37fb4cdab28",
        "ppyoloe_m_coco.pth": "742978ab9eea199252ca160ff4e55546f7a78d10203511d91d0980c300d31873",
        "ppyoloe_l_coco.pth": "fcd6e7e36c90ef965198e1523e2978ab87e4e10ed75cc2d8a00a690e78b71a31",
        "ppyoloe_x_coco.pth": "ac65ea5f383eb28d69b5cf50ed124cc827f630d73d435f41eea76c3a9c0acf8e",
    }

    # ---- registry --------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return is_ppyoloe_state_dict(weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return detect_size_from_state(weights_dict)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return detect_nb_classes_from_state(weights_dict)

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        """Accept both ``LibrePPYOLOE<size>.pt`` and native ``ppyoloe_<size>_coco.pth``.

        Tensor shapes stay authoritative; this only routes a locally staged
        native file to the right architecture before it is opened.
        """
        size = super().detect_size_from_filename(filename)
        if size is not None:
            return size
        match = _NATIVE_FILENAME_RE.search(filename)
        return match.group(1).lower() if match else None

    @classmethod
    def convert_upstream_state_dict(cls, weights_dict: dict) -> Optional[dict]:
        """Claim released PP-YOLOE tensors for runtime auto-conversion."""
        if not is_upstream_state_dict(weights_dict):
            return None
        return convert_upstream(weights_dict)

    # ---- external weights -------------------------------------------------

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        """Link to the source CDN instead of a LibreYOLO mirror.

        The source repository is Apache-2.0, but the audit for this port found
        no per-artifact license grant authorizing LibreYOLO to rehost the
        released checkpoint files, so they are linked like YOLO-NAS rather than
        mirrored. This is an artifact-evidence decision about the weights and
        says nothing about the license of the code in this family.
        """
        size = cls.detect_size_from_filename(filename)
        if size is None or size not in cls.INPUT_SIZES:
            return None
        return f"{cls._CDN_BASE}/ppyoloe_{size}_coco.pth"

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> Optional[str]:
        return (
            "PP-YOLOE pretrained weights are hosted by the source provider, not "
            f"by LibreYOLO. Fetching {url}. The file is checksum-verified "
            "against a pinned digest before it is loaded. Review the source "
            "model zoo terms before using these weights."
        )

    @classmethod
    def verify_downloaded_file(cls, local_path: str, source_url: str) -> None:
        """Fail closed unless a freshly downloaded native pickle matches its pin."""
        import hashlib
        from urllib.parse import urlparse

        name = Path(urlparse(source_url).path).name
        expected = cls._CHECKPOINT_SHA256.get(name)
        if expected is None:
            Path(local_path).unlink(missing_ok=True)
            raise RuntimeError(
                f"Refusing to auto-load PP-YOLOE checkpoint '{name}': no pinned "
                "checksum is known for it, so this freshly downloaded "
                "third-party pickle cannot be verified before loading. "
                "Download it manually from a source you trust and pass its path."
            )
        digest = hashlib.sha256()
        with open(local_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            Path(local_path).unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for PP-YOLOE checkpoint '{name}': expected "
                f"{expected}, got {actual}. The download was discarded and not "
                "deserialized. Re-audit the artifact provenance before retrying."
            )

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
        return LibrePPYOLOEModel(size=self.size, nb_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "neck": self.model.neck,
            "head": self.model.head,
        }

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        """Accept both lean LibreYOLO states and released native layouts.

        Released files are ``{"net": {"module.<key>": tensor}}``. Unwrapping
        and stripping the single ``module.`` prefix here means the rest of the
        base loader (class-count detection, strict load) works unchanged.
        """
        unwrapped = unwrap_ppyoloe_checkpoint(state_dict)
        if is_upstream_state_dict(unwrapped):
            return convert_upstream(unwrapped)
        return dict(unwrapped)

    def _strict_loading(self) -> bool:
        # The converter keeps the full parameter set and the port materialises
        # every upstream buffer, so a missing or unexpected key is a real
        # architecture mismatch, not tolerable drift.
        return True

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
        return _ppyoloe_preprocess(image, input_size=eff, color_format=color_format)

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        output = self.model(input_tensor)
        # Normalise to the named-slot dict the rest of the stack expects.
        # ``raw_predictions`` is what the validation-loss adapter needs, and it
        # is only present in eager eval (traced graphs return the pair alone).
        if isinstance(output, tuple) and len(output) == 2:
            if isinstance(output[0], tuple):
                boxes, scores = output[0]
                return {
                    "boxes": boxes,
                    "scores": scores,
                    "raw_predictions": output[1],
                }
            if all(isinstance(x, torch.Tensor) for x in output):
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
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        from ...postprocess.ppyoloe import postprocess as _postprocess

        return _postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=kwargs.get("input_size", self.input_size),
            original_size=original_size,
            max_det=max_det,
        )

    # ---- training --------------------------------------------------------

    @ddp_aware()
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
        static_assigner_epochs: int = _TRAIN_DEFAULTS.static_assigner_epochs,
        allow_download_scripts: bool = False,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs: Any,
    ) -> dict:
        """Fine-tune PP-YOLOE on a YOLO-format dataset.

        Follows the source recipe's two-stage assignment: ATSS for the first
        ``static_assigner_epochs`` epochs, then TaskAlignedAssigner. The source
        500-epoch COCO recipe switches at epoch 150; the default here is
        scaled to the requested epoch budget the same way as that recipe
        (30% of total) so short fine-tunes still get both phases. Pass
        ``static_assigner_epochs`` to pin it explicitly.

        Fine-tuning on a dataset with a different class count rebuilds only
        the class-prediction convolutions and keeps every other learned
        weight.

        Args:
            callbacks: Optional training callback or iterable of callbacks.
            loggers: Optional built-in experiment loggers.
        """
        from libreyolo.data import load_data_config

        from .trainer import PPYOLOETrainer

        if imgsz is None:
            imgsz = self.input_size

        try:
            data_config = load_data_config(
                data, autodownload=True, allow_scripts=allow_download_scripts
            )
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")
        if yaml_nc is None and yaml_names is not None:
            yaml_nc = len(yaml_names)
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

        trainer = PPYOLOETrainer(
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
            static_assigner_epochs=static_assigner_epochs,
            allow_download_scripts=allow_download_scripts,
            callbacks=callbacks,
            loggers=loggers,
            **kwargs,
        )

        if resume:
            trainer.setup()
            trainer.resume(self._resume_checkpoint(resume))

        results = trainer.train()
        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self._load_weights(best_ckpt)
        return results
