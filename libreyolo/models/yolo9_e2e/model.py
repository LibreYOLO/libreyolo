"""LibreYOLO9E2E inference and training wrapper.

YOLOv9 end-to-end (NMS-free) variant.  Shares the backbone and neck with
standard YOLOv9 but replaces the detection head with YOLO9E2EDetect, which
adds a one-to-one matching branch alongside the standard one-to-many branch.

Inference uses only the one-to-one branch and applies top-K selection instead
of NMS, making the model deployment-friendly on runtimes that lack an NMS op.

Color space: RGB 0–1 (same as standard YOLOv9).
Sizes: t / s / m / c (same backbone configs as yolo9).
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..yolo9.model import LibreYOLO9
from .config import YOLO9E2EConfig
from .nn import LibreYOLO9E2EModel
from .utils import postprocess
from ...training.config import YOLO9Config
from ...validation.preprocessors import YOLO9E2EValPreprocessor

# Use parent's training defaults as the baseline; only the name differs.
_TRAIN_DEFAULTS = YOLO9Config()


class LibreYOLO9E2E(LibreYOLO9):
    """YOLOv9 model with end-to-end NMS-free training and inference.

    Args:
        model_path: Path to weights, pre-loaded state_dict, or None.
        size: Model size variant ("t", "s", "m", "c").
        reg_max: Regression max for DFL (default: 16).
        nb_classes: Number of classes (default: 80).
        device: Device for inference.

    Example::

        >>> model = LibreYOLO9E2E("LibreYOLO9E2Es.pt", size="s")
        >>> detections = model(image_path, save=True)
    """

    FAMILY = "yolo9_e2e"
    FILENAME_PREFIX = "LibreYOLO9E2E"
    # INPUT_SIZES inherited from LibreYOLO9 (t/s/m/c → 640)
    TRAIN_CONFIG = YOLO9E2EConfig
    val_preprocessor_class = YOLO9E2EValPreprocessor

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Match checkpoints that contain the one-to-one head keys.

        The discriminating tokens are ``one2one_cv2`` and ``one2one_cv3``
        which are unique to the E2E head and absent from standard YOLOv9
        checkpoints.  This must be checked *before* LibreYOLO9.can_load in
        the registry because E2E checkpoints also contain repncspelan / adown /
        sppelan keys that would otherwise cause a false LibreYOLO9 match.
        """
        return any(
            "one2one_cv2" in key or "one2one_cv3" in key for key in weights_dict
        )

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        patterns = (
            r"head\.one2one_cv3\.\d+\.2\.weight",
            r"detect\.one2one_cv3\.\d+\.2\.weight",
            r"head\.cv3\.\d+\.2\.weight",
        )
        for key, tensor in weights_dict.items():
            if any(re.match(pattern, key) for pattern in patterns):
                return tensor.shape[0]
        return None

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return LibreYOLO9E2EModel(
            config=self.size, reg_max=self.reg_max, nb_classes=self.nb_classes
        )

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        """Remap legacy ``detect.*`` head keys to ``head.*``."""
        remapped = {}
        for key, value in state_dict.items():
            new_key = (
                key.replace("detect.", "head.", 1)
                if key.startswith("detect.")
                else key
            )
            remapped[new_key] = value
        return remapped

    def _rebuild_for_new_classes(self, new_nc: int):
        """Replace both class-output branches for a new class count."""
        self.nb_classes = new_nc
        self.model.nc = new_nc
        head = self.model.head
        head.nc = new_nc
        head.no = new_nc + head.reg_max * 4

        for branches in (head.cv3, head.one2one_cv3):
            for seq in branches:
                old_final = seq[-1]
                in_channels = old_final.weight.shape[1]
                seq[-1] = nn.Conv2d(in_channels, new_nc, 1)

        head._init_bias()
        head._init_one2one_bias()
        head._loss_fn = None
        head.to(next(self.model.parameters()).device)

    # =========================================================================
    # Inference pipeline
    # =========================================================================

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        actual_input_size = kwargs.get("input_size", 640)
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            max_det=max_det,
            letterbox=kwargs.get("letterbox", False),
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def train(
        self,
        data: str,
        *,
        epochs: int = _TRAIN_DEFAULTS.epochs,
        batch: int = _TRAIN_DEFAULTS.batch,
        imgsz: int = _TRAIN_DEFAULTS.imgsz,
        lr0: float = _TRAIN_DEFAULTS.lr0,
        optimizer: str = _TRAIN_DEFAULTS.optimizer,
        device: str = "",
        workers: int = _TRAIN_DEFAULTS.workers,
        seed: int = _TRAIN_DEFAULTS.seed,
        project: str = _TRAIN_DEFAULTS.project,
        name: str = "yolo9_e2e_exp",
        exist_ok: bool = _TRAIN_DEFAULTS.exist_ok,
        resume: bool = _TRAIN_DEFAULTS.resume,
        amp: bool = _TRAIN_DEFAULTS.amp,
        patience: int = _TRAIN_DEFAULTS.patience,
        allow_download_scripts: bool = False,
        **kwargs,
    ) -> dict:
        """Train the YOLOv9 E2E model on a dataset.

        Args:
            data: Path to data.yaml file (required).
            epochs: Number of training epochs.
            batch: Batch size.
            imgsz: Input image size.
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
            patience: Early stopping patience.
            allow_download_scripts: Allow embedded Python in dataset YAML downloads.

        Returns:
            Training results dict with final_loss, best_mAP50, best_mAP50_95, etc.
        """
        from libreyolo.data import load_data_config

        from .trainer import YOLO9E2ETrainer

        try:
            data_config = load_data_config(
                data, autodownload=True, allow_scripts=allow_download_scripts
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

        trainer = YOLO9E2ETrainer(
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
                    "model = LibreYOLO9E2E('path/to/last.pt', size='t'); "
                    "model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()

        if Path(results["best_checkpoint"]).exists():
            self._load_weights(results["best_checkpoint"])

        return results


__all__ = ["LibreYOLO9E2E"]
