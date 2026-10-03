"""LibreYOLO9E2E inference and training wrapper.

YOLOv9 end-to-end (NMS-free) variant.  Shares the backbone and neck with
standard YOLOv9 but replaces the detection head with YOLO9E2EHead, which
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
from libreyolo.training.ddp_spawn import ddp_aware

from ...training.callbacks import TrainCallbacks
from ..yolo9.model import LibreYOLO9, _E2E_KEY_MARKERS, _upgraded_keys
from .config import YOLO9E2EConfig
from .nn import LibreYOLO9E2EModel
from ...postprocess.yolo9_e2e import postprocess
from ...training.config import YOLO9Config
from ...validation.preprocessors import YOLO9E2EValPreprocessor

# Use parent's training defaults as the baseline; only the name differs.
_TRAIN_DEFAULTS = YOLO9Config()

_CLASS_OUTPUT_KEY_RE = re.compile(
    r"head\.(one_to_one_class_convs|class_convs)\.\d+\.2\.weight"
)
_ONE_TO_ONE_CLASS_TOWER_HIDDEN_KEY = "head.one_to_one_class_convs.0.0.conv.weight"


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
    SUPPORTED_TASKS = ("detect",)
    TRAIN_CONFIG = YOLO9E2EConfig
    val_preprocessor_class = YOLO9E2EValPreprocessor

    # =====================================================================
    # Registry classmethods
    # =====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Match checkpoints that contain the one-to-one head towers.

        The discriminating tokens are the one-to-one tower names, legacy
        (``one2one_cv2`` / ``one2one_cv3``) or current
        (``one_to_one_anchor_convs`` / ``one_to_one_class_convs``), which
        standard YOLOv9 checkpoints never contain.  This must be checked
        *before* LibreYOLO9.can_load in the registry because E2E checkpoints
        also contain repncspelan / adown / sppelan keys that would otherwise
        cause a false LibreYOLO9 match.
        """
        return any(
            marker in key.lower()
            for key in weights_dict
            if isinstance(key, str)
            for marker in _E2E_KEY_MARKERS
        )

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        for key, tensor in _upgraded_keys(weights_dict).items():
            if _CLASS_OUTPUT_KEY_RE.match(key):
                return int(tensor.shape[0])
        return None

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        """Claim native-keyed E2E dicts only.

        The numbered upstream layout belongs to LibreYOLO9 (its remap converts
        the detection head); a numbered dict that happens to carry a
        one-to-one key must not be passed through raw here, or the
        subclass-wins rule would hand LibreYOLO9's correct claim to a garbage
        E2E wrap.
        """
        from ..yolo9.convert import is_upstream_state_dict

        if is_upstream_state_dict(state_dict):
            return None
        return dict(state_dict) if cls.can_load(state_dict) else None

    # =====================================================================
    # Model lifecycle
    # =====================================================================

    def _init_model(self) -> nn.Module:
        return LibreYOLO9E2EModel(
            config=self.size, reg_max=self.reg_max, nb_classes=self.nb_classes
        )

    # Legacy key renaming (``_prepare_state_dict``) and the class-count
    # rebuild (``_rebuild_for_new_classes``) are inherited from LibreYOLO9:
    # ``upgrade_legacy_key`` also renames the legacy one-to-one towers, and
    # ``YOLO9E2EHead.set_num_classes`` / ``init_bias`` cover both branches.

    def _align_class_towers_for_transfer(self, state_dict: dict) -> None:
        """Match both class-tower sets' hidden width to the checkpoint's."""
        super()._align_class_towers_for_transfer(state_dict)
        if _ONE_TO_ONE_CLASS_TOWER_HIDDEN_KEY in state_dict:
            self._rebuild_one_to_one_class_towers(
                int(state_dict[_ONE_TO_ONE_CLASS_TOWER_HIDDEN_KEY].shape[0])
            )

    def _rebuild_one_to_one_class_towers(self, class_neck: int) -> None:
        """Rebuild the one-to-one class towers at ``class_neck`` hidden width.

        Counterpart of :meth:`LibreYOLO9._rebuild_class_towers` for the second
        branch; no-op when the width already matches.
        """
        head = self.model.head
        towers = head.one_to_one_class_convs
        if int(towers[0][0].conv.weight.shape[0]) == class_neck:
            return
        channels = [int(tower[0].conv.weight.shape[1]) for tower in towers]
        head.one_to_one_class_convs = head.build_class_convs(
            channels, class_neck, self.nb_classes
        )
        head.init_bias()
        head._loss_fn = None
        head.to(next(self.model.parameters()).device)

    # =====================================================================
    # Inference pipeline
    # =====================================================================

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
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
            letterbox_pad=getattr(self, "letterbox_pad", None),
        )

    # =====================================================================
    # Public API
    # =====================================================================

    @ddp_aware()
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
        resume: bool | str | Path = _TRAIN_DEFAULTS.resume,
        amp: bool = _TRAIN_DEFAULTS.amp,
        patience: int = _TRAIN_DEFAULTS.patience,
        allow_download_scripts: bool = False,
        callbacks: TrainCallbacks = None,
        loggers=None,
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
            resume: True resumes the loaded training checkpoint, a path
                resumes that one, with its saved training arguments and run
                directory; explicit arguments override the saved ones.
            amp: Enable automatic mixed precision training.
            patience: Early stopping patience.
            allow_download_scripts: Allow embedded Python in dataset YAML downloads.
            callbacks: Optional training callback or iterable of callbacks.
            loggers: Optional built-in experiment loggers: a registered name,
                a configured logger instance, or an iterable mixing both.

        Returns:
            Training results dict with final_loss, best_mAP50, best_mAP50_95, etc.
        """
        from libreyolo.data import load_data_config

        from .trainer import YOLO9E2ETrainer

        resume_path = self._resume_checkpoint(resume) if resume else None
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
            resume=bool(resume_path),
            amp=amp,
            patience=patience,
            allow_download_scripts=allow_download_scripts,
            callbacks=callbacks,
            loggers=loggers,
            **kwargs,
        )

        if resume_path:
            trainer.setup()
            trainer.resume(resume_path)

        results = trainer.train()

        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self._load_weights(best_ckpt)

        return results


__all__ = ["LibreYOLO9E2E"]
