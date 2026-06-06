"""LibreYOLO9 inference and training wrapper."""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from libreyolo.training.ddp_spawn import ddp_aware
from PIL import Image

from ..base import BaseModel
from ...training.config import YOLO9Config, YOLO9PoseConfig
from ...utils.image_loader import ImageInput
from .nn import LibreYOLO9Model
from .utils import preprocess_image, postprocess
from ...validation.preprocessors import YOLO9ValPreprocessor

# Single source of truth for training defaults
_TRAIN_DEFAULTS = YOLO9Config()


class LibreYOLO9(BaseModel):
    """YOLOv9 model for object detection.

    Args:
        model_path: Path to weights, pre-loaded state_dict, or None for fresh model.
        size: Model size variant ("t", "s", "m", "c").
        reg_max: Regression max value for DFL (default: 16).
        nb_classes: Number of classes (default: 80 for COCO).
        device: Device for inference.

    Example::

        >>> model = LibreYOLO9(model_path="path/to/weights.pt", size="s")
        >>> detections = model(image=image_path, save=True)
    """

    # Class-level metadata
    FAMILY = "yolo9"
    FILENAME_PREFIX = "LibreYOLO9"
    INPUT_SIZES = {"t": 640, "s": 640, "m": 640, "c": 640}
    SUPPORTED_TASKS = ("detect", "segment", "pose")
    TRAIN_CONFIG = YOLO9Config
    val_preprocessor_class = YOLO9ValPreprocessor

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys_lower = [k.lower() for k in weights_dict]
        # Explicitly exclude E2E checkpoints so LibreYOLO9E2E.can_load wins first.
        if any("one2one_cv2" in k or "one2one_cv3" in k for k in keys_lower):
            return False
        return any(
            "repncspelan" in k or "adown" in k or "sppelan" in k for k in keys_lower
        ) or any("backbone.elan" in k or "neck.elan" in k for k in weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        key = "backbone.conv0.conv.weight"
        if key not in weights_dict:
            return None
        first_channel = weights_dict[key].shape[0]
        if first_channel == 16:
            return "t"
        if first_channel == 64:
            return "c"
        if first_channel == 32:
            secondary_key = "backbone.elan1.cv1.conv.weight"
            if secondary_key in weights_dict:
                mid_channel = weights_dict[secondary_key].shape[0]
                if mid_channel == 64:
                    return "s"
                elif mid_channel == 128:
                    return "m"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        for key, tensor in weights_dict.items():
            if re.match(r"head\.cv3\.\d+\.2\.weight", key):
                return tensor.shape[0]
        return None

    @classmethod
    def detect_num_keypoints(cls, weights_dict: dict) -> Optional[int]:
        for key, tensor in weights_dict.items():
            if re.match(r"head\.cv4\.\d+\.2\.weight", key):
                channels = int(tensor.shape[0])
                if channels % 3 == 0:
                    return channels // 3
        return None

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        model_path,
        size: str,
        reg_max: int = 16,
        num_masks: int = 32,
        proto_channels: int = 256,
        num_keypoints: int = 17,
        keypoint_dim: int = 3,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        if kwargs.get("task") == "pose" and nb_classes == 80:
            nb_classes = 1
        self.reg_max = reg_max
        self.num_masks = num_masks
        self.proto_channels = proto_channels
        self.num_keypoints = int(num_keypoints)
        self.keypoint_dim = int(keypoint_dim)
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )
        if self._is_pose and self.nb_classes == 1 and self.names.get(0) == "class_0":
            self.names = {0: "person"}

        if isinstance(model_path, str):
            self._load_weights(model_path)
            if self._is_pose and self.nb_classes != 1:
                self._rebuild_for_new_classes(1)
            if self._is_pose:
                self.names = {0: "person"}

    @property
    def _is_segmentation(self) -> bool:
        return self.task == "segment"

    @property
    def _is_pose(self) -> bool:
        return self.task == "pose"

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return LibreYOLO9Model(
            config=self.size,
            reg_max=self.reg_max,
            nb_classes=self.nb_classes,
            segmentation=self._is_segmentation,
            pose=self._is_pose,
            num_masks=self.num_masks,
            proto_channels=self.proto_channels,
            num_keypoints=self.num_keypoints,
            keypoint_dim=self.keypoint_dim,
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone_conv0": self.model.backbone.conv0,
            "backbone_conv1": self.model.backbone.conv1,
            "backbone_elan1": self.model.backbone.elan1,
            "backbone_down2": self.model.backbone.down2,
            "backbone_elan2": self.model.backbone.elan2,
            "backbone_down3": self.model.backbone.down3,
            "backbone_elan3": self.model.backbone.elan3,
            "backbone_down4": self.model.backbone.down4,
            "backbone_elan4": self.model.backbone.elan4,
            "backbone_spp": self.model.backbone.spp,
            "neck_elan_up1": self.model.neck.elan_up1,
            "neck_elan_up2": self.model.neck.elan_up2,
            "neck_elan_down1": self.model.neck.elan_down1,
            "neck_elan_down2": self.model.neck.elan_down2,
        }

    def _strict_loading(self) -> bool:
        return False

    def _allow_checkpoint_task_mismatch(self, checkpoint_task: str) -> bool:
        return self._is_pose and checkpoint_task == "detect"

    def _adapt_checkpoint_num_classes(
        self,
        ckpt_nc: int | None,
        checkpoint_task: str | None = None,
    ) -> int | None:
        if self._is_pose and checkpoint_task == "detect":
            return self.nb_classes
        return ckpt_nc

    def _filter_incoming_state_dict(
        self,
        state_dict: dict,
        *,
        loaded: dict | None = None,
        checkpoint_task: str | None = None,
    ) -> dict:
        has_pose_head = any(key.startswith("head.cv4.") for key in state_dict)
        if self._is_pose and (checkpoint_task == "detect" or not has_pose_head):
            return {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("head.cv3.")
            }
        return state_dict

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        """Remap legacy 'detect.*' keys to 'head.*' for backward compatibility."""
        remapped = {}
        for key, value in state_dict.items():
            new_key = (
                key.replace("detect.", "head.", 1) if key.startswith("detect.") else key
            )
            remapped[new_key] = value
        if self._is_pose:
            ckpt_k = self.detect_num_keypoints(remapped)
            if ckpt_k is not None and ckpt_k != self.num_keypoints:
                self._rebuild_for_new_keypoints(ckpt_k)
        return remapped

    def _rebuild_for_new_classes(self, new_nc: int):
        """Replace only the final classification layers for different number of classes."""
        self.nb_classes = new_nc
        self.model.nc = new_nc
        detect = self.model.head
        detect.nc = new_nc
        detect.no = new_nc + detect.reg_max * 4

        for seq in detect.cv3:
            old_final = seq[-1]
            in_channels = old_final.weight.shape[1]
            seq[-1] = nn.Conv2d(in_channels, new_nc, 1)

        detect._init_bias()
        detect._loss_fn = None
        if hasattr(detect, "_seg_loss_fn"):
            detect._seg_loss_fn = None
        if hasattr(detect, "_pose_loss_fn"):
            detect._pose_loss_fn = None
        detect.to(next(self.model.parameters()).device)

    def _rebuild_for_new_keypoints(self, new_num_keypoints: int):
        """Replace only YOLO9 pose keypoint prediction layers."""
        new_num_keypoints = int(new_num_keypoints)
        if new_num_keypoints == self.num_keypoints:
            return
        if not hasattr(self.model.head, "replace_num_keypoints"):
            raise RuntimeError("Cannot rebuild keypoints on a non-pose YOLO9 head")
        self.model.head.replace_num_keypoints(new_num_keypoints)
        self.model.num_keypoints = new_num_keypoints
        self.num_keypoints = new_num_keypoints
        self.model.to(next(self.model.parameters()).device)

    def _restore_after_training(self, results: dict) -> None:
        """Reload the saved checkpoint and leave the model ready for inference."""
        checkpoint = None
        for key in ("best_checkpoint", "last_checkpoint"):
            path = results.get(key)
            if path and Path(path).exists():
                checkpoint = str(path)
                break

        if checkpoint is not None:
            self.model_path = checkpoint
            self._load_weights(checkpoint)

        self.model.to(self.device).eval()

    # =========================================================================
    # Inference pipeline
    # =========================================================================

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
        effective_size = input_size if input_size is not None else self._get_input_size()
        tensor, img, size = preprocess_image(
            image, input_size=effective_size, color_format=color_format
        )
        return tensor, img, size, 1.0

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

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
        actual_input_size = kwargs.get("input_size", self._get_input_size())
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            max_det=max_det,
            letterbox=kwargs.get("letterbox", True),
        )

    # =========================================================================
    # Public API
    # =========================================================================

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
        name: str = _TRAIN_DEFAULTS.name,
        exist_ok: bool = _TRAIN_DEFAULTS.exist_ok,
        resume: bool = _TRAIN_DEFAULTS.resume,
        amp: bool = _TRAIN_DEFAULTS.amp,
        patience: int = _TRAIN_DEFAULTS.patience,
        allow_download_scripts: bool = False,
        callbacks=None,
        **kwargs,
    ) -> dict:
        """Train the YOLOv9 model on a dataset.

        Args:
            data: Path to data.yaml file (required).
            epochs: Number of epochs to train.
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
            callbacks: Optional training callback or iterable of callbacks.

        Returns:
            Training results dict with final_loss, best_mAP50, best_mAP50_95, etc.
        """
        from .trainer import YOLO9Trainer
        from libreyolo.data import load_data_config

        try:
            data_config = load_data_config(
                data,
                autodownload=True,
                allow_scripts=allow_download_scripts,
            )
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")
        kpt_shape = data_config.get("kpt_shape")

        if self._is_pose:
            if not kpt_shape or len(kpt_shape) < 1:
                raise ValueError(
                    "YOLO9 pose training requires 'kpt_shape: [num_keypoints, 2|3]' "
                    "in the dataset YAML."
                )
            num_keypoints = int(kpt_shape[0])
            keypoint_dim = int(kpt_shape[1]) if len(kpt_shape) > 1 else 3
            if keypoint_dim not in (2, 3):
                raise ValueError(
                    f"YOLO9 pose training requires keypoint_dim 2 or 3, got {keypoint_dim}."
                )
            yaml_nc = 1 if yaml_nc is None else int(yaml_nc)
            if yaml_nc != 1:
                raise ValueError("YOLO9 pose v1 supports one class: person")
            if yaml_names is None:
                yaml_names = {0: "person"}
            self.keypoint_dim = 3
            if num_keypoints != self.num_keypoints:
                self._rebuild_for_new_keypoints(num_keypoints)

        # If no nc in data.yaml, infer it by counting.
        if yaml_nc is None and yaml_names is not None:
            yaml_nc = len(yaml_names)

        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)

        # Apply custom class names from data config
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

        trainer_cls = YOLO9Trainer
        if self._is_pose:
            from .pose_trainer import YOLO9PoseTrainer

            trainer_cls = YOLO9PoseTrainer

        trainer_kwargs = dict(
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
            callbacks=callbacks,
            **kwargs,
        )
        if self._is_pose:
            trainer_kwargs.update(
                {
                    "num_keypoints": self.num_keypoints,
                    "keypoint_dim": self.keypoint_dim,
                    "oks_sigmas": data_config.get("oks_sigmas"),
                }
            )
        trainer = trainer_cls(**trainer_kwargs)

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreYOLO9('path/to/last.pt', size='t'); model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()

        self._restore_after_training(results)

        return results
