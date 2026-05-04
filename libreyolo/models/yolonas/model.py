"""LibreYOLO YOLO-NAS wrapper (detect + pose)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..base import BaseModel
from ...tasks import normalize_task
from ...utils.image_loader import ImageInput
from ...utils.serialization import load_untrusted_torch_file
from ...validation.preprocessors import YOLONASValPreprocessor
from .nn import LibreYOLONASModel, LibreYOLONASPoseModel
from .utils import (
    postprocess,
    postprocess_pose,
    preprocess_image,
    unwrap_yolonas_checkpoint,
)

_POSE_HEAD_KEY = "heads.head1.pose_pred.weight"


class LibreYOLONAS(BaseModel):
    FAMILY = "yolonas"
    FILENAME_PREFIX = "LibreYOLONAS"
    INPUT_SIZES = {"s": 640, "m": 640, "l": 640}
    POSE_INPUT_SIZES = {"n": 640, "s": 640, "m": 640, "l": 640}
    SUPPORTED_TASKS = ("detect", "pose")
    DEFAULT_TASK = "detect"
    TASK_INPUT_SIZES = {
        "detect": INPUT_SIZES,
        "pose": POSE_INPUT_SIZES,
    }
    POSE_NUM_KEYPOINTS = 17
    val_preprocessor_class = YOLONASValPreprocessor

    _REQUIRED_SIGNATURE_KEYS = (
        "backbone.stem.conv.branch_3x3.conv.weight",
        "backbone.stem.conv.branch_1x1.weight",
        "backbone.stem.conv.rbr_reparam.weight",
        "heads.head1.cls_pred.weight",
        "heads.head1.reg_pred.weight",
    )
    _SIZE_FROM_HEAD_WIDTH = {64: "s", 96: "m", 128: "l"}
    _SIZE_FROM_HEAD_WIDTH_POSE = {48: "n", 64: "s", 96: "m", 128: "l"}
    _NUM_CLASSES_KEY = "heads.head1.cls_pred.weight"

    _DECI_CDN_BASE = "https://d2gjn4b69gu75n.cloudfront.net/models"

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return all(key in weights_dict for key in cls._REQUIRED_SIGNATURE_KEYS)

    @classmethod
    def is_pose_state_dict(cls, weights_dict: dict) -> bool:
        return _POSE_HEAD_KEY in weights_dict

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        # YOLO-NAS weights are under Deci's proprietary license — LibreYOLO
        # links to Deci's public CDN instead of mirroring on its own HF org.
        size = cls.detect_size_from_filename(filename)
        if size is None:
            return None
        task = cls.detect_task_from_filename(filename)
        if task == "pose":
            return f"{cls._DECI_CDN_BASE}/yolo_nas_pose_{size}_coco_pose.pth"
        return f"{cls._DECI_CDN_BASE}/yolo_nas_{size}_coco.pth"

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        tensor = weights_dict.get(cls._NUM_CLASSES_KEY)
        if tensor is None or tensor.ndim < 2:
            return None
        size_map = (
            cls._SIZE_FROM_HEAD_WIDTH_POSE
            if cls.is_pose_state_dict(weights_dict)
            else cls._SIZE_FROM_HEAD_WIDTH
        )
        return size_map.get(tensor.shape[1])

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        tensor = weights_dict.get(cls._NUM_CLASSES_KEY)
        if tensor is None or tensor.ndim == 0:
            return None
        if cls.is_pose_state_dict(weights_dict):
            # Pose has 1 detection class (person); the cls head's extra
            # channels are per-keypoint visibility logits.
            return 1
        return int(tensor.shape[0])

    @staticmethod
    def _detect_pose(model_path) -> bool:
        if not isinstance(model_path, str):
            return False
        try:
            ckpt = load_untrusted_torch_file(
                model_path, map_location="cpu", context="YOLO-NAS task probe"
            )
            if isinstance(ckpt, dict) and isinstance(ckpt.get("task"), str):
                return normalize_task(ckpt["task"]) == "pose"
            state = unwrap_yolonas_checkpoint(ckpt)
            return _POSE_HEAD_KEY in state
        except Exception:
            return False

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        reg_max: int = 16,
        task: str | None = None,
        **kwargs,
    ):
        self.reg_max = reg_max
        if isinstance(model_path, dict):
            model_path = unwrap_yolonas_checkpoint(model_path)
        # For pose, override classes to single-class person detection regardless
        # of how many classes the user passed (which defaults to 80 for COCO).
        resolved_task = normalize_task(task) if task is not None else None
        if resolved_task == "pose":
            nb_classes = 1
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )
        if self.task == "pose":
            self.names = {0: "person"}
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        if self.task == "pose":
            return LibreYOLONASPoseModel(
                config=self.size,
                num_keypoints=self.POSE_NUM_KEYPOINTS,
                reg_max=self.reg_max,
            )
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
        if self.task == "pose":
            # Pose head has fixed single-class detection; classes are not
            # configurable at load time.
            return
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
        if self.task == "pose":
            # Heads return the inference 4-tuple
            # (bboxes, scores, pose_xy, pose_scores).
            if isinstance(output, tuple) and len(output) == 4:
                bboxes, scores, pose_xy, pose_scores = output
                return {
                    "boxes": bboxes,
                    "scores": scores,
                    "keypoints_xy": pose_xy,
                    "keypoints_conf": pose_scores,
                }
            return output
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
        if self.task == "pose":
            return postprocess_pose(
                output,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                input_size=actual_input_size,
                original_size=original_size,
                post_nms_max_predictions=max_det,
                letterbox=kwargs.get("letterbox", True),
            )
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

            ckpt_is_pose = self.is_pose_state_dict(state_dict)
            if ckpt_is_pose and self.task != "pose":
                raise RuntimeError(
                    "Checkpoint is a YOLO-NAS pose model but this instance was "
                    "initialized for detection. Pass task='pose' or use a "
                    "detection checkpoint."
                )
            if not ckpt_is_pose and self.task == "pose":
                raise RuntimeError(
                    "Checkpoint is a YOLO-NAS detection model but this instance "
                    "was initialized for pose. Pass task='detect' or use a pose "
                    "checkpoint."
                )

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
        if self.task == "pose":
            raise NotImplementedError(
                "YOLO-NAS pose training is not yet implemented. Use task='detect' "
                "to train detection variants, or run pose inference from the "
                "pretrained checkpoints."
            )

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
