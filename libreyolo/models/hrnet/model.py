"""LibreHRNet top-down human-pose wrapper (inference only).

HRNet keeps a high-resolution stream throughout repeated multi-scale fusion.
The pose family consumes person crops and emits one COCO-17 heatmap per crop.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import torch
from torch import nn

from ...postprocess.hrnet import postprocess_hrnet
from ..base import BaseModel
from .detector import PersonDetector, resolve_person_detector
from .nn import HRNetPoseModel
from .utils import box_to_center_scale, preprocess_crop_image, preprocess_numpy


class LibreHRNet(BaseModel):
    """Top-down HRNet pose estimator: person crops to COCO-17 heatmaps."""

    FAMILY = "hrnet"
    FILENAME_PREFIX = "LibreHRNet"
    INPUT_SIZES: ClassVar[dict[str, tuple[int, int]]] = {
        "w32": (256, 192),
        "w48": (384, 288),
    }
    SUPPORTED_TASKS = ("pose",)
    DEFAULT_TASK = "pose"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    TTA_ENABLED = False
    SUPPORTS_BATCHED_PREDICT = False
    POSE_NUM_KEYPOINTS = 17

    _STAGE_KEY = "stage3.0.branches.0.0.conv1.weight"
    _SIGNATURE_KEYS = (
        "transition1.0.0.weight",
        _STAGE_KEY,
        "final_layer.weight",
    )

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        if not all(key in weights_dict for key in cls._SIGNATURE_KEYS):
            return False
        stem = weights_dict.get("conv1.weight")
        stage = weights_dict[cls._STAGE_KEY]
        head = weights_dict["final_layer.weight"]
        return bool(
            getattr(stem, "shape", None) == torch.Size((64, 3, 3, 3))
            and getattr(stage, "ndim", 0) == 4
            and int(stage.shape[0]) in (32, 48)
            and getattr(head, "shape", None)
            in (torch.Size((17, 32, 1, 1)), torch.Size((17, 48, 1, 1)))
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        stage = weights_dict.get(cls._STAGE_KEY)
        if stage is None or getattr(stage, "ndim", 0) != 4:
            return None
        return {32: "w32", 48: "w48"}.get(int(stage.shape[0]))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_num_keypoints(cls, weights_dict: dict) -> Optional[int]:
        head = weights_dict.get("final_layer.weight")
        return int(head.shape[0]) if head is not None and getattr(head, "ndim", 0) == 4 else None

    def __init__(
        self,
        model_path=None,
        size: str = "w32",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        person_detector: PersonDetector | object | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        self.num_keypoints = self.POSE_NUM_KEYPOINTS
        self.keypoint_dim = 3
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        # HRNet's released pose head is fixed to the COCO person category.
        # Keep this semantic name even when a metadata-less upstream file was
        # auto-wrapped with the generic one-class fallback.
        self.names = {0: "person"}
        self.person_detector = resolve_person_detector(
            person_detector,
            device=str(self.device),
        )
        self.model.eval()

    def _init_model(self) -> nn.Module:
        width = 32 if self.size == "w32" else 48
        return HRNetPoseModel(width=width, num_keypoints=self.POSE_NUM_KEYPOINTS)

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {
            "stem": self.model.conv1,
            "backbone": self.model.stage3,
            "head": self.model.final_layer,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(self, image, color_format="auto", input_size=None):
        return preprocess_crop_image(
            image,
            input_size=input_size or self._get_input_size(),
            color_format=color_format,
        )

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output,
        conf_thres,
        iou_thres,
        original_size,
        max_det=1,
        ratio=1.0,
        **kwargs,
    ) -> dict:
        del conf_thres, iou_thres, ratio
        original_width, original_height = original_size
        box = [0.0, 0.0, float(original_width), float(original_height)]
        center, scale = box_to_center_scale(box, self._get_input_size())
        return postprocess_hrnet(
            output,
            centers=center[None, :],
            scales=scale[None, :],
            boxes=np.asarray([box], dtype=np.float32),
            box_scores=np.ones((1,), dtype=np.float32),
            keypoint_threshold=float(kwargs.get("keypoint_threshold", 0.2)),
            oks_threshold=float(kwargs.get("oks_threshold", 0.9)),
            max_det=min(int(max_det), 1),
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreHRNet is inference-only. Pose training requires a keypoint-aware "
            "data path and augmentations that LibreYOLO does not yet provide."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        supported = {"onnx", "torchscript", "openvino", "tensorrt", "axelera"}
        if format.lower() not in supported:
            raise NotImplementedError(
                f"LibreHRNet export to {format!r} is not implemented. The pose-head "
                "export contract supports ONNX, TorchScript, OpenVINO and "
                "TensorRT, and runs hybrid on Axelera Metis."
            )
        return super().export(format=format, **kwargs)

    @property
    def _runner(self):
        if getattr(self, "_runner_instance", None) is None:
            from .inference import HRNetPoseInferenceRunner

            self._runner_instance = HRNetPoseInferenceRunner(self)
        return self._runner_instance
