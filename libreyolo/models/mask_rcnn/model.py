"""LibreMaskRCNN: wire Mask R-CNN into the LibreYOLO factory."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional, Tuple

import torch.nn as nn

from ...postprocess.mask_rcnn import postprocess
from ...utils.coco import COCO91_TO_COCO80
from ...validation.preprocessors import FasterRCNNValPreprocessor
from ..faster_rcnn.model import LibreFasterRCNN
from ..faster_rcnn.validator import FasterRCNNValidator
from .nn import LibreMaskRCNNModel
from .validator import MaskRCNNValidator


class LibreMaskRCNN(LibreFasterRCNN):
    """Mask R-CNN, the defining two-stage instance-segmentation architecture."""

    FAMILY = "mask_rcnn"
    FILENAME_PREFIX = "LibreMaskRCNN"
    INPUT_SIZES = {"r50": 800}
    SUPPORTED_TASKS = ("detect", "segment")
    DEFAULT_TASK = "segment"
    TASK_INPUT_SIZES = {
        "detect": INPUT_SIZES,
        "segment": INPUT_SIZES,
    }
    TRAIN_CONFIG = None
    val_preprocessor_class = FasterRCNNValPreprocessor

    def __init__(
        self,
        model_path=None,
        size: str = "r50",
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
        self.validator_class = (
            MaskRCNNValidator if self.task == "segment" else FasterRCNNValidator
        )

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Claim only two-stage checkpoints carrying the Mask R-CNN head."""
        return (
            "roi_heads.mask_predictor.mask_fcn_logits.weight" in weights_dict
            and "roi_heads.box_predictor.cls_score.weight" in weights_dict
            and any(key.startswith("rpn.head.") for key in weights_dict)
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return "r50" if cls.can_load(weights_dict) else None

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        detected = super().detect_size_from_filename(filename)
        if detected is not None:
            return detected
        lower = Path(filename).name.lower()
        if lower.startswith("maskrcnn_resnet50_fpn_v2"):
            return "r50"
        return None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "segment" if cls.can_load(state_dict) else None

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        """Reject the ``-seg`` spelling instead of building a dead URL.

        The hosted repo is suffixless (``LibreYOLO/LibreMaskRCNNr50``), so the
        base implementation turns ``LibreMaskRCNNr50-seg.pt`` into a URL for a
        repo that does not exist and the download fails with an HTTP error
        that reads like an outage. Name the correct spelling instead.

        The size guard matters: ``download_weights`` asks every registered
        family in turn and takes the first non-None answer, so raising on a
        filename that is not ours would hijack another family's download.
        Torchvision spellings (``maskrcnn_resnet50_fpn_v2*``) carry no task
        suffix and keep resolving through the base implementation.
        """
        name = Path(filename).name
        size = cls.detect_size_from_filename(name)
        if size is None:
            return None
        if cls.detect_task_from_filename(name) is not None:
            canonical = f"{cls.FILENAME_PREFIX}{size}{cls.WEIGHT_EXT}"
            raise FileNotFoundError(
                f"{name}: LibreYOLO hosts {cls.FILENAME_PREFIX} weights without "
                f"a task suffix. Use '{canonical}' instead."
            )
        return super().get_download_url(name)

    def _allow_checkpoint_task_mismatch(self, checkpoint_task: str) -> bool:
        """Allow the shared instance-segmentation checkpoint in detect mode."""
        return checkpoint_task == "segment" and self.task == "detect"

    def _init_model(self) -> nn.Module:
        head_width = 91 if self.nb_classes == 80 else self.nb_classes + 1
        return LibreMaskRCNNModel(
            size=self.size,
            num_classes=head_width,
            return_masks=self.task == "segment",
        )

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> dict:
        class_map = (
            COCO91_TO_COCO80
            if self._arch_num_classes == 91 and self.nb_classes == 80
            else None
        )
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
            class_map=class_map,
            include_masks=self.task == "segment",
            **kwargs,
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Mask R-CNN is currently inference-only; sampled-RoI and mask "
            "training are not implemented."
        )

    def export(
        self,
        format: str = "onnx",
        *,
        opset: int = 18,
        **kwargs,
    ) -> str:
        if format.lower() == "onnx":
            if int(kwargs.get("batch", 1)) != 1:
                raise NotImplementedError(
                    "Mask R-CNN ONNX export supports batch=1 only."
                )
            if kwargs.get("dynamic") is False:
                warnings.warn(
                    "Mask R-CNN ONNX keeps its upstream resize and mask paste "
                    "inside the graph; forcing dynamic=True for source parity.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            kwargs["dynamic"] = True
        return super().export(format=format, opset=opset, **kwargs)
