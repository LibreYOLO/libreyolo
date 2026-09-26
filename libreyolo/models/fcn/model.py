"""LibreFCN: wire ResNet-backed FCN into the LibreYOLO factory."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.fcn import postprocess, resize_logits
from ...tasks import normalize_task
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .nn import LibreFCNModel
from .utils import preprocess_image, preprocess_numpy


VOC_NAMES = (
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


class LibreFCN(BaseModel):
    """ResNet-50/101 FCN for dense semantic segmentation.

    FCN established end-to-end pixels-to-pixels semantic prediction. This is
    torchvision's modern dilated-ResNet adaptation, not the paper's VGG-based
    FCN-8s skip-fusion graph. Inputs are RGB floats in ``[0, 1]`` at 520 pixels
    and use ImageNet mean/std normalization inside the network.
    """

    FAMILY: ClassVar[str] = "fcn"
    FILENAME_PREFIX: ClassVar[str] = "LibreFCN"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"r50": 520, "r101": 520}
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("semantic",)
    DEFAULT_TASK: ClassVar[str] = "semantic"
    TRAIN_CONFIG: ClassVar[None] = None

    semantic_resize_mode: ClassVar[str] = "stretch"
    semantic_imgsz_divisor: ClassVar[int] = 8

    _HEAD_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "backbone.conv1.weight",
            "backbone.layer4.0.conv2.weight",
            "classifier.0.weight",
            "classifier.1.running_mean",
            "classifier.4.weight",
            "aux_classifier.0.weight",
            "aux_classifier.4.weight",
        }
    )

    def __init__(
        self,
        model_path=None,
        size: str = "r50",
        nb_classes: int = 21,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else "semantic"
        if resolved_task != "semantic":
            raise ValueError(f"LibreFCN supports only task='semantic'; got {task!r}.")
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )
        if self.nb_classes == len(VOC_NAMES):
            self.names = dict(enumerate(VOC_NAMES))
        if isinstance(model_path, str):
            self._load_weights(str(self.model_path))
        self.model.eval()

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Match the FCN head together with its embedded ResNet backbone."""
        return cls._HEAD_KEYS.issubset(weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Infer the ResNet depth from the number of layer-3 bottlenecks."""
        if not cls.can_load(weights_dict):
            return None
        pattern = re.compile(r"^backbone\.layer3\.(\d+)\.conv3\.weight$")
        indices = [
            int(match.group(1)) for key in weights_dict if (match := pattern.match(key))
        ]
        if not indices:
            return None
        return {6: "r50", 23: "r101"}.get(max(indices) + 1)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        weight = weights_dict.get("classifier.4.weight")
        if weight is None or getattr(weight, "ndim", 0) < 1:
            return None
        return int(weight.shape[0])

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        """Reject the ``-sem`` spelling instead of building a dead URL.

        This family's hosted repos are suffixless (``LibreYOLO/LibreFCNr50``),
        unlike the ``-sem`` convention of its semantic siblings, so the base
        implementation turns ``LibreFCNr50-sem.pt`` into a URL for a repo that
        does not exist and the download fails with an HTTP error that reads
        like an outage. Name the correct spelling instead.

        The size guard matters: ``download_weights`` asks every registered
        family in turn and takes the first non-None answer, so raising on a
        filename that is not ours would hijack another family's download.
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

    @classmethod
    def default_checkpoint_names(
        cls, nc: int, task: Optional[str] = None
    ) -> Optional[Dict[int, str]]:
        """Supply the published VOC-style labels for bare official weights."""
        if nc != len(VOC_NAMES):
            return None
        return dict(enumerate(VOC_NAMES))

    def _init_model(self) -> nn.Module:
        return LibreFCNModel(size=self.size, num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "head": self.model.classifier,
            "aux_head": self.model.aux_classifier,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ):
        effective_size = (
            input_size if input_size is not None else self._get_input_size()
        )
        if effective_size % self.semantic_imgsz_divisor:
            raise ValueError(
                f"LibreFCN semantic imgsz={effective_size} must be divisible by "
                f"{self.semantic_imgsz_divisor}."
            )
        return preprocess_image(image, effective_size, color_format=color_format)

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> dict:
        return postprocess(
            output,
            conf_thres,
            iou_thres,
            original_size,
            max_det=max_det,
            **kwargs,
        )

    def _postprocess_semantic_logits(
        self,
        output: Any,
        original_size: Tuple[int, int],
        **kwargs,
    ) -> torch.Tensor:
        """Resize primary FCN logits to the source image before argmax or TTA."""
        return resize_logits(output, original_size)

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreFCN is inference-only; semantic training and auxiliary-head loss "
            "are outside this port."
        )

    def _strict_loading(self) -> bool:
        return True


__all__ = ["LibreFCN", "VOC_NAMES"]
