"""LibreHVI-CIDNet low-light image-restoration family."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.hvi_cidnet import postprocess as _hvi_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .nn import CIDNet
from .utils import _pad_to_multiple, preprocess_image, preprocess_numpy


_GAMMA: ContextVar[float] = ContextVar("hvi_cidnet_gamma", default=1.0)
_SATURATION: ContextVar[float] = ContextVar("hvi_cidnet_saturation", default=1.0)
_INTENSITY: ContextVar[float] = ContextVar("hvi_cidnet_intensity", default=1.0)


@contextmanager
def _control_scope(controls: Dict[str, float]) -> Iterator[None]:
    """Apply one call's controls without leaking them into its caller."""

    gamma_token = _GAMMA.set(controls["gamma"])
    saturation_token = _SATURATION.set(controls["saturation"])
    intensity_token = _INTENSITY.set(controls["intensity"])
    try:
        yield
    finally:
        _INTENSITY.reset(intensity_token)
        _SATURATION.reset(saturation_token)
        _GAMMA.reset(gamma_token)


def _controlled_stream(results: Any, controls: Dict[str, float]) -> Iterator[Any]:
    """Pull each lazy result inside its controls, then reset before yielding."""

    iterator = iter(results)
    try:
        while True:
            with _control_scope(controls):
                try:
                    result = next(iterator)
                except StopIteration:
                    return
            yield result
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            with _control_scope(controls):
                close()


class LibreHVICIDNet(BaseModel):
    """Tiny HVI-CIDNet low-light enhancer.

    Prediction runs at native resolution with right/bottom reflection padding
    to a multiple of eight. ``gamma``, ``saturation``, and ``intensity`` are
    per-call controls; all default to upstream's neutral evaluation values.
    """

    FAMILY = "hvi_cidnet"
    FILENAME_PREFIX = "LibreHVICIDNet"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"t": 256}
    SUPPORTED_TASKS = ("restore",)
    DEFAULT_TASK = "restore"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    SUPPORTS_BATCHED_PREDICT = True
    TTA_ENABLED = False

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return cls.detect_size(weights_dict) is not None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        density = weights_dict.get("trans.density_k")
        hv_input = weights_dict.get("HVE_block0.1.weight")
        intensity_output = weights_dict.get("ID_block0.1.weight")
        attention = weights_dict.get("I_LCA6.ffn.temperature")
        if (
            getattr(density, "shape", None) == torch.Size((1,))
            and getattr(hv_input, "shape", None) == torch.Size((36, 3, 3, 3))
            and getattr(intensity_output, "shape", None) == torch.Size((1, 36, 3, 3))
            and getattr(attention, "shape", None) == torch.Size((2, 1, 1))
        ):
            return "t"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "restore" if cls.can_load(state_dict) else None

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> str:
        del url
        return (
            f"{Path(filename).name} is published under MIT by HVI-CIDNet's "
            "authors. It was trained on LOLv2-Synthetic; the canonical LOLv2 "
            "source does not state a dataset license. No training images are "
            "bundled or downloaded by LibreYOLO."
        )

    def __init__(
        self,
        model_path=None,
        size: str = "t",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        del nb_classes
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.nb_classes = 1
        self.names = {0: "image"}
        self.model.eval()

    def __call__(
        self,
        source=None,
        *,
        gamma: float = 1.0,
        saturation: float = 1.0,
        intensity: float = 1.0,
        **kwargs,
    ):
        """Enhance ``source`` with optional upstream HVI controls.

        ``gamma`` shapes the input exposure. ``saturation`` and ``intensity``
        scale the inverse-HVI output. Values must be positive; ``1.0`` exactly
        reproduces the official generalization checkpoint's evaluation setup.
        Context-local values keep concurrent calls independent.
        """

        controls = {
            "gamma": float(gamma),
            "saturation": float(saturation),
            "intensity": float(intensity),
        }
        for name, value in controls.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        with _control_scope(controls):
            results = super().__call__(source, **kwargs)
        if kwargs.get("stream", False):
            return _controlled_stream(results, controls)
        return results

    def _init_model(self) -> nn.Module:
        return CIDNet()

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "hvi_transform": self.model.trans,
            "hue_value_encoder": self.model.HVE_block0,
            "intensity_encoder": self.model.IE_block0,
            "hue_value_decoder": self.model.HVD_block0,
            "intensity_decoder": self.model.ID_block0,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        del input_size
        return preprocess_image(
            image,
            color_format=color_format,
            gamma=_GAMMA.get(),
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(
            _pad_to_multiple(input_tensor, 8),
            saturation_scale=_SATURATION.get(),
            intensity_scale=_INTENSITY.get(),
        )

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
        del conf_thres, iou_thres, max_det, ratio, kwargs
        return {"restored": _hvi_postprocess(output, original_size)}

    def train(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "LibreHVICIDNet currently ships inference and paired PSNR/SSIM "
            "validation only. Training is not implemented for this family."
        )


__all__ = ["LibreHVICIDNet"]
