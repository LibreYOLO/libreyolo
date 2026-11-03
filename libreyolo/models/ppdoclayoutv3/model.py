"""LibrePPDocLayoutV3: BaseModel integration for PP-DocLayoutV3."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput
from ...validation.preprocessors import PPDocLayoutV3ValPreprocessor
from ..base import BaseModel
from .convert import is_native_state_dict
from .nn import PPDocLayoutV3Model
from .utils import DOC_LAYOUT_CLASSES, preprocess_image

_NUM_DOC_CLASSES = len(DOC_LAYOUT_CLASSES)


class LibrePPDocLayoutV3(BaseModel):
    """PP-DocLayoutV3 document layout detector with reading order.

    An RT-DETR-style set-prediction detector (HGNetV2-L backbone, hybrid
    encoder, six-layer deformable decoder, 300 queries) with mask-enhanced
    query initialisation and a global-pointer head that ranks every region
    pair. Predicting a page returns detect ``Results`` over 25 layout classes
    whose rows are sorted in reading order, plus ``Results.reading_order``,
    the one-based rank of each box.

    Input is a fixed 800x800 RGB canvas in ``[0, 1]`` (stretch resize, no
    normalization). The upstream pipeline uses ``conf=0.5``; pass it to
    reproduce upstream output. No NMS is applied.

    Inference-only: the training recipe needs reading-order and mask
    supervision that LibreYOLO's detect datasets do not carry.

    Devices: pure PyTorch (deformable attention through ``grid_sample``), so it
    runs on CPU, CUDA and MPS. On MPS the positional encoding frequencies are
    computed in float32 because MPS has no float64; results can differ from
    CUDA/CPU in the last bits.

    Example::

        >>> model = LibrePPDocLayoutV3("LibrePPDocLayoutV3l.pt")
        >>> result = model("page.png", conf=0.5)
        >>> for order, box, cls in zip(result.reading_order, result.boxes.xyxy, result.boxes.cls):
        ...     print(int(order), result.names[int(cls)], box.tolist())
    """

    FAMILY = "ppdoclayoutv3"
    FILENAME_PREFIX = "LibrePPDocLayoutV3"
    INPUT_SIZES = {"l": 800}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = None
    TTA_FIXED_SIZE = True
    # A score-ranked TTA merge would discard the reading order; augment=True
    # runs the plain forward instead.
    TTA_ENABLED = False
    # The FPN joins stride-8/16/32 maps, so the canvas must be square and a
    # multiple of 32 (800 by default).
    IMGSZ_STRIDE = 32
    SQUARE_IMGSZ_CALLS = frozenset({"predict", "val"})
    val_preprocessor_class = PPDocLayoutV3ValPreprocessor

    # ------------------------------------------------------------------
    # Registry classmethods
    # ------------------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Require the reading-order pointer, order heads and mask branch."""
        return is_native_state_dict(weights_dict)

    @classmethod
    def convert_upstream_state_dict(cls, weights_dict: dict) -> Optional[dict]:
        """Accept the native layout and the official ``model.``-prefixed release."""
        from .convert import convert_upstream, is_upstream_state_dict

        if is_upstream_state_dict(weights_dict):
            return convert_upstream(weights_dict)
        return super().convert_upstream_state_dict(weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Only the HGNetV2-L variant exists; confirm it by the last stage width."""
        key = "encoder_input_proj.2.0.weight"
        weight = weights_dict.get(key)
        if weight is None or getattr(weight, "ndim", 0) != 4:
            return None
        return "l" if int(weight.shape[1]) == 2048 else None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        weight = weights_dict.get("enc_score_head.weight")
        if weight is None or getattr(weight, "ndim", 0) != 2:
            return None
        return int(weight.shape[0])

    @classmethod
    def default_checkpoint_names(
        cls, nc: int, task: Optional[str] = None
    ) -> Optional[Dict[int, str]]:
        if nc == _NUM_DOC_CLASSES:
            return dict(enumerate(DOC_LAYOUT_CLASSES))
        return None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "l",
        nb_classes: int = _NUM_DOC_CLASSES,
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
        if nb_classes == _NUM_DOC_CLASSES:
            self.names = dict(enumerate(DOC_LAYOUT_CLASSES))
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        return PPDocLayoutV3Model(nc=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "encoder": self.model.encoder,
            "decoder": self.model.decoder,
            "decoder_global_pointer": self.model.decoder_global_pointer,
        }

    def _strict_loading(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Inference pipeline
    # ------------------------------------------------------------------

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
        return preprocess_image(
            image,
            input_size=input_size if input_size is not None else self.input_size,
            color_format=color_format,
        )

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
    ) -> Dict:
        from ...postprocess.ppdoclayoutv3 import postprocess

        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
        )

    def export(self, format: str = "onnx", *, opset: int = 17, **kwargs) -> str:
        """Export; deformable attention needs opset 17 (``grid_sample``)."""
        return super().export(format, opset=opset, **kwargs)

    def train(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "PP-DocLayoutV3 is shipped inference-only. Its training recipe "
            "needs reading-order and per-region mask supervision that "
            "LibreYOLO's detect datasets do not carry."
        )
