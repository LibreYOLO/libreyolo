"""LibreLaMa image inpainting through the pinned OpenCV Zoo ONNX graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.lama import postprocess as _lama_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from . import nn as lama_nn
from .utils import LaMaPredictionContext, preprocess_image_and_mask, preprocess_numpy
from .validator import LaMaRestoreValidator


class LibreLaMa(BaseModel):
    """Mask-guided image inpainting with an opaque, immutable ONNX payload.

    ``mask=`` is required. White or any other nonzero mask value means fill;
    zero means preserve. Inference always uses the official 512x512 graph, then
    returns an RGB image on the source canvas with unmasked pixels copied from
    the source exactly.
    """

    FAMILY = "lama"
    FILENAME_PREFIX = "LibreLaMa"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"b": lama_nn.ONNX_INPUT_SIZE}
    SUPPORTED_TASKS = ("restore",)
    DEFAULT_TASK = "restore"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    SUPPORTS_BATCHED_PREDICT = False
    SUPPORTS_CUDA_GRAPH = False
    TTA_ENABLED = False
    TTA_FIXED_SIZE = True
    validator_class = LaMaRestoreValidator
    PREDICT_INPUT_KWARGS = ("mask",)
    REQUIRED_PREDICT_INPUT_KWARGS = ("mask",)
    restore_scale = 1

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return lama_nn.is_official_onnx_graph(weights_dict.get("onnx_graph"))

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return "b" if cls.can_load(weights_dict) else None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "restore" if cls.can_load(state_dict) else None

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> Optional[str]:
        del filename, url
        return (
            "LibreLaMa contains OpenCV Zoo's Apache-2.0 ONNX artifact. The "
            "model was trained on Places365-Challenge, whose image-download "
            "terms restrict the data to non-commercial research and education; "
            "commercial training-data clearance was not independently established."
        )

    def __init__(
        self,
        model_path=None,
        size: str = "b",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
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
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.nb_classes = 1
        self.names = {0: "image"}
        self.model.eval()

    def _init_model(self) -> nn.Module:
        return lama_nn.OpaqueLaMaONNX()

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"opaque_onnx_runtime": self.model}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _validate_loaded_state_dict_for_task(
        self,
        state_dict: dict,
        checkpoint: dict | None = None,
    ) -> None:
        if set(state_dict) != {"onnx_graph"}:
            raise ValueError(
                "LibreLaMa state_dict must contain only the opaque "
                f"'onnx_graph' tensor; got keys={sorted(state_dict)}."
            )
        digest = lama_nn.validate_onnx_graph_tensor(state_dict["onnx_graph"])
        if checkpoint is not None:
            recorded = checkpoint.get("source_sha256")
            if recorded is not None and str(recorded).lower() != digest:
                raise ValueError(
                    "LibreLaMa checkpoint source_sha256 does not match its "
                    f"embedded graph: metadata={recorded}, graph={digest}."
                )

    def _prepare_model_for_state_dict(self, state_dict: dict) -> None:
        graph = state_dict.get("onnx_graph")
        if not lama_nn.is_official_onnx_graph(graph):
            lama_nn.validate_onnx_graph_tensor(graph)
        self.model.allocate_graph_buffer(graph.numel())

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ):
        del image, color_format, input_size
        raise ValueError(
            "LibreLaMa prediction requires an aligned mask: "
            "model.predict(image, mask=mask)."
        )

    def _preprocess_predict(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
        *,
        mask,
    ) -> Tuple[
        torch.Tensor,
        Any,
        Tuple[int, int],
        LaMaPredictionContext,
    ]:
        effective_size = self.input_size if input_size is None else input_size
        tensor, pil, original_size, _, context = preprocess_image_and_mask(
            image,
            mask,
            color_format=color_format,
            input_size=effective_size,
        )
        return tensor, pil, original_size, context

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float | LaMaPredictionContext = 1.0,
        **kwargs,
    ) -> Dict:
        del conf_thres, iou_thres, max_det, kwargs
        context = ratio
        if not isinstance(context, LaMaPredictionContext):
            raise RuntimeError(
                "LibreLaMa postprocessing requires the image/mask context "
                "returned by guided preprocessing. Use "
                "model.predict(image, mask=mask)."
            )
        return {
            "restored": _lama_postprocess(
                output,
                original_size,
                original_rgb=context.original_rgb,
                fill_mask=context.fill_mask,
            )
        }

    def _save_extra_metadata(self) -> dict[str, Any]:
        return {
            "scale": 1,
            "degradation": "inpaint",
            "dataset": "Places365-Challenge",
            "source_url": (
                "https://huggingface.co/opencv/inpainting_lama/resolve/"
                "aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f/"
                "inpainting_lama_2025jan.onnx"
            ),
            "source_revision": "aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f",
            "source_sha256": lama_nn.OFFICIAL_ONNX_SHA256,
            "onnx_opset": 21,
            "runtime": "onnxruntime>=1.18",
            "inference_only": True,
        }

    def train(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "LibreLaMa is inference-only. Its checkpoint embeds the immutable "
            "official OpenCV Zoo ONNX graph; no native training graph is present."
        )

    def export(self, format: str = "onnx", **kwargs: Any) -> str:
        del format, kwargs
        raise NotImplementedError(
            "LibreLaMa already executes the pinned upstream ONNX graph embedded "
            "inside its .pt checkpoint. Re-export is intentionally unsupported."
        )

    def quantize(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "LibreLaMa's embedded OpenCV Zoo graph is already a QDQ-quantized "
            "deployment artifact and cannot be quantized through PyTorch."
        )


__all__ = ["LibreLaMa"]
