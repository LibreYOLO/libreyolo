"""LibreRiODETRv2: real-time oriented-box DETR with distributional refinement."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

from ...postprocess.rtdetr import postprocess_obb
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import RTDETRv2OBBValPreprocessor
from ..base import BaseModel
from ..rtdetrv2.utils import preprocess_obb_topleft
from .convert import convert_upstream, is_riodetrv2_state_dict, is_upstream_state_dict
from .nn import RIODETRV2_CONFIGS, RiODETRv2Model

DOTA_NAMES = {
    0: "plane", 1: "baseball-diamond", 2: "bridge", 3: "ground-track-field",
    4: "small-vehicle", 5: "large-vehicle", 6: "ship", 7: "tennis-court",
    8: "basketball-court", 9: "storage-tank", 10: "soccer-ball-field",
    11: "roundabout", 12: "harbor", 13: "swimming-pool", 14: "helicopter",
}

DIORR_NAMES = {
    0: "airplane", 1: "airport", 2: "baseballfield", 3: "basketballcourt",
    4: "bridge", 5: "chimney", 6: "expressway-service-area",
    7: "expressway-toll-station", 8: "dam", 9: "golffield", 10: "groundtrackfield",
    11: "harbor", 12: "overpass", 13: "ship", 14: "stadium", 15: "storagetank",
    16: "tenniscourt", 17: "trainstation", 18: "vehicle", 19: "windmill",
}

# encoder hidden width -> size; m and l share 256 and differ by backbone stem.
_HIDDEN_TO_SIZE = {128: "n", 224: "s", 384: "x"}


class LibreRiODETRv2(BaseModel):
    """RiO-DETRv2 oriented-box detector (DOTA-v1.0 default, DIOR-R variant).

    Adapted from ``RicePasteM/RiO-DETRv2`` commit ``ab8b22df5d80`` under the
    Apache License 2.0 (see ``NOTICE``).  Upstream relicensed to AGPL-3.0 in a
    later commit without code changes; this port relies only on the Apache
    grant of the initial release.

    Sizes ``n/s/m/l/x``.  ``LibreRiODETRv2<size>-obb.pt`` is the DOTA-v1.0
    checkpoint (15 classes, 1024 px canvas); ``-obb-diorr`` is the DIOR-R
    checkpoint (20 classes, 800 px canvas).  The native canvas is taken from the
    checkpoint's saved anchor buffers.  Inference only: the upstream recipe
    trains with a DINOv3 feature teacher, an OBB Hungarian matcher, and
    boundary/angle distribution losses that are not ported.
    """

    FAMILY = "riodetrv2"
    FILENAME_PREFIX = "LibreRiODETRv2"
    INPUT_SIZES: ClassVar[dict[str, int]] = {s: 1024 for s in RIODETRV2_CONFIGS}
    SUPPORTED_TASKS = ("obb",)
    DEFAULT_TASK = "obb"
    REQUIRE_TASK_SUFFIX = True
    TASK_INPUT_SIZES: ClassVar[dict[str, dict[str, int]]] = {"obb": INPUT_SIZES}
    WEIGHT_VARIANTS = ("diorr",)
    TRAIN_CONFIG = None
    val_preprocessor_class = RTDETRv2OBBValPreprocessor
    # Immutable LibreYOLO mirror revisions, one per published weight file.
    HF_REVISIONS: ClassVar[dict[str, str]] = {
        "LibreRiODETRv2l-obb": "3e56cb1df77725b58427a5c12b9d707a721239b2",
        "LibreRiODETRv2l-obb-diorr": "cb3c07ed8628e390f417617104ebc04a6335273e",
        "LibreRiODETRv2m-obb": "35e59ba258bb5aa15f4408380264b164ac00da93",
        "LibreRiODETRv2m-obb-diorr": "27c41556f8374a403f33fe576e09c0989b3090e3",
        "LibreRiODETRv2n-obb": "fdaef0cb28ba1ef36fa84652667b4a0161fced63",
        "LibreRiODETRv2n-obb-diorr": "fb6f81748b210e0bf19d5184605f8a4af8b320f8",
        "LibreRiODETRv2s-obb": "64dfd16194404d24489173f593032dc355b269a1",
        "LibreRiODETRv2s-obb-diorr": "631a650beffd4e0a44998778b947a988c7a322b5",
        "LibreRiODETRv2x-obb": "43892014eca2161a84687e9f63a14c7a0aafc21a",
        "LibreRiODETRv2x-obb-diorr": "2d7b1e590db93f466996864ab10d69a6c49ab93e",
    }

    # ------------------------------------------------------------------
    # Checkpoint recognition
    # ------------------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return is_riodetrv2_state_dict(weights_dict)

    @classmethod
    def convert_upstream_state_dict(cls, weights_dict: dict) -> dict | None:
        if is_upstream_state_dict(weights_dict):
            return convert_upstream(weights_dict)
        return dict(weights_dict) if cls.can_load(weights_dict) else None

    @classmethod
    def get_download_url(cls, filename: str) -> str | None:
        url = super().get_download_url(filename)
        if url is None:
            return None
        name = url.rsplit("/", 1)[-1][: -len(cls.WEIGHT_EXT)]
        revision = cls.HF_REVISIONS.get(name)
        return None if revision is None else url.replace("/resolve/main/", f"/resolve/{revision}/")

    @classmethod
    def detect_size(cls, weights_dict: dict) -> str | None:
        weight = weights_dict.get("encoder.input_proj.0.0.weight")
        if weight is None:
            weight = weights_dict.get("encoder.input_proj.0.conv.weight")
        if weight is None:
            return None
        hidden = int(weight.shape[0])
        if hidden in _HIDDEN_TO_SIZE:
            return _HIDDEN_TO_SIZE[hidden]
        if hidden == 256:
            stem = weights_dict.get("backbone.stem.stem1.conv.weight")
            if stem is not None:
                return "m" if int(stem.shape[0]) == 24 else "l"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> int | None:
        weight = weights_dict.get("decoder.enc_score_head.weight")
        return None if weight is None else int(weight.shape[0])

    @classmethod
    def detect_checkpoint_task(cls, weights_dict: dict) -> str | None:
        return "obb" if cls.can_load(weights_dict) else None

    @classmethod
    def default_checkpoint_names(cls, nc: int, task: str | None = None):
        return {15: dict(DOTA_NAMES), 20: dict(DIORR_NAMES)}.get(nc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, model_path=None, size: str = "n", nb_classes: int = 15,
                 device: str = "auto", **kwargs):
        super().__init__(model_path=model_path, size=size, nb_classes=nb_classes,
                         device=device, **kwargs)
        default_names = self.default_checkpoint_names(self.nb_classes)
        if default_names is not None and model_path is None:
            self.names = default_names
        if isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))

    def _init_model(self) -> nn.Module:
        size = self.input_size
        canvas = (size, size) if isinstance(size, int) else tuple(size)
        return RiODETRv2Model(self.size, num_classes=self.nb_classes, eval_spatial_size=canvas)

    def _prepare_model_for_state_dict(self, state_dict: dict) -> None:
        """Size the fixed-canvas buffers to the checkpoint's saved anchors."""
        anchors = state_dict.get("decoder.anchors")
        if anchors is None:
            return
        canvas = self._canvas_from_anchor_count(int(anchors.shape[1]))
        if canvas is None:
            raise ValueError(
                f"RiO-DETRv2 checkpoint anchors ({anchors.shape[1]}) match no square canvas"
            )
        self.model.set_eval_spatial_size((canvas, canvas))
        self.input_size = canvas

    def _canvas_from_anchor_count(self, count: int) -> int | None:
        strides = self.model.decoder.feat_strides
        # count = sum((S / s) ** 2); solve for S and confirm.
        canvas = int(round(math.sqrt(count / sum(1.0 / (s * s) for s in strides))))
        if sum((canvas // s) ** 2 for s in strides) == count:
            return canvas
        return None

    def _validate_loaded_state_dict_for_task(self, state_dict: dict, checkpoint=None) -> None:
        if not self.can_load(state_dict):
            raise ValueError("Not a RiO-DETRv2 checkpoint (missing angle head / angle LQE keys)")

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "encoder": self.model.encoder,
            "decoder": self.model.decoder,
        }

    def train(self, *args, **kwargs):
        """Not implemented: RiO-DETRv2 ships inference-only in LibreYOLO."""
        raise NotImplementedError(
            "RiO-DETRv2 is inference-only in LibreYOLO. Its training recipe (DINOv3 "
            "feature distillation, OBB Hungarian matching, boundary and angle "
            "distribution losses) is not implemented. Use LibreGTR or LibreYOLONAS "
            "for trainable oriented-box detection."
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @staticmethod
    def _get_preprocess_numpy():
        from ...preprocess.rtdetr import preprocess_numpy

        return preprocess_numpy

    def _get_val_preprocessor(self, img_size: int | None = None):
        if img_size is None:
            img_size = self._get_input_size()
        hw = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        return RTDETRv2OBBValPreprocessor(img_size=hw)

    def _preprocess(self, image: ImageInput, color_format: str = "auto",
                    input_size: int | None = None):
        effective = input_size if input_size is not None else self.input_size
        return preprocess_obb_topleft(image, color_format, effective)

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        with torch.no_grad():
            return self.model(input_tensor)

    def _postprocess(self, output: Any, conf_thres: float, iou_thres: float,
                     original_size: tuple[int, int], max_det: int = 300, **kwargs) -> dict:
        return postprocess_obb(
            output,
            conf_thres,
            iou_thres,
            original_size,
            max_det=max_det,
            input_size=kwargs.get("input_size", self.input_size),
            float32_scale=True,
        )
