"""LibreDOMEDETR — BaseModel wrapper for the Dome-DETR tiny-object family."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from libreyolo.training.ddp_spawn import ddp_aware

from ...postprocess.domedetr import postprocess
from ...training.callbacks import TrainCallbacks
from ...training.config import DOMEDETRConfig
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import DOMEDETRValPreprocessor
from ..base import BaseModel
from .nn import DEFAULT_VARIANT, VARIANT_QUERY_BUDGET, LibreDOMEDETRModel
from .utils import preprocess_image, unwrap_domedetr_checkpoint

_TRAIN_DEFAULTS = DOMEDETRConfig()

logger = logging.getLogger(__name__)

# The stride-4 encoder projection width is the cheapest size fingerprint:
# B0/B2/B4 stage-1 outputs are 64/96/128 channels.
_STEM_CHANNELS_TO_SIZE = {64: "s", 96: "m", 128: "l"}

# Published class counts per dataset variant. Used only as a fallback when a
# checkpoint carries no explicit variant marker.
_NC_TO_VARIANT = {9: "aitod", 12: "visdrone"}


class LibreDOMEDETR(BaseModel):
    """LibreYOLO wrapper for Dome-DETR (ACM MM 2025).

    A tiny-object specialist for aerial, drone and remote-sensing imagery, not
    a general-purpose detector. It is D-FINE plus three modules: DeFE predicts
    a density map, MWAS restricts encoder attention to occupied windows, and
    PAQI sizes the query set from that density instead of using a fixed 300.

    Scope notes that matter before reaching for this family:

    - **No COCO checkpoint exists.** Upstream publishes AI-TOD-V2 (9 classes)
      and VisDrone (12 classes) weights only, so canonical filenames always
      carry a dataset suffix (``LibreDOMEDETRs-visdrone.pt``) and ``names``
      comes from checkpoint metadata, never from a family constant.
    - **The advantage narrows as objects grow.** Upstream's own ablation moves
      AP-verytiny 14.0 -> 17.8 but AP-medium only 45.4 -> 46.4. It sits beside
      D-FINE rather than replacing it.
    - **Training is wired.** The full upstream objective is ported: the
      D-FINE losses plus DeFE density and count supervision, padded queries
      masked out of the classification terms, and per-image denoising
      attention masks. Convergence against upstream's published 160-epoch
      schedule has not been reproduced here, so treat the paper's AP numbers
      as unverified rather than as a promise this recipe reaches them.
    - **Official weights are academic-research-only.** LibreYOLO mirrors the
      six upstream checkpoints under the stricter use restriction stated on
      the upstream model card. They are not covered by LibreYOLO's MIT license,
      and every auto-download prints the restriction before fetching them.
    """

    FAMILY = "domedetr"
    FILENAME_PREFIX = "LibreDOMEDETR"
    INPUT_SIZES = {"s": 800, "m": 800, "l": 800}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TASK_INPUT_SIZES = {"detect": INPUT_SIZES}
    TRAIN_CONFIG = DOMEDETRConfig
    RESUME_RESTORES_TRAIN_ARGS = True
    WEIGHT_VARIANTS = ("aitod", "visdrone")
    val_preprocessor_class = DOMEDETRValPreprocessor
    TTA_FIXED_SIZE = True  # fixed square resize; multi-scale TTA is a no-op
    # PAQI's query count is data dependent (boolean masking + a greedy NMS
    # loop), so the forward has host syncs and a shape that changes per image.
    # That is exactly what CUDA graph capture cannot do.
    SUPPORTS_CUDA_GRAPH = False

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # DeFE lives only in Dome-DETR. Deliberately *not* keyed on
        # ``decoder.pre_bbox_head.``: Dome-DETR is a D-FINE derivative and
        # carries that key too, which is why this family must also register
        # ahead of LibreDFINE.
        return any(k.startswith("encoder.DeFE.") for k in weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        key = "encoder.input_proj.0.conv.weight"
        if key not in weights_dict:
            return None
        return _STEM_CHANNELS_TO_SIZE.get(int(weights_dict[key].shape[1]))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "decoder.enc_score_head.weight"
        if key not in weights_dict:
            return None
        return int(weights_dict[key].shape[0])

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ):
        # The variant has to be known before ``_init_model`` runs: it selects
        # the PAQI query budget *and* the decoder depth for L (4 layers on
        # AI-TOD-V2, 6 on VisDrone), so it changes the module tree, not just a
        # label. Resolve it from the most explicit signal available.
        kwargs["weight_variant"] = self._resolve_weight_variant(
            explicit=kwargs.pop("weight_variant", None),
            model_path=model_path,
            nb_classes=nb_classes,
        )
        if isinstance(model_path, dict):
            model_path = unwrap_domedetr_checkpoint(model_path)
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task,
            **kwargs,
        )
        if isinstance(model_path, str):
            self._load_weights(model_path)

    @classmethod
    def _resolve_weight_variant(
        cls, explicit: str | None, model_path: Any, nb_classes: int
    ) -> str:
        """Pick the dataset variant: explicit > filename suffix > class count."""
        if explicit is not None:
            if explicit not in VARIANT_QUERY_BUDGET:
                raise ValueError(
                    f"Unknown weight_variant {explicit!r}; "
                    f"expected one of {tuple(VARIANT_QUERY_BUDGET)}"
                )
            return explicit

        if isinstance(model_path, (str, Path)):
            name = Path(model_path).name.lower()
            for variant in cls.WEIGHT_VARIANTS:
                if f"-{variant}" in name:
                    return variant

        variant = _NC_TO_VARIANT.get(nb_classes)
        if variant is None:
            logger.warning(
                "Dome-DETR checkpoint has no -aitod/-visdrone filename suffix and "
                "nc=%s matches neither AI-TOD-V2 (9) nor VisDrone (12); falling back "
                "to the %r query budget. Pass weight_variant= to be explicit.",
                nb_classes,
                DEFAULT_VARIANT,
            )
            return DEFAULT_VARIANT
        return variant

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        """Resolve hosted dataset variants and reject the nonexistent bare name."""
        name = Path(filename).name
        size = cls.detect_size_from_filename(name)
        if size is None:
            # Not one of ours. ``download_weights`` asks every registered
            # family in turn and takes the first non-None answer, so raising
            # here would hijack every other family's download with a
            # Dome-DETR error message.
            return None

        variant = cls.detect_variant_from_filename(name)
        if variant is None:
            raise FileNotFoundError(
                f"{name} has no dataset suffix. Dome-DETR has no COCO checkpoint, "
                f"so there is no bare {cls.FILENAME_PREFIX}{size}.pt -- the canonical "
                f"names are {cls.FILENAME_PREFIX}{size}-aitod.pt (AI-TOD-V2, 9 classes) "
                f"and {cls.FILENAME_PREFIX}{size}-visdrone.pt (VisDrone, 12 classes). "
                "Pick the one matching your data."
            )
        return super().get_download_url(name)

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> Optional[str]:
        del url
        return (
            f"{Path(filename).name} contains official Dome-DETR pretrained weights. "
            "The upstream model card restricts these weights to ACADEMIC RESEARCH "
            "PURPOSES ONLY. They are NOT covered by LibreYOLO's MIT license and "
            "must not be treated as commercially cleared. The same card also calls "
            "the project Apache-2.0 but provides no license metadata or weight-repo "
            "LICENSE file; LibreYOLO mirrors the weights under the stricter academic-"
            "use reading and documents that unresolved context in each mirror. You "
            "are responsible for reviewing and complying with the upstream terms."
        )

    def _init_model(self) -> nn.Module:
        return LibreDOMEDETRModel(
            config=self.size,
            nb_classes=self.nb_classes,
            variant=getattr(self, "weight_variant", DEFAULT_VARIANT),
            eval_spatial_size=(self.input_size, self.input_size),
            train_from_scratch=self._is_scratch_build(),
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "backbone_stem": self.model.backbone.stem,
            "encoder": self.model.encoder,
            "encoder_input_proj": self.model.encoder.input_proj,
            "encoder_defe": self.model.encoder.DeFE,
            "encoder_mwas": self.model.encoder.mwas_processor,
            "encoder_fpn": self.model.encoder.fpn_blocks,
            "encoder_pan": self.model.encoder.pan_blocks,
            "decoder": self.model.decoder,
            "decoder_input_proj": self.model.decoder.input_proj,
            "dec_bbox_head": self.model.decoder.dec_bbox_head,
            "dec_score_head": self.model.decoder.dec_score_head,
        }

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
        return preprocess_image(image, input_size=effective_size, color_format=color_format)

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
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
            **kwargs,
        )

    def _strict_loading(self) -> bool:
        # Anchors and valid_mask are regenerated at forward time from
        # eval_spatial_size rather than restored, as in every DETR family here.
        return False

    @staticmethod
    def unwrap_checkpoint(checkpoint):
        return unwrap_domedetr_checkpoint(checkpoint)

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = 160,
        batch: int = 4,
        imgsz: int = 800,
        lr0: float = 2e-4,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "domedetr_exp",
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = _TRAIN_DEFAULTS.amp,
        patience: int = 50,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Fine-tune Dome-DETR on a YOLO-format dataset config.

        Trains against upstream's full objective: the D-FINE losses plus
        DeFE density and count supervision, padded queries masked out of the
        classification terms, and per-image denoising attention masks.

        Two things worth knowing before a long run. Upstream trains 160 epochs
        with ``MultiStepLR(milestones=[80, 120], gamma=0.8)`` while these
        defaults use D-FINE's flat-cosine, so reproducing the paper's numbers
        means supplying the upstream schedule rather than taking these as is.
        And PAQI makes the query count vary per image, so a batch is padded to
        its widest member: memory tracks the busiest image in each batch, not
        the average, which is why ``batch`` defaults lower than D-FINE's.

        Args:
            data: Path to the dataset YAML file.
            callbacks: Optional training callback or iterable of callbacks.
            loggers: Optional built-in experiment loggers.
        """
        from libreyolo.data import load_data_config

        from .trainer import DOMEDETRTrainer

        try:
            data_config = load_data_config(data, autodownload=True)
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")
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

        trainer = DOMEDETRTrainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            patience=patience,
            callbacks=callbacks,
            loggers=loggers,
            **kwargs,
        )

        if resume:
            trainer.setup()
            trainer.resume(self._resume_checkpoint(resume))
            return trainer.train()

        return trainer.train()
