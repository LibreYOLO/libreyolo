"""SAM 3 image segmentation adapter for the LibreSAM tier.

Visual prompts use Transformers' tracker model and keep the existing
encode-once LibreSAM contract. Concept text prompts lazily load the separate
PCS model from the same gated checkpoint snapshot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict

import torch

from .base import _INSTALL_HINT, _is_empty_prompt, LibreSAMModel

logger = logging.getLogger(__name__)

_GATED_REPO = "facebook/sam3"


def _http_status(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction without depending on Hub internals."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


class LibreSAM3(LibreSAMModel):
    """SAM 3 visual and concept-prompted image segmenter.

    The visual path uses ``Sam3TrackerModel``. The first ``text=`` call loads a
    separate ``Sam3Model`` instance, which increases peak RAM/VRAM because the
    two Transformers models do not share their instantiated encoder weights.
    """

    FAMILY = "sam3"
    FILENAME_PREFIX = "LibreSAM3"
    HF_REPOS: ClassVar[Dict[str, str]] = {"large": _GATED_REPO}
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"large": 1008}
    DEFAULT_PCS_SCORE_THRESH: ClassVar[float] = 0.3
    _LICENSE_NOTICE_SHOWN: ClassVar[bool] = False
    _LICENSE_NOTICE: ClassVar[str] = (
        "SAM 3 weights are provided by Meta under the custom SAM License, not "
        "LibreYOLO's MIT license. Use is subject to the terms accepted at "
        "https://huggingface.co/facebook/sam3."
    )
    _GATED_HELP: ClassVar[str] = (
        "SAM 3 weights are gated. Visit https://huggingface.co/facebook/sam3, "
        "accept the terms, then run `hf auth login` (or set HF_TOKEN) and retry."
    )
    SNAPSHOT_IGNORE_PATTERNS: ClassVar[tuple[str, ...]] = (
        *LibreSAMModel.SNAPSHOT_IGNORE_PATTERNS,
        "*.pt",
    )

    def __init__(self, size: str = "large", **kwargs):
        super().__init__(size=size, **kwargs)
        self._pcs_model = None
        self._pcs_processor = None

    def _ensure_weights(self) -> str:
        if not type(self)._LICENSE_NOTICE_SHOWN:
            type(self)._LICENSE_NOTICE_SHOWN = True
            logger.warning(self._LICENSE_NOTICE)
        try:
            return super()._ensure_weights()
        except Exception as exc:
            if _http_status(exc) in (401, 403):
                raise RuntimeError(self._GATED_HELP) from exc
            raise

    def _load_pretrained(self, snapshot_dir: str) -> tuple:
        try:
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        # Absolute: the PCS model loads lazily, possibly after a cwd change.
        self._snapshot_dir = str(Path(snapshot_dir).resolve())
        model = Sam3TrackerModel.from_pretrained(
            snapshot_dir, dtype=self._resolve_dtype()
        )
        processor = Sam3TrackerProcessor.from_pretrained(snapshot_dir)
        return model, processor

    def _load_pcs(self):
        if self._pcs_model is not None:
            return self._pcs_model, self._pcs_processor
        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        model = Sam3Model.from_pretrained(
            self._snapshot_dir, dtype=self._resolve_dtype()
        ).to(self.device)
        model.eval()
        self._pcs_model = model
        self._pcs_processor = Sam3Processor.from_pretrained(self._snapshot_dir)
        return self._pcs_model, self._pcs_processor

    def _set_device(self, device: str) -> "LibreSAM3":
        super()._set_device(device)
        pcs_model = getattr(self, "_pcs_model", None)
        if pcs_model is not None:
            pcs_model.to(self.device)
        return self

    def _move_embeddings(self, embeddings, device: torch.device):
        """SAM 3 tracker caches a list of vision feature tensors."""
        if isinstance(embeddings, list):
            return [embedding.to(device) for embedding in embeddings]
        return super()._move_embeddings(embeddings, device)

    def _post_process_masks(self, pred_masks, enc):
        """SAM 3 tracker uses original sizes without a reshaped-size argument."""
        fn = getattr(self.processor, "post_process_masks", None)
        if fn is None:
            fn = self.processor.image_processor.post_process_masks
        return fn(pred_masks.cpu(), enc["original_sizes"])

    @staticmethod
    def _move_pcs_inputs(inputs: Any, device: torch.device) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(device)
        if isinstance(inputs, dict):
            return {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        return inputs

    def predict(
        self,
        source=None,
        *,
        points=None,
        bboxes=None,
        labels=None,
        masks=None,
        text: str | None = None,
        conf=None,
        multimask=None,
        max_det: int = 300,
        device=None,
        color_format: str = "auto",
        points_per_side=None,
    ):
        """Segment with spatial prompts or find all instances matching ``text``.

        On the text path, ``conf`` is the PCS detection score; ``None`` uses
        ``DEFAULT_PCS_SCORE_THRESH`` (0.3), while an explicit ``0.0`` keeps all
        candidates. On the visual path it remains the predicted mask-IoU score.
        Text is mutually exclusive with points, boxes, labels, masks, and
        segment-everything controls.
        """
        if text is None:
            return super().predict(
                source,
                points=points,
                bboxes=bboxes,
                labels=labels,
                masks=masks,
                text=None,
                conf=conf,
                multimask=multimask,
                max_det=max_det,
                device=device,
                color_format=color_format,
                points_per_side=points_per_side,
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty concept string.")
        if not _is_empty_prompt(points) or not _is_empty_prompt(bboxes):
            raise ValueError("text prompts cannot be combined with points or bboxes.")
        if labels is not None or masks is not None:
            raise ValueError("text prompts cannot be combined with labels or masks.")
        if points_per_side is not None:
            raise ValueError("points_per_side only applies to segment-everything mode.")
        if multimask not in (None, False):
            raise ValueError("multimask only applies to visual prompts.")
        if device is not None:
            self._set_device(device)

        img, image_path, _ = self._resolve_image(source, color_format)
        model, processor = self._load_pcs()
        inputs = processor(images=img, text=text.strip(), return_tensors="pt")
        inputs = self._move_pcs_inputs(inputs, self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = inputs["original_sizes"].detach().cpu().tolist()
        processed = processor.post_process_instance_segmentation(
            outputs,
            # Keep candidates here so the shared Results assembly owns both
            # the default/explicit threshold semantics and max_det ordering.
            threshold=0.0,
            mask_threshold=0.5,
            target_sizes=target_sizes,
        )[0]
        result_masks = torch.as_tensor(processed["masks"]).detach().cpu().bool()
        scores = torch.as_tensor(processed["scores"]).detach().cpu().float()
        conf_thres = (
            self.DEFAULT_PCS_SCORE_THRESH if conf is None else float(conf)
        )
        return self._assemble_results(
            result_masks,
            scores,
            img,
            image_path,
            conf_thres=conf_thres,
            max_det=max_det,
            names={0: text.strip()},
        )

    def track(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreSAM3 supports image inference only; SAM 3 video models are "
            "outside the LibreSAM image contract."
        )


class LibreSAM31(LibreSAM3):
    """SAM 3.1 image segmenter on the SAM 3 path.

    SAM 3.1 keeps the SAM 3 image architecture with retrained detector and
    interactive-head weights; its Object Multiplex addition is video-only.
    Meta publishes only an original-format checkpoint, so LibreYOLO mirrors
    its image path converted to the Transformers SAM 3 layout
    (``weights/convert_sam31_weights.py``). Everything else, including text
    prompts, is inherited from ``LibreSAM3``. The mirror is gated and the
    weights stay under Meta's SAM License.
    """

    FAMILY = "sam31"
    FILENAME_PREFIX = "LibreSAM31"
    HF_REPOS: ClassVar[Dict[str, str]] = {"large": "LibreYOLO/LibreSAM31large"}
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"large": 1008}
    _LICENSE_NOTICE_SHOWN: ClassVar[bool] = False
    _LICENSE_NOTICE: ClassVar[str] = (
        "SAM 3.1 weights are provided by Meta under the custom SAM License, not "
        "LibreYOLO's MIT license. Use is subject to the terms accepted at "
        "https://huggingface.co/LibreYOLO/LibreSAM31large."
    )
    _GATED_HELP: ClassVar[str] = (
        "SAM 3.1 weights are gated. Visit "
        "https://huggingface.co/LibreYOLO/LibreSAM31large, accept the SAM "
        "License, then run `hf auth login` (or set HF_TOKEN) and retry."
    )

    def track(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreSAM31 supports image inference only; SAM 3.1 Object Multiplex "
            "video tracking is outside the LibreSAM image contract."
        )


__all__ = ["LibreSAM3", "LibreSAM31"]
