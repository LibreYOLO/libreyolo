"""LibreVocab1 Phase 2 — BaseModel-style wrapper for the trainable variant.

Mirrors ``LibreVocab1`` (Phase 1) on the ``BaseModel`` surface but builds
the LibreYOLO-owned decoder instead of vendoring SAM3's. Like Phase 1, it
does not load LibreYOLO-format checkpoints — there are none yet — and
therefore does not register itself for the factory's ``can_load`` routing.

Construct directly::

    model = LibreVocab1Phase2(size="s", device="cpu")
    out = model(image, prompts=["a photo of a person"])
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

import torch
import torch.nn as nn

from ...base import BaseModel
from .nn import LibreVocab1Phase2Network


class LibreVocab1Phase2(BaseModel):
    """Phase 2 LibreVocab1: CRADIOv4 backbone + LibreYOLO-owned open-vocab decoder.

    Sizes mirror Phase 1:
        - ``s`` -> CRADIOv4-SO400M (1152-d backbone)
        - ``h`` -> CRADIOv4-H      (1280-d backbone)

    Inference inputs are 800x800 by default (smaller than Phase 1's 1152x1152
    because the lighter decoder doesn't need SAM3's segmentation resolution).
    """

    FAMILY = "librevocab1_phase2"
    FILENAME_PREFIX = "LibreVocab1Phase2"
    INPUT_SIZES = {"s": 800, "h": 800}
    TRAIN_CONFIG = None  # populated when phase2.trainer lands
    SUPPORTS_SEG = False  # Phase 2 is detection-only by design

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # No checkpoint format ships yet. Once Phase 2 is trained and we
        # adopt the LibreYOLO checkpoint metadata convention, this returns
        # True for checkpoints with model_family == "librevocab1_phase2".
        return False

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # Open-vocab: prompt list determines K at call time, not training time.
        return None

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        m = re.search(r"librevocab1phase2([sh])(?:_|\.|$)", filename.lower())
        return m.group(1) if m else None

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        size: str = "s",
        device: str = "cuda",
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_layers: int = 6,
        nb_classes: Optional[int] = None,  # accepted for BaseModel parity, unused
        **kwargs,
    ) -> None:
        if size not in self.INPUT_SIZES:
            raise ValueError(
                f"size must be one of {sorted(self.INPUT_SIZES)}; got {size!r}"
            )
        # Stash builder args before super().__init__() because BaseModel.__init__
        # will call self._init_model() and we need them readable from there.
        self._hidden_dim = hidden_dim
        self._num_queries = num_queries
        self._num_layers = num_layers
        super().__init__(
            model_path=None,
            size=size,
            nb_classes=nb_classes if nb_classes is not None else 80,
            device=device,
        )
        self.model.eval()

    def _init_model(self) -> nn.Module:
        return LibreVocab1Phase2Network(
            size=self.size,
            hidden_dim=self._hidden_dim,
            num_queries=self._num_queries,
            num_layers=self._num_layers,
            device=str(self.device),
        )

    # ------------------------------------------------------------------
    # BaseModel surface (placeholders until trainer + inference parity land)
    # ------------------------------------------------------------------

    def _get_available_layers(self) -> List[str]:
        return []

    def _preprocess(self, image_input: Any, **kwargs):
        raise NotImplementedError(
            "LibreVocab1Phase2 preprocess is pending: see PHASE2_PLAN.md "
            "Day 3 (open-vocab dataset wrapper)."
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        raise NotImplementedError(
            "LibreVocab1Phase2 forward is pending: see PHASE2_PLAN.md "
            "Day 1 (network) — currently scaffolded with NotImplementedError "
            "stubs in nn.py."
        )

    def _postprocess(self, predictions: Any, metadata: Any, **kwargs) -> Any:
        raise NotImplementedError("LibreVocab1Phase2 postprocess pending.")

    def _get_preprocess_numpy(self):
        raise NotImplementedError("LibreVocab1Phase2 preprocess_numpy pending.")
