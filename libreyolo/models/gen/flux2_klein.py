"""LibreYOLO adapter for FLUX.2 [klein] (Black Forest Labs).

FLUX.2 [klein] 4B is a 4B-parameter step-distilled rectified-flow transformer
with a Qwen3 text encoder. Code and weights are Apache-2.0. The pipeline and
its weights load through ``diffusers`` (``Flux2KleinPipeline``) at a pinned
Hub revision. This module is LibreYOLO's own adapter code and calls only the
upstream public API; no upstream source is ported.

The distilled checkpoint is trained for 4 steps at guidance 1.0, which are
the defaults here.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .diffusers_family import DiffusersGenFamily


class LibreFlux2Klein(DiffusersGenFamily):
    """FLUX.2 [klein] behind the LibreGen generate surface."""

    FAMILY = "flux2_klein"
    FILENAME_PREFIX = "LibreFlux2Klein"
    HF_REPOS: ClassVar[dict[str, str]] = {"4b": "black-forest-labs/FLUX.2-klein-4B"}
    HF_REVISIONS: ClassVar[dict[str, str]] = {
        "4b": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    }
    INPUT_SIZES: ClassVar[dict[str, int]] = {"4b": 1024}
    SIZE_MULTIPLE = 16
    PIPELINE_CLASS = "Flux2KleinPipeline"
    KNOBS = ("steps", "guidance")
    DEFAULTS: ClassVar[dict[str, Any]] = {"steps": 4, "guidance": 1.0}

    def _pipeline_arguments(self, knobs: dict[str, Any]) -> dict[str, Any]:
        # A step-distilled checkpoint runs without guidance: the pipeline
        # drops any other value, so refuse it instead of recording it.
        config = getattr(self._pipeline, "config", None)
        if getattr(config, "is_distilled", False) and knobs.get("guidance") != 1.0:
            raise ValueError(
                f"{type(self).__name__} is step-distilled and ignores guidance; "
                f"got guidance={knobs.get('guidance')!r}. Leave it at 1.0."
            )
        return super()._pipeline_arguments(knobs)
