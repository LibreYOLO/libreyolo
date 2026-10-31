"""LibreYOLO adapter for SANA-Sprint (NVIDIA).

SANA-Sprint 0.6B is a 0.6B-parameter linear diffusion transformer distilled
for 1 to 4 sampling steps, with a Gemma 2 2B text encoder and a 32x DC-AE
autoencoder. The pipeline and its weights load through ``diffusers``
(``SanaSprintPipeline``) at a pinned Hub revision. This module is LibreYOLO's
own adapter code and calls only the upstream public API; no upstream source
is ported.

The SANA-Sprint weights are Apache-2.0. The bundled Gemma 2 text encoder
stays under Google's Gemma Terms of Use, which the loader announces.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .diffusers_family import DiffusersGenFamily


class LibreSanaSprint(DiffusersGenFamily):
    """SANA-Sprint behind the LibreGen generate surface."""

    FAMILY = "sana_sprint"
    FILENAME_PREFIX = "LibreSanaSprint"
    HF_REPOS: ClassVar[dict[str, str]] = {
        "0.6b": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    }
    HF_REVISIONS: ClassVar[dict[str, str]] = {
        "0.6b": "aa76e7f4f4928f378716b6716a2130fba3caf5b1",
    }
    INPUT_SIZES: ClassVar[dict[str, int]] = {"0.6b": 1024}
    SIZE_MULTIPLE = 32
    PIPELINE_CLASS = "SanaSprintPipeline"
    KNOBS = ("steps", "guidance")
    DEFAULTS: ClassVar[dict[str, Any]] = {"steps": 2, "guidance": 4.5}

    _LICENSE_NOTICE = (
        "SANA-Sprint weights are Apache-2.0. The bundled Gemma 2 text encoder "
        "is under the Gemma Terms of Use (https://ai.google.dev/gemma/terms) "
        "and the Gemma Prohibited Use Policy."
    )

    def _pipeline_arguments(self, knobs: dict[str, Any]) -> dict[str, Any]:
        arguments = super()._pipeline_arguments(knobs)
        # The pipeline's default intermediate timestep is defined for the
        # 2-step schedule only; any other step count uses the even schedule.
        if arguments.get("num_inference_steps") != 2:
            arguments["intermediate_timesteps"] = None
        return arguments
