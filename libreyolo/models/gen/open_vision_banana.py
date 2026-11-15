"""LibreYOLO adapter for Open Vision Banana (FunAILab).

Open Vision Banana fine-tunes the transformer of FLUX.2 [klein] Base so that
one model answers perception tasks as images: semantic, instance and
referring segmentation as colour masks, relative inverse depth as grey, and
surface normals as RGB. The prompt picks the task and the colours; the input
photo goes in as ``source``. It still edits images and generates from text.
The architecture is unchanged, so the pipeline is diffusers'
``Flux2KleinPipeline``.

LibreYOLO mirrors the paper's non-zero-shot 4B and 9B checkpoints as
complete pipelines: BFL's scheduler, Qwen3 text encoder, tokenizer and VAE
byte-identical, and FunAILab's transformer cast from fp32 to bf16, the
precision it runs in. ``UPSTREAM_REPOS`` records the sources. This module is
LibreYOLO's own adapter code and calls only the upstream public API; no
upstream source is ported. Reading masks, depth and normals out of the
output colours is shown in ``docs/open_vision_banana.md``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar

from .diffusers_family import DiffusersGenFamily

_OVB_REPO = "FunAILab/Open-Vision-Banana"
_OVB_REVISION = "be5da66250cc385ff9c75ebbe4114e13ec305848"


class LibreOpenVisionBanana(DiffusersGenFamily):
    """Open Vision Banana behind the LibreGen surface: a prompt, and an optional source image."""

    FAMILY = "open_vision_banana"
    FILENAME_PREFIX = "LibreOpenVisionBanana"
    UPSTREAM_REPOS: ClassVar[dict[str, str]] = {
        "4b": f"{_OVB_REPO}/final_models/4B/nonzeroshot/checkpoint-avg90000-92500-97500-100000_uniform",
        "9b": f"{_OVB_REPO}/final_models/9B/nonzeroshot/checkpoint-avg82500-85000-87500-92500-97500_uniform",
    }
    UPSTREAM_REVISIONS: ClassVar[dict[str, str]] = {
        "4b": _OVB_REVISION,
        "9b": _OVB_REVISION,
    }
    HF_REPOS: ClassVar[dict[str, str]] = {
        "4b": "LibreYOLO/LibreOpenVisionBanana4b",
        "9b": "LibreYOLO/LibreOpenVisionBanana9b",
    }
    HF_REVISIONS: ClassVar[dict[str, str]] = {
        "4b": "f279d483b6b7bab1ed9e9e503436e8d25e630496",
        "9b": "7a01bb2c1e9dd72678032beb126d2670cb724dd6",
    }
    INPUT_SIZES: ClassVar[dict[str, int]] = {"4b": 1024, "9b": 1024}
    PIPELINE_CLASS = "Flux2KleinPipeline"
    # The mirror's NOTICE names the model; it tells an OVB folder apart from
    # a plain FLUX.2 [klein] one, which has the same pipeline class.
    SNAPSHOT_ALLOW_PATTERNS = (*DiffusersGenFamily.SNAPSHOT_ALLOW_PATTERNS, "NOTICE")
    KNOBS = ("steps", "guidance")
    # The paper's sampling settings.
    DEFAULTS: ClassVar[dict[str, Any]] = {"steps": 28, "guidance": 3.5}

    _LICENSE_NOTICE = (
        "Open Vision Banana weights are derived from FLUX.2 [klein] Base and are "
        "under the FLUX Non-Commercial License "
        "(https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/blob/main/LICENSE.md): "
        "no commercial or production use."
    )

    @classmethod
    def matches_directory(cls, path: Any) -> bool:
        """A ``Flux2KleinPipeline`` folder whose NOTICE names Open Vision Banana."""
        if not super().matches_directory(path):
            return False
        try:
            return "Open Vision Banana" in (Path(path) / "NOTICE").read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            return False

    def generate(
        self,
        prompt: Any = None,
        *,
        source: Any = None,
        seed: int | None = None,
        save: bool = False,
        output_path: str | None = None,
        **knobs,
    ):
        """Run ``prompt`` on ``source``; without ``source``, generate from text.

        Args:
            prompt: The task prompt, e.g. a semantic segmentation prompt with
                its class colours (see ``docs/open_vision_banana.md``).
            source: One image (path, URL, ``PIL.Image``, NumPy array in BGR).
                The output keeps its aspect ratio, capped at one megapixel.
            seed: Recorded on the result; ``None`` draws a fresh one.
            save: Write the output image under ``runs/generate/predict``.
            output_path: Explicit output file, with ``save=True``.
            **knobs: ``steps`` (28) and ``guidance`` (3.5). Without a source,
                every LibreGen text-to-image argument (``imgsz``, ``n``,
                ``stream``) works as usual.

        Returns:
            ``Results`` whose ``generated`` holds the output RGB image.
        """
        if source is None:
            return super().generate(
                prompt, seed=seed, save=save, output_path=output_path, **knobs
            )
        import torch

        from .base3d import LibreGen3DModel

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "Open Vision Banana needs a task prompt with the source image, e.g. "
                'model("Estimate a dense relative inverse-depth map ...", source="photo.jpg").'
            )
        if output_path is not None and not save:
            raise ValueError("output_path requires save=True.")
        (seed,) = self._seeds(seed, 1)
        settings = self._resolve_knobs(knobs)
        image, _ = LibreGen3DModel._load_source(source)
        image = image.convert("RGB")
        self._ensure_loaded()
        started = time.perf_counter()
        output = self._pipeline(
            prompt=prompt,
            image=image,
            generator=torch.Generator("cpu").manual_seed(seed),
            **self._pipeline_arguments(settings),
        ).images[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = self._result(output, prompt, seed, settings, elapsed_ms)
        if save:
            self._save_image(result, 0, output_path, None)
        return result

    predict = generate
    __call__ = generate
