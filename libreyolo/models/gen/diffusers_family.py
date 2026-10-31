"""Shared adapter over the public ``diffusers`` pipeline API.

Original LibreYOLO adapter code; no upstream implementation is ported. A
family names its pipeline class and the knobs it accepts; loading, device
placement, seeding and the call itself are shared.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence, Tuple

from .base import _INSTALL_HINT, LibreGenModel

# LibreGen knob name -> diffusers pipeline argument.
_PIPELINE_ARGUMENTS: Dict[str, str] = {
    "steps": "num_inference_steps",
    "guidance": "guidance_scale",
    "negative": "negative_prompt",
}


class DiffusersGenFamily(LibreGenModel):
    """Common lazy loading and call contract for diffusers pipelines."""

    PIPELINE_CLASS: ClassVar[str] = ""
    MIN_DIFFUSERS: ClassVar[str] = "0.38.0"
    # Component folders only: a repo's root-level single-file checkpoint
    # duplicates the transformer and is not read by ``from_pretrained``.
    SNAPSHOT_ALLOW_PATTERNS: ClassVar[Tuple[str, ...]] = (
        "model_index.json",
        "*/*.json",
        "*/*.safetensors",
        "*/*.txt",
        "*/*.model",
        "*/*.jinja",
    )

    @classmethod
    def matches_directory(cls, path: Any) -> bool:
        """True when ``path`` holds a diffusers pipeline of this family's class."""
        try:
            index = Path(path) / "model_index.json"
            if not index.is_file():
                return False
            recorded = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return False
        return recorded.get("_class_name") == cls.PIPELINE_CLASS

    @classmethod
    def _require_pipeline_class(cls):
        try:
            import diffusers
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        pipeline_class = getattr(diffusers, cls.PIPELINE_CLASS, None)
        if pipeline_class is None:
            raise ImportError(
                f"{cls.__name__} needs diffusers>={cls.MIN_DIFFUSERS} "
                f"({cls.PIPELINE_CLASS} is missing from diffusers "
                f"{diffusers.__version__}).\n{_INSTALL_HINT}"
            )
        return pipeline_class

    def _check_runtime(self) -> None:
        self._require_pipeline_class()

    def _load(self, snapshot_dir: str) -> None:
        pipeline_class = self._require_pipeline_class()
        import diffusers

        # ``dtype`` replaced ``torch_dtype`` in diffusers 0.40.
        release = tuple(int(part) for part in diffusers.__version__.split(".")[:2])
        dtype_argument = "dtype" if release >= (0, 40) else "torch_dtype"
        pipeline = pipeline_class.from_pretrained(
            snapshot_dir, **{dtype_argument: self._resolve_dtype()}
        )
        if self.offload:
            pipeline.enable_model_cpu_offload(device=self.device)
        else:
            pipeline.to(self.device)
        self._pipeline = pipeline

    def _pipeline_arguments(self, knobs: Dict[str, Any]) -> Dict[str, Any]:
        """Map LibreGen knobs to pipeline arguments; families may extend it."""
        return {_PIPELINE_ARGUMENTS[name]: value for name, value in knobs.items()}

    def _generate(
        self,
        prompt: str,
        height: int,
        width: int,
        seeds: Sequence[int],
        knobs: Dict[str, Any],
    ) -> List[Any]:
        import torch

        # CPU generators make a seed reproduce across devices.
        generators = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]
        arguments = self._pipeline_arguments(knobs)
        output = self._pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_images_per_prompt=len(seeds),
            generator=generators,
            **arguments,
        )
        return list(output.images)
