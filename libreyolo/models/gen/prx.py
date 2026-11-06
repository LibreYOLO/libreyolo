"""LibreYOLO adapters for PRX and PRXPixel (Photoroom).

PRX is a 1.3B-parameter text-to-image transformer trained from scratch by
Photoroom, with a T5Gemma 2B encoder and either the FLUX.1 VAE or a 32x DC-AE
autoencoder. It ships as nine checkpoints that differ in resolution (256,
512, 1024 beta), training stage (base, SFT, SFT plus 8-step distillation) and
autoencoder. PRXPixel is a 7B variant that denoises RGB directly, with no
autoencoder, conditioned on a Qwen3-VL text encoder.

Both load through ``diffusers`` (``PRXPipeline``, ``PRXPixelPipeline``) from
LibreYOLO's byte-identical mirrors of Photoroom's repos, each pinned to a
commit; ``UPSTREAM_REPOS`` and ``UPSTREAM_REVISIONS`` record the source. This
module is LibreYOLO's own adapter code and calls only the upstream public API;
no upstream source is ported.

PRX and PRXPixel weights are Apache-2.0. PRX bundles the FLUX.1 [schnell] VAE
(Apache-2.0) or the DC-AE autoencoder (MIT), and a T5Gemma text encoder that
stays under Google's Gemma Terms of Use, which the loader announces.
PRXPixel bundles the Qwen3-VL-Embedding-2B text encoder (Apache-2.0).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from .base import _INSTALL_HINT
from .diffusers_family import DiffusersGenFamily

_PHOTOROOM = "Photoroom"


class _NeedsFtfy:
    """Both PRX pipelines clean prompts with ftfy, an optional diffusers
    dependency; without it the first call dies on a NameError inside
    diffusers, so check it up front."""

    def _check_runtime(self) -> None:
        super()._check_runtime()
        try:
            import ftfy  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"{type(self).__name__} needs ftfy to clean prompts the way "
                f"the model was trained.\n{_INSTALL_HINT}"
            ) from exc

# Model card settings: 28 steps at guidance 5.0, and 8 steps without
# guidance for the LADD-distilled checkpoints.
_FULL = {"steps": 28, "guidance": 5.0, "negative": ""}
_DISTILLED = {"steps": 8, "guidance": 1.0, "negative": ""}


class LibrePRX(_NeedsFtfy, DiffusersGenFamily):
    """Photoroom PRX behind the LibreGen generate surface.

    Size codes name the checkpoint: resolution, then ``-dc-ae`` for the DC-AE
    autoencoder, then the training stage (``-sft``, ``-sft-distilled``), and
    ``1024-beta`` for the 1024 preview.
    """

    FAMILY = "prx"
    FILENAME_PREFIX = "LibrePRX"
    # Photoroom's repos and the commits LibreYOLO mirrored.
    UPSTREAM_REPOS: ClassVar[Dict[str, str]] = {
        "1024-beta": f"{_PHOTOROOM}/prx-1024-t2i-beta",
        "512-sft-distilled": f"{_PHOTOROOM}/prx-512-t2i-sft-distilled",
        "512-sft": f"{_PHOTOROOM}/prx-512-t2i-sft",
        "512": f"{_PHOTOROOM}/prx-512-t2i",
        "512-dc-ae-sft-distilled": f"{_PHOTOROOM}/prx-512-t2i-dc-ae-sft-distilled",
        "512-dc-ae-sft": f"{_PHOTOROOM}/prx-512-t2i-dc-ae-sft",
        "512-dc-ae": f"{_PHOTOROOM}/prx-512-t2i-dc-ae",
        "256-sft": f"{_PHOTOROOM}/prx-256-t2i-sft",
        "256": f"{_PHOTOROOM}/prx-256-t2i",
    }
    UPSTREAM_REVISIONS: ClassVar[Dict[str, str]] = {
        "1024-beta": "0370504b9a433fbdff814b8fcba9da03bf72a7a3",
        "512-sft-distilled": "cb72bdc1f4c77126fb80d0c03fab156ee5f4df3a",
        "512-sft": "2996423bc26e8eaca48774fac1797484214dfea0",
        "512": "fff12bc02365e07860f2567caa44f1f249946cd4",
        "512-dc-ae-sft-distilled": "03d133506d11f7b3087a9c9509259acd36ede975",
        "512-dc-ae-sft": "da9ca7af3f87383677ff7c63817a02c340a73e85",
        "512-dc-ae": "3db6a639a9782f6e05f952b75c4943c3875b19c8",
        "256-sft": "1b8e70f4dd51170fe7c03402f1891c15455d7447",
        "256": "39a6b3bc5adc5a0271acdcc20c19be2c355e227c",
    }
    HF_REPOS: ClassVar[Dict[str, str]] = {
        size: f"LibreYOLO/LibrePRX{size}" for size in UPSTREAM_REPOS
    }
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "1024-beta": "f0d9e1eedc9542a6cede46c685f23e51fcd61b1d",
        "512-sft-distilled": "5a7a28f300513257ca8a86053677fbafc1ac9f4e",
        "512-sft": "ec429cfb8171a3aa3a9ccdd416f987ae7a501705",
        "512": "5041bb88f9fc6d5e82ae596612c3bf324af40fbc",
        "512-dc-ae-sft-distilled": "22b50fc5e758eb10ba4a29bab6051bb035cd0fdb",
        "512-dc-ae-sft": "75f5738fbaf204648adbc8bef76405fc782a21de",
        "512-dc-ae": "8b8b57d42cd46f5ac3873977274409c1a7e8d615",
        "256-sft": "5217e6f68e5f4fd7db9e120f280bd8d8d2d0a475",
        "256": "c59d733f91685ce75880e9f3d1f5693e308cd0f1",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        size: int(size.split("-")[0]) for size in HF_REPOS
    }
    # FLUX VAE (8x) times patch 2 is 16; DC-AE (32x) times patch 1 is 32.
    SIZE_MULTIPLES: ClassVar[Dict[str, int]] = {
        size: 32 if "dc-ae" in size else 16 for size in HF_REPOS
    }
    SIZE_DEFAULTS: ClassVar[Dict[str, Dict[str, Any]]] = {
        size: dict(_DISTILLED if size.endswith("distilled") else _FULL)
        for size in HF_REPOS
    }
    PIPELINE_CLASS = "PRXPipeline"
    KNOBS = ("steps", "guidance", "negative")
    DEFAULTS: ClassVar[Dict[str, Any]] = dict(_FULL)

    _LICENSE_NOTICE = (
        "PRX weights are Apache-2.0. The bundled T5Gemma text encoder is under "
        "the Gemma Terms of Use (https://ai.google.dev/gemma/terms) and the "
        "Gemma Prohibited Use Policy."
    )

    def __init__(self, size: Optional[str] = None, **kwargs: Any):
        if size is None and kwargs.get("checkpoint_dir"):
            size = self._size_from_directory(kwargs["checkpoint_dir"])
            if size is None:
                # Falling back to the first size code would silently sample
                # at the wrong resolution with the wrong defaults.
                raise ValueError(
                    f"Cannot tell which PRX checkpoint {kwargs['checkpoint_dir']} "
                    f"is from its model_index.json; pass size= (one of "
                    f"{', '.join(self.HF_REPOS)})."
                )
        super().__init__(size, **kwargs)
        # One class serves nine checkpoints: the size code picks the
        # autoencoder's size multiple and the model card's sampling defaults.
        self.SIZE_MULTIPLE = self.SIZE_MULTIPLES[self.size]
        self.DEFAULTS = dict(self.SIZE_DEFAULTS[self.size])

    @classmethod
    def _size_from_directory(cls, path: Any) -> Optional[str]:
        """Closest size code for a local pipeline folder, from its config."""
        try:
            index = json.loads((Path(path) / "model_index.json").read_text("utf-8"))
        except (OSError, TypeError, ValueError):
            return None
        resolution = index.get("default_sample_size")
        dc_ae = (index.get("vae") or [None, None])[1] == "AutoencoderDC"
        if resolution == 1024 and not dc_ae:
            return "1024-beta"
        candidate = f"{resolution}{'-dc-ae' if dc_ae else ''}-sft"
        if candidate in cls.HF_REPOS:
            return candidate
        return None


class LibrePRXPixel(_NeedsFtfy, DiffusersGenFamily):
    """Photoroom PRXPixel (7B, pixel space) behind the LibreGen generate surface."""

    FAMILY = "prxpixel"
    FILENAME_PREFIX = "LibrePRXPixel"
    UPSTREAM_REPOS: ClassVar[Dict[str, str]] = {"7b": f"{_PHOTOROOM}/prxpixel-t2i"}
    UPSTREAM_REVISIONS: ClassVar[Dict[str, str]] = {
        "7b": "bcd5e63f072257a220c5d0ba039c97657398b1c2",
    }
    HF_REPOS: ClassVar[Dict[str, str]] = {"7b": "LibreYOLO/LibrePRXPixel7b"}
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "7b": "9fc490838d5f892c41b7d883af7bd67a9f445b25",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"7b": 1024}
    # No autoencoder: the pixel patch is 16.
    SIZE_MULTIPLE = 16
    PIPELINE_CLASS = "PRXPixelPipeline"
    MIN_DIFFUSERS = "0.39.0"
    KNOBS = ("steps", "guidance", "negative")
    DEFAULTS: ClassVar[Dict[str, Any]] = dict(_FULL)
