"""TripoSplat: one image in, the whole object out as 3D Gaussians.

TripoSplat (https://www.tripo3d.ai/research/triposplat, "Generative 3D
Gaussians with Learned Density Control", SIGGRAPH 2026) is TripoAI's open
image-to-3D model. A 0.66B rectified-flow transformer denoises an object
latent and the input camera, conditioned on DINOv3 and FLUX.2 VAE tokens of
the 1024px input; an octree decoder places anchors and a Gaussian decoder
turns each anchor into 32 Gaussians, from 32,768 to 262,144 in all. Code and
weights are MIT.

The networks are ported from upstream (see ``NOTICE``) with unchanged
parameter names. Following upstream, the seed drives a generator on the
model's device, so objects differ across devices. Upstream draws the octree
sampling from the global generator; the port draws it from the seeded one,
so a seed reproduces the whole object.

Nothing is gated and no login is needed. TripoSplat's own files (denoiser,
decoder, BiRefNet) come from LibreYOLO's mirror ``LibreYOLO/LibreTripoSplat0.6b``,
byte-identical to the pinned ``VAST-AI/TripoSplat`` snapshot (MIT); the FLUX.2 VAE from
``FLUX.2-klein-4B`` (Apache-2.0); the DINOv3 encoder from LibreYOLO's bf16
mirror ``LibreYOLO/LibreDINOv3vith16plus`` (Meta's DINOv3 License). All at
pinned revisions. Background removal runs on LibreYOLO's BiRefNet network
with the BiRefNet weights from the TripoSplat snapshot.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

from ....utils.results import Gaussians
from ..base import _INSTALL_HINT
from ..base3d import LibreGen3DModel

_DENOISER_FILE = "diffusion_models/triposplat_fp16.safetensors"
_DECODER_FILE = "vae/triposplat_vae_decoder_fp16.safetensors"
_BIREFNET_FILE = "background_removal/birefnet.safetensors"
_WEIGHT_FILES = (_DENOISER_FILE, _DECODER_FILE, _BIREFNET_FILE)
# BiRefNet heads used only for training supervision; the snapshot may ship
# without them.
_BIREFNET_TRAINING_ONLY = ("decoder.conv_ms_spvn_", "decoder.gdt_convs_pred_")
_GAUSSIANS_RANGE = (32_768, 262_144)


class LibreTripoSplat(LibreGen3DModel):
    """TripoSplat behind the LibreGen surface: ``model("chair.jpg").save("chair.ply")``."""

    FAMILY = "triposplat"
    FILENAME_PREFIX = "LibreTripoSplat"
    HF_REPOS: ClassVar[Dict[str, str]] = {"0.6b": "LibreYOLO/LibreTripoSplat0.6b"}
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "0.6b": "dbef04688a6b3af979b735e20dbf09acbb941af7"
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"0.6b": 1024}
    SNAPSHOT_ALLOW_PATTERNS: ClassVar[Tuple[str, ...]] = _WEIGHT_FILES
    KNOBS = ("steps", "guidance", "shift", "gaussians", "margin")
    DEFAULTS: ClassVar[Dict[str, Any]] = {
        "steps": 20,
        "guidance": 3.0,
        "shift": 3.0,
        "gaussians": 262144,
        "margin": 1.2,
    }
    _LICENSE_NOTICE = (
        "TripoSplat code and weights are MIT. Its DINOv3 image encoder is Meta's, "
        "mirrored at LibreYOLO/LibreDINOv3vith16plus under the DINOv3 License, "
        "which restricts some uses (see the repository's LICENSE)."
    )

    def __init__(self, size: Optional[str] = None, *, compile: bool = False, **kwargs):
        if compile:
            raise ValueError(
                "LibreTripoSplat has no compiled sampler; compile=True is a "
                "Speridlabs Idol option."
            )
        super().__init__(size, compile=False, **kwargs)

    @classmethod
    def matches_directory(cls, path: Any) -> bool:
        """True when ``path`` holds the TripoSplat snapshot files (in their upstream folders)."""
        try:
            folder = Path(path)
            return all((folder / name).is_file() for name in _WEIGHT_FILES)
        except (OSError, TypeError, ValueError):
            return False

    def _check_runtime(self) -> None:
        try:
            import diffusers.models  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _network_dtype(device):
        """fp16 on CUDA and MPS, as upstream loads the networks; fp32 on CPU."""
        import torch

        return (
            torch.float16
            if torch.device(device).type in ("cuda", "mps")
            else torch.float32
        )

    def _load(self, snapshot_dir: str) -> None:
        from safetensors.torch import load_file

        from ...birefnet.nn import LibreBiRefNetModel
        from ..image3d import load_image_encoder
        from .nn import (
            FLOW_MODEL_ARGS,
            GS_DECODER_ARGS,
            OCTREE_DECODER_ARGS,
            LatentSeqMMFlowModel,
            OctreeGaussianDecoder,
        )

        folder = Path(snapshot_dir)
        dtype = self._network_dtype(self.device)

        def load(module, name, strict=True):
            state = load_file(str(folder / name))
            missing, unexpected = module.load_state_dict(state, strict=strict)
            if not strict:
                missing = [
                    key
                    for key in missing
                    if not key.startswith(_BIREFNET_TRAINING_ONLY)
                ]
                if missing or unexpected:
                    raise RuntimeError(
                        f"{name}: missing {missing[:5]}, unexpected {unexpected[:5]}"
                    )
            return module.eval().requires_grad_(False)

        denoiser = load(LatentSeqMMFlowModel(**FLOW_MODEL_ARGS), _DENOISER_FILE)
        decoder = load(
            OctreeGaussianDecoder(OCTREE_DECODER_ARGS, GS_DECODER_ARGS), _DECODER_FILE
        )
        birefnet = load(LibreBiRefNetModel(size="l"), _BIREFNET_FILE, strict=False)
        encoder = load_image_encoder(self.device)
        self._pipeline = _TripoSplatModules(denoiser, decoder, birefnet, encoder).to(
            self.device, dtype
        )

    def _move_loaded(self, device) -> None:
        self._pipeline.to(device, self._network_dtype(device))

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _resolve_knobs(self, knobs: Dict[str, Any]) -> Dict[str, Any]:
        settings = super()._resolve_knobs(knobs)
        count = settings["gaussians"]
        low, high = _GAUSSIANS_RANGE
        if (
            isinstance(count, bool)
            or not isinstance(count, Integral)
            or not low <= count <= high
        ):
            raise ValueError(
                f"gaussians must be an integer from {low:,} to {high:,}, got {count!r}."
            )
        settings["gaussians"] = int(count)
        steps = settings["steps"]
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError(f"steps must be a positive integer, got {steps!r}.")
        settings["steps"] = int(steps)
        margin = settings["margin"]
        if (
            isinstance(margin, bool)
            or not isinstance(margin, Real)
            or not math.isfinite(margin)
            or margin < 1
        ):
            raise ValueError(
                f"margin must be a number >= 1 (crop side / object size), got {margin!r}."
            )
        shift = settings["shift"]
        if (
            isinstance(shift, bool)
            or not isinstance(shift, Real)
            or not math.isfinite(shift)
            or shift <= 0
        ):
            raise ValueError(f"shift must be a positive number, got {shift!r}.")
        guidance = settings["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, Real)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError(
                f"guidance must be a number >= 0 (1 or less turns it off), got {guidance!r}."
            )
        return settings

    def _generate_3d(
        self, image, seeds: Sequence[int], knobs: Dict[str, Any]
    ) -> List[Gaussians]:
        import torch

        from ..image3d import decode_camera, pixels_of, preprocess_image
        from .nn import MAX_VOXEL_LEVEL
        from .pipeline import sample_flow

        modules = self._pipeline
        margin = float(knobs["margin"])
        with torch.no_grad():
            prepared = preprocess_image(
                image, modules.birefnet, image_size=self.input_size, crop_margin=margin
            )
            pixels = pixels_of(prepared, self.device)
            denoiser = modules.denoiser
            objects = []
            for seed in seeds:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                cond = modules.encoder(pixels, generator=generator)
                latent = torch.randn(
                    1,
                    denoiser.q_token_length,
                    denoiser.in_channels,
                    device=self.device,
                    generator=generator,
                )
                camera = torch.randn(
                    1, 1, denoiser.cam_channels, device=self.device, generator=generator
                )
                latent, camera = sample_flow(
                    denoiser,
                    latent,
                    camera,
                    cond,
                    steps=knobs["steps"],
                    shift=knobs["shift"],
                    guidance=knobs["guidance"],
                )
                splats = modules.decoder.decode(
                    latent,
                    knobs["gaussians"],
                    level=MAX_VOXEL_LEVEL,
                    generator=generator,
                )
                objects.append(
                    Gaussians(
                        splats["xyz"],
                        splats["scaling"],
                        splats["rotation"],
                        splats["opacity"].reshape(-1),
                        splats["features_dc"],
                        camera={**decode_camera(camera[0, 0]), "margin": margin},
                        seed=seed,
                        params={
                            "model": self.model_path
                            if self._checkpoint_dir
                            else self.model_name,
                            **knobs,
                        },
                        input_image=prepared,
                    )
                )
        return objects


class _TripoSplatModules:
    """The loaded networks; ``model.pipeline`` returns this."""

    def __init__(self, denoiser, decoder, birefnet, encoder):
        self.denoiser = denoiser
        self.decoder = decoder
        self.birefnet = birefnet
        self.encoder = encoder

    def to(self, device, dtype=None) -> "_TripoSplatModules":
        for module in (self.denoiser, self.decoder):
            module.to(device=device, dtype=dtype)
        self.birefnet.to(device=device, dtype=dtype)
        self.encoder.to(device)
        return self
