"""Speridlabs Idol: one image in, the whole object out as 3D Gaussians.

Idol (https://speridlabs.com/research/idol) is an open image-to-3D model by
Speridlabs. A 0.66B rectified-flow transformer denoises an object latent and
the input camera, conditioned on DINOv3 and FLUX.2 VAE tokens of the
1024px input; an octree decoder places anchors and a Gaussian decoder turns
each anchor into 32 Gaussians. Code and weights are Apache-2.0.

The networks and pipeline are ported from the Idol SDK (see ``NOTICE``) and
reproduce it: on CUDA, a seed gives the object the SDK gives for the same
seed and settings. Following the SDK, the seed drives a generator on the
model's device, so objects differ across devices.

Weights are not redistributed. Idol's own files come from the pinned
``speridlabs/idol`` snapshot; DINOv3 (gated, Meta's DINOv3 License) and the
FLUX.2 VAE (Apache-2.0, read from ``FLUX.2-klein-4B``, tensor-identical in
bf16 to the copy in the gated ``FLUX.2-dev`` repo the SDK names) load from
their publishers at pinned revisions. Background removal runs on LibreYOLO's
BiRefNet network with the BiRefNet weights from the Idol snapshot.
"""

from __future__ import annotations

import json
import math
import os
import threading
from contextlib import contextmanager
from numbers import Integral, Real
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence, Tuple

from ....utils.results import Gaussians
from ..base import _INSTALL_HINT
from ..base3d import LibreGen3DModel

_DINOV3_REPO = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
_DINOV3_REVISION = "c807c9eeea853df70aec4069e6f56b28ddc82acc"
_FLUX2_VAE_REPO = "black-forest-labs/FLUX.2-klein-4B"
_FLUX2_VAE_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
_WEIGHT_FILES = (
    "config.json",
    "denoiser.safetensors",
    "decoder.safetensors",
    "decoder_gs.safetensors",
    "birefnet.safetensors",
)
# BiRefNet heads used only for training supervision; Idol ships without them.
_BIREFNET_TRAINING_ONLY = ("decoder.conv_ms_spvn_", "decoder.gdt_convs_pred_")
_MODES = ("synthetic", "real")


class EncoderAccessError(RuntimeError):
    """The Hugging Face account in use cannot download a gated image encoder."""


# What ``AutoModel.from_pretrained`` reads from an encoder repo.
_ENCODER_FILES = ("config.json", "model.safetensors")


def _encoder_cached(repo: str, revision: str) -> bool:
    """True when the pinned encoder files are already in the Hugging Face cache."""
    from huggingface_hub import try_to_load_from_cache

    return all(
        isinstance(try_to_load_from_cache(repo, name, revision=revision), str)
        for name in _ENCODER_FILES
    )


def require_encoder_access(repos: Dict[str, str]) -> None:
    """Stop before anything large downloads when a gated encoder is not readable.

    ``repos`` maps each gated repo to its pinned revision. An encoder already
    in the cache needs no access, and network problems are not treated as
    missing access.
    """
    if os.environ.get("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes"):
        return
    from huggingface_hub import auth_check, get_token
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    denied = []
    for repo, revision in repos.items():
        if _encoder_cached(repo, revision):
            continue
        try:
            auth_check(repo)
        except (GatedRepoError, RepositoryNotFoundError):
            denied.append(repo)
        except Exception:
            pass
    if not denied:
        return
    pages = "\n".join(f"  https://huggingface.co/{repo}" for repo in denied)
    if get_token() is None:
        reason = (
            "You are not logged in to Hugging Face, and this image encoder is gated:"
        )
        steps = (
            "1. Accept the terms on the page above with your Hugging Face account.\n"
            "2. Log in: run `hf auth login`, or set HF_TOKEN to a token of that account."
        )
    else:
        reason = "Your Hugging Face account does not have access yet to this gated image encoder:"
        steps = (
            "1. Accept the terms on the page above (approval can take a while).\n"
            "2. Check that the token in use (`hf auth whoami`) belongs to the account that accepted them."
        )
    raise EncoderAccessError(
        f"Speridlabs Idol reads its input through DINOv3.\n{reason}\n{pages}\n{steps}"
    )


class LibreSperidlabsIdol(LibreGen3DModel):
    """Speridlabs Idol behind the LibreGen surface: ``model("chair.jpg").save("chair.ply")``."""

    FAMILY = "speridlabs_idol"
    FILENAME_PREFIX = "LibreSperidlabsIdol"
    HF_REPOS: ClassVar[Dict[str, str]] = {"0.6b": "speridlabs/idol"}
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "0.6b": "2731c2291fc65a797c5cb627a5e5526fc1490902"
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"0.6b": 1024}
    SNAPSHOT_ALLOW_PATTERNS: ClassVar[Tuple[str, ...]] = _WEIGHT_FILES
    KNOBS = ("steps", "guidance", "shift", "gaussians", "mode", "margin")
    DEFAULTS: ClassVar[Dict[str, Any]] = {
        "steps": 50,
        "guidance": 3.0,
        "shift": 3.0,
        "gaussians": 262144,
        "mode": "synthetic",
        "margin": None,
    }
    _LICENSE_NOTICE = (
        "Speridlabs Idol code and weights are Apache-2.0. Its DINOv3 image encoder "
        "is downloaded from Meta under the DINOv3 License."
    )

    @classmethod
    def matches_directory(cls, path: Any) -> bool:
        """True when ``path`` holds the Idol snapshot files."""
        try:
            folder = Path(path)
            if not all((folder / name).is_file() for name in _WEIGHT_FILES):
                return False
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return False
        return (
            "image_encoders" in config
            and "denoiser" in config
            and "decoder_gs" in config
        )

    def _check_runtime(self) -> None:
        try:
            import diffusers.models  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        require_encoder_access({_DINOV3_REPO: _DINOV3_REVISION})

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, snapshot_dir: str) -> None:
        import diffusers
        import torch
        from diffusers.models import AutoencoderKLFlux2
        from safetensors.torch import load_file
        from transformers import AutoModel

        from ...birefnet.nn import LibreBiRefNetModel
        from .nn import (
            GaussianFixedlenDecoder,
            LatentSeqMMFlowModel,
            OctreeProbabilityFixedlenDecoder,
            compile_for_sampling,
        )
        from .pipeline import ImageEncoder

        folder = Path(snapshot_dir)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))

        def load(module, name, strict=True):
            state = load_file(str(folder / f"{name}.safetensors"))
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
            return module.eval().requires_grad_(False).to(self.device)

        denoiser = load(LatentSeqMMFlowModel(**config["denoiser"]["args"]), "denoiser")
        decoder = load(
            OctreeProbabilityFixedlenDecoder(**config["decoder"]["args"]), "decoder"
        )
        decoder_gs = load(
            GaussianFixedlenDecoder(**config["decoder_gs"]["args"]), "decoder_gs"
        )
        birefnet = load(LibreBiRefNetModel(size="l"), "birefnet", strict=False)
        if self.device.type == "cuda":
            birefnet.half()

        # ``dtype`` replaced ``torch_dtype`` in diffusers 0.40.
        release = tuple(int(part) for part in diffusers.__version__.split(".")[:2])
        vae_dtype_argument = "dtype" if release >= (0, 40) else "torch_dtype"
        dinov3 = AutoModel.from_pretrained(
            _DINOV3_REPO, revision=_DINOV3_REVISION, dtype=torch.bfloat16
        )
        flux2_vae = AutoencoderKLFlux2.from_pretrained(
            _FLUX2_VAE_REPO,
            subfolder="vae",
            revision=_FLUX2_VAE_REVISION,
            **{vae_dtype_argument: torch.bfloat16},
        )
        encoder = ImageEncoder(
            dinov3.eval().requires_grad_(False).to(self.device),
            flux2_vae.eval().requires_grad_(False).to(self.device),
        )
        if self.compile:
            compile_for_sampling(denoiser)
        self._pipeline = _IdolModules(
            config, denoiser, decoder, decoder_gs, birefnet, encoder
        )

    def _move_loaded(self, device) -> None:
        self._pipeline.to(device)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _resolve_knobs(self, knobs: Dict[str, Any]) -> Dict[str, Any]:
        settings = super()._resolve_knobs(knobs)
        if settings["mode"] not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {settings['mode']!r}.")
        count = settings["gaussians"]
        if (
            isinstance(count, bool)
            or not isinstance(count, Integral)
            or not 32_768 <= count <= 1_048_576
        ):
            raise ValueError(
                f"gaussians must be an integer from 32,768 to 1,048,576, got {count!r}."
            )
        settings["gaussians"] = int(count)
        steps = settings["steps"]
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError(f"steps must be a positive integer, got {steps!r}.")
        settings["steps"] = int(steps)
        margin = settings["margin"]
        if margin is not None and (
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
                f"guidance must be a number >= 0 (0 turns it off), got {guidance!r}."
            )
        return settings

    def _generate_3d(
        self, image, seeds: Sequence[int], knobs: Dict[str, Any]
    ) -> List[Gaussians]:
        import numpy as np
        import torch

        from .pipeline import CROP_MARGIN, decode_camera, preprocess_image, sample_flow

        modules = self._pipeline
        margin = (
            knobs["margin"]
            if knobs["margin"] is not None
            else CROP_MARGIN[knobs["mode"]]
        )
        # One generation at a time per process: compiled sampling switches
        # PyTorch's process-wide TF32 flag, and no other Idol call may run
        # under it (see ``_tf32_matmuls``).
        with _GENERATION_LOCK, torch.no_grad():
            prepared = preprocess_image(
                image,
                modules.birefnet,
                image_size=modules.config["image_size"],
                crop_margin=margin,
            )
            pixels = (
                torch.from_numpy(np.array(prepared))
                .permute(2, 0, 1)
                .float()
                .div(255.0)[None]
                .to(self.device)
            )
            denoiser = modules.denoiser
            objects = []
            for seed in seeds:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                cond = modules.encoder(pixels, generator=generator)
                latent = torch.randn(
                    1,
                    denoiser.q_token_length,
                    denoiser.in_channels,
                    generator=generator,
                    device=self.device,
                )
                camera = torch.randn(
                    1, 1, denoiser.cam_channels, generator=generator, device=self.device
                )
                with self._sampling_settings():
                    latent, camera = sample_flow(
                        denoiser,
                        latent,
                        camera,
                        cond,
                        steps=knobs["steps"],
                        shift=knobs["shift"],
                        guidance=knobs["guidance"],
                        batch_guidance=self.compile,
                    )
                per_anchor = modules.decoder_gs.rep_config["num_gaussians"]
                anchors = modules.decoder.sample(
                    latent,
                    num_points=max(1, int(knobs["gaussians"]) // per_anchor),
                    level=modules.config["max_voxel_level"],
                    generator=generator,
                )
                splats = modules.decoder_gs.to_gaussians(
                    anchors, modules.decoder_gs(anchors, latent)
                )
                objects.append(
                    Gaussians(
                        splats["xyz"],
                        splats["scaling"],
                        splats["rotation"],
                        splats["opacity"].reshape(-1),
                        splats["features_dc"],
                        camera={**decode_camera(camera[0, 0]), "margin": float(margin)},
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

    @contextmanager
    def _sampling_settings(self):
        """compile=True: TF32 matmuls while the denoiser samples (its attention is pinned to cuDNN at load)."""
        if not self.compile:
            yield
            return
        with _tf32_matmuls():
            yield


_GENERATION_LOCK = threading.Lock()


@contextmanager
def _tf32_matmuls():
    """Allow TF32 matmuls for the block, then restore the user's setting.

    PyTorch keeps this flag process-wide. Callers hold ``_GENERATION_LOCK``,
    so no other Idol generation (eager, or another compiled call's
    decoders) runs while it is on.
    """
    import torch

    saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved


class _IdolModules:
    """The loaded networks; ``model.pipeline`` returns this."""

    def __init__(self, config, denoiser, decoder, decoder_gs, birefnet, encoder):
        self.config = config
        self.denoiser = denoiser
        self.decoder = decoder
        self.decoder_gs = decoder_gs
        self.birefnet = birefnet
        self.encoder = encoder

    def to(self, device) -> "_IdolModules":
        for module in (
            self.denoiser,
            self.decoder,
            self.decoder_gs,
            self.birefnet,
            self.encoder.dinov3,
            self.encoder.flux2_vae,
        ):
            module.to(device)
        # BiRefNet runs in fp16 on CUDA only, as upstream loads it.
        if torch_device_type(device) == "cuda":
            self.birefnet.half()
        else:
            self.birefnet.float()
        return self


def torch_device_type(device) -> str:
    import torch

    return torch.device(device).type
