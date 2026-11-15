"""LGM: one image in, the whole object out as 3D Gaussians.

LGM ("Large Multi-View Gaussian Model for High-Resolution 3D Content
Creation", ECCV 2024, https://me.kiui.moe/lgm/) works in two stages.
ImageDream, a multi-view diffusion model fine-tuned from Stable Diffusion
2.1, samples four views of the object around it from the input image; LGM,
an asymmetric UNet with cross-view attention, reads the four views with
their camera rays and returns one Gaussian per output pixel of each view,
4 x 128 x 128 = 65,536 Gaussians, of which those above opacity 0.005 are
kept (as upstream's saved files keep them). LGM's code and weights are MIT;
ImageDream's weights are OpenRAIL, which carries use restrictions.

Upstream needs xformers (CUDA) for every attention and renders with a
non-commercial rasterizer; this port runs the attention through PyTorch's
SDPA and does not render, so the same model runs on CUDA, Apple MPS and CPU.
The networks run in fp16 on CUDA and MPS, as upstream on CUDA, and in fp32
on CPU. Noise comes from a generator on the model's device seeded like
upstream's demo seeds the global one, so on CUDA a seed reproduces
upstream's object; other devices draw other noise for the same seed.

Nothing is gated and no login is needed. LGM's network and the BiRefNet
background remover come from ``LibreYOLO/LibreLGM0.4b``, a mirror of
``ashawkey/LGM``; ImageDream from ``LibreYOLO/LibreImageDream``, a mirror of
``ashawkey/imagedream-ipmv-diffusers``; both at pinned revisions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any, ClassVar

from ....utils.results import Gaussians
from ..base import _INSTALL_HINT
from ..base3d import LibreGen3DModel

_LGM_FILE = "model_fp16_fixrot.safetensors"
_BIREFNET_FILE = "birefnet.safetensors"
# The multi-view stage: LibreYOLO's byte-identical mirror of
# ashawkey/imagedream-ipmv-diffusers@73a034178e748421506492e91790cc62d6aefef5,
# without upstream's remote-code .py files.
IMAGEDREAM_REPO = "LibreYOLO/LibreImageDream"
IMAGEDREAM_REVISION = "9076f3a181027f8172ec1abed27faddfeca7f7d5"
_IMAGEDREAM_DIR = "imagedream"  # an optional local copy inside a model directory
_IMAGEDREAM_COMPONENTS = (
    "feature_extractor",
    "image_encoder",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "unet",
    "vae",
)
_BIREFNET_TRAINING_ONLY = ("decoder.conv_ms_spvn_", "decoder.gdt_convs_pred_")


class LibreLGM(LibreGen3DModel):
    """LGM behind the LibreGen surface: ``model("chair.jpg").save("chair.ply")``."""

    FAMILY = "lgm"
    FILENAME_PREFIX = "LibreLGM"
    HF_REPOS: ClassVar[dict[str, str]] = {"0.4b": "LibreYOLO/LibreLGM0.4b"}
    HF_REVISIONS: ClassVar[dict[str, str]] = {
        "0.4b": "492f8ab3a45e79073df505ebc28f5c3beb479303"
    }
    INPUT_SIZES: ClassVar[dict[str, int]] = {"0.4b": 256}
    SNAPSHOT_ALLOW_PATTERNS: ClassVar[tuple[str, ...]] = (_LGM_FILE, _BIREFNET_FILE)
    KNOBS = ("steps", "guidance", "elevation")
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "steps": 30,
        "guidance": 5.0,
        "elevation": 0.0,
    }
    _LICENSE_NOTICE = (
        "LGM's multi-view stage (ImageDream, after Stable Diffusion 2.1) is under "
        "the OpenRAIL license, whose use restrictions apply to you; see "
        "https://huggingface.co/LibreYOLO/LibreImageDream."
    )

    def __init__(self, size=None, *, compile: bool = False, **kwargs):
        if compile:
            raise ValueError(
                "LibreLGM has no compiled sampler; compile=True is a Speridlabs Idol option."
            )
        super().__init__(size, compile=False, **kwargs)

    @classmethod
    def matches_directory(cls, path: Any) -> bool:
        """True when ``path`` holds LGM's ``model_fp16_fixrot.safetensors`` (``birefnet.safetensors`` and ``imagedream/`` optional)."""
        try:
            return (Path(path) / _LGM_FILE).is_file()
        except (OSError, TypeError, ValueError):
            return False

    def _check_runtime(self) -> None:
        try:
            import diffusers  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _download(self, repo: str, revision: str, filename: str | None = None) -> Path:
        from huggingface_hub import hf_hub_download, snapshot_download

        if filename is not None:
            return Path(hf_hub_download(repo, filename, revision=revision))
        return Path(snapshot_download(repo, revision=revision))

    def _load(self, snapshot_dir: str) -> None:
        import torch
        from diffusers import AutoencoderKL, DDIMScheduler
        from safetensors.torch import load_file
        from transformers import (
            CLIPImageProcessor,
            CLIPTextModel,
            CLIPTokenizer,
            CLIPVisionModel,
        )

        from ...birefnet.nn import LibreBiRefNetModel
        from .nn import LGMNetwork, MultiViewUNet
        from .pipeline import encode_text

        folder = Path(snapshot_dir)
        device = self.device
        # fp16 on CUDA and MPS, as upstream on CUDA; fp32 on CPU.
        half = device.type in ("cuda", "mps")
        dtype = torch.float16 if half else torch.float32

        lgm = LGMNetwork()
        lgm.load_state_dict(load_file(str(folder / _LGM_FILE)), strict=True)
        lgm = lgm.eval().requires_grad_(False).to(device)
        if half:
            lgm.half()

        birefnet_file = folder / _BIREFNET_FILE
        if not birefnet_file.is_file():
            birefnet_file = self._download(
                self.HF_REPOS[self.size], self.HF_REVISIONS[self.size], _BIREFNET_FILE
            )
        birefnet = LibreBiRefNetModel(size="l")
        missing, unexpected = birefnet.load_state_dict(
            load_file(str(birefnet_file)), strict=False
        )
        missing = [k for k in missing if not k.startswith(_BIREFNET_TRAINING_ONLY)]
        if missing or unexpected:
            raise RuntimeError(
                f"birefnet: missing {missing[:5]}, unexpected {unexpected[:5]}"
            )
        birefnet = birefnet.eval().requires_grad_(False).to(device)
        if device.type == "cuda":
            birefnet.half()

        mv_dir = folder / _IMAGEDREAM_DIR
        if not (mv_dir / "model_index.json").is_file():
            mv_dir = self._download(IMAGEDREAM_REPO, IMAGEDREAM_REVISION)
        unet_config = json.loads(
            (mv_dir / "unet" / "config.json").read_text(encoding="utf-8")
        )
        unet = MultiViewUNet(**unet_config)
        unet.load_state_dict(
            load_file(str(mv_dir / "unet" / "diffusion_pytorch_model.safetensors")),
            strict=True,
        )
        unet = unet.eval().requires_grad_(False).to(device=device, dtype=dtype)

        def pretrained(cls, name):
            return cls.from_pretrained(str(mv_dir / name), dtype=dtype)

        import diffusers

        release = tuple(int(p) for p in diffusers.__version__.split(".")[:2])
        vae_dtype = "dtype" if release >= (0, 40) else "torch_dtype"
        vae = AutoencoderKL.from_pretrained(str(mv_dir / "vae"), **{vae_dtype: dtype})
        vae = vae.eval().requires_grad_(False).to(device)
        image_encoder = pretrained(CLIPVisionModel, "image_encoder")
        image_encoder = image_encoder.eval().requires_grad_(False).to(device)
        text_encoder = pretrained(CLIPTextModel, "text_encoder").eval().to(device)
        tokenizer = CLIPTokenizer.from_pretrained(str(mv_dir / "tokenizer"))
        with torch.no_grad():
            # Image-to-3D always runs ImageDream with the empty prompt, so its
            # text tokens are a constant; the text encoder is not kept.
            text_tokens = encode_text(tokenizer, text_encoder, device)
        del text_encoder
        self._pipeline = _LGMModules(
            device=device,
            lgm=lgm,
            autocast=half,
            birefnet=birefnet,
            unet=unet,
            vae=vae,
            image_encoder=image_encoder,
            feature_extractor=CLIPImageProcessor.from_pretrained(
                str(mv_dir / "feature_extractor")
            ),
            scheduler=DDIMScheduler.from_pretrained(str(mv_dir / "scheduler")),
            text_tokens=text_tokens,
        )

    def to(self, device) -> LibreLGM:
        """Move before loading; a loaded model keeps its device, whose precision it was built for."""
        target = self._resolve_device(device)
        if self._pipeline is not None and target != self.device:
            raise ValueError(
                "LibreLGM picks its precision per device at load; construct it "
                "with device=... instead of to() after it has loaded."
            )
        self.device = target
        return self

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _resolve_knobs(self, knobs: dict[str, Any]) -> dict[str, Any]:
        settings = super()._resolve_knobs(knobs)
        steps = settings["steps"]
        if (
            isinstance(steps, bool)
            or not isinstance(steps, Integral)
            or not 1 <= steps <= 1000
        ):
            raise ValueError(f"steps must be an integer in [1, 1000], got {steps!r}.")
        settings["steps"] = int(steps)
        for name, low, high in (
            ("guidance", 1.0, math.inf),
            ("elevation", -89.0, 89.0),
        ):
            value = settings[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                bound = f">= {low}" if high == math.inf else f"in [{low}, {high}]"
                raise ValueError(f"{name} must be a number {bound}, got {value!r}.")
            settings[name] = float(value)
        return settings

    def _generate_3d(
        self, image, seeds: Sequence[int], knobs: dict[str, Any]
    ) -> list[Gaussians]:
        import torch
        from PIL import Image

        from .pipeline import encode_image, input_camera, preprocess_image, run_stages

        modules = self._pipeline
        objects = []
        with torch.no_grad():
            prepared = preprocess_image(image, modules.birefnet)
            # One image, so one set of CLIP tokens for every seed.
            tokens = encode_image(modules, prepared)
            seen = Image.fromarray((prepared * 255).round().astype("uint8"))
            for seed in seeds:
                splats = run_stages(
                    modules,
                    prepared,
                    seed,
                    steps=knobs["steps"],
                    guidance=knobs["guidance"],
                    elevation=knobs["elevation"],
                    image_tokens=tokens,
                )["gaussians"]
                objects.append(
                    Gaussians(
                        splats["xyz"],
                        splats["scales"],
                        splats["rotations"],
                        splats["opacities"],
                        splats["sh"],
                        camera=input_camera(knobs["elevation"]),
                        seed=seed,
                        params={
                            "model": self.model_path
                            if self._checkpoint_dir
                            else self.model_name,
                            **knobs,
                        },
                        input_image=seen,
                    )
                )
        return objects


class _LGMModules:
    """The loaded networks; ``model.pipeline`` returns this."""

    def __init__(self, **modules):
        self.__dict__.update(modules)
