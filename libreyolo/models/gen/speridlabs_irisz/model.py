"""Speridlabs Iris-Z: one generative model for images and their depth.

Iris-Z (https://speridlabs.com/research/iris-z) by Speridlabs is FLUX.2
fine-tuned to denoise an RGB grid and a depth grid together. The same weights
estimate depth from a photo in 4 steps and generate images from a prompt.
LibreYOLO exposes both:

    from libreyolo import LibreGen

    model = LibreGen("speridlabs-iris-z", task="depth")
    model("photo.jpg").save("depth.png")       # Results.depth_map, log depth

    model = LibreGen("speridlabs-iris-z")
    model("a lighthouse at dusk").save("lighthouse.png")

Two sizes: ``9b`` (Iris-Z-small, from FLUX.2 [klein] Base 9B, Qwen3 8B text
encoder) and ``32b`` (Iris-Z-big, from FLUX.2 [dev], Mistral Small 3.2 24B text
encoder). Depth never loads a text encoder: it conditions on the empty-prompt
embedding Speridlabs ships with the weights.

Weights are never redistributed by LibreYOLO. The transformer comes from
Speridlabs' own repo, ``speridlabs/iris-z`` (FLUX Non-Commercial License), at
a pinned revision; the FLUX.2 VAE from ``FLUX.2-klein-4B`` (Apache-2.0); the
text encoders from Mistral AI and Qwen (Apache-2.0). Nothing is gated.

The networks and samplers are ported from the Iris-Z SDK (Apache-2.0, see
``NOTICE``). As upstream, a seed drives a generator on the model's device, so
outputs differ across devices.
"""

from __future__ import annotations

import json
import logging
import time
from numbers import Integral
from pathlib import Path
from typing import Any, ClassVar, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np

from ....utils.results import DepthMap, Results
from ..base import _INSTALL_HINT, _SEED_RANGE, LibreGenModel
from ..image3d import FLUX2_VAE_REPO, FLUX2_VAE_REVISION

logger = logging.getLogger(__name__)

HF_REPO = "speridlabs/iris-z"
HF_REVISION = "16239e1a5444c484b11a04b61d633eceb768574c"
VARIANT_FOLDERS: Dict[str, str] = {"9b": "Iris-Z-small", "32b": "Iris-Z-big"}
_VARIANT_SIZES = {"small": "9b", "big": "32b"}
_WEIGHT_FILES = ("config.json", "transformer.safetensors.index.json", "empty_prompt.safetensors")
_BASE_MODELS = {
    "flux.2-dev": ("Flux2Params", "mistral_small"),
    "flux.2-klein-base-9b": ("Klein9BParams", "qwen3_8b"),
}
# Upstream's defaults: 50 steps for generation, 4 for depth with the median
# of the last 2 clean-depth estimates.
_GENERATE_DEFAULTS = {"9b": {"steps": 50, "guidance": 4.0}, "32b": {"steps": 50, "guidance": 3.5}}
_DEPTH_DEFAULTS = {"steps": 4, "ensemble": 1}
_DEPTH_AVERAGE_LAST = 2
_DEPTH_SIZES = (512, 1024)


def _variant_folder(path: Path, size: Optional[str]) -> Optional[Path]:
    """The Iris-Z variant folder at ``path``: the folder itself or its size's subfolder."""
    candidates = [path]
    if size is not None:
        candidates.append(path / VARIANT_FOLDERS[size])
    else:
        candidates.extend(path / folder for folder in VARIANT_FOLDERS.values())
    for folder in candidates:
        if all((folder / name).is_file() for name in _WEIGHT_FILES):
            return folder
    return None


def _folder_size(folder: Path) -> Optional[str]:
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _VARIANT_SIZES.get(config.get("variant"))


class _IrisZNetworks:
    """The loaded networks; ``to()`` moves the ones on the model's device."""

    def __init__(self, transformer, latents, empty_context, text_encoder, config):
        self.transformer = transformer
        self.latents = latents
        self.empty_context = empty_context
        self.text_encoder = text_encoder
        self.config = config

    def to(self, device, dtype=None):
        self.transformer.to(device)
        self.latents.to(device)
        self.empty_context = tuple(part.to(device) for part in self.empty_context)
        return self


class LibreSperidlabsIrisZ(LibreGenModel):
    """Speridlabs Iris-Z behind LibreGen: text-to-image (``generate``) and image-to-depth (``depth``)."""

    FAMILY = "speridlabs_irisz"
    FILENAME_PREFIX = "LibreSperidlabsIrisZ"
    HF_REPOS: ClassVar[Dict[str, str]] = {"9b": HF_REPO, "32b": HF_REPO}
    HF_REVISIONS: ClassVar[Dict[str, str]] = {"9b": HF_REVISION, "32b": HF_REVISION}
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"9b": 1024, "32b": 1024}
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("generate", "depth")
    DEFAULT_TASK: ClassVar[str] = "generate"
    SIZE_MULTIPLE = 16
    _LICENSE_NOTICE = (
        "Speridlabs Iris-Z weights are under the FLUX Non-Commercial License "
        "(https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.txt) "
        "and are downloaded from speridlabs/iris-z; LibreYOLO does not redistribute "
        "them. Generated images must be filtered or reviewed before you share them "
        "(section 2(e))."
    )

    def __init__(
        self,
        size: Optional[str] = None,
        *,
        device: str = "auto",
        dtype: str = "auto",
        offload: bool = False,
        task: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        if dtype not in ("auto", "bf16"):
            raise ValueError(
                f"{type(self).__name__} runs in its published bfloat16 precision; "
                f"dtype must be 'auto' or 'bf16', got {dtype!r}."
            )
        if size is None and checkpoint_dir is not None:
            folder = _variant_folder(Path(checkpoint_dir), None)
            size = _folder_size(folder) if folder is not None else None
        super().__init__(
            size, device=device, dtype=dtype, offload=offload, task=task, checkpoint_dir=checkpoint_dir
        )
        if self.offload and self.task != "generate":
            raise ValueError(
                "offload=True keeps the text encoder off the GPU between prompts; "
                "depth has no text encoder."
            )
        if self._checkpoint_dir is not None and _variant_folder(self._checkpoint_dir, self.size) is None:
            raise FileNotFoundError(
                f"{self._checkpoint_dir} holds no Iris-Z {VARIANT_FOLDERS[self.size]} weights "
                f"({', '.join(_WEIGHT_FILES)} and the transformer shards)."
            )
        # Only the size's own folder is downloaded from the shared repo.
        self.SNAPSHOT_ALLOW_PATTERNS = (f"{VARIANT_FOLDERS[self.size]}/*",)
        if self.task == "depth":
            self.KNOBS = tuple(_DEPTH_DEFAULTS)
            self.DEFAULTS = dict(_DEPTH_DEFAULTS)
        else:
            self.KNOBS = ("steps", "guidance")
            self.DEFAULTS = dict(_GENERATE_DEFAULTS[self.size])

    @property
    def model_name(self) -> str:
        return f"speridlabs-iris-z-{self.size}"

    @classmethod
    def matches_directory(cls, path: Any) -> bool:
        """True for a ``speridlabs/iris-z`` snapshot or one of its variant folders."""
        try:
            folder = _variant_folder(Path(path), None)
        except (TypeError, ValueError):
            return False
        return folder is not None and _folder_size(folder) is not None

    def _check_runtime(self) -> None:
        try:
            import diffusers.models  # noqa: F401
            import safetensors  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, snapshot_dir: str) -> None:
        import torch
        from safetensors.torch import load_file

        from . import nn as irisz_nn
        from .pipeline import Flux2Latents, text_context
        from .text import TEXT_ENCODERS

        folder = _variant_folder(Path(snapshot_dir), self.size)
        if folder is None:
            raise FileNotFoundError(f"No Iris-Z {VARIANT_FOLDERS[self.size]} weights in {snapshot_dir}.")
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if _VARIANT_SIZES.get(config.get("variant")) != self.size:
            raise ValueError(
                f"{folder} holds Iris-Z variant {config.get('variant')!r}, not {VARIANT_FOLDERS[self.size]}."
            )
        params_name, encoder_name = _BASE_MODELS[config["base_model"]]
        params = getattr(irisz_nn, params_name)()
        if config.get("guidance_mode") == "remove":
            params.use_guidance_embed = False
        device = self.device
        with torch.device("meta"):
            transformer = irisz_nn.IrisZTransformer(params, num_modalities=config["num_modalities"]).to(
                torch.bfloat16
            )
        index = json.loads((folder / "transformer.safetensors.index.json").read_text(encoding="utf-8"))
        state: Dict[str, Any] = {}
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(str(folder / shard), device=str(device)))
        transformer.load_state_dict(state, strict=True, assign=True)
        del state
        transformer = transformer.to(device).eval().requires_grad_(False)

        latents = Flux2Latents(self._load_vae()).to(device)
        empty = load_file(str(folder / "empty_prompt.safetensors"), device=str(device))["empty_prompt"]
        empty_context = text_context(empty)

        text_encoder = None
        if self.task == "generate":
            encoder_device = "cpu" if self.offload else device
            text_encoder = TEXT_ENCODERS[encoder_name](device=encoder_device).eval().requires_grad_(False)
        self._pipeline = _IrisZNetworks(transformer, latents, empty_context, text_encoder, config)

    @staticmethod
    def _load_vae():
        import diffusers
        import torch
        from diffusers.models import AutoencoderKLFlux2

        # ``dtype`` replaced ``torch_dtype`` in diffusers 0.40.
        release = tuple(int(part) for part in diffusers.__version__.split(".")[:2])
        dtype_argument = "dtype" if release >= (0, 40) else "torch_dtype"
        vae = AutoencoderKLFlux2.from_pretrained(
            FLUX2_VAE_REPO,
            subfolder="vae",
            revision=FLUX2_VAE_REVISION,
            **{dtype_argument: torch.bfloat16},
        )
        # Upstream keeps the VAE in float32 and runs it under bfloat16
        # autocast; this copy is published in bfloat16, so convolutions match
        # and only the float32 paths (norms) see bfloat16-rounded parameters.
        return vae.float().eval().requires_grad_(False)

    def to(self, device) -> "LibreSperidlabsIrisZ":
        device = self._resolve_device(device)
        if self.offload:
            raise ValueError("to() is not available with offload=True.")
        self.device = device
        if self._pipeline is not None:
            self._pipeline.to(device)
            if self._pipeline.text_encoder is not None:
                self._pipeline.text_encoder.to(device)
        return self

    def info(self, verbose: bool = True) -> Dict[str, Any]:
        data = super().info(verbose=False)
        data["weights"] = f"{HF_REPO}/{VARIANT_FOLDERS[self.size]}"
        if self.task == "depth":
            data["imgsz"] = "auto (512 under 0.5 MP, else 1024)"
        if verbose:
            logger.info(json.dumps(data, indent=2))
        return data

    def _autocast(self):
        import torch

        return torch.autocast(self.device.type, dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # Text -> image
    # ------------------------------------------------------------------

    def _prompt_context(self, prompt: str):
        """[unconditional, conditional] prompt embeddings as one two-row context."""
        import torch

        from .pipeline import text_context

        encoder = self._pipeline.text_encoder
        if self.offload:
            encoder.to(self.device)
        try:
            embedding = encoder(["", prompt]).to(torch.bfloat16)
        finally:
            if self.offload:
                encoder.to("cpu")
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        return text_context(embedding.to(self.device))

    def _generate(
        self,
        prompt: str,
        height: int,
        width: int,
        seeds: Sequence[int],
        knobs: Dict[str, Any],
    ) -> List[Any]:
        import torch
        from PIL import Image

        from .pipeline import sample_rgb

        if self.task != "generate":
            raise RuntimeError("This model was loaded for depth; load it with task='generate' to generate images.")
        steps, guidance = _check_steps(knobs["steps"]), _check_guidance(knobs["guidance"])
        networks = self._pipeline
        images = []
        with torch.no_grad():
            context = self._prompt_context(prompt)
            for seed in seeds:
                with self._autocast():
                    latent = sample_rgb(networks.transformer, context, height, width, seed, guidance, steps)
                    rgb = networks.latents.decode(latent).float().clamp(-1, 1)
                pixels = (127.5 * (rgb[0] + 1.0)).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                images.append(Image.fromarray(pixels))
        return images

    # ------------------------------------------------------------------
    # Image -> depth
    # ------------------------------------------------------------------

    def generate(self, prompt: Any = None, *, source: Any = None, **kwargs):
        """Text -> image with ``task="generate"``; image -> depth with ``task="depth"``.

        With ``task="depth"`` the first argument is the image source; see
        :meth:`predict_depth`.
        """
        if self.task == "depth":
            if prompt is not None and source is not None:
                raise TypeError("Pass the image once, as the first argument or as source=.")
            return self.predict_depth(source if source is not None else prompt, **kwargs)
        if source is not None:
            raise NotImplementedError(
                f"{type(self).__name__} generates from a text prompt with task='generate'; "
                "for depth from an image, load it with task='depth'."
            )
        return super().generate(prompt, **kwargs)

    predict = generate
    __call__ = generate

    def predict_depth(
        self,
        source: Any = None,
        *,
        imgsz: Optional[int] = None,
        seed: Optional[int] = None,
        stream: bool = False,
        save: bool = False,
        output_path: Optional[str] = None,
        **knobs,
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Estimate relative depth for each source image.

        Args:
            source: An image path, URL, ``PIL.Image``, NumPy array (BGR),
                tensor, a directory of images, or a list of any of these.
            imgsz: Working resolution, 512 or 1024 (about that many pixels
                squared, keeping the aspect ratio). ``None`` picks 512 for
                images under half a megapixel and 1024 above, as upstream.
            seed: Seed of the first ensemble member (default 0, as upstream).
            stream: Return a generator instead of a list.
            save: Write each coloured depth map under ``runs/depth/predict``.
            output_path: Explicit file for one source's depth plot.
            **knobs: ``steps`` (4) and ``ensemble`` (1): the per-pixel median
                over that many seeds, each affinely aligned to the first.
                Speridlabs' benchmarks use ``ensemble=10``.

        Returns:
            ``Results`` for one source, a list otherwise, a generator with
            ``stream=True``. ``Results.depth_map`` is relative log depth at the
            source's size (``encoding="log_depth"``, larger is farther), with
            sky in ``valid_mask=False`` at the far end.
        """
        from ..base3d import LibreGen3DModel

        if source is None:
            raise ValueError('Depth needs a source image, e.g. model("photo.jpg").')
        if imgsz is not None and (isinstance(imgsz, bool) or imgsz not in _DEPTH_SIZES):
            raise ValueError(f"imgsz for depth must be 512, 1024 or None (auto), got {imgsz!r}.")
        if seed is None:
            seed = 0
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < _SEED_RANGE:
            raise ValueError(f"seed must be an integer in [0, 2**32), got {seed!r}.")
        settings = self._resolve_knobs(knobs)
        steps = _check_steps(settings["steps"])
        ensemble = settings["ensemble"]
        if isinstance(ensemble, bool) or not isinstance(ensemble, Integral) or ensemble < 1:
            raise ValueError(f"ensemble must be a positive integer, got {ensemble!r}.")
        if steps <= _DEPTH_AVERAGE_LAST:
            raise ValueError(f"Depth needs more than {_DEPTH_AVERAGE_LAST} steps, got steps={steps}.")
        sources, single = LibreGen3DModel._expand_sources(source)
        if output_path is not None and not save:
            raise ValueError("output_path requires save=True.")
        if output_path is not None and not single:
            raise ValueError("output_path names one file; it needs one source.")
        seeds = [(int(seed) + offset) % _SEED_RANGE for offset in range(int(ensemble))]

        def run() -> Generator[Results, None, None]:
            from ....utils.general import increment_path, log_saved_result

            self._ensure_loaded()
            save_dir: Optional[Path] = None
            for index, item in enumerate(sources):
                image, path = LibreGen3DModel._load_source(item)
                image = image.convert("RGB")
                started = time.perf_counter()
                log_depth, sky = self._depth(image, imgsz, steps, seeds)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                shape = (int(image.height), int(image.width))
                result = Results(
                    None,
                    shape,
                    path=path,
                    names={},
                    speed={"inference": elapsed_ms},
                    orig_img=image,
                    depth_map=DepthMap(log_depth, shape, encoding="log_depth", valid_mask=~sky),
                )
                if save:
                    if output_path is not None:
                        destination = Path(output_path)
                    else:
                        if save_dir is None:
                            save_dir = increment_path(Path("runs/depth/predict"), mkdir=True)
                            self._save_dir = save_dir
                        stem = Path(path).stem if path else "image"
                        destination = save_dir / f"{index:03d}_{stem}.png"
                    result.save(str(destination))
                    log_saved_result(result, destination)
                yield result

        if stream:
            return run()
        results = list(run())
        return results[0] if single else results

    def _depth(self, image, imgsz: Optional[int], steps: int, seeds: Sequence[int]):
        """``(log_depth, sky)`` at the image's size, as upstream's ``image_to_depth``."""
        import torch
        from PIL import Image

        from .pipeline import (
            SKY_VALUE,
            SMALL_IMAGE_PIXELS,
            working_size,
            ensemble,
            sample_depth,
            unsqueeze_from_ceiling,
        )

        networks = self._pipeline
        sky_ceiling = float(networks.config.get("sky_ceiling", 1.0))
        if imgsz is None:
            imgsz = 512 if image.width * image.height < SMALL_IMAGE_PIXELS else 1024
        width, height = working_size(image.width, image.height, imgsz)
        rgb = np.asarray(image.resize((width, height), Image.LANCZOS))

        def decode(latent):
            encoded = networks.latents.decode(latent).float().mean(dim=1).clamp(-1, 1)[0].cpu().numpy()
            return unsqueeze_from_ceiling(encoded, sky_ceiling)

        with torch.no_grad(), self._autocast():
            pixels = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
            rgb_latent = networks.latents.encode(pixels.to(self.device))
            members = sample_depth(
                networks.transformer,
                networks.empty_context,
                rgb_latent,
                list(seeds),
                steps,
                average_last=_DEPTH_AVERAGE_LAST,
                decode=decode,
            )
        log_depth = (members[0] if len(members) == 1 else ensemble(members)).astype(np.float32)
        sky = log_depth >= SKY_VALUE if sky_ceiling < 1.0 else np.zeros(log_depth.shape, dtype=bool)
        if (image.width, image.height) != (width, height):
            log_depth = np.asarray(
                Image.fromarray(log_depth).resize((image.width, image.height), Image.BILINEAR)
            )
            sky = (
                np.asarray(
                    Image.fromarray(sky.astype(np.uint8) * 255).resize((image.width, image.height), Image.NEAREST)
                )
                > 127
            )
        return np.ascontiguousarray(log_depth, dtype=np.float32), np.ascontiguousarray(sky)


def _check_steps(steps: Any) -> int:
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}.")
    return int(steps)


def _check_guidance(guidance: Any) -> float:
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or guidance < 0:
        raise ValueError(f"guidance must be a non-negative number, got {guidance!r}.")
    return float(guidance)
