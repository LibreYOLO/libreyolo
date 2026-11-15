"""TRELLIS: one image in, the whole object out as 3D Gaussians.

TRELLIS (https://microsoft.github.io/TRELLIS, "Structured 3D Latents for
Scalable and Versatile 3D Generation", CVPR 2025) is Microsoft's open
image-to-3D model. A dense flow transformer denoises the object's occupancy
at 16^3, a 3D CNN decodes it to the active voxels of a 64^3 grid, a sparse
flow transformer denoises a structured latent on those voxels, and a
shifted-window sparse transformer turns each voxel into 32 Gaussians. Both
flow stages are conditioned on DINOv2 tokens of the 518px input. Code and
weights are MIT.

Upstream runs on CUDA only: its sparse convolutions come from spconv and its
sparse attention from xformers or flash-attn. This port replaces them with
PyTorch operations (``sparse.py``) and keeps the networks' parameter names,
so the released weights load unchanged and the same model runs on CUDA,
Apple MPS and CPU. Noise is drawn on the CPU as upstream does, so a seed
gives the same object on every device.

Nothing is gated and no login is needed. The four networks used here come
from LibreYOLO's mirror ``LibreYOLO/LibreTRELLISl`` of
``microsoft/TRELLIS-image-large`` (MIT), the image encoder from
``facebook/dinov2-with-registers-large`` (Apache-2.0), both at pinned
revisions. Background removal runs on LibreYOLO's BiRefNet network with the
BiRefNet weights shipped in the mirror.
"""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence, Tuple

from ....utils.results import Gaussians
from ..base import _INSTALL_HINT
from ..base3d import LibreGen3DModel

_DINOV2_REPO = "facebook/dinov2-with-registers-large"
_DINOV2_REVISION = "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"
_NETWORKS = {
    "structure_flow": "ckpts/ss_flow_img_dit_L_16l8_fp16",
    "structure_decoder": "ckpts/ss_dec_conv3d_16l8_fp16",
    "latent_flow": "ckpts/slat_flow_img_dit_L_64l8p2_fp16",
    "gaussian_decoder": "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16",
}
_BIREFNET_FILE = "birefnet.safetensors"
# What upstream's TRELLIS-image-large layout must hold for LibreYOLO to load it.
_UPSTREAM_FILES = (
    "pipeline.json",
    *(
        f"{stem}{ext}"
        for stem in _NETWORKS.values()
        for ext in (".json", ".safetensors")
    ),
)
# LibreYOLO's mirror adds the BiRefNet weights; an upstream folder without
# them reads that one file from the pinned mirror.
_WEIGHT_FILES = (*_UPSTREAM_FILES, _BIREFNET_FILE)
# BiRefNet heads used only for training supervision.
_BIREFNET_TRAINING_ONLY = ("decoder.conv_ms_spvn_", "decoder.gdt_convs_pred_")
# Upstream's sampler settings (pipeline.json), shared by both stages.
_INTERVAL = (0.5, 1.0)


class LibreTRELLIS(LibreGen3DModel):
    """TRELLIS behind the LibreGen surface: ``model("chair.jpg").save("chair.ply")``."""

    FAMILY = "trellis"
    FILENAME_PREFIX = "LibreTRELLIS"
    HF_REPOS: ClassVar[Dict[str, str]] = {"l": "LibreYOLO/LibreTRELLISl"}
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "l": "75c784a8fa56ebfb76f46d9b9d830caec25826a7"
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"l": 518}
    SNAPSHOT_ALLOW_PATTERNS: ClassVar[Tuple[str, ...]] = _WEIGHT_FILES
    KNOBS = ("steps", "guidance", "shift", "margin")
    DEFAULTS: ClassVar[Dict[str, Any]] = {
        "steps": 25,
        "guidance": 5.0,
        "shift": 3.0,
        "margin": 1.2,
    }
    _LICENSE_NOTICE = ""

    def __init__(self, size=None, *, compile: bool = False, **kwargs):
        if compile:
            raise ValueError(
                "LibreTRELLIS has no compiled sampler; compile=True is a Speridlabs Idol option."
            )
        super().__init__(size, compile=False, **kwargs)

    @classmethod
    def matches_directory(cls, path: Any) -> bool:
        """True when ``path`` holds upstream's TRELLIS-image-large files (``birefnet.safetensors`` optional)."""
        try:
            folder = Path(path)
            if not all((folder / name).is_file() for name in _UPSTREAM_FILES):
                return False
            config = json.loads((folder / "pipeline.json").read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return False
        return config.get("name") == "TrellisImageTo3DPipeline"

    def _check_runtime(self) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, snapshot_dir: str) -> None:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel

        from ...birefnet.nn import LibreBiRefNetModel
        from . import nn as networks

        folder = Path(snapshot_dir)
        pipeline = json.loads((folder / "pipeline.json").read_text(encoding="utf-8"))[
            "args"
        ]
        # The torso is fp16 on CUDA and MPS, as upstream converts it; fp32 on
        # CPU, where half-precision 3D convolution is not available.
        use_fp16 = self.device.type in ("cuda", "mps")
        modules = {}
        for name, stem in _NETWORKS.items():
            config = json.loads((folder / f"{stem}.json").read_text(encoding="utf-8"))
            args = dict(config["args"], use_fp16=use_fp16)
            module = getattr(networks, config["name"])(**args)
            module.load_state_dict(
                load_file(str(folder / f"{stem}.safetensors")), strict=True
            )
            modules[name] = module.eval().requires_grad_(False).to(self.device)
        birefnet_file = folder / _BIREFNET_FILE
        if not birefnet_file.is_file():
            from huggingface_hub import hf_hub_download

            birefnet_file = Path(
                hf_hub_download(
                    self.HF_REPOS[self.size],
                    _BIREFNET_FILE,
                    revision=self.HF_REVISIONS[self.size],
                )
            )
        birefnet = LibreBiRefNetModel(size="l")
        missing, unexpected = birefnet.load_state_dict(
            load_file(str(birefnet_file)), strict=False
        )
        missing = [
            key for key in missing if not key.startswith(_BIREFNET_TRAINING_ONLY)
        ]
        if missing or unexpected:
            raise RuntimeError(
                f"birefnet: missing {missing[:5]}, unexpected {unexpected[:5]}"
            )
        birefnet = birefnet.eval().requires_grad_(False).to(self.device)
        if self.device.type == "cuda":
            birefnet.half()
        dinov2 = AutoModel.from_pretrained(
            _DINOV2_REPO, revision=_DINOV2_REVISION, dtype=torch.float32
        )
        dinov2 = dinov2.eval().requires_grad_(False).to(self.device)
        self._pipeline = _TRELLISModules(
            device=self.device,
            dinov2=dinov2,
            birefnet=birefnet,
            normalization=pipeline["slat_normalization"],
            **modules,
        )

    def to(self, device) -> "LibreTRELLIS":
        """Move before loading; a loaded model keeps its device, whose precision it was built for."""
        target = self._resolve_device(device)
        if self._pipeline is not None and target != self.device:
            raise ValueError(
                "LibreTRELLIS picks its precision per device at load; construct it "
                "with device=... instead of to() after it has loaded."
            )
        self.device = target
        return self

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _resolve_knobs(self, knobs: Dict[str, Any]) -> Dict[str, Any]:
        settings = super()._resolve_knobs(knobs)
        steps = settings["steps"]
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError(f"steps must be a positive integer, got {steps!r}.")
        settings["steps"] = int(steps)
        for name, low, inclusive in (
            ("guidance", 0, True),
            ("shift", 0, False),
            ("margin", 1, True),
        ):
            value = settings[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or (value < low if inclusive else value <= low)
            ):
                relation = ">=" if inclusive else ">"
                raise ValueError(
                    f"{name} must be a number {relation} {low}, got {value!r}."
                )
            settings[name] = float(value)
        return settings

    def _generate_3d(
        self, image, seeds: Sequence[int], knobs: Dict[str, Any]
    ) -> List[Gaussians]:
        import torch

        from .pipeline import encode_image, preprocess_image, run_stages

        modules = self._pipeline
        stage = {
            "steps": knobs["steps"],
            "guidance": knobs["guidance"],
            "interval": _INTERVAL,
            "shift": knobs["shift"],
        }
        objects = []
        with torch.no_grad():
            prepared = preprocess_image(
                image, modules.birefnet, crop_margin=knobs["margin"]
            )
            # One image, so one set of tokens for every seed.
            cond = encode_image(modules.dinov2, prepared, modules.device)
            for seed in seeds:
                splats = run_stages(
                    modules, prepared, seed, structure=stage, latent=stage, cond=cond
                )["gaussians"]
                objects.append(
                    Gaussians(
                        splats["xyz"],
                        splats["scales"],
                        splats["rotations"],
                        splats["opacities"],
                        splats["sh"],
                        # TRELLIS infers no camera; the margin is the crop it saw.
                        camera={"margin": knobs["margin"]},
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


class _TRELLISModules:
    """The loaded networks; ``model.pipeline`` returns this."""

    def __init__(
        self,
        *,
        device,
        dinov2,
        birefnet,
        normalization,
        structure_flow,
        structure_decoder,
        latent_flow,
        gaussian_decoder,
    ):
        self.device = device
        self.dinov2 = dinov2
        self.birefnet = birefnet
        self.normalization = normalization
        self.structure_flow = structure_flow
        self.structure_decoder = structure_decoder
        self.latent_flow = latent_flow
        self.gaussian_decoder = gaussian_decoder
