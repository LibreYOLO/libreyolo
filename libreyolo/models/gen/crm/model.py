"""CRM: one image in, the whole object out as a textured triangle mesh.

CRM (https://ml.cs.tsinghua.edu.cn/~zhengyi/CRM, "Single Image to 3D
Textured Mesh with Convolutional Reconstruction Model", ECCV 2024) is
Tsinghua's open image-to-3D model. Two ImageDream-style multi-view diffusion
models first generate six orthographic views of the object and then their
canonical coordinate maps; a convolutional UNet turns them into a triplane,
and FlexiCubes extracts a mesh coloured by the triplane. Code and weights are
MIT (the diffusion weights are fine-tuned from Stable Diffusion 2.1, whose
CreativeML Open RAIL++-M use restrictions the card carries).

Upstream needs CUDA-only extensions (xformers, and nvdiffrast, whose licence
is non-commercial) and a mesh library under the GPL for cleanup. This port
runs the same weights in plain PyTorch on CUDA, Apple MPS and CPU: attention
through PyTorch's SDPA, FlexiCubes from NVIDIA's Apache-2.0 release, the
cleanup and the UV rasterization written for LibreYOLO. Noise is drawn on the
CPU, so a seed gives the same starting noise on every device.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any, ClassVar

from ....utils.results import Meshes
from ..base import _INSTALL_HINT
from ..base3d import LibreGen3DModel

_WEIGHT_FILES = ("pixel-diffusion.pth", "ccm-diffusion.pth", "CRM.pth")
_BIREFNET_REPO = "LibreYOLO/LibreTRELLISl"
_BIREFNET_REVISION = "75c784a8fa56ebfb76f46d9b9d830caec25826a7"
_BIREFNET_FILE = "birefnet.safetensors"
_BIREFNET_TRAINING_ONLY = ("decoder.conv_ms_spvn_", "decoder.gdt_convs_pred_")


class LibreCRM(LibreGen3DModel):
    """CRM behind the LibreGen surface: ``model("chair.png").save("chair.obj")``."""

    FAMILY = "crm"
    FILENAME_PREFIX = "LibreCRM"
    HF_REPOS: ClassVar[dict[str, str]] = {"": "LibreYOLO/LibreCRM"}
    HF_REVISIONS: ClassVar[dict[str, str]] = {"": "386740e20da920f1e73f4604dbdb484a6ddee9bb"}
    INPUT_SIZES: ClassVar[dict[str, int]] = {"": 256}
    SNAPSHOT_ALLOW_PATTERNS: ClassVar[tuple[str, ...]] = _WEIGHT_FILES
    SAVE_SUFFIXES: ClassVar[tuple[str, ...]] = (".obj",)
    KNOBS = ("steps", "guidance", "texture")
    DEFAULTS: ClassVar[dict[str, Any]] = {"steps": 50, "guidance": 5.0, "texture": False}
    _LICENSE_NOTICE = (
        "CRM's diffusion weights are fine-tuned from Stable Diffusion 2.1; the "
        "CreativeML Open RAIL++-M use restrictions apply to them "
        "(https://huggingface.co/LibreYOLO/LibreCRM)."
    )

    def __init__(self, size=None, *, compile: bool = False, **kwargs):
        if compile:
            raise ValueError("LibreCRM has no compiled sampler; compile=True is a Speridlabs Idol option.")
        super().__init__(size, compile=False, **kwargs)

    @property
    def model_name(self) -> str:
        return "crm"

    @classmethod
    def matches_directory(cls, path: Any) -> bool:
        """True when ``path`` holds upstream's three ``.pth`` files."""
        try:
            folder = Path(path)
            return all((folder / name).is_file() for name in _WEIGHT_FILES)
        except (OSError, TypeError, ValueError):
            return False

    def _check_runtime(self) -> None:
        try:
            import diffusers  # noqa: F401
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, snapshot_dir: str) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from ...birefnet.nn import LibreBiRefNetModel
        from .diffusion import MultiViewDiffusion
        from .reconstruction import CRMNet, Grid

        folder = Path(snapshot_dir)
        # fp16 on CUDA and MPS, as upstream casts the diffusion models; fp32 on CPU.
        half = self.device.type in ("cuda", "mps")
        stages = []
        for name, stage2 in (("pixel-diffusion.pth", False), ("ccm-diffusion.pth", True)):
            module = MultiViewDiffusion(stage2=stage2)
            state = torch.load(folder / name, map_location="cpu", weights_only=True)
            module.load_state_dict(state, strict=True)
            module = module.eval().requires_grad_(False)
            module = module.half() if half else module.float()
            stages.append(module.to(self.device))
            del state
        net = CRMNet()
        net.load_state_dict(
            torch.load(folder / "CRM.pth", map_location="cpu", weights_only=True), strict=True
        )
        net = net.eval().requires_grad_(False).to(self.device)
        birefnet = LibreBiRefNetModel(size="l")
        missing, unexpected = birefnet.load_state_dict(
            load_file(hf_hub_download(_BIREFNET_REPO, _BIREFNET_FILE, revision=_BIREFNET_REVISION)),
            strict=False,
        )
        missing = [k for k in missing if not k.startswith(_BIREFNET_TRAINING_ONLY)]
        if missing or unexpected:
            raise RuntimeError(f"birefnet: missing {missing[:5]}, unexpected {unexpected[:5]}")
        birefnet = birefnet.eval().requires_grad_(False).to(self.device)
        if self.device.type == "cuda":
            birefnet.half()
        self._pipeline = _CRMModules(
            device=self.device,
            half=half,
            pixel=stages[0],
            ccm=stages[1],
            net=net,
            grid=Grid(self.device),
            birefnet=birefnet,
        )

    def to(self, device) -> LibreCRM:
        """Move before loading; a loaded model keeps its device, whose precision it was built for."""
        target = self._resolve_device(device)
        if self._pipeline is not None and target != self.device:
            raise ValueError(
                "LibreCRM picks its precision per device at load; construct it with "
                "device=... instead of to() after it has loaded."
            )
        self.device = target
        return self

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _resolve_knobs(self, knobs: dict[str, Any]) -> dict[str, Any]:
        settings = super()._resolve_knobs(knobs)
        steps = settings["steps"]
        if isinstance(steps, bool) or not isinstance(steps, Integral) or not 1 <= steps <= 1000:
            raise ValueError(f"steps must be an integer in [1, 1000], got {steps!r}.")
        settings["steps"] = int(steps)
        guidance = settings["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, Real)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError(f"guidance must be a number >= 0, got {guidance!r}.")
        settings["guidance"] = float(guidance)
        if not isinstance(settings["texture"], bool):
            raise ValueError(f"texture must be True or False, got {settings['texture']!r}.")
        return settings

    def _generate_3d(self, image, seeds: Sequence[int], knobs: dict[str, Any]) -> list[Meshes]:
        import torch

        from .pipeline import generate_mesh, preprocess_image

        modules = self._pipeline
        prepared = preprocess_image(image, modules.birefnet)
        objects = []
        for seed in seeds:
            with torch.no_grad():
                out = generate_mesh(
                    modules, prepared, seed, steps=knobs["steps"], guidance=knobs["guidance"],
                    texture=knobs["texture"],
                )
            objects.append(
                mesh_payload(
                    out,
                    seed=seed,
                    params={
                        "model": self.model_path if self._checkpoint_dir else self.model_name,
                        **knobs,
                    },
                )
            )
        return objects


def mesh_payload(out: dict[str, Any], *, seed: int, params: dict[str, Any]) -> Meshes:
    """One generated object as a one-row ``Meshes`` payload (ADR 0031)."""
    import numpy as np

    extras = {"vertex_colors": out["vertex_colors"][None]}
    if "texture" in out:
        extras.update(
            uv=out["uv"][None], uv_faces=out["uv_faces"][None], texture=out["texture"][None]
        )
    return Meshes(
        np.zeros((1, 0), np.float32),
        np.zeros((1, 0), np.float32),
        np.zeros((1, 0), np.float32),
        np.zeros((1, 3), np.float32),
        body_model="object",
        vertices=out["vertices"][None],
        faces=out["faces"],
        extras=extras,
        seed=int(seed),
        generation_params=params,
    )


class _CRMModules:
    """The loaded networks; ``model.pipeline`` returns this."""

    def __init__(self, *, device, half, pixel, ccm, net, grid, birefnet):
        self.device = device
        self.half = half
        self.pixel = pixel
        self.ccm = ccm
        self.net = net
        self.grid = grid
        self.birefnet = birefnet
