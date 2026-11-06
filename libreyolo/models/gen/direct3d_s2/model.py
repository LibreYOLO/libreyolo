"""Direct3D-S2: one image in, the whole object out as a mesh.

Direct3D-S2 (https://www.neural4d.com/research/direct3d-s2, "Gigascale 3D
Generation Made Easy with Spatial Sparse Attention") is DreamTech's
image-to-3D model. It generates a signed distance field on sparse voxels: a
dense flow transformer finds the occupied 64^3 voxels, a sparse flow
transformer with spatial sparse attention denoises a latent on them, a sparse
decoder turns it into an SDF at 512^3, and a second pass repeats this at
1024^3. Marching cubes gives the mesh. Code and weights are MIT.

Upstream runs on CUDA only: torchsparse convolutions, flash-attn and Triton
kernels for the attention, a CUDA kernel for voxelising the intermediate mesh.
This port replaces them with PyTorch operations (``sparse.py``,
``pipeline.py``) and keeps the networks' parameter names, so the released
checkpoints load unchanged and the same model runs on CUDA, Apple MPS and
CPU. Noise is drawn from a generator on the model's device seeded with the
seed; on CUDA a seed gives the starting noise of upstream's
``torch.manual_seed(seed)``.

Nothing is gated and no login is needed. The networks come from LibreYOLO's
mirror ``LibreYOLO/LibreDirect3DS2v11`` of ``wushuang98/Direct3D-S2``
(``direct3d-s2-v-1-1``, MIT), the image encoder from
``facebook/dinov2-with-registers-large`` (Apache-2.0), both at pinned
revisions. Background removal runs on LibreYOLO's BiRefNet network with the
BiRefNet weights shipped in the mirror.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence, Tuple

import numpy as np

from ....utils.results import Meshes
from ..base import _INSTALL_HINT
from ..base3d import LibreGen3DModel

_RELEASE = "direct3d-s2-v-1-1"
_CHECKPOINTS = {
    "dense": "model_dense.ckpt",
    "sparse512": "model_sparse_512.ckpt",
    "sparse1024": "model_sparse_1024.ckpt",
    "refiner": "model_refiner.ckpt",
    "refiner1024": "model_refiner_1024.ckpt",
}
_BIREFNET_FILE = "birefnet.safetensors"
_UPSTREAM_FILES = (
    f"{_RELEASE}/config.yaml",
    *(f"{_RELEASE}/{name}" for name in _CHECKPOINTS.values()),
)
_WEIGHT_FILES = (*_UPSTREAM_FILES, _BIREFNET_FILE)
_BIREFNET_TRAINING_ONLY = ("decoder.conv_ms_spvn_", "decoder.gdt_convs_pred_")
# Upstream's per-stage sampler steps (dense, sparse 512, sparse 1024).
_STEPS = (50, 30, 15)


class LibreDirect3DS2(LibreGen3DModel):
    """Direct3D-S2 behind the LibreGen surface: ``model("chair.png").save("chair.obj")``."""

    FAMILY = "direct3d_s2"
    FILENAME_PREFIX = "LibreDirect3DS2"
    HF_REPOS: ClassVar[Dict[str, str]] = {"v11": "LibreYOLO/LibreDirect3DS2v11"}
    HF_REVISIONS: ClassVar[Dict[str, str]] = {"v11": None}
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"v11": 518}
    SNAPSHOT_ALLOW_PATTERNS: ClassVar[Tuple[str, ...]] = _WEIGHT_FILES
    SAVE_SUFFIXES: ClassVar[Tuple[str, ...]] = (".obj",)
    KNOBS = ("resolution", "steps", "guidance")
    DEFAULTS: ClassVar[Dict[str, Any]] = {
        "resolution": 1024,
        "steps": _STEPS,
        "guidance": 7.0,
    }
    _LICENSE_NOTICE = ""

    def __init__(self, size=None, *, compile: bool = False, **kwargs):
        if compile:
            raise ValueError(
                "LibreDirect3DS2 has no compiled sampler; compile=True is a Speridlabs Idol option."
            )
        super().__init__(size, compile=False, **kwargs)

    @classmethod
    def matches_directory(cls, path: Any) -> bool:
        """True when ``path`` holds upstream's ``direct3d-s2-v-1-1`` folder (``birefnet.safetensors`` optional)."""
        try:
            folder = Path(path)
            return all((folder / name).is_file() for name in _UPSTREAM_FILES)
        except (OSError, TypeError, ValueError):
            return False

    def _check_runtime(self) -> None:
        try:
            import diffusers  # noqa: F401
            import skimage  # noqa: F401
            import transformers  # noqa: F401
            import yaml  # noqa: F401
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, snapshot_dir: str) -> None:
        import torch
        import yaml
        from safetensors.torch import load_file
        from transformers import AutoModel

        from ...birefnet.nn import LibreBiRefNetModel
        from ..trellis.model import _DINOV2_REPO, _DINOV2_REVISION
        from . import nn as networks

        folder = Path(snapshot_dir)
        release = folder / _RELEASE
        config = yaml.safe_load((release / "config.yaml").read_text(encoding="utf-8"))
        # fp16 on CUDA and MPS, as upstream converts the networks; fp32 on CPU,
        # where half-precision 3D convolution is not available.
        use_fp16 = self.device.type in ("cuda", "mps")
        builders = {
            "dense_vae": networks.DenseShapeVAE,
            "dense_dit": networks.DenseDiT,
            "sparse_vae_512": networks.SparseSDFVAE,
            "sparse_dit_512": networks.SparseDiT,
            "sparse_vae_1024": networks.SparseSDFVAE,
            "sparse_dit_1024": networks.SparseDiT,
            "refiner": networks.VoxelRefiner,
            "refiner_1024": networks.VoxelSignRefiner,
        }
        sources = {
            "dense_vae": ("dense", "vae"),
            "dense_dit": ("dense", "dit"),
            "sparse_vae_512": ("sparse512", "vae"),
            "sparse_dit_512": ("sparse512", "dit"),
            "sparse_vae_1024": ("sparse1024", "vae"),
            "sparse_dit_1024": ("sparse1024", "dit"),
            "refiner": ("refiner", "refiner"),
            "refiner_1024": ("refiner1024", "refiner"),
        }
        loaded: Dict[str, Any] = {}
        modules = {}
        for name, builder in builders.items():
            checkpoint, key = sources[name]
            if checkpoint not in loaded:
                loaded = {
                    checkpoint: torch.load(
                        release / _CHECKPOINTS[checkpoint],
                        map_location="cpu",
                        weights_only=True,
                    )
                }
            state = loaded[checkpoint][key]
            if key == "vae":
                # The VAE encoders are training-only.
                state = {k: v for k, v in state.items() if not k.startswith("encoder.")}
            args = dict(config[name]["params"], use_fp16=use_fp16)
            module = builder(**args)
            module.load_state_dict(state, strict=True)
            modules[name] = module.eval().requires_grad_(False).to(self.device)
        del loaded
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
        missing = [k for k in missing if not k.startswith(_BIREFNET_TRAINING_ONLY)]
        if missing or unexpected:
            raise RuntimeError(f"birefnet: missing {missing[:5]}, unexpected {unexpected[:5]}")
        birefnet = birefnet.eval().requires_grad_(False).to(self.device)
        if self.device.type == "cuda":
            birefnet.half()
        dinov2 = AutoModel.from_pretrained(
            _DINOV2_REPO, revision=_DINOV2_REVISION, dtype=torch.float32
        )
        dinov2 = dinov2.eval().requires_grad_(False).to(self.device)
        shifts = tuple(
            float(config[name]["params"]["shift"])
            for name in ("dense_scheduler", "sparse_scheduler_512", "sparse_scheduler_1024")
        )
        self._pipeline = _Direct3DS2Modules(
            device=self.device,
            dtype=torch.float16 if use_fp16 else torch.float32,
            dinov2=dinov2,
            birefnet=birefnet,
            shifts=shifts,
            **modules,
        )

    def to(self, device) -> "LibreDirect3DS2":
        """Move before loading; a loaded model keeps its device, whose precision it was built for."""
        target = self._resolve_device(device)
        if self._pipeline is not None and target != self.device:
            raise ValueError(
                "LibreDirect3DS2 picks its precision per device at load; construct it "
                "with device=... instead of to() after it has loaded."
            )
        self.device = target
        return self

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _resolve_knobs(self, knobs: Dict[str, Any]) -> Dict[str, Any]:
        settings = super()._resolve_knobs(knobs)
        resolution = settings["resolution"]
        if isinstance(resolution, bool) or resolution not in (512, 1024):
            raise ValueError(f"resolution must be 512 or 1024, got {resolution!r}.")
        settings["resolution"] = int(resolution)
        steps = settings["steps"]
        if isinstance(steps, Integral) and not isinstance(steps, bool):
            steps = (steps,) * 3
        if (
            not isinstance(steps, (tuple, list))
            or len(steps) != 3
            or any(
                isinstance(s, bool) or not isinstance(s, Integral) or s < 1 for s in steps
            )
        ):
            raise ValueError(
                "steps must be a positive integer or three of them (dense, 512, 1024 "
                f"stages), got {settings['steps']!r}."
            )
        settings["steps"] = tuple(int(s) for s in steps)
        guidance = settings["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, Real)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError(f"guidance must be a number >= 0, got {guidance!r}.")
        settings["guidance"] = float(guidance)
        return settings

    def _generate_3d(
        self, image, seeds: Sequence[int], knobs: Dict[str, Any]
    ) -> List[Meshes]:
        import torch

        from ..trellis.pipeline import encode_image
        from .pipeline import preprocess_image, run_stages

        modules = self._pipeline
        objects = []
        with torch.no_grad():
            prepared = preprocess_image(image, modules.birefnet)
            # One image, so one set of tokens for every seed.
            cond = encode_image(modules.dinov2, prepared, modules.device)
            for seed in seeds:
                generator = torch.Generator(device=modules.device).manual_seed(seed)
                vertices, faces = run_stages(
                    modules,
                    prepared,
                    generator,
                    resolution=knobs["resolution"],
                    steps=knobs["steps"],
                    guidance=knobs["guidance"],
                    cond=cond,
                )["mesh"]
                objects.append(object_mesh(vertices, faces))
        return objects


def object_mesh(vertices: np.ndarray, faces: np.ndarray) -> Meshes:
    """Upstream's mesh (y up, in [-1, 1)) as a one-row ``Meshes`` in the object frame.

    The object frame is z up and inside ``[-0.5, 0.5]^3``: coordinates are
    halved and ``(x, y, z)`` becomes ``(x, -z, y)``. Marching cubes on an SDF
    that is negative inside winds faces clockwise seen from outside, so the
    winding is reversed to counter-clockwise.
    """
    import torch

    v = np.asarray(vertices, dtype=np.float32) * 0.5
    v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    f = np.ascontiguousarray(np.asarray(faces, dtype=np.int64)[:, ::-1])
    empty = torch.zeros(1, 0)
    return Meshes(
        empty,
        empty.clone(),
        empty.clone(),
        torch.zeros(1, 3),
        body_model="object",
        vertices=torch.from_numpy(v)[None],
        faces=torch.from_numpy(f),
    )


class _Direct3DS2Modules:
    """The loaded networks; ``model.pipeline`` returns this."""

    def __init__(
        self,
        *,
        device,
        dtype,
        dinov2,
        birefnet,
        shifts,
        dense_vae,
        dense_dit,
        sparse_vae_512,
        sparse_dit_512,
        sparse_vae_1024,
        sparse_dit_1024,
        refiner,
        refiner_1024,
    ):
        self.device = device
        self.dtype = dtype
        self.dinov2 = dinov2
        self.birefnet = birefnet
        self.shifts = shifts
        self.dense_vae = dense_vae
        self.dense_dit = dense_dit
        self.sparse_vae_512 = sparse_vae_512
        self.sparse_dit_512 = sparse_dit_512
        self.sparse_vae_1024 = sparse_vae_1024
        self.sparse_dit_1024 = sparse_dit_1024
        self.refiner = refiner
        self.refiner_1024 = refiner_1024
