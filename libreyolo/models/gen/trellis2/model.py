"""TRELLIS.2: one image in, the whole object out as a triangle mesh.

TRELLIS.2 (https://microsoft.github.io/TRELLIS.2, "Native and Compact
Structured Latents for 3D Generation", 2025) is Microsoft's open 4B
image-to-3D model. A dense flow transformer denoises the object's occupancy
at 16^3, TRELLIS's 3D CNN decodes it to the occupied voxels of a 32^3 grid,
sparse flow transformers denoise a 32-channel shape latent on those voxels
(at 512, then, for the cascades, again at 1024 or 1536 on the voxels the
first latent predicts), and a sparse ConvNeXt decoder upsamples the latent
16 times into a flexible dual grid: one vertex per surface voxel and the
quads between them. Both flow stages read DINOv3 ViT-L/16 tokens. Code and
weights are MIT; the DINOv3 encoder is under Meta's DINOv3 License.

Upstream runs on CUDA only (FlexGEMM sparse convolution, flash-attn, an
o-voxel CUDA hash map for meshing). This port replaces them with PyTorch
(``trellis/sparse.py``, ``mesh.py``) and keeps the networks' parameter
names, so the released weights load unchanged and the same model runs on
CUDA, Apple MPS and CPU. Noise is drawn on the CPU as upstream does, so a
seed gives the same object on every device up to kernel rounding.

The output is ``Results.meshes`` (ADR 0031): one vertex per voxel of the
decoded grid, as upstream returns it (some are referenced by no face), and
the dual grid's triangles. The dual grid carries no inside or outside, so
faces are not consistently wound; upstream orients them later with CuMesh.

Geometry only: upstream's texture stage (PBR voxels baked to a UV texture
with NVIDIA's nvdiffrast, a non-commercial library) and its CuMesh
post-processing (hole filling, orientation, simplification) are not part of
this family.

Nothing is gated and no login is needed. The TRELLIS.2 networks come from
LibreYOLO's mirror ``LibreYOLO/LibreTRELLIS2-4b`` of
``microsoft/TRELLIS.2-4B``; the sparse-structure decoder TRELLIS.2 reuses
and the BiRefNet weights from ``LibreYOLO/LibreTRELLISl``; DINOv3 ViT-L/16
from LibreYOLO's mirror ``LibreYOLO/LibreDINOv3vitl16``. All at pinned
revisions.
"""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence, Tuple

from ....utils.results import Meshes
from ..base import _INSTALL_HINT
from ..base3d import LibreGen3DModel

_DINOV3_REPO = "LibreYOLO/LibreDINOv3vitl16"
_DINOV3_REVISION = "7f681024a32a03b7f56a1732455b373834d3ee76"
_TRELLIS_REPO = "LibreYOLO/LibreTRELLISl"
_TRELLIS_REVISION = "75c784a8fa56ebfb76f46d9b9d830caec25826a7"
_STRUCTURE_DECODER = "ckpts/ss_dec_conv3d_16l8_fp16"
_BIREFNET_FILE = "birefnet.safetensors"
_NETWORKS = {
    "structure_flow": "ckpts/ss_flow_img_dit_1_3B_64_bf16",
    "shape_flow_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
    "shape_flow_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
    "shape_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
}
# What upstream's TRELLIS.2-4B layout must hold for LibreYOLO to load it.
_UPSTREAM_FILES = (
    "pipeline.json",
    *(
        f"{stem}{ext}"
        for stem in _NETWORKS.values()
        for ext in (".json", ".safetensors")
    ),
)
# BiRefNet heads used only for training supervision.
_BIREFNET_TRAINING_ONLY = ("decoder.conv_ms_spvn_", "decoder.gdt_convs_pred_")
_RESOLUTIONS = (512, 1024, 1536)


class LibreTRELLIS2(LibreGen3DModel):
    """TRELLIS.2 behind the LibreGen surface: ``model("chair.jpg").save("chair.obj")``."""

    FAMILY = "trellis2"
    FILENAME_PREFIX = "LibreTRELLIS2-"
    HF_REPOS: ClassVar[Dict[str, str]] = {"4b": "LibreYOLO/LibreTRELLIS2-4b"}
    HF_REVISIONS: ClassVar[Dict[str, str]] = {"4b": "266ce97c4ecf81b9fb090644609aaccc6fa2ce5d"}
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"4b": 1024}
    SNAPSHOT_ALLOW_PATTERNS: ClassVar[Tuple[str, ...]] = _UPSTREAM_FILES
    SAVE_SUFFIXES: ClassVar[Tuple[str, ...]] = (".obj",)
    KNOBS = ("steps", "guidance", "resolution", "margin", "max_tokens")
    DEFAULTS: ClassVar[Dict[str, Any]] = {
        "steps": 12,
        "guidance": 7.5,
        "resolution": 1024,
        "margin": 1.0,
        "max_tokens": 49152,
    }
    _LICENSE_NOTICE = (
        "TRELLIS.2 conditions on DINOv3 ViT-L/16, under Meta's DINOv3 License; "
        "read it before use."
    )

    def __init__(self, size=None, *, compile: bool = False, **kwargs):
        if compile:
            raise ValueError(
                "LibreTRELLIS2 has no compiled sampler; compile=True is a Speridlabs Idol option."
            )
        super().__init__(size, compile=False, **kwargs)

    @classmethod
    def matches_directory(cls, path: Any) -> bool:
        """True when ``path`` holds upstream's TRELLIS.2-4B geometry files."""
        try:
            folder = Path(path)
            if not all((folder / name).is_file() for name in _UPSTREAM_FILES):
                return False
            config = json.loads((folder / "pipeline.json").read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return False
        return config.get("name") == "Trellis2ImageTo3DPipeline"

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
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoModel

        from ...birefnet.nn import LibreBiRefNetModel
        from . import nn as networks

        folder = Path(snapshot_dir)
        pipeline = json.loads((folder / "pipeline.json").read_text(encoding="utf-8"))[
            "args"
        ]
        # The published precision (bf16 transformers, fp16 decoders) on CUDA
        # and MPS, as upstream runs; fp32 on CPU, where half-precision 3D
        # convolution and fast bf16 matmuls are not available.
        reduced = self.device.type in ("cuda", "mps")
        # MPS has bf16 only from macOS 14; there the bf16 flows run in fp32,
        # the shared _resolve_dtype() choice. fp16 decoders work on any MPS.
        bf16_flows = reduced and not (
            self.device.type == "mps" and self._resolve_dtype() == torch.float32
        )

        def build(config: Dict[str, Any], weights: Path):
            args = dict(config["args"])
            if "use_fp16" in args or config["name"] == "SparseStructureDecoder":
                args["use_fp16"] = reduced
            if "dtype" in args and not bf16_flows:
                args["dtype"] = "float32"
            module = getattr(networks, config["name"])(**args)
            module.load_state_dict(load_file(str(weights)), strict=True)
            return module.eval().requires_grad_(False).to(self.device)

        modules = {}
        for name, stem in _NETWORKS.items():
            config = json.loads((folder / f"{stem}.json").read_text(encoding="utf-8"))
            modules[name] = build(config, folder / f"{stem}.safetensors")

        def trellis_file(name: str) -> Path:
            local = folder / name
            if local.is_file():
                return local
            return Path(hf_hub_download(_TRELLIS_REPO, name, revision=_TRELLIS_REVISION))

        config = json.loads(
            trellis_file(f"{_STRUCTURE_DECODER}.json").read_text(encoding="utf-8")
        )
        modules["structure_decoder"] = build(
            config, trellis_file(f"{_STRUCTURE_DECODER}.safetensors")
        )

        birefnet = LibreBiRefNetModel(size="l")
        missing, unexpected = birefnet.load_state_dict(
            load_file(str(trellis_file(_BIREFNET_FILE))), strict=False
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
        # Upstream runs DINOv3 in fp32 on every device.
        dinov3 = AutoModel.from_pretrained(
            _DINOV3_REPO, revision=_DINOV3_REVISION, dtype=torch.float32
        )
        dinov3 = dinov3.eval().requires_grad_(False).to(self.device)
        self._pipeline = _TRELLIS2Modules(
            device=self.device,
            dinov3=dinov3,
            birefnet=birefnet,
            samplers={
                "structure": pipeline["sparse_structure_sampler"]["params"],
                "shape": pipeline["shape_slat_sampler"]["params"],
            },
            shape_normalization=pipeline["shape_slat_normalization"],
            **modules,
        )

    def to(self, device) -> "LibreTRELLIS2":
        """Move before loading; a loaded model keeps its device, whose precision it was built for."""
        target = self._resolve_device(device)
        if self._pipeline is not None and target != self.device:
            raise ValueError(
                "LibreTRELLIS2 picks its precision per device at load; construct it "
                "with device=... instead of to() after it has loaded."
            )
        self.device = target
        return self

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _resolve_knobs(self, knobs: Dict[str, Any]) -> Dict[str, Any]:
        settings = super()._resolve_knobs(knobs)
        for name, low in (("steps", 1), ("max_tokens", 1)):
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, Integral) or value < low:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
            settings[name] = int(value)
        resolution = settings["resolution"]
        if isinstance(resolution, bool) or resolution not in _RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {', '.join(map(str, _RESOLUTIONS))}, "
                f"got {resolution!r}."
            )
        settings["resolution"] = int(resolution)
        for name, low in (("guidance", 0), ("margin", 1)):
            value = settings[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < low
            ):
                raise ValueError(f"{name} must be a number >= {low}, got {value!r}.")
            settings[name] = float(value)
        return settings

    def _generate_3d(
        self, image, seeds: Sequence[int], knobs: Dict[str, Any]
    ) -> List[Meshes]:
        import torch

        from .pipeline import encode_image, preprocess_image, run_stages, sampler_settings

        modules = self._pipeline
        structure = sampler_settings(
            modules.samplers["structure"], knobs["steps"], knobs["guidance"]
        )
        shape = sampler_settings(
            modules.samplers["shape"], knobs["steps"], knobs["guidance"]
        )
        objects = []
        with torch.no_grad():
            prepared = preprocess_image(
                image, modules.birefnet, crop_margin=knobs["margin"]
            )
            # One image, so one set of tokens for every seed.
            sizes = (512,) if knobs["resolution"] == 512 else (512, 1024)
            conds = {
                size: encode_image(modules.dinov3, prepared, size, modules.device)
                for size in sizes
            }
            for seed in seeds:
                out = run_stages(
                    modules,
                    prepared,
                    seed,
                    resolution=knobs["resolution"],
                    structure=structure,
                    shape=shape,
                    max_tokens=knobs["max_tokens"],
                    conds=conds,
                )
                # A generated object in Meshes (ADR 0031): one row, no body
                # parameters, z up in [-0.5, 0.5]^3.
                objects.append(
                    Meshes(
                        torch.zeros(1, 0),
                        torch.zeros(1, 0),
                        torch.zeros(1, 0),
                        torch.zeros(1, 3),
                        body_model="object",
                        vertices=out["vertices"].unsqueeze(0),
                        faces=out["faces"],
                        seed=seed,
                        generation_params={
                            "model": self.model_path
                            if self._checkpoint_dir
                            else self.model_name,
                            **knobs,
                            # The cascade may lower the requested resolution to
                            # fit max_tokens; record what the mesh was built at.
                            "output_resolution": out["resolution"],
                        },
                    )
                )
                # Only the mesh is kept; free the stages before the next seed.
                del out
        return objects


class _TRELLIS2Modules:
    """The loaded networks; ``model.pipeline`` returns this."""

    def __init__(
        self,
        *,
        device,
        dinov3,
        birefnet,
        samplers,
        shape_normalization,
        structure_flow,
        structure_decoder,
        shape_flow_512,
        shape_flow_1024,
        shape_decoder,
    ):
        self.device = device
        self.dinov3 = dinov3
        self.birefnet = birefnet
        self.samplers = samplers
        self.shape_normalization = shape_normalization
        self.structure_flow = structure_flow
        self.structure_decoder = structure_decoder
        self.shape_flow_512 = shape_flow_512
        self.shape_flow_1024 = shape_flow_1024
        self.shape_decoder = shape_decoder
