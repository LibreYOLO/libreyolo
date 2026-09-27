"""LibreYOLO adapter for SAM 3D Body (inference only).

This is a **wrapper, not a port**. Meta's SAM 3D Body code is published under
the SAM License, which is not one of the permissive licenses LibreYOLO's own
code may be derived from, so none of it is vendored here. Instead this module
calls the upstream package's public API and translates its output into
LibreYOLO's ``Meshes`` result payload.

The practical consequence is that the upstream package is an optional
dependency the user installs themselves. LibreYOLO ships no SAM-licensed bytes,
and a user who never touches the mesh task never encounters those terms. The
weights are a separate matter: the SAM License does permit redistribution with
passthrough, so they are mirrored on the LibreYOLO org behind an automatic
gate that records license acceptance plus self-attested identity and
jurisdiction. It is not manual approval or screening by Meta.

The body model underneath is MHR, which is Apache 2.0 and fetched from its
public release by :func:`~libreyolo.models.sam3dbody.mhr_body.ensure_mhr_model`.

Security boundary: the upstream public constructor accepts checkpoint and MHR
pathnames, not already-open descriptors. LibreYOLO hashes both exact file sets
immediately before and after construction and rejects linked parent chains,
but a same-user process could still perform a transient valid-to-invalid-to-
valid replacement between those checks. These paths therefore must be on a
trusted, quiescent local filesystem while the model is being constructed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..base.model import BaseModel
from .mhr_body import ensure_mhr_model, inspect_mhr_model
from .person import PersonDetector, resolve_person_detector
from .snapshot import (
    acquire_sam_snapshot,
    inspect_sam_snapshot,
)

logger = logging.getLogger(__name__)

UPSTREAM_REPO = "https://github.com/facebookresearch/sam-3d-body"
UPSTREAM_REVISION = "b5c765a0d89d789985e186d396315e7590887b94"
UPSTREAM_SOURCE_URL = f"{UPSTREAM_REPO}/tree/{UPSTREAM_REVISION}"
SAM_LICENSE_URL = f"{UPSTREAM_REPO}/blob/{UPSTREAM_REVISION}/LICENSE"
UPSTREAM_PATH_TRUST_BOUNDARY = (
    "checkpoint and MHR paths must be on a trusted, quiescent same-user local "
    "filesystem while the upstream path-only constructor runs"
)


class LibreSAM3DBody(BaseModel):
    """SAM 3D Body: image + person boxes -> per-person MHR body meshes."""

    FAMILY = "sam3dbody"
    FILENAME_PREFIX = "LibreSAM3DBody"
    WEIGHT_EXT = ".ckpt"
    # Size codes name the backbone: DINOv3 ViT-H/16+ and the original ViT-H.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"d3": 512, "h": 512}
    SUPPORTED_TASKS = ("mesh",)
    DEFAULT_TASK = "mesh"

    TTA_ENABLED = False
    SUPPORTS_BATCHED_PREDICT = False

    def __init__(
        self,
        model_path=None,
        size: str = "d3",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        person_detector: Optional[PersonDetector] = None,
        sam_3d_body_path: str | Path | None = None,
        mhr_path: str | Path | None = None,
        **kwargs,
    ):
        self._sam_3d_body_path = sam_3d_body_path or os.environ.get("SAM_3D_BODY_PATH")
        self._mhr_path = mhr_path
        self._ckpt_path = self._resolve_checkpoint(model_path, size)
        self._estimator = None

        super().__init__(
            model_path=None,  # upstream's loader builds and loads in one step
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        self.names = {0: "person"}
        self.person_detector = (
            resolve_person_detector(person_detector)
            if person_detector is not None
            else None
        )
        self.model.eval()

    # =========================================================================
    # Upstream package plumbing
    # =========================================================================

    def _import_upstream(self):
        """Import the upstream package, with actionable guidance if absent."""
        if self._sam_3d_body_path:
            path = str(Path(self._sam_3d_body_path).resolve())
            import sys

            if path not in sys.path:
                sys.path.insert(0, path)
        # The MHR head prefers the pymomentum backend when this is unset, which
        # has no reliable Windows wheel. Selecting the TorchScript path keeps
        # the dependency surface to plain PyTorch.
        os.environ.setdefault("MOMENTUM_ENABLED", "")
        try:
            import sam_3d_body  # noqa: F401

            return sam_3d_body
        except ImportError as e:
            raise ImportError(
                "LibreSAM3DBody wraps Meta's sam-3d-body package, which is not "
                "bundled with LibreYOLO because it is published under the SAM "
                "License rather than a permissive one.\n\n"
                "Install it yourself:\n"
                f"  git clone {UPSTREAM_REPO}\n"
                f"  git -C sam-3d-body checkout --detach {UPSTREAM_REVISION}\n"
                "  pip install roma einops yacs omegaconf braceexpand "
                "pytorch-lightning timm\n\n"
                "For automatic checkpoint acquisition, also install\n"
                '  pip install "libreyolo[hf]"\n\n'
                "Then point LibreYOLO at the clone, either with\n"
                "  LibreSAM3DBody(..., sam_3d_body_path='/path/to/sam-3d-body')\n"
                "or by setting the SAM_3D_BODY_PATH environment variable.\n\n"
                f"Its terms, which you accept by using it: {SAM_LICENSE_URL}"
            ) from e

    def _resolve_checkpoint(self, model_path, size: str) -> Path:
        """Locate and byte-validate the exact reviewed upstream checkpoint."""
        if model_path is not None:
            path = Path(model_path).absolute()
            if path.is_dir():
                root = path
            else:
                if path.name != "model.ckpt":
                    raise ValueError(
                        "SAM 3D Body explicit checkpoints must be named model.ckpt "
                        "and sit beside their reviewed model_config.yaml and LICENSE"
                    )
                root = path.parent
            if not os.path.lexists(path):
                raise FileNotFoundError(f"SAM 3D Body checkpoint not found: {path}")
            identity = inspect_sam_snapshot(root, size, managed=False)
        else:
            identity = acquire_sam_snapshot(size)
        self._checkpoint_snapshot_identity = identity
        return identity.root / "model.ckpt"

    @classmethod
    def _download_checkpoint(cls, cache: Path, size: str) -> Path:
        """Fetch a revision-keyed checkpoint view (legacy private helper)."""
        identity = acquire_sam_snapshot(size, cache_root=cache)
        return identity.root / "model.ckpt"

    # =========================================================================
    # BaseModel surface
    # =========================================================================

    def _init_model(self) -> nn.Module:
        """Construct upstream under the documented path-only trust boundary."""
        self._import_upstream()
        from sam_3d_body import load_sam_3d_body

        mhr = (
            Path(self._mhr_path).absolute()
            if self._mhr_path is not None
            else ensure_mhr_model()
        )
        mhr_before = inspect_mhr_model(mhr)
        before = inspect_sam_snapshot(
            self._checkpoint_snapshot_identity.root,
            self.size,
            managed=None,
        )
        if before != self._checkpoint_snapshot_identity:
            raise RuntimeError(
                "SAM 3D Body checkpoint/config bytes changed before the upstream "
                "loader could construct the model"
            )
        # Honour the caller's device rather than grabbing any available GPU:
        # device="cpu" on a CUDA machine is a legitimate request, and silently
        # overriding it would also desynchronize the guard in estimate().
        model, cfg = load_sam_3d_body(
            str(self._ckpt_path), device=str(self.device), mhr_path=str(mhr)
        )
        mhr_observed = inspect_mhr_model(mhr)
        if mhr_observed != mhr_before:
            raise RuntimeError(
                "MHR bytes changed while the upstream loader was constructing the model"
            )
        observed = inspect_sam_snapshot(
            self._checkpoint_snapshot_identity.root,
            self.size,
            managed=None,
        )
        if observed != self._checkpoint_snapshot_identity:
            raise RuntimeError(
                "SAM 3D Body checkpoint/config bytes changed while the upstream "
                "loader was constructing the model"
            )
        self._cfg = cfg
        return model

    @property
    def estimator(self):
        """The upstream estimator, built lazily on first use."""
        if self._estimator is None:
            self._import_upstream()
            from sam_3d_body import SAM3DBodyEstimator

            self._estimator = SAM3DBodyEstimator(self.model, self._cfg)
        return self._estimator

    @property
    def faces(self) -> Optional[torch.Tensor]:
        """Shared MHR mesh topology, or None if upstream stops exposing it."""
        head = getattr(self.model, "head_pose", None)
        faces = getattr(head, "faces", None)
        if faces is None:
            logger.warning(
                "The loaded sam-3d-body build does not expose head_pose.faces, "
                "so mesh results will carry no topology: OBJ export and surface "
                "rendering will be unavailable."
            )
            return None
        return faces.detach().cpu()

    def estimate(
        self,
        image_rgb: np.ndarray,
        boxes_xyxy: np.ndarray,
        focal_length: Optional[float] = None,
    ) -> List[dict]:
        """Run the upstream estimator over one image and its person boxes."""
        # Guard on this model's device, not on global CUDA availability: with
        # device="cpu" on a CUDA machine the weights are on CPU, and a global
        # check would wave that through into a device-mismatch crash inside the
        # upstream estimator.
        if self.device.type != "cuda":
            raise RuntimeError(
                "SAM 3D Body inference requires a CUDA device, but this model "
                f"is on {self.device}. The upstream estimator moves its batch "
                "to the GPU unconditionally, so there is no CPU path to fall "
                "back to. Construct with device='cuda'."
                + (
                    ""
                    if torch.cuda.is_available()
                    else " No CUDA device is visible to PyTorch here."
                )
            )
        cam_int = None
        if focal_length is not None:
            h, w = image_rgb.shape[:2]
            cam_int = torch.tensor(
                [
                    [
                        [float(focal_length), 0.0, w / 2.0],
                        [0.0, float(focal_length), h / 2.0],
                        [0.0, 0.0, 1.0],
                    ]
                ],
                dtype=torch.float32,
            )
        return self.estimator.process_one_image(
            image_rgb,
            bboxes=np.asarray(boxes_xyxy, dtype=np.float32),
            cam_int=cam_int,
            inference_type="body",
        )

    @property
    def _runner(self):
        if getattr(self, "_runner_instance", None) is None:
            from .inference import MeshInferenceRunner

            self._runner_instance = MeshInferenceRunner(self)
        return self._runner_instance

    # The detection-shaped hooks do not apply: the upstream estimator owns
    # preprocessing, the forward pass and decoding.
    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return False  # foreign checkpoint format; constructed explicitly

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {name: module for name, module in self.model.named_modules() if name}

    @staticmethod
    def _get_preprocess_numpy():
        raise NotImplementedError(
            "LibreSAM3DBody preprocesses per person crop inside the upstream "
            "estimator, not through the detection-shaped numpy path."
        )

    def _preprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreSAM3DBody preprocesses inside the upstream estimator."
        )

    def _forward(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreSAM3DBody runs the forward pass inside the upstream estimator."
        )

    def _postprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreSAM3DBody decodes inside the upstream estimator; "
            "MeshInferenceRunner maps the result into Meshes."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            "Body-mesh validation needs a ground-truth mesh dataset, and the "
            "usual benchmarks (3DPW, EMDB, AGORA) are research-license only, so "
            "none is bundled. The metrics are available as "
            "libreyolo.validation.mesh_metrics (MPJPE, PA-MPJPE, PVE) for "
            "evaluating against a dataset you already hold."
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training is out of scope for LibreSAM3DBody. Train upstream at "
            f"{UPSTREAM_SOURCE_URL}."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        raise NotImplementedError(
            "Body-mesh export is not implemented yet; the exported-graph "
            "metadata contract for the mesh task is still to be defined."
        )
