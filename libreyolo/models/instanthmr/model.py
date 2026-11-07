"""LibreInstantHMR: real-time human mesh recovery in the MHR body model.

Native PyTorch port of InstantHMR (https://github.com/Naturalpad-studio/InstantHMR,
code Apache-2.0). A person box goes in; MHR pose and identity parameters, 70
keypoints in 2D and 3D, the camera translation and the 18,439-vertex mesh come
out. People are found by a LibreYOLO RF-DETR detector unless boxes or another
detector are supplied.

Weights: InstantHMR publishes ONNX only; ``weights/convert_instanthmr_weights.py``
rebuilds the PyTorch checkpoints from those graphs. Sizes ``n``/``t``/``s``/``m``
are CC BY-NC 4.0 (non-commercial); ``m`` with the ``commercial`` variant was
trained only on commercially usable images and is Apache-2.0. All of them were
trained on the SAM 3D Body dataset annotations (SAM License), whose terms ship
with the weights.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch import nn

from ..base.model import BaseModel
from ..sam3dbody.person import (
    LibreYOLOPersonDetector,
    PersonDetector,
    resolve_person_detector,
)
from .nn import BACKBONES, InstantHMRExport, InstantHMRNet
from .utils import crop_to_image_coords, default_focal, preprocess_person

logger = logging.getLogger(__name__)

UPSTREAM_REPO = "https://github.com/Naturalpad-studio/InstantHMR"
DEFAULT_PERSON_DETECTOR = "LibreRFDETRm.pt"

# HGNetV2 stem width and last-stage width identify the size.
_SIZE_SIGNATURES = {(16, 1024): "n", (24, 1024): "t", (24, 1536): "s", (32, 2048): "m"}


class LibreInstantHMR(BaseModel):
    """InstantHMR: image + person boxes -> per-person MHR body meshes."""

    FAMILY = "instanthmr"
    FILENAME_PREFIX = "LibreInstantHMR"
    INPUT_SIZES: ClassVar[dict[str, int]] = {size: 288 for size in BACKBONES}
    SUPPORTED_TASKS = ("mesh",)
    DEFAULT_TASK = "mesh"
    REQUIRE_TASK_SUFFIX = True
    WEIGHT_VARIANTS = ("commercial",)

    TTA_ENABLED = False
    SUPPORTS_BATCHED_PREDICT = False

    def __init__(
        self,
        model_path=None,
        size: str = "m",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        person_detector=DEFAULT_PERSON_DETECTOR,
        focal_length: float | None = None,
        max_batch: int = 16,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        self.names = {0: "person"}
        self.focal_length = focal_length
        self.max_batch = int(max_batch)
        self._person_detector_spec = person_detector
        self._person_detector: PersonDetector | None = None
        if isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.model.eval()

    # =========================================================================
    # Person source
    # =========================================================================

    @property
    def person_detector(self) -> PersonDetector | None:
        """The default person source, built on first use."""
        if self._person_detector is None and self._person_detector_spec is not None:
            spec = self._person_detector_spec
            if isinstance(spec, (str, Path)):
                from .. import LibreYOLO

                logger.info("InstantHMR: loading person detector %s", spec)
                spec = LibreYOLOPersonDetector(
                    model=LibreYOLO(str(spec), device=str(self.device))
                )
            self._person_detector = resolve_person_detector(spec)
        return self._person_detector

    @person_detector.setter
    def person_detector(self, value) -> None:
        self._person_detector_spec = value
        self._person_detector = None

    # =========================================================================
    # Inference
    # =========================================================================

    @property
    def faces(self) -> torch.Tensor | None:
        faces = self.model.body.faces
        return faces.detach().cpu() if faces.numel() else None

    @torch.no_grad()
    def estimate(
        self,
        image_rgb: np.ndarray,
        boxes_xyxy: np.ndarray,
        focal_length: float | None = None,
    ) -> list[dict]:
        """Run InstantHMR on every box of one RGB image.

        Returns one dict per box with root-relative ``pred_vertices`` and
        ``pred_keypoints_3d`` (metres, camera axes), ``pred_cam_t``,
        ``pred_keypoints_2d`` in image pixels, the MHR parameter blocks and the
        focal length the 3D is expressed in.
        """
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return []
        h, w = image_rgb.shape[:2]
        focal = focal_length if focal_length is not None else self.focal_length
        focal = float(focal) if focal is not None else default_focal(h, w)
        size = self.model.image_size
        prepared = [preprocess_person(image_rgb, box, focal, size) for box in boxes]
        crops = torch.from_numpy(np.stack([p[0] for p in prepared]))
        cliffs = torch.from_numpy(np.stack([p[1] for p in prepared]))

        chunks = []
        for i in range(0, len(boxes), self.max_batch):
            image = crops[i : i + self.max_batch].to(self.device)
            cond = cliffs[i : i + self.max_batch].to(self.device)
            out = self.model(image, cond)
            joints_3d, vertices = self.model.body(
                out["mhr_params"], out["shape_params"]
            )
            chunks.append(
                {
                    "mhr_params": out["mhr_params"],
                    "shape_params": out["shape_params"],
                    "cam_trans": out["cam_trans"],
                    "joints_2d": out["joints_2d"],
                    "joints_3d": joints_3d,
                    "vertices": vertices,
                }
            )
        outs = {
            k: torch.cat([c[k] for c in chunks]).float().cpu().numpy()
            for k in chunks[0]
        }

        results = []
        for i, (_, _, square) in enumerate(prepared):
            params = outs["mhr_params"][i]
            results.append(
                {
                    "pred_cam_t": outs["cam_trans"][i],
                    "pred_vertices": outs["vertices"][i],
                    "pred_keypoints_3d": outs["joints_3d"][i],
                    "pred_keypoints_2d": crop_to_image_coords(
                        outs["joints_2d"][i], square
                    ),
                    "focal_length": np.float32(focal),
                    "global_rot": params[3:6],
                    "body_pose_params": params[6:136],
                    "scale_params": params[136:],
                    "shape_params": outs["shape_params"][i],
                    "mhr_params": params,
                }
            )
        return results

    @property
    def _runner(self):
        if getattr(self, "_runner_instance", None) is None:
            from ..sam3dbody.inference import MeshInferenceRunner

            self._runner_instance = MeshInferenceRunner(self)
        return self._runner_instance

    # =========================================================================
    # BaseModel surface
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return InstantHMRNet(self.size)

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return (
            "cont_head.full_gather" in weights_dict
            and "head_2d_logits.weight" in weights_dict
            and "backbone.stem.stem1.conv.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> str | None:
        stem = weights_dict.get("backbone.stem.stem1.conv.weight")
        last = [
            v
            for k, v in weights_dict.items()
            if k.startswith("backbone.stages.3.")
            and k.endswith("aggregation.1.conv.weight")
        ]
        if stem is None or not last:
            return None
        return _SIZE_SIGNATURES.get((int(stem.shape[0]), int(last[-1].shape[0])))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> int | None:
        return 1

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> str | None:
        return "mesh" if cls.can_load(state_dict) else None

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> str | None:
        commercial = cls.detect_variant_from_filename(filename) == "commercial"
        licence = (
            "Apache-2.0" if commercial else "CC BY-NC 4.0 (non-commercial use only)"
        )
        return (
            f"InstantHMR weights ({Path(filename).name}) are licensed {licence}. "
            "They were trained on the SAM 3D Body dataset annotations, released "
            "under Meta's SAM License, whose terms ship with the weights."
        )

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {name: module for name, module in self.model.named_modules() if name}

    @staticmethod
    def _get_preprocess_numpy():
        raise NotImplementedError(
            "LibreInstantHMR preprocesses per person crop; see "
            "libreyolo.models.instanthmr.utils.preprocess_person."
        )

    def _preprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreInstantHMR preprocesses per person crop in estimate()."
        )

    def _forward(self, *args, **kwargs):
        raise NotImplementedError("LibreInstantHMR runs the network in estimate().")

    def _postprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreInstantHMR decodes in estimate(); MeshInferenceRunner maps the result into Meshes."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            "Body-mesh validation needs a ground-truth mesh dataset, and the usual "
            "benchmarks (3DPW, EMDB) are research-license only, so none is bundled. "
            "The metrics are available as libreyolo.validation.mesh_metrics "
            "(MPJPE, PA-MPJPE, PVE) for evaluating against a dataset you hold."
        )

    def train(
        self,
        data: str,
        *,
        epochs: int = 20,
        batch: int = 16,
        lr0: float = 5e-5,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "exp",
        exist_ok: bool = False,
        resume: bool = False,
        patience: int = 0,
        freeze_backbone: bool = False,
        freeze_bn: bool = True,
        **kwargs,
    ) -> dict:
        """Fine-tune on a YOLO-pose dataset with COCO-17 person keypoints.

        Best-effort recipe for adapting a pretrained checkpoint to your images:
        2D keypoint and reprojection losses on the labelled keypoints, plus a
        prior that keeps the mesh close to the starting model. Training from
        scratch is not supported. See ``libreyolo.models.instanthmr.trainer``.
        Extra keyword arguments ``weight_decay``, ``backbone_lr_ratio``,
        ``w_2d``, ``w_reproj`` and ``w_prior`` tune the recipe.
        """
        from .trainer import InstantHMRFineTuner

        if resume:
            raise NotImplementedError(
                "InstantHMR fine-tuning does not support resume yet."
            )
        if kwargs.pop("pretrained", True) is False:
            raise NotImplementedError(
                "InstantHMR training from scratch is not supported; fine-tune a checkpoint."
            )
        validate = bool(kwargs.pop("val", True)) and kwargs.pop("eval_interval", 1) != 0
        allowed = {"weight_decay", "backbone_lr_ratio", "w_2d", "w_reproj", "w_prior"}
        unknown = sorted(set(kwargs) - allowed - {"imgsz"})
        if unknown:
            raise TypeError(
                f"Unsupported InstantHMR train arguments: {', '.join(unknown)}"
            )
        if kwargs.pop("imgsz", None) not in (None, self.model.image_size):
            raise ValueError(
                "InstantHMR trains on fixed 288 px person crops; imgsz cannot change."
            )
        trainer = InstantHMRFineTuner(
            self,
            data,
            epochs=epochs,
            batch=batch,
            lr0=lr0,
            device=device,
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            patience=patience,
            freeze_backbone=freeze_backbone,
            freeze_bn=freeze_bn,
            validate=validate,
            **kwargs,
        )
        results = trainer.train()
        best = results.get("best_checkpoint")
        if best and Path(best).exists():
            self._load_weights(best)
        self.model.to(self.device).eval()
        return results

    # =========================================================================
    # Export
    # =========================================================================

    def export(
        self,
        format: str = "onnx",
        *,
        output_path: str | Path | None = None,
        opset: int | None = 17,
        dynamic: bool = True,
        batch: int = 1,
        half: bool = False,
        int8: bool = False,
        imgsz=None,
        nms: bool = False,
        simplify: bool = False,
        device=None,
        verbose: bool = False,
        **kwargs,
    ) -> str:
        """Export ``image (B,3,288,288), cliff_cond (B,3)`` -> parameters, keypoints and mesh.

        Outputs: ``mhr_params`` (B,204), ``shape_params`` (B,45), ``cam_trans``
        (B,3) metres, ``joints_2d`` (B,70,2) in [-1, 1] over the crop,
        ``joints_3d`` (B,70,3) and ``vertices`` (B,18439,3), body-centred metres
        in camera axes. The mesh faces, crop size and conditioning form are
        stored in the graph metadata.
        """
        # simplify, device and verbose are accepted for CLI compatibility and
        # have no effect: the graph is traced on CPU and not simplified.
        del simplify, device, verbose
        if kwargs:
            raise TypeError(
                f"Unsupported InstantHMR export arguments: {', '.join(sorted(kwargs))}"
            )
        if str(format).lower() != "onnx":
            raise NotImplementedError("LibreInstantHMR export supports ONNX only.")
        if half or int8:
            raise NotImplementedError("LibreInstantHMR exports FP32 ONNX only.")
        if nms:
            raise ValueError("nms does not apply to the body-mesh export.")
        size = self.model.image_size
        if imgsz is not None and imgsz not in (size, (size, size), [size, size]):
            raise ValueError(f"InstantHMR takes fixed {size} px person crops; imgsz cannot change.")
        opset = 17 if opset is None else int(opset)
        batch = max(1, int(batch or 1))
        from ...export.onnx import embed_onnx_metadata

        default = (
            Path(self.model_path).with_suffix(".onnx").name
            if self.model_path
            else f"{self.FILENAME_PREFIX}{self.size}-mesh.onnx"
        )
        destination = Path(output_path or default)
        destination.parent.mkdir(parents=True, exist_ok=True)

        module = InstantHMRExport(self.model).eval().to("cpu")
        inputs = (torch.zeros(batch, 3, size, size), torch.zeros(batch, 3))
        names = list(InstantHMRExport.OUTPUTS)
        axes = (
            {n: {0: "batch"} for n in ["image", "cliff_cond", *names]}
            if dynamic
            else None
        )
        try:
            # Traced with autograd enabled on purpose: under no_grad the
            # transformer layers take PyTorch's fused attention fast path,
            # which has no ONNX symbolic.
            with torch.enable_grad():
                torch.onnx.export(
                    module,
                    inputs,
                    str(destination),
                    input_names=["image", "cliff_cond"],
                    output_names=names,
                    dynamic_axes=axes,
                    opset_version=opset,
                    dynamo=False,
                )
        finally:
            self.model.to(self.device)
        faces = self.model.body.faces.cpu().numpy().astype("<i4")
        embed_onnx_metadata(
            str(destination),
            {
                "model_family": self.FAMILY,
                "model_size": self.size,
                "task": "mesh",
                "body_model": "mhr",
                "image_size": str(size),
                "imgsz": str(size),
                "cliff_focal": "true",
                "mesh_faces": base64.b64encode(faces.tobytes()).decode(),
            },
        )
        return str(destination)


__all__ = ["LibreInstantHMR"]
