"""Inference orchestrator for the face-embedding (``embed``) task.

Wraps the two-stage pipeline (face detection -> 5-pt align -> recognition head
-> L2-normalized embedding) behind the same ``__call__`` shape the detection
runner provides, so ``LibreFaceEmbedder`` integrates with the rest of the
framework. Mirrors ``GazeInferenceRunner``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np
import torch

from ...utils.general import log_saved_result, resolve_save_path
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.results import Boxes, Embeddings, Results
from ..l2cs.face import FaceBox, FaceDetector, resolve_face_detector
from .align import align_face

if TYPE_CHECKING:
    from .model import LibreFaceEmbedder

logger = logging.getLogger(__name__)


class FaceEmbedRunner:
    """Runs face-embedding inference for a ``LibreFaceEmbedder``."""

    def __init__(self, model: "LibreFaceEmbedder"):
        self.model = model

    # ------------------------------------------------------------------
    def __call__(
        self,
        source: ImageInput | None = None,
        *,
        face_boxes: Optional[Sequence] = None,
        face_detector: Optional[FaceDetector] = None,
        face_conf: float = 0.5,
        save: bool = False,
        output_path: Optional[str] = None,
        color_format: str = "auto",
        show: bool = False,
        output_file_format: Optional[str] = None,
        augment: bool = False,
        tiling: bool = False,
        **_: object,
    ):
        if augment:
            raise ValueError("TTA (augment=True) is not meaningful for face embedding.")
        if tiling:
            raise ValueError("Tiled inference is not supported for face embedding.")

        detector = self._resolve_runtime_detector(face_detector, face_boxes)

        if isinstance(source, (str, Path)) and Path(source).is_dir():
            return [
                self._predict_single(
                    p, detector=detector, face_boxes=None, face_conf=face_conf,
                    save=save, output_path=output_path, color_format=color_format,
                    output_file_format=output_file_format,
                )
                for p in ImageLoader.collect_images(source)
            ]

        return self._predict_single(
            source, detector=detector, face_boxes=face_boxes, face_conf=face_conf,
            save=save, output_path=output_path, color_format=color_format,
            output_file_format=output_file_format,
        )

    # ------------------------------------------------------------------
    def _predict_single(
        self,
        image: ImageInput,
        *,
        detector: Optional[FaceDetector],
        face_boxes: Optional[Sequence],
        face_conf: float,
        save: bool,
        output_path: Optional[str],
        color_format: str,
        output_file_format: Optional[str],
    ) -> Results:
        image_path = image if isinstance(image, (str, Path)) else None
        pil = ImageLoader.load(image, color_format=color_format)
        rgb_np = np.asarray(pil)
        h, w = rgb_np.shape[:2]

        faces = self._collect_faces(rgb_np, detector, face_boxes, face_conf)
        result = self._run_embed(rgb_np, faces, (h, w), image_path)

        if save:
            ext = (output_file_format or "jpg").lower().lstrip(".")
            save_path = resolve_save_path(output_path, image_path, ext=ext)
            annotated = self._annotate(pil, result)
            annotated.save(save_path)
            log_saved_result(result, save_path)
        return result

    # ------------------------------------------------------------------
    def _resolve_runtime_detector(self, explicit, face_boxes):
        if face_boxes is not None:
            return None
        if explicit is not None:
            return resolve_face_detector(explicit)
        return self.model.face_detector

    def _collect_faces(self, image_rgb, detector, face_boxes, face_conf) -> List[FaceBox]:
        if face_boxes is not None:
            from ..l2cs.face import _normalize_boxes

            return _normalize_boxes(face_boxes, min_score=0.0)
        if detector is None:
            raise RuntimeError(
                "LibreFaceEmbedder has no face source. Pass face_boxes=[...] for BYO "
                "boxes, or face_detector=... (a callable, a LibreYOLO model, or an "
                "OpenCVFaceDetector) when constructing or calling the model."
            )
        return [f for f in detector(image_rgb) if f.score >= face_conf]

    def _run_embed(self, rgb_np, faces, orig_shape, image_path) -> Results:
        def _empty() -> Results:
            dim = self.model.dim or 0
            return Results(
                boxes=Boxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                ),
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=self.model.names,
                embeddings=Embeddings(np.zeros((0, dim), dtype=np.float32), orig_shape),
            )

        if not faces:
            return _empty()

        crops, kept = [], []
        for f in faces:
            try:
                crops.append(align_face(rgb_np, f.xyxy, f.landmarks, image_size=112))
                kept.append(f)
            except ValueError as e:
                logger.warning("Skipping degenerate face crop: %s", e)
        if not crops:
            return _empty()

        emb = self.model.embed_aligned(crops)  # (N, D) L2-normalized float32

        xyxy = torch.tensor([list(f.xyxy) for f in kept], dtype=torch.float32)
        conf = torch.tensor([f.score for f in kept], dtype=torch.float32)
        cls = torch.zeros(len(kept), dtype=torch.float32)
        return Results(
            boxes=Boxes(xyxy, conf, cls),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.model.names,
            embeddings=Embeddings(emb, orig_shape),
        )

    # ------------------------------------------------------------------
    def _annotate(self, pil_img, result):
        if result.boxes is None or len(result.boxes) == 0:
            return pil_img
        from ...utils.drawing import draw_boxes

        return draw_boxes(
            pil_img,
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
            class_names=result.names,
        )
