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
from ...utils.video import collect_video_results, is_video_file, run_video_inference
from ..l2cs.face import FaceBox, FaceDetector, resolve_face_detector
from .align import align_face

if TYPE_CHECKING:
    from .model import LibreFaceEmbedder

logger = logging.getLogger(__name__)


def _is_single_box_list(source) -> bool:
    """A list/tuple of numbers is not a list of images (guards ``predict([...])`` misuse)."""
    return len(source) > 0 and all(isinstance(v, (int, float)) for v in source)


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
        gallery=None,
        threshold: float = 0.4,
        save: bool = False,
        output_path: Optional[str] = None,
        color_format: str = "auto",
        show: bool = False,
        output_file_format: Optional[str] = None,
        augment: bool = False,
        tiling: bool = False,
        stream: bool = False,
        vid_stride: int = 1,
        batch: Optional[int] = None,
        **_: object,
    ):
        if augment:
            raise ValueError("TTA (augment=True) is not meaningful for face embedding.")
        if tiling:
            raise ValueError("Tiled inference is not supported for face embedding.")

        detector = self._resolve_runtime_detector(face_detector, face_boxes)
        batch = self._resolve_batch(batch, detector)

        if is_video_file(source):
            gen = self._predict_video(
                source,
                detector=detector,
                face_boxes=face_boxes,
                face_conf=face_conf,
                gallery=gallery,
                threshold=threshold,
                save=save,
                show=show,
                vid_stride=vid_stride,
                output_path=output_path,
                batch=batch,
            )
            if stream:
                return gen
            return collect_video_results(gen, source, vid_stride)

        many = None
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            many, many_boxes = list(ImageLoader.collect_images(source)), None
        elif isinstance(source, (list, tuple)) and not _is_single_box_list(source):
            many, many_boxes = list(source), face_boxes
        if many is not None:
            return self._predict_many(
                many, batch=batch, detector=detector, face_boxes=many_boxes, face_conf=face_conf,
                gallery=gallery, threshold=threshold, save=save, output_path=output_path,
                color_format=color_format, output_file_format=output_file_format,
            )

        return self._predict_single(
            source, detector=detector, face_boxes=face_boxes, face_conf=face_conf,
            gallery=gallery, threshold=threshold,
            save=save, output_path=output_path, color_format=color_format,
            output_file_format=output_file_format,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_batch(batch: Optional[int], detector) -> int:
        """Images per detector call for lists, directories and video files: the detector's
        ``default_predict_batch`` (8 on the Axelera Metis) unless ``batch`` is given."""
        if batch is None:
            return int(getattr(detector, "default_predict_batch", 1))
        if int(batch) < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        return int(batch)

    def _detect_many(self, images_rgb, detector, face_boxes, face_conf: float):
        """Faces of several images: one batched detector call when the detector has ``detect_batch``."""
        if face_boxes is None and detector is not None and len(images_rgb) > 1 and hasattr(detector, "detect_batch"):
            return [[f for f in faces if f.score >= face_conf] for faces in detector.detect_batch(images_rgb)]
        return [self._collect_faces(rgb, detector, face_boxes, face_conf) for rgb in images_rgb]

    def _predict_many(self, images, *, batch: int, detector, face_boxes, face_conf: float, gallery, threshold: float,
                      save: bool, output_path, color_format: str, output_file_format) -> List[Results]:
        results: List[Results] = []
        for start in range(0, len(images), batch):
            chunk = images[start : start + batch]
            pils = [ImageLoader.load(image, color_format=color_format) for image in chunk]
            rgbs = [np.asarray(pil) for pil in pils]
            faces_lists = self._detect_many(rgbs, detector, face_boxes, face_conf)
            for image, pil, rgb, faces in zip(chunk, pils, rgbs, faces_lists):
                image_path = image if isinstance(image, (str, Path)) else None
                results.append(
                    self._finish(pil, rgb, faces, image_path, gallery=gallery, threshold=threshold, save=save,
                                 output_path=output_path, output_file_format=output_file_format)
                )
        return results

    def _finish(self, pil, rgb_np, faces, image_path, *, gallery, threshold, save, output_path, output_file_format):
        h, w = rgb_np.shape[:2]
        result = self._run_embed(rgb_np, faces, (h, w), image_path)
        if gallery is not None:
            self._identify(result, gallery, threshold)
        if save:
            ext = (output_file_format or "jpg").lower().lstrip(".")
            save_path = resolve_save_path(output_path, image_path, ext=ext)
            annotated = self._annotate(pil, result)
            annotated.save(save_path)
            log_saved_result(result, save_path)
        return result

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
        gallery=None,
        threshold: float = 0.4,
    ) -> Results:
        image_path = image if isinstance(image, (str, Path)) else None
        pil = ImageLoader.load(image, color_format=color_format)
        rgb_np = np.asarray(pil)
        faces = self._collect_faces(rgb_np, detector, face_boxes, face_conf)
        return self._finish(pil, rgb_np, faces, image_path, gallery=gallery, threshold=threshold, save=save,
                            output_path=output_path, output_file_format=output_file_format)

    # ------------------------------------------------------------------
    def _resolve_runtime_detector(self, explicit, face_boxes):
        if face_boxes is not None:
            return None
        if explicit is not None:
            from .model import _resolve_facerec_detector

            return _resolve_facerec_detector(explicit)
        if self.model.face_detector is not None:
            return self.model.face_detector
        try:
            return self.model.default_face_detector()
        except Exception as e:
            raise RuntimeError(
                "LibreFaceEmbedder could not obtain its default face detector "
                f"({e}). Pass face_boxes=[...] for BYO boxes, or "
                "face_detector=... (a callable, a LibreYOLO model, or an "
                "OpenCVFaceDetector) when constructing or calling the model."
            ) from e

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
    def _predict_video(
        self,
        source,
        *,
        detector: Optional[FaceDetector],
        face_boxes: Optional[Sequence],
        face_conf: float,
        gallery,
        threshold: float,
        save: bool,
        show: bool,
        vid_stride: int,
        output_path: Optional[str],
        batch: int = 1,
    ):
        """Embed (and optionally identify) faces frame by frame (``batch`` frames per detector call).

        Caller-supplied ``face_boxes`` apply to every frame (a fixed region on
        a static camera); dropping them here would leave the frame callback
        with neither a detector nor boxes.
        """

        def predict_frame(pil_img):
            rgb_np = np.asarray(pil_img)
            h, w = rgb_np.shape[:2]
            faces = self._collect_faces(rgb_np, detector, face_boxes, face_conf)
            result = self._run_embed(rgb_np, faces, (h, w), str(source))
            if gallery is not None:
                self._identify(result, gallery, threshold)
            return result

        def predict_frames(pil_imgs):
            rgbs = [np.asarray(p) for p in pil_imgs]
            out = []
            for rgb, faces in zip(rgbs, self._detect_many(rgbs, detector, face_boxes, face_conf)):
                h, w = rgb.shape[:2]
                result = self._run_embed(rgb, faces, (h, w), str(source))
                if gallery is not None:
                    self._identify(result, gallery, threshold)
                out.append(result)
            return out

        yield from run_video_inference(
            source,
            predict_frame,
            predict_frames_fn=predict_frames if batch > 1 else None,
            batch=batch,
            vid_stride=vid_stride,
            save=save,
            show=show,
            output_path=output_path,
            annotate_fn=self._annotate,
        )

    # ------------------------------------------------------------------
    def _identify(self, result: Results, gallery, threshold: float) -> None:
        """Attach an Identities payload by matching against ``gallery``."""
        from ...utils.results import Identities

        if result.embeddings is None or len(result.embeddings) == 0:
            result.identities = Identities([], np.zeros((0,), dtype=np.float32))
            return
        if len(gallery) == 0:
            raise ValueError(
                "Cannot identify against an empty FaceGallery — enroll at "
                "least one identity first."
            )
        # Match unthresholded to keep the best score visible for unknowns,
        # then apply the threshold to the name only.
        matches = gallery.match(
            result.embeddings, top_k=1, threshold=-1.0, model=self.model
        )
        names, scores = [], []
        for m in matches:
            best_name, best_score = m[0] if m else (None, float("nan"))
            names.append(best_name if best_score >= threshold else None)
            scores.append(best_score)
        result.identities = Identities(names, np.asarray(scores, dtype=np.float32))

    # ------------------------------------------------------------------
    def _annotate(self, pil_img, result):
        if result.boxes is None or len(result.boxes) == 0:
            return pil_img
        from ...utils.drawing import draw_boxes

        if result.identities is not None and len(result.identities) == len(result.boxes):
            # Label each face with its matched identity (or "unknown"),
            # colored per identity, with the match score as the confidence.
            labels = [n if n is not None else "unknown" for n in result.identities.name]
            palette = {label: i for i, label in enumerate(dict.fromkeys(labels))}
            return draw_boxes(
                pil_img,
                result.boxes.xyxy.tolist(),
                [max(float(s), 0.0) for s in result.identities.score],
                [palette[label] for label in labels],
                class_names={i: label for label, i in palette.items()},
            )

        return draw_boxes(
            pil_img,
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
            class_names=result.names,
        )
