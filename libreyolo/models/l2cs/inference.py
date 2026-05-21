"""Inference orchestrator for L2CS gaze estimation.

Wraps the two-stage pipeline (face detection → L2CS forward → bin decode)
behind the same ``__call__`` shape that ``InferenceRunner`` provides for
detection models, so ``LibreL2CS`` integrates with the rest of the
framework via the standard ``BaseModel`` runner property.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Generator, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from ...utils.general import log_saved_result, resolve_save_path
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.results import Boxes, Gaze, Results
from ...utils.video import collect_video_results, is_video_file, run_video_inference
from .face import FaceBox, FaceDetector, resolve_face_detector
from .utils import bin_logits_to_angles, crop_face, preprocess_face_crops

if TYPE_CHECKING:
    from .model import LibreL2CS


logger = logging.getLogger(__name__)


class GazeInferenceRunner:
    """Runs gaze inference for a ``LibreL2CS`` model.

    Mirrors the public surface of ``InferenceRunner`` for the parameters
    that make sense for gaze (source dispatch, save, streaming) and
    explicitly rejects the ones that do not (``augment``, ``tiling``).
    """

    def __init__(self, model: "LibreL2CS"):
        self.model = model

    # =========================================================================
    # Public entry point
    # =========================================================================

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
        stream: bool = False,
        vid_stride: int = 1,
        show: bool = False,
        output_file_format: Optional[str] = None,
        device: Optional[str] = None,
        # Rejected with a clear message so detection-shaped kwargs fail loudly
        augment: bool = False,
        tiling: bool = False,
        **_: object,
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        if augment:
            raise ValueError(
                "TTA (augment=True) is not meaningful for gaze inference; "
                "horizontal flip negates yaw and there is nothing sensible to merge."
            )
        if tiling:
            raise ValueError(
                "Tiled inference is not supported for gaze (face crops would be split)."
            )
        if device is not None:
            self._set_device(device)

        detector = self._resolve_runtime_detector(face_detector, face_boxes)

        if is_video_file(source):
            gen = self._predict_video(
                source,
                detector=detector,
                face_conf=face_conf,
                save=save,
                show=show,
                vid_stride=vid_stride,
                output_path=output_path,
                output_file_format=output_file_format,
            )
            if stream:
                return gen
            return collect_video_results(gen, source, vid_stride)

        if isinstance(source, (str, Path)) and Path(source).is_dir():
            image_paths = ImageLoader.collect_images(source)
            return [
                self._predict_single(
                    p,
                    detector=detector,
                    face_boxes=None,
                    face_conf=face_conf,
                    save=save,
                    output_path=output_path,
                    color_format=color_format,
                    output_file_format=output_file_format,
                )
                for p in image_paths
            ]

        return self._predict_single(
            source,
            detector=detector,
            face_boxes=face_boxes,
            face_conf=face_conf,
            save=save,
            output_path=output_path,
            color_format=color_format,
            output_file_format=output_file_format,
        )

    # =========================================================================
    # Single-frame path
    # =========================================================================

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
        orig_shape = (h, w)

        faces = self._collect_faces(rgb_np, detector, face_boxes, face_conf)
        result = self._run_gaze(pil, rgb_np, faces, orig_shape, image_path)

        if save:
            ext = (output_file_format or "jpg").lower().lstrip(".")
            save_path = resolve_save_path(output_path, image_path, ext=ext)
            self._save_annotated_image(result, pil, save_path)

        return result

    def _predict_video(
        self,
        source: Union[str, Path],
        *,
        detector: Optional[FaceDetector],
        face_conf: float,
        save: bool,
        show: bool,
        vid_stride: int,
        output_path: Optional[str],
        output_file_format: Optional[str],
    ) -> Generator[Results, None, None]:
        def predict_frame(pil_img: Image.Image) -> Results:
            rgb_np = np.asarray(pil_img)
            h, w = rgb_np.shape[:2]
            faces = self._collect_faces(rgb_np, detector, None, face_conf)
            return self._run_gaze(pil_img, rgb_np, faces, (h, w), str(source))

        def annotate(pil_img: Image.Image, result: Results) -> Image.Image:
            return self._annotate(pil_img, result)

        yield from run_video_inference(
            source,
            predict_frame,
            vid_stride=vid_stride,
            save=save,
            show=show,
            output_path=output_path,
            annotate_fn=annotate,
        )

    # =========================================================================
    # Internals
    # =========================================================================

    def _resolve_runtime_detector(
        self,
        explicit: Optional[FaceDetector],
        face_boxes: Optional[Sequence],
    ) -> Optional[FaceDetector]:
        if face_boxes is not None:
            return None
        if explicit is not None:
            return resolve_face_detector(explicit)
        return self.model.face_detector

    def _collect_faces(
        self,
        image_rgb: np.ndarray,
        detector: Optional[FaceDetector],
        face_boxes: Optional[Sequence],
        face_conf: float,
    ) -> List[FaceBox]:
        if face_boxes is not None:
            from .face import _normalize_boxes  # local to avoid wider import
            return _normalize_boxes(face_boxes, min_score=0.0)
        if detector is None:
            raise RuntimeError(
                "LibreL2CS has no face source. Pass face_boxes=[...] for BYO bboxes "
                "or face_detector=... (a callable, a LibreYOLO model, or a "
                "RetinaFaceAdapter) when constructing or calling the model."
            )
        faces = detector(image_rgb)
        return [f for f in faces if f.score >= face_conf]

    def _run_gaze(
        self,
        pil: Image.Image,
        rgb_np: np.ndarray,
        faces: List[FaceBox],
        orig_shape: tuple,
        image_path: Optional[Union[str, Path]],
    ) -> Results:
        names = {0: "face"}

        def _empty() -> Results:
            return Results(
                boxes=Boxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                ),
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=names,
                gaze=Gaze(
                    torch.zeros((0, 2), dtype=torch.float32),
                    orig_shape=orig_shape,
                ),
            )

        if not faces:
            return _empty()

        crops: list = []
        kept_faces: list = []
        for f in faces:
            try:
                crops.append(crop_face(rgb_np, f.xyxy))
                kept_faces.append(f)
            except ValueError as e:
                logger.warning("Skipping degenerate face crop: %s", e)

        if not crops:
            return _empty()

        device = self.model.device
        batch = preprocess_face_crops(crops, device=device)
        with torch.no_grad():
            yaw_logits, pitch_logits = self.model.model(batch)
        angles = bin_logits_to_angles(
            yaw_logits, pitch_logits, num_bins=self.model.num_bins
        ).cpu()

        xyxy = torch.tensor(
            [list(f.xyxy) for f in kept_faces], dtype=torch.float32
        )
        conf = torch.tensor([f.score for f in kept_faces], dtype=torch.float32)
        cls = torch.zeros(len(kept_faces), dtype=torch.float32)
        return Results(
            boxes=Boxes(xyxy, conf, cls),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=names,
            gaze=Gaze(angles, orig_shape=orig_shape),
        )

    # =========================================================================
    # Rendering
    # =========================================================================

    def _annotate(self, pil_img: Image.Image, result: Results) -> Image.Image:
        if result.boxes is None or len(result.boxes) == 0:
            return pil_img
        from ...utils.drawing import draw_boxes, draw_gaze_arrows

        boxes_xyxy = result.boxes.xyxy.tolist()
        confs = result.boxes.conf.tolist()
        clses = result.boxes.cls.tolist()
        annotated = draw_boxes(
            pil_img,
            boxes_xyxy,
            confs,
            clses,
            class_names=result.names,
        )
        if result.gaze is not None and len(result.gaze) > 0:
            gaze_np = result.gaze.numpy() if isinstance(result.gaze.data, torch.Tensor) else result.gaze
            annotated = draw_gaze_arrows(
                annotated,
                boxes_xyxy,
                gaze_np.data[:, 0],
                gaze_np.data[:, 1],
            )
        return annotated

    def _save_annotated_image(
        self,
        result: Results,
        original_img: Image.Image,
        save_path: Path,
    ) -> None:
        annotated = self._annotate(original_img, result)
        annotated.save(save_path)
        log_saved_result(result, save_path)

    def _set_device(self, device: str) -> None:
        device_str = str(device).strip().lower()
        if device_str in ("", "auto"):
            return
        if device_str.isdigit():
            device_str = f"cuda:{device_str}"
        target = torch.device(device_str)
        if target != self.model.device:
            self.model.device = target
            self.model.model.to(target)
