"""Inference orchestrator for SAM 3D Body mesh recovery.

Wraps the upstream estimator behind the same ``__call__`` shape the standard
``InferenceRunner`` provides, so the family integrates with the rest of the
framework. Mirrors the arrangement the gaze task uses: a two-stage pipeline
(person detection then per-person regression) with its own runner, accepting
the parameters that make sense and rejecting the ones that do not.
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
from ...utils.results import Boxes, Meshes, Results
from ...utils.video import collect_video_results, is_video_file, run_video_inference
from .camera import perspective_project
from .person import PersonBox, PersonDetector, normalize_person_boxes, resolve_person_detector

if TYPE_CHECKING:
    from .model import LibreSAM3DBody

logger = logging.getLogger(__name__)


class MeshInferenceRunner:
    """Runs body-mesh inference for a ``LibreSAM3DBody`` model."""

    def __init__(self, model: "LibreSAM3DBody"):
        self.model = model

    def __call__(
        self,
        source: ImageInput | None = None,
        *,
        person_boxes: Optional[Sequence] = None,
        person_detector: Optional[PersonDetector] = None,
        conf: float = 0.5,
        save: bool = False,
        output_path: Optional[str] = None,
        color_format: str = "auto",
        stream: bool = False,
        vid_stride: int = 1,
        show: bool = False,
        output_file_format: Optional[str] = None,
        device: Optional[str] = None,
        focal_length: Optional[float] = None,
        # Rejected loudly so detection-shaped kwargs do not silently no-op.
        augment: bool = False,
        tiling: bool = False,
        **_: object,
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        if augment:
            raise ValueError(
                "TTA (augment=True) is not supported for body mesh: a horizontal "
                "flip swaps left and right body parts, so flipped mesh parameters "
                "cannot be merged by averaging."
            )
        if tiling:
            raise ValueError(
                "Tiled inference is not supported for body mesh (a person crop "
                "would be split across tiles)."
            )
        if output_file_format is not None:
            output_file_format = output_file_format.lower().lstrip(".")
            if output_file_format not in ("jpg", "jpeg", "png", "webp"):
                raise ValueError(
                    f"Invalid output_file_format: {output_file_format}. "
                    "Must be one of: 'jpg', 'png', 'webp'."
                )

        detector = self._resolve_runtime_detector(person_detector, person_boxes)

        if person_boxes is not None and is_video_file(source):
            # Refuse rather than drop them: a single set of boxes cannot follow
            # a moving person across frames, and letting this through would
            # surface as a per-frame "pass person_boxes" error telling the user
            # to do the thing they already did.
            raise ValueError(
                "person_boxes applies to a single image and cannot be reused "
                "across video frames, where people move between them. Pass "
                "person_detector=... for video, or call the model per frame "
                "with the boxes for that frame."
            )

        if is_video_file(source):
            gen = self._predict_video(
                source,
                detector=detector,
                conf=conf,
                save=save,
                show=show,
                vid_stride=vid_stride,
                output_path=output_path,
                output_file_format=output_file_format,
                focal_length=focal_length,
            )
            if stream:
                return gen
            return collect_video_results(gen, source, vid_stride)

        if isinstance(source, (str, Path)) and Path(source).is_dir():
            return [
                self._predict_single(
                    p,
                    detector=detector,
                    person_boxes=None,
                    conf=conf,
                    save=save,
                    output_path=output_path,
                    color_format=color_format,
                    output_file_format=output_file_format,
                    focal_length=focal_length,
                )
                for p in ImageLoader.collect_images(source)
            ]

        return self._predict_single(
            source,
            detector=detector,
            person_boxes=person_boxes,
            conf=conf,
            save=save,
            output_path=output_path,
            color_format=color_format,
            output_file_format=output_file_format,
            focal_length=focal_length,
        )

    # =========================================================================
    # Single-frame path
    # =========================================================================

    def _predict_single(
        self,
        image: ImageInput,
        *,
        detector: Optional[PersonDetector],
        person_boxes: Optional[Sequence],
        conf: float,
        save: bool,
        output_path: Optional[str],
        color_format: str,
        output_file_format: Optional[str],
        focal_length: Optional[float],
    ) -> Results:
        image_path = image if isinstance(image, (str, Path)) else None
        pil = ImageLoader.load(image, color_format=color_format)
        rgb = np.asarray(pil)
        result = self._run_mesh(rgb, detector, person_boxes, conf, image_path, focal_length)

        if save:
            ext = (output_file_format or "jpg").lower().lstrip(".")
            save_path = resolve_save_path(output_path, image_path, ext=ext)
            annotated = self._annotate(pil, result)
            annotated.save(save_path)
            log_saved_result(result, save_path)
        return result

    def _predict_video(
        self,
        source,
        *,
        detector,
        conf,
        save,
        show,
        vid_stride,
        output_path,
        output_file_format,
        focal_length,
    ) -> Generator[Results, None, None]:
        def predict_frame(pil_img: Image.Image) -> Results:
            return self._run_mesh(
                np.asarray(pil_img), detector, None, conf, str(source), focal_length
            )

        yield from run_video_inference(
            source,
            predict_frame,
            vid_stride=vid_stride,
            save=save,
            show=show,
            output_path=output_path,
            annotate_fn=self._annotate,
        )

    # =========================================================================
    # Internals
    # =========================================================================

    def _resolve_runtime_detector(self, explicit, person_boxes):
        if person_boxes is not None:
            return None
        if explicit is not None:
            return resolve_person_detector(explicit)
        return self.model.person_detector

    def _collect_people(self, rgb, detector, person_boxes, conf) -> List[PersonBox]:
        if person_boxes is not None:
            return normalize_person_boxes(person_boxes, min_score=0.0)
        if detector is None:
            raise RuntimeError(
                f"{type(self.model).__name__} has no person source. Pass person_boxes=[...] for "
                "BYO boxes, or person_detector=... (a callable, a LibreYOLO model, "
                "or a PersonDetector) when constructing or calling the model."
            )
        return [p for p in detector(rgb) if p.score >= conf]

    def _empty(self, orig_shape, image_path) -> Results:
        return Results(
            boxes=Boxes(
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
            ),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.model.names,
        )

    def _run_mesh(
        self, rgb, detector, person_boxes, conf, image_path, focal_length
    ) -> Results:
        h, w = rgb.shape[:2]
        orig_shape = (h, w)
        people = self._collect_people(rgb, detector, person_boxes, conf)
        if not people:
            return self._empty(orig_shape, image_path)

        boxes_np = np.array([list(p.xyxy) for p in people], dtype=np.float32)
        raw = self.model.estimate(rgb, boxes_np, focal_length=focal_length)
        if not raw:
            return self._empty(orig_shape, image_path)

        def stack(key):
            return torch.as_tensor(
                np.stack([np.asarray(r[key], dtype=np.float32) for r in raw])
            )

        def stack_optional(key):
            """Stack a field the upstream estimator may not emit.

            Which parameter blocks come back depends on the inference type
            (body-only runs omit the hand decoder's output), so a missing key
            must degrade to "not reported" rather than raising.
            """
            if any(key not in r or r[key] is None for r in raw):
                return None
            return stack(key)

        transl = stack("pred_cam_t")
        vertices = stack("pred_vertices")
        joints3d = stack("pred_keypoints_3d")
        joints2d = stack("pred_keypoints_2d")
        focal = torch.as_tensor(
            np.array([float(np.asarray(r["focal_length"]).reshape(-1)[0]) for r in raw],
                     dtype=np.float32)
        )

        # Vertices and joints come back root-relative; the camera translation is
        # carried separately. Add it so the payload is in the camera frame, as
        # the mesh contract requires.
        vertices_cam = vertices + transl[:, None, :]
        vertices2d = perspective_project(vertices_cam, focal, image_size=orig_shape)

        meshes = Meshes(
            stack("global_rot"),
            stack("body_pose_params"),
            stack("shape_params"),
            transl,
            body_model="mhr",
            vertices=vertices_cam,
            faces=self.model.faces,
            joints3d=joints3d + transl[:, None, :],
            joints2d=joints2d,
            conf=torch.as_tensor(np.array([p.score for p in people], dtype=np.float32)),
            focal_length=focal,
            extras={
                key: value
                for key, value in (
                    ("scale", stack_optional("scale_params")),
                    ("hand_pose", stack_optional("hand_pose_params")),
                    ("expression", stack_optional("expr_params")),
                    ("vertices2d", vertices2d),
                )
                if value is not None
            },
            orig_shape=orig_shape,
        )
        return Results(
            boxes=Boxes(
                torch.as_tensor(boxes_np),
                torch.as_tensor(np.array([p.score for p in people], dtype=np.float32)),
                torch.zeros(len(people), dtype=torch.float32),
            ),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.model.names,
            meshes=meshes,
        )

    def _annotate(self, pil_img: Image.Image, result: Results) -> Image.Image:
        from ...utils.drawing import draw_boxes, draw_mesh

        if result.boxes is None or len(result.boxes) == 0:
            return pil_img
        annotated = draw_boxes(
            pil_img,
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
            class_names=result.names,
        )
        if result.meshes is not None and len(result.meshes) > 0:
            m = result.meshes.numpy()
            annotated = draw_mesh(
                annotated,
                joints2d=m.joints2d,
                vertices2d=m.extras.get("vertices2d"),
                faces=m.faces,
                # Metric camera-space vertices drive the shading normals and
                # the far-to-near draw order.
                vertices3d=m.vertices,
            )
        return annotated
