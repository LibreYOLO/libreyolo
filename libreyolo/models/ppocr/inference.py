"""Inference orchestrator for the LibrePPOCR two-stage pipeline.

Runs detection, crops each text quad, batches the crops through recognition,
and assembles ``Results.ocr``. Wraps the pipeline behind the same ``__call__``
shape that ``InferenceRunner`` provides for detection models (mirroring the
L2CS gaze runner), so ``LibrePPOCR`` integrates via the standard ``BaseModel``
runner property.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Generator, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from ...postprocess.ppocr import ctc_decode, db_postprocess, sort_quads_reading_order
from ...utils.general import log_saved_result, resolve_save_path
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.results import OCRRegions, Results, keep_source
from ...utils.video import collect_video_results, is_video_file, run_video_inference
from .preprocess import det_normalize, det_resize, get_rotate_crop_image, rec_batches

if TYPE_CHECKING:
    from .model import LibrePPOCR

logger = logging.getLogger(__name__)


class OCRInferenceRunner:
    """Runs the det -> crop -> rec pipeline for a ``LibrePPOCR`` model."""

    def __init__(self, model: "LibrePPOCR"):
        self.model = model

    def __call__(
        self,
        source: ImageInput | list | tuple | None = None,
        *,
        conf: float = 0.25,
        save: bool = False,
        output_path: Optional[str] = None,
        color_format: str = "auto",
        stream: bool = False,
        vid_stride: int = 1,
        show: bool = False,
        output_file_format: Optional[str] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
        rec_batch: int = 6,
        # augment/tiling would silently change semantics, so they are rejected
        # loudly. Other detection-shaped kwargs (iou, max_det, half, ...) are
        # accepted and ignored so generic callers like the CLI can pass them.
        augment: bool = False,
        tiling: bool = False,
        **_: object,
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Run OCR on an image, list of images, directory, or video.

        Args:
            source: Image path/PIL/array, list of those, directory, or video.
            conf: Minimum recognition confidence for a kept region.
            imgsz: Detection long-side limit override (default 960).
            rec_batch: Recognition crops per forward pass.
        """
        if augment:
            raise ValueError(
                "TTA (augment=True) is not supported for OCR models. "
                "Use augment=False."
            )
        if tiling:
            raise ValueError(
                "Tiled inference is not supported for OCR models (text quads "
                "would be split across tiles)."
            )
        if output_file_format is not None:
            output_file_format = output_file_format.lower().lstrip(".")
            if output_file_format not in ("jpg", "jpeg", "png", "webp"):
                raise ValueError(
                    f"Invalid output_file_format: {output_file_format}. "
                    "Must be one of: 'jpg', 'png', 'webp'."
                )
        if device is not None:
            self._set_device(device)

        if is_video_file(source):
            gen = self._predict_video(
                source,
                conf=conf,
                imgsz=imgsz,
                rec_batch=rec_batch,
                save=save,
                show=show,
                vid_stride=vid_stride,
                output_path=output_path,
            )
            if stream:
                return gen
            return collect_video_results(gen, source, vid_stride)

        if isinstance(source, (list, tuple)):
            return [
                self._predict_single(
                    img,
                    conf=conf,
                    imgsz=imgsz,
                    rec_batch=rec_batch,
                    save=save,
                    output_path=output_path,
                    color_format=color_format,
                    output_file_format=output_file_format,
                    save_stem=None if isinstance(img, (str, Path)) else f"image{idx}",
                )
                for idx, img in enumerate(source)
            ]

        if isinstance(source, (str, Path)) and Path(source).is_dir():
            image_paths = ImageLoader.collect_images(source)
            return [
                self._predict_single(
                    p,
                    conf=conf,
                    imgsz=imgsz,
                    rec_batch=rec_batch,
                    save=save,
                    output_path=output_path,
                    color_format=color_format,
                    output_file_format=output_file_format,
                )
                for p in image_paths
            ]

        return self._predict_single(
            source,
            conf=conf,
            imgsz=imgsz,
            rec_batch=rec_batch,
            save=save,
            output_path=output_path,
            color_format=color_format,
            output_file_format=output_file_format,
        )

    # =========================================================================
    # Pipeline
    # =========================================================================

    def _predict_single(
        self,
        image: ImageInput,
        *,
        conf: float,
        imgsz: Optional[int],
        rec_batch: int,
        save: bool,
        output_path: Optional[str],
        color_format: str,
        output_file_format: Optional[str],
        save_stem: Optional[str] = None,
    ) -> Results:
        image_path = image if isinstance(image, (str, Path)) else None
        pil = ImageLoader.load(image, color_format=color_format)
        result = self.run_pipeline(
            pil,
            conf=conf,
            imgsz=imgsz,
            rec_batch=rec_batch,
            image_path=image_path,
        )
        keep_source(result, pil, image)
        if save:
            ext = (output_file_format or "jpg").lower().lstrip(".")
            save_path = resolve_save_path(
                output_path,
                image_path if image_path is not None else save_stem,
                ext=ext,
            )
            self._save_annotated_image(result, pil, save_path)
        return result

    def _predict_video(
        self,
        source: Union[str, Path],
        *,
        conf: float,
        imgsz: Optional[int],
        rec_batch: int,
        save: bool,
        show: bool,
        vid_stride: int,
        output_path: Optional[str],
    ) -> Generator[Results, None, None]:
        def predict_frame(pil_img: Image.Image) -> Results:
            result = self.run_pipeline(
                pil_img,
                conf=conf,
                imgsz=imgsz,
                rec_batch=rec_batch,
                image_path=str(source),
            )
            result.orig_img = pil_img
            return result

        yield from run_video_inference(
            source,
            predict_frame,
            vid_stride=vid_stride,
            save=save,
            show=show,
            output_path=output_path,
            annotate_fn=lambda pil_img, result: self._annotate(pil_img, result),
        )

    def run_pipeline(
        self,
        pil: Image.Image,
        *,
        conf: float,
        imgsz: Optional[int],
        rec_batch: int,
        image_path,
    ) -> Results:
        model = self.model
        charset = model.charset
        if charset is None:
            raise RuntimeError(
                "This LibrePPOCR instance has no recognition charset. Load an "
                "official LibrePPOCR*-ocr.pt checkpoint (the charset ships in "
                "the checkpoint metadata)."
            )

        rgb = np.asarray(pil.convert("RGB"))
        img_bgr = rgb[:, :, ::-1].copy()  # upstream networks consume BGR
        src_h, src_w = img_bgr.shape[:2]
        orig_shape = (src_h, src_w)

        pipeline = model.pipeline_config
        limit = int(imgsz) if imgsz else int(pipeline.get("det_limit_side_len", 960))

        # Stage 1: detection.
        resized, _, _ = det_resize(img_bgr, limit_side_len=limit)
        det_input = det_normalize(resized).to(model.device)
        with torch.no_grad():
            # Replays a captured graph when one is in scope, otherwise eager.
            prob = model.forward_det(det_input)
        prob_map = prob[0, 0].detach().float().cpu().numpy()
        quads, det_scores = db_postprocess(
            prob_map,
            src_w,
            src_h,
            thresh=float(pipeline.get("det_db_thresh", 0.3)),
            box_thresh=float(pipeline.get("det_db_box_thresh", 0.6)),
            unclip_ratio=float(pipeline.get("det_db_unclip_ratio", 1.5)),
        )

        def _empty() -> Results:
            return Results(
                boxes=None,
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=model.names,
                ocr=OCRRegions(
                    torch.zeros((0, 4, 2), dtype=torch.float32),
                    [],
                    None,
                    None,
                    orig_shape,
                ),
            )

        if quads.shape[0] == 0:
            return _empty()

        det_scores = np.asarray(det_scores, dtype=np.float32)
        order = sort_quads_reading_order(quads)
        quads = quads[order]
        det_scores = det_scores[order]

        # Stage 2: recognition on warped crops.
        crops = [get_rotate_crop_image(img_bgr, quads[i].copy()) for i in range(len(quads))]
        keep = [i for i, c in enumerate(crops) if c.shape[0] > 0 and c.shape[1] > 0]
        if not keep:
            return _empty()
        quads = quads[keep]
        det_scores = det_scores[keep]
        crops = [crops[i] for i in keep]

        texts: List[str] = [""] * len(crops)
        rec_scores = np.zeros(len(crops), dtype=np.float32)
        for chunk_indices, batch in rec_batches(crops, batch_size=max(1, int(rec_batch))):
            batch_t = torch.from_numpy(batch).to(model.device)
            with torch.no_grad():
                # fp32 by default: CTC confidences are sensitive to reduced
                # precision, so autocast stays opt-in at the caller level.
                probs = model.model.rec(batch_t)
            decoded = ctc_decode(probs.detach().float().cpu().numpy(), charset)
            for local_idx, (text, score) in zip(chunk_indices, decoded):
                texts[local_idx] = text
                rec_scores[local_idx] = score

        kept = rec_scores >= float(conf)
        if not bool(kept.any()):
            return _empty()

        return Results(
            boxes=None,
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=model.names,
            ocr=OCRRegions(
                torch.from_numpy(quads[kept]).float(),
                [t for t, k in zip(texts, kept) if k],
                torch.from_numpy(rec_scores[kept]),
                torch.from_numpy(det_scores[kept]),
                orig_shape,
            ),
        )

    # =========================================================================
    # Rendering / device
    # =========================================================================

    def _annotate(self, pil_img: Image.Image, result: Results) -> Image.Image:
        if result.ocr is None or len(result.ocr) == 0:
            return pil_img
        from ...utils.drawing import draw_ocr_regions

        ocr_np = result.ocr.numpy()
        return draw_ocr_regions(pil_img, ocr_np.data, ocr_np.texts, ocr_np.conf)

    def _save_annotated_image(self, result: Results, original_img: Image.Image, save_path: Path) -> None:
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


__all__ = ["OCRInferenceRunner"]
