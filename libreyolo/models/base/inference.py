"""
Inference runner for LibreYOLO models.

Encapsulates all inference-related logic: single-image prediction,
tiled inference, batch processing, video inference, and result wrapping.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
from torchvision.ops import batched_nms

from ...postprocess.slicing import slice_batch_outputs
from ...utils.drawing import (
    draw_boxes,
    draw_keypoints,
    draw_masks,
    draw_obb,
    draw_depth_map,
    draw_edge_map,
    draw_normal_map,
    draw_mesh,
    draw_ocr_regions,
    draw_panoptic,
    draw_points,
    draw_semantic_mask,
    draw_tile_grid,
)
from ...utils.general import (
    get_safe_stem,
    get_slice_bboxes,
    log_saved_result,
    resolve_save_path,
)
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.predict_args import normalize_predict_kwargs
from ...utils.results import (
    AlbedoMap,
    Boxes,
    DepthMap,
    Embeddings,
    EdgeMap,
    Keypoints,
    Masks,
    Matte,
    Meshes,
    NormalMap,
    OBB,
    OCRRegions,
    PanopticSegmentation,
    Points,
    Probs,
    Results,
    RestoredImage,
    SemanticMask,
)
from ...utils.screen import ScreenSource, grab_screen
from ...utils.source import SourceKind, build_stream_source, classify_source
from ...utils.video import (
    FrameSource,
    collect_video_results,
    run_video_inference,
    sample_clip_frames,
)
from .cuda_graph import forward_maybe_graphed, with_cuda_graph_scope

logger = logging.getLogger(__name__)

# Tasks whose result is one row for a whole clip rather than one per frame.
# A family still has to opt in via ``VIDEO_EMBED_MODE = "clip"``.
_WHOLE_CLIP_TASKS = frozenset({"embed", "classify"})
_COMMON_PREDICT_PASSTHROUGH = frozenset(
    {"num_select", "gallery", "threshold", "clip_frames"}
)


def _as_float_tensor(
    value,
    shape: Tuple[int, ...],
    default: float | None = None,
) -> torch.Tensor:
    """Coerce a payload entry to a float tensor, or synthesize a constant one."""
    if value is None:
        if default is None:
            return torch.zeros(shape, dtype=torch.float32)
        return torch.full(shape, float(default), dtype=torch.float32)
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _build_meshes(mesh_data: dict, orig_shape: Tuple[int, int]) -> Meshes:
    """Turn a family's mesh payload dict into a ``Meshes`` result slot."""
    required = ("global_orient", "body_pose", "betas", "transl")
    missing = [key for key in required if mesh_data.get(key) is None]
    if missing:
        raise ValueError(
            "Mesh-task models must return a 'meshes' payload containing "
            f"{', '.join(required)}; missing: {', '.join(missing)}."
        )
    body_model = mesh_data.get("body_model")
    if not body_model:
        raise ValueError(
            "Mesh payloads must name their parameterization via 'body_model' "
            "(for example 'mhr'), since field shapes depend on it."
        )

    def optional(key):
        value = mesh_data.get(key)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(np.asarray(value))

    known = set(required) | {
        "body_model",
        "vertices",
        "faces",
        "joints3d",
        "joints2d",
        "conf",
        "focal_length",
        "extras",
    }
    extras = dict(mesh_data.get("extras") or {})
    # Model-specific parameters (skeleton scale, hand pose, expression) may be
    # passed at the top level rather than nested; keep them rather than drop
    # them silently.
    extras.update(
        {k: v for k, v in mesh_data.items() if k not in known and v is not None}
    )

    return Meshes(
        optional("global_orient"),
        optional("body_pose"),
        optional("betas"),
        optional("transl"),
        body_model=str(body_model),
        vertices=optional("vertices"),
        faces=optional("faces"),
        joints3d=optional("joints3d"),
        joints2d=optional("joints2d"),
        conf=optional("conf"),
        focal_length=optional("focal_length"),
        extras=extras,
        orig_shape=orig_shape,
    )


if TYPE_CHECKING:
    from .model import BaseModel


class InferenceRunner:
    """Handles all inference logic on behalf of a BaseModel instance."""

    def __init__(self, model: BaseModel):
        self.model = model

    def _predict_input_contract(self) -> tuple[frozenset[str], frozenset[str]]:
        """Return and validate the loaded family's auxiliary-input contract."""
        declared = frozenset(getattr(self.model, "PREDICT_INPUT_KWARGS", ()) or ())
        required = frozenset(
            getattr(self.model, "REQUIRED_PREDICT_INPUT_KWARGS", ()) or ()
        )
        invalid = required - declared
        if invalid:
            raise ValueError(
                f"{type(self.model).__name__} declares required prediction "
                "input option(s) that it does not accept: "
                f"{', '.join(sorted(invalid))}."
            )
        return declared, required

    def _preprocess_model_input(
        self,
        image: ImageInput,
        color_format: str,
        input_size,
        predict_input_kwargs: Optional[Dict[str, object]] = None,
    ):
        """Call the prediction hook, retaining support for duck-typed stubs."""
        preprocess = getattr(self.model, "_preprocess_predict", None)
        if preprocess is None:
            preprocess = self.model._preprocess
        return preprocess(
            image,
            color_format,
            input_size=input_size,
            **(predict_input_kwargs or {}),
        )

    @staticmethod
    def _reject_guided_source_modes(
        source_spec,
        predict_input_kwargs: Dict[str, object],
        *,
        stream: bool,
        tiling: bool,
        augment: bool,
    ) -> None:
        """Keep auxiliary guides scoped to an unambiguous single image."""
        if not predict_input_kwargs:
            return

        unsupported = []
        if source_spec.kind == SourceKind.VIDEO:
            unsupported.append("video")
        if source_spec.live:
            unsupported.append("live source")
        if source_spec.kind == SourceKind.SCREEN:
            unsupported.append("screen capture")
        if source_spec.kind == SourceKind.DIRECTORY:
            unsupported.append("directory source")
        if source_spec.kind == SourceKind.IMAGE_BATCH:
            unsupported.append("image batch")
        if stream:
            unsupported.append("stream=True")
        if tiling:
            unsupported.append("tiling=True")
        if augment:
            unsupported.append("augment=True")

        if unsupported:
            keys = ", ".join(sorted(predict_input_kwargs))
            raise ValueError(
                f"Guided prediction input(s) {keys} currently support only "
                "one non-streamed, non-tiled, non-augmented image. "
                f"Unsupported mode(s): {', '.join(unsupported)}."
            )

    @with_cuda_graph_scope
    def __call__(
        self,
        source: ImageInput
        | int
        | list[ImageInput | int]
        | tuple[ImageInput | int, ...]
        | None = None,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        device: str | None = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        augment: bool = False,
        save: bool = False,
        batch: int = 1,
        # video parameters
        stream: bool = False,
        stream_buffer: bool = False,
        vid_stride: int = 1,
        show: bool = False,
        # libreyolo-specific
        output_path: str | None = None,
        color_format: str = "auto",
        tiling: bool = False,
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        cuda_graph: bool | str = False,
        **kwargs,
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """
        Run inference on an image, list of images, directory, or video.

        Args:
            source: Input image, list/tuple of in-memory images, directory
                path, video file path, or a screen-capture source such as
                ``"screen"``, ``"screen 1"``, or ``"screen 1 100 200 512 256"``
                (monitor index, then ``left top width height``).
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            imgsz: Input size override (None = model default).
            classes: Filter to specific class IDs.
            max_det: Maximum detections per image.
            save: If True, saves annotated image or video.
            batch: Images per forward pass for directory and list sources.
                With batch > 1, supported families run a single stacked
                forward per chunk (true batched inference); batch=1 keeps
                the sequential one-forward-per-image behavior.
            stream: If True, return a generator yielding Results one at a time
                instead of a materialized list. Recommended for video, screen
                capture, and large directories to avoid high memory usage. A
                single-image source yields exactly one Results.
            stream_buffer: Keep every captured live frame instead of retaining
                only the newest frame per camera. Buffering preserves frames but
                can increase latency when inference is slower than capture.
            vid_stride: Process every N-th video or screen frame (default: 1).
            show: If True, display annotated frames in a window (video and
                screen sources only).
            output_path: Optional output path.
            color_format: Color format hint.
            tiling: Enable tiled inference for large images.
            overlap_ratio: Tile overlap ratio.
            output_file_format: Output format ("jpg", "png", "webp").
            cuda_graph: Replay the forward pass from a captured CUDA graph.
                Small detectors are launch-bound, so collapsing the forward's
                kernel launches into one replay is a large batch-1 win, and
                output is bit-identical to eager. ``True`` captures on first
                use per input shape; ``"auto"`` waits until a shape repeats, so
                one-shot and shape-varying work never pays capture. Requires
                CUDA and a family that opts in via ``SUPPORTS_CUDA_GRAPH``.
                Call ``model.capture_graph()`` to move capture cost off the
                first request.
            **kwargs: Additional arguments for postprocessing.

        Returns:
            Results, list of Results, or generator of Results (stream=True).
        """
        declared_predict_inputs, required_predict_inputs = (
            self._predict_input_contract()
        )
        kwargs = normalize_predict_kwargs(
            kwargs,
            passthrough=_COMMON_PREDICT_PASSTHROUGH | declared_predict_inputs,
        )
        predict_input_kwargs = {
            key: kwargs.pop(key) for key in declared_predict_inputs if key in kwargs
        }
        missing_predict_inputs = sorted(
            key
            for key in required_predict_inputs
            if key not in predict_input_kwargs or predict_input_kwargs[key] is None
        )
        if missing_predict_inputs:
            raise ValueError(
                f"{type(self.model).__name__} requires prediction input "
                f"option(s): {', '.join(missing_predict_inputs)}."
            )
        source_spec = None
        if predict_input_kwargs:
            source_spec = classify_source(source)
            self._reject_guided_source_modes(
                source_spec,
                predict_input_kwargs,
                stream=stream,
                tiling=tiling,
                augment=augment,
            )
        if device is not None:
            self._set_device(device)
        if (
            kwargs.get("gallery") is not None
            and getattr(self.model, "task", None) != "embed"
        ):
            raise ValueError(
                "gallery= is only supported by models loaded with task='embed'."
            )
        if (
            kwargs.get("threshold") is not None
            and getattr(self.model, "task", None) != "embed"
        ):
            raise ValueError(
                "threshold= is only supported by models loaded with "
                "task='embed' (the gallery match threshold). For detection "
                "confidence use conf=."
            )
        if augment and getattr(self.model, "task", None) == "embed":
            raise ValueError(
                "Test-time augmentation does not support embedding models. "
                "Use augment=False."
            )

        if output_file_format is not None:
            output_file_format = output_file_format.lower().lstrip(".")
            if output_file_format not in ("jpg", "jpeg", "png", "webp"):
                raise ValueError(
                    f"Invalid output_file_format: {output_file_format}. "
                    "Must be one of: 'jpg', 'png', 'webp'"
                )

        if tiling and augment:
            raise ValueError(
                "tiling and augment cannot be used together. Disable one of them."
            )
        if augment and getattr(self.model, "task", None) == "point":
            raise ValueError(
                "Test-time augmentation does not support point-task models yet. "
                "Use augment=False for point models."
            )
        if augment and getattr(self.model, "task", None) == "edge":
            raise ValueError(
                "Test-time augmentation does not support edge detection yet. "
                "Use augment=False for edge models."
            )

        if source_spec is None:
            source_spec = classify_source(source)

        # Whole-clip inference. Opt-in per family: only when the family declares
        # clip support, the resolved task consumes a whole clip, and the source
        # is a finite video. Every other family keeps the frame-by-frame path
        # below.
        #
        # "embed" pools a clip into one row; "classify" scores a clip with a
        # video head (V-JEPA 2's attentive probe). Both collapse the video to a
        # single result, so they share this route rather than adding a second
        # one.
        if (
            getattr(self.model, "VIDEO_EMBED_MODE", "frames") == "clip"
            and getattr(self.model, "task", None) in _WHOLE_CLIP_TASKS
        ):
            if source_spec.live or source_spec.kind == SourceKind.SCREEN:
                raise ValueError(
                    f"{type(self.model).__name__} embeds a whole video as a "
                    "single vector, which requires a finite source with a known "
                    "end. Live cameras, network streams and screen captures are "
                    "unbounded and would have to be buffered indefinitely. Pass "
                    "a video file, or use task='classify' to score frames as "
                    "they arrive."
                )
            if source_spec.kind == SourceKind.VIDEO:
                if vid_stride != 1:
                    raise ValueError(
                        "vid_stride does not apply when a whole video is "
                        "embedded as a single vector: frames are sampled "
                        "uniformly across the entire clip. Use clip_frames= to "
                        "control how many frames are sampled."
                    )
                if show:
                    raise NotImplementedError(
                        "show=True displays annotated frames as they are "
                        "processed, which does not apply to a whole-clip "
                        "embedding that yields one result for the entire video."
                    )
                clip_results = self._predict_video_clip(
                    source_spec.source,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    classes=classes,
                    max_det=max_det,
                    save=save,
                    output_path=output_path,
                    output_file_format=output_file_format,
                    **kwargs,
                )
                # Honor the streaming contract: callers passing stream=True
                # expect an iterator, even though a whole clip collapses to a
                # single Results.
                return iter(clip_results) if stream else clip_results

        # Handle finite video input.
        if source_spec.kind == SourceKind.VIDEO:
            gen = self._predict_video(
                source_spec.source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                save=save,
                show=show,
                vid_stride=vid_stride,
                output_path=output_path,
                **kwargs,
            )
            if stream:
                return gen
            return collect_video_results(gen, source_spec.source, vid_stride)

        # Webcams and network streams are unbounded and always stay lazy.
        if source_spec.live:
            if not stream:
                raise ValueError(
                    "Live stream sources require stream=True so results are "
                    "consumed incrementally."
                )
            frame_source = build_stream_source(
                source_spec,
                vid_stride=vid_stride,
                stream_buffer=stream_buffer,
            )
            return self._predict_video(
                frame_source,
                source_label="stream",
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                save=save,
                show=show,
                output_path=output_path,
                **kwargs,
            )

        # Handle screen-capture input ("screen", "screen 1", "screen 1 x y w h")
        if source_spec.kind == SourceKind.SCREEN:
            return self._predict_screen(
                source_spec.source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                save=save,
                show=show,
                stream=stream,
                vid_stride=vid_stride,
                output_path=output_path,
                output_file_format=output_file_format,
                **kwargs,
            )

        # Handle in-memory batch input (list/tuple of images) and directories.
        # Both collapse to a list of images run in batches.
        images: Optional[List[ImageInput]] = None
        if source_spec.kind == SourceKind.IMAGE_BATCH:
            images = list(source_spec.items)
        elif source_spec.kind == SourceKind.DIRECTORY:
            images = ImageLoader.collect_images(source_spec.source)
            if not images:
                return iter(()) if stream else []

        if images is not None or stream:
            if images is None:
                images = [source]
            batch_kwargs = dict(
                batch=batch,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
                tiling=tiling,
                overlap_ratio=overlap_ratio,
                output_file_format=output_file_format,
                augment=augment,
                **kwargs,
            )
            if stream:
                return self._stream_in_batches(images, **batch_kwargs)
            return self._process_in_batches(images, **batch_kwargs)

        # Use tiled inference if enabled
        if tiling:
            return self._predict_tiled(
                source,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
                overlap_ratio=overlap_ratio,
                output_file_format=output_file_format,
                **kwargs,
            )

        if augment and getattr(self.model, "TTA_ENABLED", False):
            result = self.model._predict_augment(
                source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
                **kwargs,
            )
            if save:
                image_path = source if isinstance(source, (str, Path)) else None
                img_pil = ImageLoader.load(source, color_format=color_format)
                ext = output_file_format or "jpg"
                save_path = resolve_save_path(output_path, image_path, ext=ext)
                self._save_annotated_image(result, img_pil, save_path)
            return result

        return self._predict_single(
            source,
            save=save,
            output_path=output_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
            max_det=max_det,
            color_format=color_format,
            output_file_format=output_file_format,
            predict_input_kwargs=predict_input_kwargs,
            **kwargs,
        )

    def _set_device(self, device: str) -> None:
        """Move the wrapped model when predict(device=...) is supplied."""
        device_str = str(device).strip().lower()
        if device_str in ("", "auto"):
            return
        if device_str.isdigit():
            device_str = f"cuda:{device_str}"
        target = torch.device(device_str)
        if target != self.model.device:
            # ``.to()`` reallocates parameters, so any captured graph now points
            # at freed storage. This matters even when moving back to a device
            # that already has a cache entry: the addresses it recorded are gone.
            if hasattr(self.model, "_invalidate_cuda_graphs"):
                self.model._invalidate_cuda_graphs("device change")
            self.model.device = target
            if hasattr(self.model.model, "to"):
                dtype = None
                if hasattr(self.model, "_resolve_dtype") and hasattr(
                    self.model, "_model_dtype"
                ):
                    dtype = self.model._resolve_dtype()
                    self.model._model_dtype = dtype
                if dtype is not None:
                    self.model.model.to(device=target, dtype=dtype)
                else:
                    self.model.model.to(target)

    def _stream_in_batches(
        self,
        images: Sequence[ImageInput],
        batch: int = 1,
        **kwargs,
    ) -> Generator[Results, None, None]:
        """Yield Results for *images* one at a time, ``batch`` images per step.

        Peak memory holds one chunk of Results rather than the whole source.
        Each chunk still goes through ``_process_in_batches``, so family
        overrides (batched forward passes, backend-specific batching) keep
        applying.
        """
        step = max(1, int(batch))
        for start in range(0, len(images), step):
            yield from self._process_in_batches(
                images[start : start + step],
                batch=batch,
                start_idx=start,
                **kwargs,
            )

    def _process_in_batches(
        self,
        images: Sequence[ImageInput],
        batch: int = 1,
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        tiling: bool = False,
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        augment: bool = False,
        start_idx: int = 0,
        **kwargs,
    ) -> List[Results]:
        """Process multiple images (file paths or in-memory).

        With ``batch > 1`` (and no tiling/TTA) each chunk of ``batch``
        images runs as one stacked forward pass; the batched output is
        sliced back per image and fed to the family's existing batch-1
        postprocess. Otherwise images run sequentially, one forward each.

        ``start_idx`` is the position of ``images[0]`` within the caller's full
        source. It only affects the indexed filename stems given to in-memory
        images, so streaming can hand over one chunk at a time without two
        chunks both saving an ``image0``.
        """
        use_batched = (
            batch > 1
            and not tiling
            and not (augment and getattr(self.model, "TTA_ENABLED", False))
            and getattr(self.model, "SUPPORTS_BATCHED_PREDICT", False)
            # A train-mode network (weightless init, or predict mid-training)
            # would normalize the stacked chunk with cross-image BatchNorm
            # batch statistics, letting chunk-mates change each other's
            # results; keep per-image forwards there.
            and not getattr(getattr(self.model, "model", None), "training", False)
        )
        if use_batched:
            results = []
            for start in range(0, len(images), batch):
                results.extend(
                    self._predict_batch(
                        images[start : start + batch],
                        start_idx=start_idx + start,
                        save=save,
                        output_path=output_path,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        classes=classes,
                        max_det=max_det,
                        color_format=color_format,
                        output_file_format=output_file_format,
                        **kwargs,
                    )
                )
            return results

        results = []
        for idx, image in enumerate(images):
            # In-memory images have no filename to derive a save name from;
            # index them so save=True does not overwrite a single file.
            save_stem = (
                None if isinstance(image, (str, Path)) else f"image{start_idx + idx}"
            )
            if tiling:
                results.append(
                    self._predict_tiled(
                        image,
                        save=save,
                        output_path=output_path,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        classes=classes,
                        max_det=max_det,
                        color_format=color_format,
                        overlap_ratio=overlap_ratio,
                        output_file_format=output_file_format,
                        # Reaches _predict_single through the small-image and
                        # classify/semantic fallbacks so in-memory saves stay
                        # indexed there too.
                        save_stem=save_stem,
                        **kwargs,
                    )
                )
            elif augment and getattr(self.model, "TTA_ENABLED", False):
                result = self.model._predict_augment(
                    image,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    classes=classes,
                    max_det=max_det,
                    color_format=color_format,
                    **kwargs,
                )
                if save:
                    ext = output_file_format or "jpg"
                    save_path = resolve_save_path(
                        output_path,
                        image if save_stem is None else save_stem,
                        ext=ext,
                    )
                    img_pil = ImageLoader.load(image, color_format=color_format)
                    self._save_annotated_image(result, img_pil, save_path)
                results.append(result)
            else:
                results.append(
                    self._predict_single(
                        image,
                        save=save,
                        output_path=output_path,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        classes=classes,
                        max_det=max_det,
                        color_format=color_format,
                        output_file_format=output_file_format,
                        save_stem=save_stem,
                        **kwargs,
                    )
                )
        return results

    def _predict_batch(
        self,
        chunk: Sequence[ImageInput],
        start_idx: int,
        *,
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> List[Results]:
        """Run one stacked forward pass over a chunk of images.

        Mirrors ``_predict_single`` step for step; the only difference is
        that the chunk's preprocessed tensors are concatenated into a single
        batch for the forward pass and the output is sliced back per image.
        ``start_idx`` is the chunk's offset in the full source list, used for
        the indexed save names of in-memory images.
        """
        effective_imgsz = imgsz if imgsz is not None else self.model._get_input_size()
        kwargs["input_size"] = effective_imgsz

        preprocessed = []
        for image in chunk:
            input_tensor, original_img, original_size, ratio = (
                self._preprocess_model_input(
                    image, color_format, input_size=effective_imgsz
                )
            )
            preprocessed.append(
                (input_tensor, original_img, original_size, ratio, image)
            )

        tensors = [item[0] for item in preprocessed]
        stackable = all(
            isinstance(t, torch.Tensor)
            and t.dim() == 4
            and t.shape == tensors[0].shape
            and t.dtype == tensors[0].dtype
            and t.device == tensors[0].device
            for t in tensors
        )
        if not stackable:
            # Preprocess did not yield uniform (1, C, H, W) tensors; keep
            # correctness over speed and run the chunk sequentially.
            return [
                self._predict_single(
                    image,
                    save=save,
                    output_path=output_path,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    classes=classes,
                    max_det=max_det,
                    color_format=color_format,
                    output_file_format=output_file_format,
                    save_stem=(
                        None
                        if isinstance(image, (str, Path))
                        else f"image{start_idx + offset}"
                    ),
                    **kwargs,
                )
                for offset, image in enumerate(chunk)
            ]

        stacked = torch.cat(tensors, dim=0)
        with torch.no_grad():
            output = forward_maybe_graphed(self.model, stacked.to(self.model.device))

        results = []
        for offset, (_, original_img, original_size, ratio, image) in enumerate(
            preprocessed
        ):
            detections = self.model._postprocess(
                slice_batch_outputs(output, offset),
                conf,
                iou,
                original_size,
                max_det=max_det,
                ratio=ratio,
                classes=classes,
                **kwargs,
            )
            image_path = image if isinstance(image, (str, Path)) else None
            result = self._wrap_results(detections, original_size, image_path, classes)
            if save:
                ext = output_file_format or "jpg"
                save_path = resolve_save_path(
                    output_path,
                    image_path
                    if image_path is not None
                    else f"image{start_idx + offset}",
                    ext=ext,
                )
                self._save_annotated_image(result, original_img, save_path)
            results.append(result)
        return results

    def _predict_video_clip(
        self,
        source,
        *,
        conf,
        iou,
        imgsz=None,
        classes=None,
        max_det: int = 300,
        save: bool = False,
        output_path=None,
        output_file_format=None,
        **kwargs,
    ) -> List[Results]:
        """Embed a finite video as a single row (``VIDEO_EMBED_MODE == "clip"``).

        Decoding and uniform sampling are family-independent; tensor layout and
        temporal pooling belong to the family's ``_forward``. Returns a
        one-element list holding one ``Results`` for the whole clip, not one per
        frame.

        ``save=True`` writes a single annotated image built from the first
        sampled frame -- there is no per-frame video to render, because the
        whole clip collapses to one result.
        """
        clip_frames = kwargs.pop("clip_frames", getattr(self.model, "clip_frames", 8))
        clip_frames = int(clip_frames)
        if clip_frames < 1:
            raise ValueError(f"clip_frames must be positive; got {clip_frames}.")

        # Decoding stays shared. Temporal sampling is family-local when the
        # family says so: uniform sampling across the whole video is right for
        # a general clip embedder, but a checkpoint trained on a fixed window
        # (V-JEPA 2 uses 64 frames at stride 2, centered) must be fed that
        # window or its pooled statistics no longer match training.
        sampler = getattr(self.model, "sample_clip_frames", None)
        if callable(sampler):
            frames = sampler(source, clip_frames)
        else:
            frames = sample_clip_frames(source, clip_frames)
        preprocessed = [
            self._preprocess_model_input(frame, color_format="rgb", input_size=imgsz)
            for frame in frames
        ]
        tensors = [item[0] for item in preprocessed]
        original_size = preprocessed[0][2]

        # (F, C, H, W) -> (1, F, C, H, W): one clip in the batch.
        clip = torch.cat(tensors, dim=0).unsqueeze(0)
        with torch.no_grad():
            output = self.model._forward(clip)

        detections = self.model._postprocess(
            output, conf, iou, original_size, max_det=max_det, classes=classes, **kwargs
        )
        result = self._wrap_results(detections, original_size, str(source), classes)
        result.path = str(source)

        if save:
            save_file = resolve_save_path(
                output_path, str(source), ext=output_file_format or "jpg"
            )
            self._save_annotated_image(result, frames[0], save_file)
        return [result]

    def _save_annotated_image(
        self, result: Results, original_img, save_path: Path
    ) -> None:
        """Internal helper to draw boxes, masks, and keypoints and save to disk."""
        # Classification and whole-image embed results carry no boxes; there is
        # nothing to draw, so persist the source image as-is.
        if result.boxes is None and (
            getattr(result, "probs", None) is not None
            or getattr(result, "embeddings", None) is not None
        ):
            original_img.save(save_path)
            log_saved_result(result, save_path)
            return
        if result.boxes is None and getattr(result, "semantic_mask", None) is not None:
            mask_data = result.semantic_mask.data
            if isinstance(mask_data, torch.Tensor):
                mask_data = mask_data.cpu().numpy()
            annotated_img = draw_semantic_mask(original_img, mask_data)
            annotated_img.save(save_path)
            log_saved_result(result, save_path)
            return
        if result.boxes is None and getattr(result, "panoptic", None) is not None:
            pan_data = result.panoptic.data
            if isinstance(pan_data, torch.Tensor):
                pan_data = pan_data.cpu().numpy()
            annotated_img = draw_panoptic(
                original_img,
                pan_data,
                result.panoptic.segments_info,
                class_names=result.names,
            )
            annotated_img.save(save_path)
            log_saved_result(result, save_path)
            return
        if result.boxes is None and getattr(result, "depth_map", None) is not None:
            depth_data = result.depth_map.data
            if isinstance(depth_data, torch.Tensor):
                depth_data = depth_data.cpu().numpy()
            if not result.depth_map.near_is_high:
                depth_data = -depth_data
            annotated_img = draw_depth_map(original_img, depth_data)
            annotated_img.save(save_path)
            log_saved_result(result, save_path)
            return
        if result.boxes is None and getattr(result, "edges", None) is not None:
            edge_data = result.edges.data
            if isinstance(edge_data, torch.Tensor):
                edge_data = edge_data.cpu().numpy()
            annotated_img = draw_edge_map(original_img, edge_data)
            annotated_img.save(save_path)
            log_saved_result(result, save_path)
            return
        if result.boxes is None and getattr(result, "normal_map", None) is not None:
            normal_data = result.normal_map.data
            if isinstance(normal_data, torch.Tensor):
                normal_data = normal_data.cpu().numpy()
            annotated_img = draw_normal_map(original_img, normal_data)
            annotated_img.save(save_path)
            log_saved_result(result, save_path)
            return
        if result.boxes is None and getattr(result, "albedo", None) is not None:
            result.albedo.save(save_path)
            log_saved_result(result, save_path)
            return
        if result.boxes is None and getattr(result, "restored", None) is not None:
            result.restored.save(save_path)
            log_saved_result(result, save_path)
            return
        if result.boxes is None and getattr(result, "matte", None) is not None:
            # A matte result saves as a transparent-background RGBA PNG cutout
            # (source RGB + soft matte alpha), the canonical background-removal
            # deliverable. Force a .png suffix so the alpha channel survives.
            png_path = Path(save_path).with_suffix(".png")
            result.save(png_path, image=original_img)
            log_saved_result(result, png_path)
            return
        if result.boxes is None and getattr(result, "ocr", None) is not None:
            if len(result.ocr) > 0:
                ocr_np = result.ocr.numpy()
                annotated_img = draw_ocr_regions(
                    original_img,
                    ocr_np.data,
                    ocr_np.texts,
                    ocr_np.conf,
                )
            else:
                annotated_img = original_img.copy()
            annotated_img.save(save_path)
            log_saved_result(result, save_path)
            return
        if result.boxes is None and getattr(result, "points", None) is not None:
            if len(result.points) > 0:
                annotated_img = draw_points(
                    original_img,
                    result.points.xy.tolist(),
                    result.points.conf.tolist(),
                    result.points.cls.tolist(),
                    class_names=result.names,
                )
            else:
                annotated_img = original_img.copy()
            annotated_img.save(save_path)
            log_saved_result(result, save_path)
            return
        if len(result) > 0:
            annotated_img = original_img
            # Draw masks first (underneath boxes)
            if result.masks is not None:
                masks_np = result.masks.data
                if isinstance(masks_np, torch.Tensor):
                    masks_np = masks_np.cpu().numpy()
                annotated_img = draw_masks(
                    annotated_img,
                    masks_np,
                    result.boxes.cls.tolist(),
                )
            # Draw boxes
            if result.obb is not None:
                annotated_img = draw_obb(
                    annotated_img,
                    result.obb.xywhr.tolist(),
                    result.obb.conf.tolist(),
                    result.obb.cls.tolist(),
                    class_names=result.names,
                    track_ids=result.obb.id.tolist()
                    if result.obb.id is not None
                    else None,
                )
            else:
                annotated_img = draw_boxes(
                    annotated_img,
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                    class_names=result.names,
                )
            # Draw keypoints
            if result.keypoints is not None:
                kpts_np = result.keypoints.data
                if isinstance(kpts_np, torch.Tensor):
                    kpts_np = kpts_np.cpu().numpy()
                annotated_img = draw_keypoints(annotated_img, kpts_np)
            # Draw body meshes: projected vertices plus the skeleton through
            # the projected joints.
            if result.meshes is not None and len(result.meshes) > 0:
                meshes_np = result.meshes.numpy()
                annotated_img = draw_mesh(
                    annotated_img,
                    joints2d=meshes_np.joints2d,
                    vertices2d=meshes_np.extras.get("vertices2d"),
                    faces=meshes_np.faces,
                    vertices3d=meshes_np.vertices,
                )
        else:
            annotated_img = original_img.copy()

        annotated_img.save(save_path)
        log_saved_result(result, save_path)

    @staticmethod
    def _apply_classes_filter(
        boxes_t: torch.Tensor,
        conf_t: torch.Tensor,
        cls_t: torch.Tensor,
        classes: List[int],
        masks_t: Optional[torch.Tensor] = None,
        keypoints_t: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Filter detections to keep only the requested class IDs."""
        mask = torch.zeros(len(cls_t), dtype=torch.bool, device=cls_t.device)
        for cid in classes:
            mask |= cls_t == cid
        filtered_masks = masks_t[mask] if masks_t is not None else None
        filtered_kpts = keypoints_t[mask] if keypoints_t is not None else None
        return boxes_t[mask], conf_t[mask], cls_t[mask], filtered_masks, filtered_kpts

    def _wrap_results(
        self,
        detections: Dict,
        original_size: Tuple[int, int],
        image_path,
        classes: Optional[List[int]],
    ) -> Results:
        """Convert raw detection dict to a Results object.

        Args:
            detections: Dict with 'boxes', 'scores', 'classes', 'num_detections',
                and optionally 'masks' and/or 'keypoints'.
            original_size: (width, height) from preprocessing.
            image_path: Source path or None.
            classes: Optional class filter list.
        """
        # Classification: a probs vector, no boxes. Wrap into Results.probs so
        # result.probs.top1 / .top5 work like the rest of the ecosystem.
        probs_data = detections.get("probs")
        if probs_data is not None:
            orig_w, orig_h = original_size
            probs_t = (
                probs_data.float()
                if isinstance(probs_data, torch.Tensor)
                else torch.as_tensor(probs_data, dtype=torch.float32)
            )
            return Results(
                boxes=None,
                orig_shape=(orig_h, orig_w),
                path=str(image_path) if image_path else None,
                names=self.model.names,
                probs=Probs(probs_t),
            )

        # Whole-image embedding: one normalized row, no boxes. Optional
        # identities are produced by Gallery during model postprocessing.
        embeddings_data = detections.get("embeddings")
        if embeddings_data is not None:
            orig_w, orig_h = original_size
            embeddings_t = (
                embeddings_data.float()
                if isinstance(embeddings_data, torch.Tensor)
                else torch.as_tensor(embeddings_data, dtype=torch.float32)
            )
            if embeddings_t.ndim == 1:
                embeddings_t = embeddings_t.unsqueeze(0)
            return Results(
                boxes=None,
                orig_shape=(orig_h, orig_w),
                path=str(image_path) if image_path else None,
                names={},
                embeddings=Embeddings(embeddings_t, (orig_h, orig_w)),
                identities=detections.get("identities"),
            )

        # Semantic segmentation: a dense class map, no boxes.
        semantic_data = detections.get("semantic")
        if semantic_data is not None:
            orig_w, orig_h = original_size
            semantic_t = (
                semantic_data
                if isinstance(semantic_data, torch.Tensor)
                else torch.as_tensor(semantic_data)
            )
            return Results(
                boxes=None,
                orig_shape=(orig_h, orig_w),
                path=str(image_path) if image_path else None,
                names=self.model.names,
                semantic_mask=SemanticMask(semantic_t.long(), (orig_h, orig_w)),
            )

        # Panoptic: a dense non-overlapping segment-id map plus segments_info.
        panoptic_data = detections.get("panoptic")
        if panoptic_data is not None:
            orig_w, orig_h = original_size
            panoptic_t = (
                panoptic_data
                if isinstance(panoptic_data, torch.Tensor)
                else torch.as_tensor(panoptic_data)
            )
            return Results(
                boxes=None,
                orig_shape=(orig_h, orig_w),
                path=str(image_path) if image_path else None,
                names=self.model.names,
                panoptic=PanopticSegmentation(
                    panoptic_t.long(),
                    detections.get("segments_info") or [],
                    (orig_h, orig_w),
                ),
            )

        # Depth: a dense relative inverse-depth map, no boxes.
        depth_data = detections.get("depth")
        if depth_data is not None:
            orig_w, orig_h = original_size
            depth_t = (
                depth_data
                if isinstance(depth_data, torch.Tensor)
                else torch.as_tensor(depth_data)
            )
            return Results(
                boxes=None,
                orig_shape=(orig_h, orig_w),
                path=str(image_path) if image_path else None,
                names=self.model.names,
                depth_map=DepthMap(
                    depth_t.float(), (orig_h, orig_w),
                    encoding=detections.get("depth_encoding", "inverse_depth"),
                ),
            )

        albedo_data = detections.get("albedo")
        if albedo_data is not None:
            orig_w, orig_h = original_size
            return Results(
                boxes=None, orig_shape=(orig_h, orig_w),
                path=str(image_path) if image_path else None,
                names=self.model.names,
                albedo=AlbedoMap(torch.as_tensor(albedo_data), (orig_h, orig_w)),
            )

        # Edge detection: a dense (H, W) probability map, no boxes.
        edge_data = detections.get("edges", detections.get("edge"))
        if edge_data is not None:
            orig_w, orig_h = original_size
            edge_t = (
                edge_data
                if isinstance(edge_data, torch.Tensor)
                else torch.as_tensor(edge_data)
            )
            return Results(
                boxes=None,
                orig_shape=(orig_h, orig_w),
                path=str(image_path) if image_path else None,
                names=self.model.names,
                edges=EdgeMap(edge_t.float(), (orig_h, orig_w)),
            )

        # Surface normals: a dense (H, W, 3) unit-vector field, no boxes.
        normal_data = detections.get("normal", detections.get("normals"))
        if normal_data is not None:
            orig_w, orig_h = original_size
            normal_t = (
                normal_data
                if isinstance(normal_data, torch.Tensor)
                else torch.as_tensor(normal_data)
            )
            return Results(
                boxes=None,
                orig_shape=(orig_h, orig_w),
                path=str(image_path) if image_path else None,
                names=self.model.names,
                normal_map=NormalMap(normal_t.float(), (orig_h, orig_w)),
            )

        # Restore: a dense RGB image, no boxes. For super-resolution the restored
        # canvas is ``restore_scale`` times the input, so the RestoredImage carries
        # its own (HR) shape while Results.orig_shape stays the source-image shape.
        restored_data = detections.get("restored")
        if restored_data is not None:
            orig_w, orig_h = original_size
            restored_t = (
                restored_data
                if isinstance(restored_data, torch.Tensor)
                else torch.as_tensor(restored_data)
            )
            restore_scale = int(getattr(self.model, "restore_scale", 1) or 1)
            restored_hw = (int(restored_t.shape[0]), int(restored_t.shape[1]))
            return Results(
                boxes=None,
                orig_shape=(orig_h, orig_w),
                path=str(image_path) if image_path else None,
                names=self.model.names,
                restored=RestoredImage(restored_t.to(torch.uint8), restored_hw),
                restore_scale=restore_scale,
            )

        # Matte: a dense soft alpha matte in [0, 1], no boxes.
        matte_data = detections.get("matte")
        if matte_data is not None:
            orig_w, orig_h = original_size
            matte_t = (
                matte_data
                if isinstance(matte_data, torch.Tensor)
                else torch.as_tensor(matte_data)
            )
            return Results(
                boxes=None,
                orig_shape=(orig_h, orig_w),
                path=str(image_path) if image_path else None,
                names=self.model.names,
                matte=Matte(matte_t.float(), (orig_h, orig_w)),
            )

        # Mesh: per-person body meshes carried alongside their person boxes,
        # the same row-aligned arrangement pose uses for keypoints.
        mesh_data = detections.get("meshes")
        if mesh_data is not None:
            orig_w, orig_h = original_size
            meshes = _build_meshes(mesh_data, (orig_h, orig_w))
            boxes_data = detections.get("boxes")
            boxes = None
            if boxes_data is not None:
                boxes_t = _as_float_tensor(boxes_data, (0, 4))
                scores = detections.get("scores")
                classes_t = detections.get("classes")
                n = boxes_t.shape[0]
                boxes = Boxes(
                    boxes_t,
                    _as_float_tensor(scores, (n,), default=1.0),
                    _as_float_tensor(classes_t, (n,), default=0.0),
                    orig_shape=(orig_h, orig_w),
                )
            return Results(
                boxes=boxes,
                orig_shape=(orig_h, orig_w),
                path=str(image_path) if image_path else None,
                names=self.model.names,
                meshes=meshes,
            )

        # OCR: polygons + transcripts, no axis-aligned boxes.
        ocr_data = detections.get("ocr")
        if ocr_data is not None:
            orig_w, orig_h = original_size
            polygons = ocr_data.get("polygons")
            polygons_t = (
                polygons.float()
                if isinstance(polygons, torch.Tensor)
                else torch.as_tensor(np.asarray(polygons), dtype=torch.float32)
                if polygons is not None and len(polygons)
                else torch.zeros((0, 4, 2), dtype=torch.float32)
            )
            return Results(
                boxes=None,
                orig_shape=(orig_h, orig_w),
                path=str(image_path) if image_path else None,
                names=self.model.names,
                ocr=OCRRegions(
                    polygons_t,
                    ocr_data.get("texts"),
                    ocr_data.get("confidences"),
                    ocr_data.get("det_confidences"),
                    (orig_h, orig_w),
                ),
            )

        points_data = detections.get("points")
        if points_data is None and getattr(self.model, "task", "detect") == "point":
            raise ValueError(
                "Point-task models must return a 'points' payload with rows "
                "(x, y, class, confidence)."
            )
        if points_data is not None:
            orig_w, orig_h = original_size
            if isinstance(points_data, torch.Tensor):
                points_t = points_data.float()
            else:
                points_t = torch.as_tensor(points_data, dtype=torch.float32)
            if points_t.ndim == 1:
                points_t = points_t.unsqueeze(0)
            if points_t.numel() == 0:
                points_t = torch.zeros((0, 4), dtype=torch.float32)
            points_obj = Points(points_t)
            if classes is not None and len(points_t) > 0:
                cls_mask = torch.zeros(
                    len(points_t), dtype=torch.bool, device=points_t.device
                )
                for cid in classes:
                    cls_mask |= points_obj.cls == cid
                points_obj = Points(points_t[cls_mask])
            return Results(
                boxes=None,
                orig_shape=(orig_h, orig_w),
                path=str(image_path) if image_path else None,
                names=self.model.names,
                points=points_obj,
            )

        masks_t = None
        keypoints_t = None
        obb_t = None

        if detections["num_detections"] == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            conf_t = torch.zeros((0,), dtype=torch.float32)
            cls_t = torch.zeros((0,), dtype=torch.float32)
            if "obb" in detections:
                obb_t = torch.zeros((0, 7), dtype=torch.float32)
        else:
            raw_boxes = detections["boxes"]
            if isinstance(raw_boxes, torch.Tensor):
                boxes_t = raw_boxes.float()
            else:
                boxes_t = torch.tensor(raw_boxes, dtype=torch.float32)

            raw_conf = detections["scores"]
            if isinstance(raw_conf, torch.Tensor):
                conf_t = raw_conf.float()
            else:
                conf_t = torch.tensor(raw_conf, dtype=torch.float32)

            raw_cls = detections["classes"]
            if isinstance(raw_cls, torch.Tensor):
                cls_t = raw_cls.float()
            else:
                cls_t = torch.tensor(raw_cls, dtype=torch.float32)

            raw_masks = detections.get("masks")
            if raw_masks is not None:
                if isinstance(raw_masks, torch.Tensor):
                    masks_t = raw_masks
                else:
                    masks_t = torch.tensor(raw_masks)

            raw_kpts = detections.get("keypoints")
            if raw_kpts is not None:
                if isinstance(raw_kpts, torch.Tensor):
                    keypoints_t = raw_kpts.float()
                else:
                    keypoints_t = torch.as_tensor(raw_kpts).float()

            raw_obb = detections.get("obb")
            if raw_obb is not None:
                if isinstance(raw_obb, torch.Tensor):
                    obb_t = raw_obb.float()
                else:
                    obb_t = torch.as_tensor(raw_obb).float()

        # Apply class filter
        if classes is not None and len(boxes_t) > 0:
            cls_mask = torch.zeros(len(cls_t), dtype=torch.bool, device=cls_t.device)
            for cid in classes:
                cls_mask |= cls_t == cid
            boxes_t, conf_t, cls_t, masks_t, keypoints_t = self._apply_classes_filter(
                boxes_t, conf_t, cls_t, classes, masks_t, keypoints_t
            )
            if obb_t is not None:
                obb_t = obb_t[cls_mask]

        # original_size from preprocess is (W, H); orig_shape is (H, W)
        orig_w, orig_h = original_size
        orig_shape = (orig_h, orig_w)

        masks_obj = None
        if masks_t is not None:
            masks_obj = Masks(masks_t, orig_shape)

        keypoints_obj = None
        if keypoints_t is not None:
            keypoints_obj = Keypoints(keypoints_t, orig_shape)

        obb_obj = None
        if obb_t is not None:
            obb_obj = OBB(obb_t, orig_shape)

        return Results(
            boxes=Boxes(boxes_t, conf_t, cls_t),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.model.names,
            masks=masks_obj,
            keypoints=keypoints_obj,
            obb=obb_obj,
        )

    def _predict_single(
        self,
        image: ImageInput,
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        output_file_format: Optional[str] = None,
        save_stem: Optional[str] = None,
        predict_input_kwargs: Optional[Dict[str, object]] = None,
        **kwargs,
    ) -> Results:
        """Run inference on a single image.

        ``save_stem`` overrides the saved filename stem for in-memory images
        (which have no path to derive one from).
        """
        image_path = image if isinstance(image, (str, Path)) else None

        # Resolve input size
        effective_imgsz = imgsz if imgsz is not None else self.model._get_input_size()

        # Pass "effective_imgsz" immediately to kwargs
        kwargs["input_size"] = effective_imgsz

        # Preprocess
        input_tensor, original_img, original_size, ratio = self._preprocess_model_input(
            image,
            color_format,
            input_size=effective_imgsz,
            predict_input_kwargs=predict_input_kwargs,
        )

        # Forward pass
        with torch.no_grad():
            output = forward_maybe_graphed(
                self.model, input_tensor.to(self.model.device)
            )

        # Postprocess
        detections = self.model._postprocess(
            output,
            conf,
            iou,
            original_size,
            max_det=max_det,
            ratio=ratio,
            classes=classes,
            **kwargs,
        )

        # Wrap into Results
        result = self._wrap_results(detections, original_size, image_path, classes)

        # Save annotated image
        if save:
            ext = output_file_format or "jpg"
            save_path = resolve_save_path(
                output_path,
                image_path if image_path is not None else save_stem,
                ext=ext,
            )
            self._save_annotated_image(result, original_img, save_path)

        return result

    def _predict_video(
        self,
        source: Union[str, Path, FrameSource],
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        show: bool = False,
        vid_stride: int = 1,
        output_path: Optional[str] = None,
        source_label: Optional[str] = None,
        **kwargs,
    ) -> Generator[Results, None, None]:
        """Run inference on a video file, yielding per-frame Results.

        Whole-clip families are dispatched earlier, in ``__call__``, so this
        path is always frame-by-frame.
        """
        source_label = str(source) if source_label is None else source_label

        yield from run_video_inference(
            source,
            self._frame_predictor(
                source_label,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                **kwargs,
            ),
            vid_stride=vid_stride,
            save=save,
            show=show,
            output_path=output_path,
        )

    def _frame_predictor(
        self,
        source_label: str,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        **kwargs,
    ):
        """Build the per-frame ``PIL image -> Results`` callable for a stream."""
        effective_imgsz = imgsz if imgsz is not None else self.model._get_input_size()

        # Pass "effective_imgsz" immediately to kwargs
        kwargs["input_size"] = effective_imgsz

        def predict_frame(pil_img):
            input_tensor, original_img, original_size, ratio = (
                self._preprocess_model_input(pil_img, "rgb", input_size=effective_imgsz)
            )
            with torch.no_grad():
                output = forward_maybe_graphed(
                    self.model, input_tensor.to(self.model.device)
                )
            detections = self.model._postprocess(
                output,
                conf,
                iou,
                original_size,
                max_det=max_det,
                ratio=ratio,
                classes=classes,
                **kwargs,
            )
            return self._wrap_results(detections, original_size, source_label, classes)

        return predict_frame

    def _predict_screen(
        self,
        source: Union[str, Path],
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        show: bool = False,
        stream: bool = False,
        vid_stride: int = 1,
        output_path: Optional[str] = None,
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> Union[Results, Generator[Results, None, None]]:
        """Run inference on the screen.

        Without ``stream=True`` this grabs one frame and returns a single
        Results, the screenshot equivalent of predicting on an image file. With
        ``stream=True`` it captures continuously and yields Results per frame,
        which for a live screen never ends on its own: bound it with
        ``itertools.islice`` or break out of the loop.
        """
        if stream:

            def stream_results():
                yield from run_video_inference(
                    ScreenSource(source, vid_stride=vid_stride),
                    self._frame_predictor(
                        str(source),
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        classes=classes,
                        max_det=max_det,
                        **kwargs,
                    ),
                    save=save,
                    show=show,
                    output_path=output_path,
                )

            return stream_results()

        result = self._predict_single(
            grab_screen(source),
            save=save,
            output_path=output_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
            max_det=max_det,
            color_format="rgb",
            output_file_format=output_file_format,
            save_stem="screen",
            **kwargs,
        )
        result.path = str(source)
        return result

    def _merge_tile_detections(
        self,
        boxes: List,
        scores: List,
        classes: List,
        iou_thres: float,
    ) -> Tuple[List, List, List]:
        """Merge detections from tiles using class-wise NMS."""
        if not boxes:
            return [], [], []

        boxes_t = torch.tensor(boxes, dtype=torch.float32, device=self.model.device)
        scores_t = torch.tensor(scores, dtype=torch.float32, device=self.model.device)
        classes_t = torch.tensor(classes, dtype=torch.int64, device=self.model.device)

        # Drop non-finite rows — batched_nms is undefined on NaN/Inf inputs.
        finite_mask = torch.isfinite(boxes_t).all(dim=1) & torch.isfinite(scores_t)
        if not finite_mask.all():
            boxes_t = boxes_t[finite_mask]
            scores_t = scores_t[finite_mask]
            classes_t = classes_t[finite_mask]
            if boxes_t.numel() == 0:
                return [], [], []

        # Shift to non-negative coords — batched_nms's class-offset trick
        # uses (boxes.max() + 1), which only separates classes when all
        # coords are non-negative. Translation-invariant for IoU.
        nms_boxes = boxes_t - boxes_t.min().clamp(max=0)
        # Single per-class-batched dispatch instead of one NMS call per class.
        keep = batched_nms(nms_boxes, scores_t, classes_t, iou_thres)
        return (
            boxes_t[keep].cpu().tolist(),
            scores_t[keep].cpu().tolist(),
            classes_t[keep].cpu().tolist(),
        )

    def _predict_tiled(
        self,
        image: ImageInput,
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        save_stem: Optional[str] = None,
        **kwargs,
    ) -> Results:
        """Run tiled inference on large images.

        ``save_stem`` overrides the saved artifact stem for in-memory images
        (which have no path to derive one from); it is threaded into the
        single-image fallbacks below and into the tiled save directory name.
        """

        # Tiling is a detection-time technique; for whole-image classification
        # and dense semantic maps it is meaningless, so fall back to a single
        # forward pass.
        if getattr(self.model, "task", None) in ("classify", "semantic", "embed"):
            return self._predict_single(
                image,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
                output_file_format=output_file_format,
                save_stem=save_stem,
                **kwargs,
            )
        if getattr(self.model, "task", "detect") == "depth":
            raise ValueError(
                "Tiled inference does not support depth maps yet. "
                "Use non-tiled inference for depth models."
            )
        if getattr(self.model, "task", "detect") in ("edge", "normal", "albedo"):
            raise ValueError(
                "Tiled inference does not support edge, normal, or albedo maps yet. "
                "Use non-tiled inference for these dense models."
            )

        if getattr(self.model, "_is_segmentation", False):
            raise ValueError(
                "Tiled inference does not support segmentation masks. "
                "Use non-tiled inference for instance segmentation."
            )
        if getattr(self.model, "task", "detect") == "obb":
            raise ValueError(
                "Tiled inference does not support oriented boxes yet. "
                "Use non-tiled inference for OBB models."
            )
        if getattr(self.model, "task", "detect") == "point":
            raise ValueError(
                "Tiled inference does not support point results yet. "
                "Use non-tiled inference for point models."
            )

        input_size = imgsz if imgsz is not None else self.model._get_input_size()
        if isinstance(input_size, (list, tuple)):
            raise ValueError(
                "Tiled inference requires a square imgsz (tiles are square). "
                f"Got imgsz={tuple(input_size)}; pass a single int or disable tiling."
            )
        img_pil = ImageLoader.load(image, color_format=color_format)
        orig_width, orig_height = img_pil.size
        image_path = image if isinstance(image, (str, Path)) else None

        # Skip tiling if image is small enough
        if orig_width <= input_size and orig_height <= input_size:
            return self._predict_single(
                image,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
                output_file_format=output_file_format,
                save_stem=save_stem,
                **kwargs,
            )

        # Get tile coordinates
        slices = get_slice_bboxes(
            orig_width, orig_height, slice_size=input_size, overlap_ratio=overlap_ratio
        )

        # Process tiles
        all_boxes, all_scores, all_classes = [], [], []
        tiles_data = []

        for idx, (x1, y1, x2, y2) in enumerate(slices):
            tile = img_pil.crop((x1, y1, x2, y2))

            if save:
                tiles_data.append(
                    {"index": idx, "coords": (x1, y1, x2, y2), "image": tile.copy()}
                )

            tile_result = self._predict_single(
                tile,
                save=False,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                **kwargs,
            )

            # Shift boxes to original coordinates
            if len(tile_result) > 0:
                tile_boxes = tile_result.boxes.xyxy.tolist()
                for box in tile_boxes:
                    shifted_box = [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1]
                    all_boxes.append(shifted_box)
                all_scores.extend(tile_result.boxes.conf.tolist())
                all_classes.extend(tile_result.boxes.cls.tolist())

        # Merge detections
        final_boxes, final_scores, final_classes = self._merge_tile_detections(
            all_boxes, all_scores, all_classes, iou
        )
        if max_det >= 0:
            final_boxes = final_boxes[:max_det]
            final_scores = final_scores[:max_det]
            final_classes = final_classes[:max_det]

        # Build Results
        original_size = (orig_width, orig_height)
        detections = {
            "boxes": final_boxes,
            "scores": final_scores,
            "classes": final_classes,
            "num_detections": len(final_boxes),
        }
        result = self._wrap_results(detections, original_size, image_path, classes)

        # Attach tiling metadata as extra attributes
        result.tiled = True
        result.num_tiles = len(slices)

        # Save if requested
        if save:
            ext = output_file_format or "jpg"

            if isinstance(image_path, (str, Path)):
                stem = get_safe_stem(image_path)
            else:
                stem = save_stem or "inference"
            model_tag = f"{self.model._get_model_name()}_{self.model.size}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            if output_path:
                base_path = Path(output_path)
                if base_path.suffix == "":
                    save_dir = base_path / f"{stem}_{model_tag}_{timestamp}"
                else:
                    save_dir = base_path.parent / f"{stem}_{model_tag}_{timestamp}"
            else:
                save_dir = (
                    Path("runs/tiled_detections") / f"{stem}_{model_tag}_{timestamp}"
                )

            save_dir.mkdir(parents=True, exist_ok=True)

            # Save tiles
            tiles_dir = save_dir / "tiles"
            tiles_dir.mkdir(parents=True, exist_ok=True)
            for tile_data in tiles_data:
                tile_filename = f"tile_{tile_data['index']:03d}.{ext}"
                tile_data["image"].save(tiles_dir / tile_filename)

            # Save annotated image
            if len(result) > 0:
                annotated_img = draw_boxes(
                    img_pil,
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                    class_names=result.names,
                )
            else:
                annotated_img = img_pil.copy()

            annotated_img.save(save_dir / f"final_image.{ext}")

            # Save grid visualization
            grid_img = draw_tile_grid(img_pil, slices)
            grid_path = save_dir / f"grid_visualization.{ext}"
            grid_img.save(grid_path)

            # Save metadata
            metadata = {
                "model": self.model._get_model_name(),
                "size": self.model.size,
                "image_source": str(image_path) if image_path else "PIL/numpy input",
                "original_size": [orig_width, orig_height],
                "num_tiles": len(slices),
                "tile_size": input_size,
                "overlap_ratio": overlap_ratio,
                "num_detections": len(result),
                "conf": conf,
                "iou": iou,
            }
            with open(save_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            log_saved_result(result, save_dir)
            result.tiles_path = str(tiles_dir)
            result.grid_path = str(grid_path)

        return result
