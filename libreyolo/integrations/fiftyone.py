"""FiftyOne integration: predictions in, datasets in and out.

FiftyOne (https://github.com/voxel51/fiftyone, Apache-2.0) is a dataset
curation and prediction-analysis tool. This module is a thin adapter, not a
port: nothing is vendored, ``fiftyone`` is imported lazily, and every entry
point sits on LibreYOLO's public predict API, so it works for any family
whose task is covered below.

Typical use::

    import fiftyone as fo
    from libreyolo import LibreYOLO
    from libreyolo.integrations.fiftyone import apply_model, to_fiftyone

    dataset = to_fiftyone("coco128.yaml", split="val")
    apply_model(dataset, LibreYOLO("LibreYOLO9s.pt"), label_field="predictions")

    results = dataset.evaluate_detections("predictions", gt_field="ground_truth")
    session = fo.launch_app(dataset)

Coordinates follow FiftyOne's convention: boxes and points are normalized to
``[0, 1]`` against the original image, and instance masks are stored cropped
to their box. Boxes are clipped to the image before normalization, which is
what the COCO evaluators do as well.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = [
    "apply_model",
    "from_fiftyone",
    "to_fiftyone",
    "to_fiftyone_model",
]

_INSTALL_HINT = (
    "fiftyone is required by libreyolo.integrations.fiftyone. Install it "
    "with: pip install libreyolo[fiftyone]"
)

# Tasks whose Results this module knows how to express as FiftyOne labels.
SUPPORTED_TASKS = ("detect", "segment", "pose", "obb", "classify")


def _import_fiftyone():
    """Import ``fiftyone``, or raise with an install hint."""
    try:
        import fiftyone as fo  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised without fiftyone
        raise ImportError(_INSTALL_HINT) from exc
    return fo


# =========================================================================
# Results -> FiftyOne payloads
#
# These helpers are deliberately fiftyone-free: they turn a Results into
# plain dicts with FiftyOne's field names, so the geometry is unit-testable
# without the optional dependency installed.
# =========================================================================


def _as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _class_name(names, class_id: int) -> str:
    """Map a class id to its name, falling back to the stringified id."""
    class_id = int(class_id)
    if isinstance(names, dict):
        name = names.get(class_id)
    elif names is not None and 0 <= class_id < len(names):
        name = names[class_id]
    else:
        name = None
    return str(name) if name is not None else str(class_id)


def _bounding_box(xyxy, width: int, height: int) -> List[float]:
    """Convert absolute xyxy to FiftyOne's normalized ``[x, y, w, h]``."""
    x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    return [
        x1 / width,
        y1 / height,
        max(x2 - x1, 0.0) / width,
        max(y2 - y1, 0.0) / height,
    ]


def _crop_mask(mask: np.ndarray, xyxy, width: int, height: int):
    """Crop a full-canvas instance mask to its box, FiftyOne's mask layout."""
    x1 = int(max(math.floor(float(xyxy[0])), 0))
    y1 = int(max(math.floor(float(xyxy[1])), 0))
    x2 = int(min(math.ceil(float(xyxy[2])), width))
    y2 = int(min(math.ceil(float(xyxy[3])), height))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop.astype(bool)


def _detection_payloads(result) -> List[Dict[str, Any]]:
    """Per-box dicts with FiftyOne ``Detection`` field names."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    height, width = (int(v) for v in result.orig_shape)
    xyxy = _as_numpy(boxes.xyxy)
    conf = _as_numpy(boxes.conf)
    cls = _as_numpy(boxes.cls)
    track_id = _as_numpy(boxes.id)
    masks = _as_numpy(result.masks.data) if result.masks is not None else None

    payloads = []
    for i in range(len(boxes)):
        payload: Dict[str, Any] = {
            "label": _class_name(result.names, cls[i]),
            "bounding_box": _bounding_box(xyxy[i], width, height),
            "confidence": float(conf[i]),
        }
        if track_id is not None:
            payload["index"] = int(track_id[i])
        if masks is not None and i < len(masks):
            crop = _crop_mask(masks[i] > 0.5, xyxy[i], width, height)
            if crop is not None:
                payload["mask"] = crop
        payloads.append(payload)
    return payloads


def _obb_payloads(result) -> List[Dict[str, Any]]:
    """Per-instance dicts with FiftyOne ``Polyline`` field names."""
    obb = result.obb
    if obb is None or len(obb) == 0:
        return []

    height, width = (int(v) for v in result.orig_shape)
    corners = _as_numpy(obb.xyxyxyxy)
    conf = _as_numpy(obb.conf)
    cls = _as_numpy(obb.cls)
    track_id = _as_numpy(obb.id)

    payloads = []
    for i in range(len(obb)):
        ring = [(float(x) / width, float(y) / height) for x, y in corners[i]]
        payload: Dict[str, Any] = {
            "label": _class_name(result.names, cls[i]),
            "points": [ring],
            "closed": True,
            "filled": True,
            "confidence": float(conf[i]),
        }
        if track_id is not None:
            payload["index"] = int(track_id[i])
        payloads.append(payload)
    return payloads


def _polyline_payloads(result) -> List[Dict[str, Any]]:
    """Instance masks as normalized polygon rings."""
    masks = result.masks
    if masks is None or len(masks) == 0:
        return []

    boxes = result.boxes
    conf = _as_numpy(boxes.conf) if boxes is not None else None
    cls = _as_numpy(boxes.cls) if boxes is not None else None

    payloads = []
    for i, contour in enumerate(masks.xyn):
        if len(contour) < 3:
            continue
        payload: Dict[str, Any] = {
            "label": _class_name(result.names, cls[i]) if cls is not None else "object",
            "points": [[(float(x), float(y)) for x, y in contour]],
            "closed": True,
            "filled": True,
        }
        if conf is not None and i < len(conf):
            payload["confidence"] = float(conf[i])
        payloads.append(payload)
    return payloads


def _keypoint_payloads(result) -> List[Dict[str, Any]]:
    """Per-instance dicts with FiftyOne ``Keypoint`` field names."""
    keypoints = result.keypoints
    if keypoints is None or len(keypoints) == 0:
        return []

    height, width = (int(v) for v in result.orig_shape)
    xy = _as_numpy(keypoints.xy)
    point_conf = _as_numpy(keypoints.conf)
    boxes = result.boxes
    cls = _as_numpy(boxes.cls) if boxes is not None else None
    box_conf = _as_numpy(boxes.conf) if boxes is not None else None

    payloads = []
    for i in range(len(xy)):
        points = []
        for j in range(xy.shape[1]):
            visible = point_conf is None or float(point_conf[i][j]) > 0
            if visible:
                points.append([float(xy[i][j][0]) / width, float(xy[i][j][1]) / height])
            else:
                # FiftyOne's convention for a keypoint that was not detected.
                points.append([float("nan"), float("nan")])

        payload: Dict[str, Any] = {
            "label": (
                _class_name(result.names, cls[i])
                if cls is not None and i < len(cls)
                else "keypoints"
            ),
            "points": points,
            "index": i,
        }
        if point_conf is not None:
            payload["confidence"] = [float(c) for c in point_conf[i]]
        elif box_conf is not None and i < len(box_conf):
            payload["confidence"] = [float(box_conf[i])] * len(points)
        payloads.append(payload)
    return payloads


def _classification_payload(result) -> Optional[Dict[str, Any]]:
    """Top-1 dict with FiftyOne ``Classification`` field names."""
    probs = result.probs
    if probs is None:
        return None
    top1 = probs.top1
    return {
        "label": _class_name(result.names, top1),
        "confidence": float(_as_numpy(probs.top1conf)),
    }


def _to_fiftyone_labels(result, mask_format: str = "mask"):
    """Convert one ``Results`` into the matching FiftyOne label(s).

    Returns a single ``fiftyone.Label`` for single-slot tasks, or a dict of
    labels for pose (boxes plus keypoints), which FiftyOne stores as separate
    fields derived from ``label_field``.
    """
    fo = _import_fiftyone()
    result = result.numpy()

    if result.probs is not None:
        payload = _classification_payload(result)
        return fo.Classification(**payload) if payload else None

    if result.obb is not None:
        return fo.Polylines(polylines=[fo.Polyline(**p) for p in _obb_payloads(result)])

    if result.keypoints is not None:
        labels = {
            "keypoints": fo.Keypoints(
                keypoints=[fo.Keypoint(**p) for p in _keypoint_payloads(result)]
            )
        }
        if result.boxes is not None:
            labels["detections"] = fo.Detections(
                detections=[fo.Detection(**p) for p in _detection_payloads(result)]
            )
        return labels

    if result.masks is not None and mask_format == "polyline":
        return fo.Polylines(
            polylines=[fo.Polyline(**p) for p in _polyline_payloads(result)]
        )

    if result.boxes is not None:
        return fo.Detections(
            detections=[fo.Detection(**p) for p in _detection_payloads(result)]
        )

    if result.masks is not None:
        return fo.Polylines(
            polylines=[fo.Polyline(**p) for p in _polyline_payloads(result)]
        )

    raise ValueError(
        "This result carries no label FiftyOne can represent. Supported "
        f"tasks: {', '.join(SUPPORTED_TASKS)}."
    )


# =========================================================================
# Model wrapper
# =========================================================================

_MODEL_CLASS = None


def _model_class():
    """Build the ``fiftyone.core.models.Model`` subclass on first use.

    The class is defined lazily because subclassing requires fiftyone to be
    importable, and importing this module must not.
    """
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS

    _import_fiftyone()
    import fiftyone.core.models as fom  # noqa: PLC0415

    class LibreYOLOModel(fom.Model):
        """Runs a LibreYOLO model through FiftyOne's ``apply_model``."""

        def __init__(self, model, mask_format: str = "mask", **predict_kwargs):
            self.model = model
            self.mask_format = mask_format
            self.predict_kwargs = predict_kwargs

        @property
        def media_type(self) -> str:
            return "image"

        @property
        def ragged_batches(self) -> bool:
            # False is what unlocks FiftyOne's batching: it means "a batch of
            # transforms() outputs can be passed to predict_all together".
            # transforms is None here, so FiftyOne hands predict_all the raw
            # images without stacking them, and LibreYOLO letterboxes each one
            # to a common size itself. Returning True would make FiftyOne warn
            # "Model does not support batching" and fall back to one image at
            # a time.
            return False

        @property
        def transforms(self):
            return None

        @property
        def preprocess(self) -> bool:
            return False

        @preprocess.setter
        def preprocess(self, _value):
            # FiftyOne sets this while applying a model; preprocessing is
            # always ours, so the assignment is accepted and ignored.
            pass

        def predict(self, arg):
            result = self.model.predict(arg, color_format="rgb", **self.predict_kwargs)
            if isinstance(result, list):
                result = result[0]
            return _to_fiftyone_labels(result, mask_format=self.mask_format)

        def predict_all(self, args):
            args = list(args)
            if not args:
                return []
            results = self.model.predict(
                args,
                batch=len(args),
                color_format="rgb",
                **self.predict_kwargs,
            )
            if not isinstance(results, list):
                results = [results]
            return [
                _to_fiftyone_labels(r, mask_format=self.mask_format) for r in results
            ]

    _MODEL_CLASS = LibreYOLOModel
    return _MODEL_CLASS


def _resolve_model(model):
    """Accept a loaded LibreYOLO model or a checkpoint name/path."""
    if isinstance(model, (str, Path)):
        from ..models import LibreYOLO  # noqa: PLC0415

        return LibreYOLO(str(model))
    return model


def to_fiftyone_model(
    model,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
    classes: Optional[Sequence[int]] = None,
    max_det: int = 300,
    mask_format: str = "mask",
    **predict_kwargs,
):
    """Wrap a LibreYOLO model as a ``fiftyone.Model``.

    The wrapper plugs into every FiftyOne API that takes a model, including
    ``dataset.apply_model(...)``.

    Args:
        model: A loaded LibreYOLO model, or a checkpoint name/path.
        conf: Confidence threshold.
        iou: NMS IoU threshold.
        imgsz: Input size override (None uses the model default).
        device: Torch device string.
        classes: Optional class-id filter.
        max_det: Maximum detections per image.
        mask_format: ``"mask"`` stores instance masks on the detections;
            ``"polyline"`` stores them as polygons instead.
        **predict_kwargs: Forwarded to ``model.predict``.

    Returns:
        A ``fiftyone.core.models.Model`` instance.
    """
    if mask_format not in ("mask", "polyline"):
        raise ValueError(
            f"mask_format must be 'mask' or 'polyline', got {mask_format!r}"
        )
    return _model_class()(
        _resolve_model(model),
        mask_format=mask_format,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        classes=list(classes) if classes is not None else None,
        max_det=max_det,
        **predict_kwargs,
    )


def apply_model(
    samples,
    model,
    label_field: str = "predictions",
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
    classes: Optional[Sequence[int]] = None,
    max_det: int = 300,
    mask_format: str = "mask",
    batch_size: Optional[int] = None,
    skip_failures: bool = True,
    progress: Optional[bool] = None,
    **predict_kwargs,
):
    """Run a LibreYOLO model on a FiftyOne dataset or view.

    Args:
        samples: A ``fiftyone`` dataset or view.
        model: A loaded LibreYOLO model, or a checkpoint name/path.
        label_field: Field to write predictions to. Pose models write
            ``keypoints`` and ``detections`` into fields derived from this
            name, since one result carries both.
        conf: Confidence threshold.
        iou: NMS IoU threshold.
        imgsz: Input size override (None uses the model default).
        device: Torch device string.
        classes: Optional class-id filter.
        max_det: Maximum detections per image.
        mask_format: ``"mask"`` or ``"polyline"`` for instance segmentation.
        batch_size: Images per forward pass. None runs one image at a time.
        skip_failures: Keep going when a single sample fails.
        progress: Whether to render a progress bar (FiftyOne's default when
            None).
        **predict_kwargs: Forwarded to ``model.predict``.

    Returns:
        None. Predictions are written to ``samples`` in place.
    """
    fo_model = to_fiftyone_model(
        model,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        classes=classes,
        max_det=max_det,
        mask_format=mask_format,
        **predict_kwargs,
    )
    return samples.apply_model(
        fo_model,
        label_field=label_field,
        batch_size=batch_size,
        skip_failures=skip_failures,
        progress=progress,
    )


# =========================================================================
# Dataset bridges
# =========================================================================


def _yolo_rows(label_file: Path) -> List[List[float]]:
    rows = []
    if not label_file.exists():
        return rows
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        rows.append([float(p) for p in parts])
    return rows


def _sample_labels_from_rows(fo, rows, names, task: str):
    """Build the ground-truth label for one image from its YOLO label rows.

    Coordinates in LibreYOLO label files are already normalized, which is
    also FiftyOne's convention, so no image read is needed here.
    """
    if task == "segment":
        polylines = []
        for row in rows:
            class_id, coords = int(row[0]), row[1:]
            if len(coords) == 4:
                cx, cy, w, h = coords
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                ring = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            else:
                ring = list(zip(coords[0::2], coords[1::2]))
            if len(ring) < 3:
                continue
            polylines.append(
                fo.Polyline(
                    label=_class_name(names, class_id),
                    points=[[(float(x), float(y)) for x, y in ring]],
                    closed=True,
                    filled=True,
                )
            )
        return fo.Polylines(polylines=polylines)

    detections = []
    for row in rows:
        if len(row) < 5:
            continue
        class_id = int(row[0])
        cx, cy, w, h = row[1:5]
        detections.append(
            fo.Detection(
                label=_class_name(names, class_id),
                bounding_box=[
                    float(cx - w / 2),
                    float(cy - h / 2),
                    float(w),
                    float(h),
                ],
            )
        )
    return fo.Detections(detections=detections)


def to_fiftyone(
    data: Union[str, Path],
    *,
    split: str = "val",
    task: str = "detect",
    label_field: str = "ground_truth",
    name: Optional[str] = None,
    persistent: bool = False,
    max_samples: Optional[int] = None,
    autodownload: bool = True,
):
    """Load a LibreYOLO dataset yaml as a FiftyOne dataset with ground truth.

    Both dataset layouts documented in ``docs/dataset_schema.md`` are
    supported: the YOLO layout (``images/`` plus ``labels/``, including
    ``.txt`` image lists) and native COCO JSON via the yaml's ``annotations``
    mapping.

    Args:
        data: Dataset name (e.g. ``"coco128"``) or path to a dataset yaml.
        split: Split to load: ``"train"``, ``"val"``, or ``"test"``.
        task: ``"detect"`` for boxes or ``"segment"`` for instance shapes.
        label_field: Field to store ground truth in.
        name: FiftyOne dataset name (None generates one).
        persistent: Whether the FiftyOne dataset survives the session.
        max_samples: Optional cap on the number of samples loaded.
        autodownload: Allow LibreYOLO to download a missing dataset.

    Returns:
        A ``fiftyone.Dataset``.
    """
    fo = _import_fiftyone()
    from ..data.utils import load_data_config  # noqa: PLC0415

    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
    if task not in ("detect", "segment"):
        raise ValueError(f"task must be 'detect' or 'segment', got {task!r}")

    config = load_data_config(str(data), autodownload=autodownload)
    names = config.get("names")

    annotation_file = config.get(f"{split}_annotation_file")
    if annotation_file:
        # label_types must be passed explicitly: without it the importer
        # writes several label types and suffixes each field name, so the
        # ground truth would land in "<label_field>_detections" instead of
        # "<label_field>".
        return fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=str(config[split]),
            labels_path=str(annotation_file),
            label_field=label_field,
            label_types="segmentations" if task == "segment" else "detections",
            name=name,
            persistent=persistent,
            max_samples=max_samples,
        )

    img_files = config.get(f"{split}_img_files")
    label_files = config.get(f"{split}_label_files")
    if not img_files:
        raise ValueError(
            f"Dataset '{data}' has no images for split '{split}'. Checked "
            f"{config.get(split)!r}."
        )
    if max_samples is not None:
        img_files = img_files[:max_samples]
        label_files = label_files[:max_samples] if label_files else None

    dataset = fo.Dataset(name=name, persistent=persistent)
    samples = []
    for i, img_file in enumerate(img_files):
        sample = fo.Sample(filepath=str(img_file))
        if label_files is not None and i < len(label_files):
            rows = _yolo_rows(Path(label_files[i]))
            sample[label_field] = _sample_labels_from_rows(fo, rows, names, task)
        samples.append(sample)
    dataset.add_samples(samples)
    if isinstance(names, dict):
        dataset.default_classes = [names[k] for k in sorted(names)]
    elif names:
        dataset.default_classes = list(names)
    return dataset


def from_fiftyone(
    samples,
    export_dir: Union[str, Path],
    *,
    label_field: str = "ground_truth",
    split: str = "val",
    classes: Optional[Sequence[str]] = None,
    export_media: Union[bool, str] = True,
    yaml_name: str = "dataset.yaml",
) -> Path:
    """Export a FiftyOne dataset or view as a LibreYOLO-trainable dataset.

    This is the half of the round trip that makes curation actionable: filter
    or fix a view in FiftyOne, export it here, and train on the yaml.

    Calling this twice against the same ``export_dir`` with different
    ``split`` values writes both splits into one yaml, which is what training
    needs (``train`` plus ``val``).

    Args:
        samples: A ``fiftyone`` dataset or view.
        export_dir: Directory to write images, labels, and the yaml into.
        label_field: Field holding the labels to export.
        split: Split name to write: ``"train"``, ``"val"``, or ``"test"``.
        classes: Class list fixing the label ids. Defaults to the field's
            classes as FiftyOne resolves them.
        export_media: True copies images, ``"symlink"`` links them, False
            writes labels only.
        yaml_name: Name of the dataset yaml inside ``export_dir``.

    Returns:
        Path to the written dataset yaml, ready for ``model.train(data=...)``.
    """
    fo = _import_fiftyone()

    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

    export_dir = Path(export_dir)
    samples.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        split=split,
        classes=list(classes) if classes is not None else None,
        export_media=export_media,
        yaml_path=yaml_name,
    )
    return export_dir / yaml_name
