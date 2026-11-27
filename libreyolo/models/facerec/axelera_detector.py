"""Face detector on the Axelera Metis: the family's YuNet-style detector (``librefacerec-det.onnx``) compiled at a fixed
canvas, with its decode on the host.

The compiled graph stops at the detector's 12 head convolutions (class, objectness, box and 5-landmark maps at strides
8, 16 and 32, layout ``yunet_raw_heads_v1``); LibreYOLO applies the sigmoids, decodes boxes and landmarks on the grid
and runs NMS in numpy, exactly as ``cv2.FaceDetectorYN`` does on the same maps. Images are letterboxed (top-left, zero
padding) into the canvas as BGR 0..255, the detector's convention.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..l2cs.face import FaceBox

YUNET_RAW_LAYOUT = "yunet_raw_heads_v1"
YUNET_STRIDES = (8, 16, 32)
YUNET_RAW_NAMES = tuple(f"{k}_{s}" for k in ("cls", "obj", "bbox", "kps") for s in YUNET_STRIDES)
# The source graph's tensors right before each output's Transpose (librefacerec-det.onnx).
_YUNET_RAW_TENSORS = {
    "cls_8": "258", "cls_16": "260", "cls_32": "262", "obj_8": "270", "obj_16": "272", "obj_32": "274",
    "bbox_8": "264", "bbox_16": "266", "bbox_32": "268", "kps_8": "276", "kps_16": "278", "kps_32": "280",
}


def letterbox_bgr(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """``(1, 3, size, size)`` float32 BGR 0..255, the image resized into the top-left corner; and the scale."""
    import cv2

    h, w = image_rgb.shape[:2]
    ratio = min(size / h, size / w)
    resized = cv2.resize(np.ascontiguousarray(image_rgb[:, :, ::-1]), (int(round(w * ratio)), int(round(h * ratio))))
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas.astype(np.float32).transpose(2, 0, 1)[None], ratio


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_yunet_raw(maps, size: int, score_threshold: float = 0.6, nms_threshold: float = 0.3):
    """Raw head maps (``{name: [1, C, H, W]}``) -> ``(xywh [N, 4], landmarks [N, 5, 2], scores [N])`` in canvas pixels."""
    import cv2

    boxes, marks, scores = [], [], []
    for s in YUNET_STRIDES:
        cls, obj, box, kps = (
            np.asarray(maps[f"{k}_{s}"], np.float32)[0].transpose(1, 2, 0).reshape(-1, np.asarray(maps[f"{k}_{s}"]).shape[1])
            for k in ("cls", "obj", "bbox", "kps")
        )
        cls, obj = _sigmoid(cls), _sigmoid(obj)
        cols = size // s
        row, col = np.divmod(np.arange(cls.shape[0]), cols)
        scores.append(np.sqrt(np.clip(cls[:, 0], 0, 1) * np.clip(obj[:, 0], 0, 1)))
        cx, cy = (col + box[:, 0]) * s, (row + box[:, 1]) * s
        bw, bh = np.exp(box[:, 2]) * s, np.exp(box[:, 3]) * s
        boxes.append(np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1))
        marks.append(np.stack([np.stack([(kps[:, 2 * j] + col) * s, (kps[:, 2 * j + 1] + row) * s], 1) for j in range(5)], 1))
    boxes, marks, scores = np.concatenate(boxes), np.concatenate(marks), np.concatenate(scores)
    keep = np.asarray(
        cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold, nms_threshold, top_k=5000), dtype=np.int64
    ).reshape(-1)
    return boxes[keep], marks[keep], scores[keep]


class AxeleraFaceDetector:
    """``FaceDetector`` running the family's detector on an Axelera Metis (``export_face_detector_axelera``).

    ``detect_batch`` submits several images in one runtime call; the face-embedding pipeline uses it for lists,
    directories and video files (``default_predict_batch`` images at a time; ``batch=1`` opts out).
    """

    default_predict_batch = 8

    def __init__(self, path: str | Path, score_threshold: float = 0.6, nms_threshold: float = 0.3):
        import yaml

        from ...backends.axelera import AxeleraBackend, require_axelera_rt, resolve_axelera_artifact

        axm, metadata = resolve_axelera_artifact(path)
        meta = yaml.safe_load(Path(metadata).read_text()) or {}
        if meta.get("output_layout") != YUNET_RAW_LAYOUT:
            raise ValueError(f"{path} is not a face-detector Axelera export (layout {YUNET_RAW_LAYOUT}).")
        require_axelera_rt()
        self.path = str(path)
        self.size = int(meta["imgsz"][0])
        self.output_names = list(meta["output_names"])
        self.score_threshold, self.nms_threshold = float(score_threshold), float(nms_threshold)
        self._model = AxeleraBackend._load(axm)

    def _faces(self, outputs, ratio: float, shape) -> List[FaceBox]:
        outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        maps = {}
        for name, array in zip(self.output_names, outputs):
            array = np.asarray(array, np.float32)
            maps[name] = array if array.ndim == 4 else array.reshape((1, *array.shape))
        boxes, marks, scores = decode_yunet_raw(maps, self.size, self.score_threshold, self.nms_threshold)
        h, w = shape[:2]
        faces = []
        for (x, y, bw, bh), lm, sc in zip(boxes / ratio, marks / ratio, scores):
            x1, y1 = max(0.0, float(x)), max(0.0, float(y))
            x2, y2 = min(float(w), float(x + bw)), min(float(h), float(y + bh))
            faces.append(FaceBox(xyxy=(x1, y1, x2, y2), score=float(sc), landmarks=lm.astype(np.float32)))
        return faces

    def __call__(self, image_rgb: np.ndarray) -> List[FaceBox]:
        from ...backends.axelera import _no_device_error

        blob, ratio = letterbox_bgr(image_rgb, self.size)
        try:
            outputs = self._model(blob)
        except RuntimeError as exc:
            raise _no_device_error(exc) from exc
        return self._faces(outputs, ratio, image_rgb.shape)

    def detect_batch(self, images_rgb: Sequence[np.ndarray]) -> List[List[FaceBox]]:
        """Faces of several images, submitted to the runtime as one batch (results equal one-by-one calls)."""
        from ...backends.axelera import _no_device_error

        if len(images_rgb) <= 1:
            return [self(image) for image in images_rgb]
        blobs, ratios = zip(*(letterbox_bgr(image, self.size) for image in images_rgb))
        try:
            outputs = list(self._model.batch(list(blobs)))
        except RuntimeError as exc:
            raise _no_device_error(exc) from exc
        return [self._faces(o, r, image.shape) for o, r, image in zip(outputs, ratios, images_rgb)]


def export_face_detector_axelera(
    output_path: str | Path = "librefacerec-det_axelera_model",
    *,
    detector_onnx: Optional[str | Path] = None,
    data: str | Path = "coco128.yaml",
    fraction: float = 1.0,
    canvas: int = 640,
    quantization_scheme: str = "per_tensor_min_max",
    verify: bool = False,
) -> str:
    """Compile the family's face detector for an Axelera Metis at a fixed ``canvas`` (raw head maps; host decode).

    Calibration letterboxes the dataset's training images as the detector sees them. Load the result with
    ``LibreFaceEmbedder(..., face_detector=<output_path>)``.
    """
    import tempfile

    import onnx
    import onnxruntime
    import torch
    from onnx import utils as onnx_utils

    from ...data.utils import get_img_files, load_data_config
    from ...export.axelera import export_axelera
    from .weights import resolve_facerec_weight

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    source = str(detector_onnx or resolve_facerec_weight("librefacerec-det.onnx"))
    config = load_data_config(str(data), autodownload=True)
    root = Path(config.get("path", "."))
    split = config.get("train") or config.get("val")
    images = get_img_files(split, prefix=str(root))
    images = images[: max(1, int(len(images) * float(fraction)))]
    import cv2

    calibration = []
    for image_path in images:
        bgr = cv2.imread(str(image_path))
        if bgr is not None:
            calibration.append(letterbox_bgr(np.ascontiguousarray(bgr[:, :, ::-1]), canvas)[0])
    if not calibration:
        raise ValueError(f"No images to calibrate on in {data!r}.")
    destination = Path(output_path)
    with tempfile.TemporaryDirectory(prefix="facedet_axelera_") as work:
        raw = Path(work) / "facedet_raw.onnx"
        onnx_utils.extract_model(source, str(raw), ["input"], list(_YUNET_RAW_TENSORS.values()))
        graph = onnx.load(str(raw))
        rename = {v: k for k, v in _YUNET_RAW_TENSORS.items()}
        for node in graph.graph.node:
            node.output[:] = [rename.get(o, o) for o in node.output]
        for output in graph.graph.output:
            output.name = rename.get(output.name, output.name)
        graph.graph.input[0].type.tensor_type.shape.dim[2].dim_value = canvas
        graph.graph.input[0].type.tensor_type.shape.dim[3].dim_value = canvas
        onnx.save(graph, str(raw))
        session = onnxruntime.InferenceSession(str(raw), providers=["CPUExecutionProvider"])
        names = [o.name for o in session.get_outputs()]

        class _Reference(torch.nn.Module):
            output_names = names

            def forward(self, x):
                outs = session.run(None, {"input": x.detach().cpu().numpy().astype(np.float32)})
                return tuple(torch.from_numpy(o) for o in outs)

        held = calibration[-8:] if verify else None
        return export_axelera(
            onnx_path=str(raw),
            output_path=str(destination),
            stem=destination.name.removesuffix("_axelera_model") or "model",
            metadata={
                "model_family": "facerec",
                "size": "det",
                "task": "face_detect",
                "imgsz": [canvas, canvas],
                "input_names": ["input"],
                "output_layout": YUNET_RAW_LAYOUT,
                "strides": list(YUNET_STRIDES),
                "contract": f"letterboxed BGR 0..255 [1,3,{canvas},{canvas}] -> 12 raw head maps (cls/obj/bbox/kps x "
                "strides 8/16/32); sigmoids, decode and NMS in Python",
            },
            calibration=calibration,
            calibration_info={"data": str(data), "num_images": len(calibration), "fraction": float(fraction),
                              "batch": 1, "preprocess": f"face detector letterbox {canvas} (BGR 0..255)"},
            quantization_scheme=quantization_scheme,
            verify=verify,
            reference_model=_Reference(),
            verify_samples=held,
            verify_source="dataset images",
        )


__all__ = ["AxeleraFaceDetector", "decode_yunet_raw", "export_face_detector_axelera", "letterbox_bgr", "YUNET_RAW_LAYOUT"]
