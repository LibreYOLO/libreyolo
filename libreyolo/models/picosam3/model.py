"""LibrePicoSAM3: native PicoSAM3 on the LibreSAM promptable tier."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar, Dict, Optional

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import (
    load_untrusted_torch_file,
    unwrap_libreyolo_checkpoint,
)
from ..sam.base import _INSTALL_HINT, _SNAPSHOT_COMPLETE_MARKER, LibreSAMModel
from ..sam.prompts import normalize_boxes
from .nn import PicoSAM3Network
from .preprocess import padded_square_roi, place_roi_logits, preprocess_roi

logger = logging.getLogger(__name__)


class LibrePicoSAM3(LibreSAMModel):
    """PicoSAM3 edge segmenter conditioned by one or more box-defined ROIs.

    ``bboxes=`` is the only supported prompt. The box is expanded by 10%, made
    square, clipped to the image, and resized to 96x96 before the CNN runs.
    Point, text, mask, multimask, and segment-everything prompts are not part of
    the upstream model contract and fail clearly; use LibreSAM2 or LibreSAM3 for
    those modes.

    Unlike encoder/decoder SAM models, PicoSAM3 performs one complete CNN
    forward per ROI. ``set_image()`` therefore caches the source image rather
    than an embedding, preserving the common interactive API without implying
    nonexistent encoder reuse.
    """

    FAMILY = "picosam3"
    FILENAME_PREFIX = "LibrePicoSAM3"
    HF_REPOS: ClassVar[Dict[str, str]] = {"pico": "LibreYOLO/LibrePicoSAM3"}
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"pico": 96}
    WEIGHT_FILE: ClassVar[str] = "LibrePicoSAM3pico.pt"

    def __init__(self, size: str = "pico", **kwargs) -> None:
        super().__init__(size=size, **kwargs)

    @classmethod
    def _snapshot_complete(cls, local_dir: Path) -> bool:
        return (local_dir / _SNAPSHOT_COMPLETE_MARKER).exists() and (
            local_dir / cls.WEIGHT_FILE
        ).exists()

    def _ensure_weights(self) -> str:
        repo = self.HF_REPOS[self.size]
        local_dir = Path("weights") / f"{self.FILENAME_PREFIX}{self.size}"
        if self._snapshot_complete(local_dir):
            return str(local_dir)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        local_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading %s weights from %s -> %s ...",
            self.FAMILY,
            repo,
            local_dir,
        )
        hf_hub_download(
            repo_id=repo,
            filename=self.WEIGHT_FILE,
            local_dir=local_dir,
        )
        (local_dir / _SNAPSHOT_COMPLETE_MARKER).write_text(
            json.dumps({"repo": repo}) + "\n",
            encoding="utf-8",
        )
        if not self._snapshot_complete(local_dir):
            (local_dir / _SNAPSHOT_COMPLETE_MARKER).unlink(missing_ok=True)
            raise FileNotFoundError(
                f"Downloaded snapshot for {repo} is missing {self.WEIGHT_FILE} "
                f"in {local_dir}."
            )
        return str(local_dir)

    def _init_model(self) -> nn.Module:
        checkpoint_path = Path(self._ensure_weights()) / self.WEIGHT_FILE
        loaded = load_untrusted_torch_file(
            checkpoint_path,
            map_location="cpu",
            context="PicoSAM3 Apache-2.0 checkpoint",
        )
        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded, strict=False)
        if metadata:
            expected = {
                "model_family": self.FAMILY,
                "size": self.size,
                "task": self.DEFAULT_TASK,
            }
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatches:
                details = ", ".join(
                    f"{key}={got!r} (expected {want!r})"
                    for key, (got, want) in mismatches.items()
                )
                raise ValueError(
                    f"Invalid LibrePicoSAM3 checkpoint metadata: {details}"
                )

        model = PicoSAM3Network()
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            if any(key.startswith("output_head.") for key in state_dict):
                raise ValueError(
                    "This checkpoint contains the older PicoSAM2 architecture, not "
                    "PicoSAM3. Use PicoSAM3_SAM3_student_best.pt."
                ) from exc
            raise
        self.processor = None
        self._model_dtype = next(model.parameters()).dtype
        return model

    def set_image(
        self, source: ImageInput, color_format: str = "auto"
    ) -> "LibrePicoSAM3":
        """Cache an image for repeated ROI prompts."""

        self._image = ImageLoader.load(source, color_format=color_format)
        self._image_path = source if isinstance(source, (str, Path)) else None
        self._image_embeddings = None
        return self

    def predict(
        self,
        source: Optional[ImageInput] = None,
        *,
        points=None,
        bboxes=None,
        labels=None,
        masks=None,
        text: str | None = None,
        conf: Optional[float] = None,
        multimask: Optional[bool] = None,
        max_det: int = 300,
        device: Optional[str] = None,
        color_format: str = "auto",
        points_per_side: Optional[int] = None,
    ):
        """Segment the ROI inside each xyxy box prompt."""

        unsupported = []
        if points is not None:
            unsupported.append("points")
        if labels is not None:
            unsupported.append("labels")
        if masks is not None:
            unsupported.append("masks")
        if text is not None:
            unsupported.append("text")
        if points_per_side is not None:
            unsupported.append("points_per_side")
        if unsupported:
            raise ValueError(
                "PicoSAM3 supports only bboxes= ROI prompts; unsupported: "
                f"{', '.join(unsupported)}. Use LibreSAM2 or LibreSAM3 for "
                "point, text, mask, or segment-everything prompts."
            )
        if multimask not in (None, False):
            raise ValueError(
                "PicoSAM3 produces one mask per ROI and does not support "
                "multimask=True; use LibreSAM2 or LibreSAM3."
            )
        if bboxes is None or (hasattr(bboxes, "__len__") and len(bboxes) == 0):
            raise ValueError(
                "PicoSAM3 requires bboxes=; segment-everything mode is not "
                "supported. Use LibreSAM2 or LibreSAM3 for that mode."
            )
        if max_det < 1:
            raise ValueError("max_det must be >= 1.")
        conf_thres = 0.0 if conf is None else float(conf)
        if not 0.0 <= conf_thres <= 1.0:
            raise ValueError("conf must be between 0.0 and 1.0.")
        if device is not None:
            self._set_device(device)

        img, image_path, _ = self._resolve_image(source, color_format)
        boxes = normalize_boxes(bboxes)
        width, height = img.size
        rois = [padded_square_roi(box, width, height) for box in boxes]
        batch = torch.stack([preprocess_roi(img, roi) for roi in rois]).to(
            self.device, dtype=self._model_dtype
        )
        with torch.no_grad():
            logits = self.model(batch)
        full_masks, scores = place_roi_logits(logits, rois, height, width)
        return self._assemble_results(
            full_masks,
            scores,
            img,
            image_path,
            conf_thres=conf_thres,
            max_det=max_det,
        )

    #: Axelera calibration: ROIs cut from the calibration dataset's ground-truth
    #: boxes exactly as ``predict(bboxes=...)`` cuts them, in image order, at most
    #: this many per image and this many in total.
    AXELERA_CALIBRATION_ROIS_PER_IMAGE: ClassVar[int] = 2
    AXELERA_CALIBRATION_ROIS: ClassVar[int] = 128
    #: Histogram activation ranges with per-channel depthwise inputs: on 2360
    #: COCO val2017 ground-truth-box ROIs the device mask IoU vs FP32 is 0.968
    #: (0.940 with min-max ranges).
    AXELERA_QUANTIZATION_SCHEME: ClassVar[str] = "hybrid_per_tensor_per_channel"

    def export(
        self,
        format: str = "onnx",
        *,
        output: str | Path | None = None,
        output_path: str | Path | None = None,
        opset: int = 13,
        dynamic: bool | None = None,
        data: str | Path | None = None,
        fraction: float = 1.0,
        quantization_scheme: str | None = None,
        verify: bool = False,
        **kwargs,
    ) -> str:
        """Export the raw 96x96 ROI CNN as ``roi_image -> mask_logits``.

        ``format="onnx"`` writes the graph (batch axis dynamic by default).
        ``format="axelera"`` compiles the same graph, at batch 1, for an Axelera
        Metis with INT8 calibration on ROIs cut from ``data``'s ground-truth
        boxes (default ``coco128.yaml``). Both are the raw contract: cutting the
        ROI from a box and pasting the logits back stay in Python
        (``predict(bboxes=...)``).
        """

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported PicoSAM3 export arguments: {unknown}")
        fmt = str(format).lower()
        if fmt not in ("onnx", "axelera"):
            raise NotImplementedError(
                "LibrePicoSAM3 export supports ONNX and Axelera only."
            )
        if output is not None and output_path is not None:
            raise ValueError("Pass only one of output= or output_path=.")
        if fmt == "onnx":
            onnx_only = {"data": data, "quantization_scheme": quantization_scheme}
            given = sorted(k for k, v in onnx_only.items() if v is not None)
            if given or verify or fraction != 1.0:
                raise TypeError(
                    "Axelera-only PicoSAM3 export arguments for ONNX: "
                    + ", ".join(given + (["verify"] if verify else []) + (["fraction"] if fraction != 1.0 else []))
                )
            destination = Path(output_path or output or "LibrePicoSAM3pico.onnx")
            return self._export_raw_onnx(
                destination, opset=opset, dynamic=True if dynamic is None else dynamic
            )
        if dynamic:
            raise ValueError("Axelera exports are compiled at batch 1 (dynamic=False).")
        return self._export_axelera(
            Path(output_path or output or "LibrePicoSAM3pico_axelera_model"),
            opset=opset,
            data=data or "coco128.yaml",
            fraction=fraction,
            quantization_scheme=quantization_scheme or self.AXELERA_QUANTIZATION_SCHEME,
            verify=verify,
        )

    def _export_raw_onnx(
        self, destination: Path, *, opset: int, dynamic: bool, network: nn.Module | None = None
    ) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        axes = None
        if dynamic:
            axes = {"roi_image": {0: "batch"}, "mask_logits": {0: "batch"}}
        imgsz = self.INPUT_SIZES[self.size]
        network = (network if network is not None else self.model).eval()
        torch.onnx.export(
            network,
            torch.zeros((1, 3, imgsz, imgsz), device=self.device),
            str(destination),
            input_names=["roi_image"],
            output_names=["mask_logits"],
            dynamic_axes=axes,
            opset_version=int(opset),
            dynamo=False,
        )
        return str(destination)

    def _axelera_rois(self, data, fraction: float, split: str) -> list:
        """Preprocessed ``[1, 3, 96, 96]`` ROIs from a YOLO-format dataset's boxes."""

        import numpy as np
        from PIL import Image

        from ...data.utils import get_img_files, img2label_paths, load_data_config

        config = load_data_config(str(data), autodownload=True)
        root = Path(config.get("path", "."))
        source = config.get(split) or config.get("train") or config.get("val")
        if source is None:
            raise ValueError(f"Dataset config {data!r} has no {split!r} or 'val' split.")
        images = get_img_files(source, prefix=str(root))
        images = images[: max(1, int(len(images) * fraction))]
        rois = []
        for image_path, label_path in zip(images, img2label_paths(images)):
            label_path = Path(label_path)
            if not label_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            rows = [r.split() for r in label_path.read_text().splitlines() if r.strip()]
            for row in rows[: self.AXELERA_CALIBRATION_ROIS_PER_IMAGE]:
                cx, cy, w, h = (float(v) for v in row[1:5])
                box = [
                    (cx - w / 2) * width,
                    (cy - h / 2) * height,
                    (cx + w / 2) * width,
                    (cy + h / 2) * height,
                ]
                if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                    continue
                roi = padded_square_roi(box, width, height)
                rois.append(preprocess_roi(image, roi).numpy()[None].astype(np.float32))
        return rois

    def _export_axelera(
        self,
        destination: Path,
        *,
        opset: int,
        data,
        fraction: float,
        quantization_scheme: str,
        verify: bool,
    ) -> str:
        import tempfile

        from ...export.axelera import export_axelera

        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("fraction must be in (0, 1].")
        calibration = self._axelera_rois(data, fraction, "train")[
            : self.AXELERA_CALIBRATION_ROIS
        ]
        if not calibration:
            raise ValueError(f"No ground-truth boxes to cut calibration ROIs from in {data!r}.")
        samples = None
        if verify:
            samples = self._axelera_rois(data, 1.0, "val")[-32:]
        network = _axelera_network(self.model)

        class _Reference(nn.Module):
            output_names = ["mask_logits"]

            def __init__(self) -> None:
                super().__init__()
                self.network = network

            def forward(self, x):
                return (self.network(x),)

        imgsz = self.INPUT_SIZES[self.size]
        with tempfile.TemporaryDirectory(prefix="picosam3_axelera_") as work:
            onnx_path = self._export_raw_onnx(
                Path(work) / "picosam3.onnx", opset=opset, dynamic=False, network=network
            )
            return export_axelera(
                onnx_path=onnx_path,
                output_path=str(destination),
                stem=destination.name.removesuffix("_axelera_model") or "model",
                metadata={
                    "model_family": self.FAMILY,
                    "size": self.size,
                    "task": "segment",
                    "imgsz": [imgsz, imgsz],
                    "contract": "raw ROI CNN: roi_image [1,3,96,96] RGB, mean/std-normalized as preprocess_roi -> "
                    "mask_logits [1,1,96,96]; ROI cut and paste-back in Python",
                    "input_names": ["roi_image"],
                },
                calibration=calibration,
                calibration_info={
                    "data": str(data),
                    "num_images": len(calibration),
                    "fraction": float(fraction),
                    "batch": 1,
                    "preprocess": "picosam3 ROI (ground-truth boxes)",
                },
                quantization_scheme=quantization_scheme,
                verify=verify,
                reference_model=_Reference() if verify else None,
                verify_samples=samples,
                verify_source="dataset ROIs",
            )


def _axelera_network(model: nn.Module) -> nn.Module:
    """A private copy of the ROI CNN in its Axelera form (exact; the caller's model is untouched).

    The stem's depthwise 3x3 runs on the 3-channel image, and the Metis compiler's
    input-channel rearrangement (SWICR) has no depthwise form ("Depthwise convolutions are
    not supported in SWICR"): it becomes a dense 3->3 conv whose kernel is zero off the
    channel diagonal.
    """

    import copy

    network = copy.deepcopy(model).eval()
    depthwise = network.encoder_stage1[0]
    dense = nn.Conv2d(
        depthwise.in_channels,
        depthwise.out_channels,
        kernel_size=depthwise.kernel_size,
        padding=depthwise.padding,
        bias=False,
    ).to(depthwise.weight.device, depthwise.weight.dtype)
    with torch.no_grad():
        dense.weight.zero_()
        for channel in range(depthwise.out_channels):
            dense.weight[channel, channel] = depthwise.weight[channel, 0]
    network.encoder_stage1[0] = dense
    return network


__all__ = ["LibrePicoSAM3", "PicoSAM3Network"]
