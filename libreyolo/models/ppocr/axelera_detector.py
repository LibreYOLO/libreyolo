"""PP-OCR text detector on the Axelera Metis: the family's DB detection network compiled at a fixed canvas.

The compiled graph is the whole detection network (backbone, neck and DB head) and stops at the shrink map's
logits (layout ``ppocr_db_logits_v1``, one ``[1, 1, H, W]`` map, windowed to ``OUTPUT_WINDOW`` as a ReLU6 output).
LibreYOLO undoes the window, applies the sigmoid and runs the DB box extraction (``db_postprocess``) on the host,
exactly as the FP32 pipeline does. Before export the network gets exact rewrites the Metis needs
(``metis_forms``): PP-LCNetV3's multi-branch rep layers folded into one convolution each (the unfused stride-2 1x1
depthwise branch fails lowering), its scalar affines as per-channel constants and the neck's SE gates as HardSigmoid.

Recognition is not compiled: the recognizer's SVTR neck (LayerNorm and attention) has no form on the Metis, so
``LibrePPOCR(..., text_detector=<dir>)`` runs detection on the Metis and recognition with PyTorch on the host. Only
the mobile tier (``t``) is wired; the server tier's PFHeadLocal head is not.

Preprocessing: the pipeline's own ``det_resize`` (long side at most ``max(canvas)``, both sides rounded to /32) and
ImageNet normalization, written into the top-left of the canvas with zero padding. An image whose resized size fits
the canvas sees the FP32 pipeline's input plus padding; a larger one is shrunk (aspect kept, /32) until it fits.
The padding is not free: the SE blocks' global pooling averages it in (on ICDAR2015 the 960x960 canvas costs FP32
hmean against the pipeline's own 544x960 input, see docs/axelera.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

PPOCR_DB_LAYOUT = "ppocr_db_logits_v1"
DEFAULT_CANVAS = (960, 960)
# The shrink-map logits reach about +-60 on text images; INT8 over that range is ~0.47 logit per step around the DB
# thresholds (0.3 / 0.6). The export folds the window [-8, 8] into the last ConvTranspose as a ReLU6 output
# (0..6) and the host maps it back; the clipped tail changes probabilities by less than 3.4e-4.
OUTPUT_WINDOW = (-8.0, 8.0)
# PP-LCNetV3 is depthwise-heavy: per-channel depthwise inputs (hybrid) recover most of the min-max INT8 loss
# (ICDAR2015 40-image CPU screens: min-max -15.0 pt det hmean, hybrid -6.1 pt).
DEFAULT_QUANTIZATION_SCHEME = "hybrid_per_tensor_per_channel"


def canvas_input(image_bgr: np.ndarray, canvas: Tuple[int, int]) -> Tuple[np.ndarray, int, int]:
    """``(1, 3, H, W)`` float32 network input, plus the height and width of the image region inside it."""
    import cv2

    from .preprocess import det_normalize, det_resize

    height, width = int(canvas[0]), int(canvas[1])
    resized = det_resize(image_bgr, limit_side_len=max(height, width))[0]
    h, w = resized.shape[:2]
    if h > height or w > width:
        scale = min(height / h, width / w)
        resized = cv2.resize(resized, (max(32, int(w * scale) // 32 * 32), max(32, int(h * scale) // 32 * 32)))
        h, w = resized.shape[:2]
    blob = np.zeros((1, 3, height, width), np.float32)
    blob[:, :, :h, :w] = det_normalize(resized).numpy()
    return blob, h, w


def probability_map(output: np.ndarray, h: int, w: int, window: Optional[Sequence[float]] = None) -> np.ndarray:
    """Network output on the canvas -> ``(h, w)`` probability map of the image region.

    ``window`` (``(low, high)``): the output is the windowed ReLU6 form; map it back to logits first.
    """
    output = np.asarray(output, np.float32)
    logits = output.reshape(output.shape[-2:])[:h, :w]
    if window is not None:
        low, high = float(window[0]), float(window[1])
        logits = logits * ((high - low) / 6.0) + low
    return 1.0 / (1.0 + np.exp(-logits))


class AxeleraTextDetector:
    """PP-OCR detection on an Axelera Metis (``LibrePPOCR.export(format="axelera")``).

    ``prob_map(image_bgr)`` returns the probability map of the image region (the FP32 detector's ``prob[0, 0]``);
    ``prob_maps`` submits several images to the runtime as one batch.
    """

    default_predict_batch = 8

    def __init__(self, path: str | Path):
        import yaml

        from ...backends.axelera import AxeleraBackend, require_axelera_rt, resolve_axelera_artifact

        axm, metadata = resolve_axelera_artifact(path)
        meta = yaml.safe_load(Path(metadata).read_text()) or {}
        if meta.get("output_layout") != PPOCR_DB_LAYOUT:
            raise ValueError(f"{path} is not a PP-OCR text-detector Axelera export (layout {PPOCR_DB_LAYOUT}).")
        require_axelera_rt()
        self.path = str(path)
        self.metadata = meta
        self.size = str(meta.get("size"))
        self.canvas = (int(meta["imgsz"][0]), int(meta["imgsz"][1]))
        self.window = meta.get("output_window")
        self._model = AxeleraBackend._load(axm)

    @staticmethod
    def _first(outputs) -> np.ndarray:
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        return np.asarray(outputs, np.float32)

    def prob_map(self, image_bgr: np.ndarray) -> np.ndarray:
        from ...backends.axelera import _no_device_error

        blob, h, w = canvas_input(image_bgr, self.canvas)
        try:
            outputs = self._model(blob)
        except RuntimeError as exc:
            raise _no_device_error(exc) from exc
        return probability_map(self._first(outputs), h, w, self.window)

    def prob_maps(self, images_bgr: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Probability maps of several images, one runtime batch (results equal one-by-one calls)."""
        from ...backends.axelera import _no_device_error

        if len(images_bgr) <= 1:
            return [self.prob_map(image) for image in images_bgr]
        items = [canvas_input(image, self.canvas) for image in images_bgr]
        try:
            outputs = list(self._model.batch([blob for blob, _, _ in items]))
        except RuntimeError as exc:
            raise _no_device_error(exc) from exc
        return [probability_map(self._first(o), h, w, self.window) for o, (_, h, w) in zip(outputs, items)]


def _fold_bn(conv_weight, bn):
    std = (bn.running_var + bn.eps).sqrt()
    scale = bn.weight / std
    return conv_weight * scale.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * scale


class _FusedRepLayer:
    """Factory: a ``LearnableRepLayer`` folded into one convolution (exact up to float rounding).

    The identity BN, the 1x1 branch (zero-padded to the kxk kernel's centre) and the kxk branches are summed into one
    kxk kernel and bias; the layer's scalar affine (``lab``) folds in as well. The Metis rejects the unfused form: the
    stride-2 1x1 depthwise branch at 480x480 exceeds a depthwise-unit command limit (``DwpuStandardCommand``).
    """

    @staticmethod
    def build(layer):
        import torch

        first = layer.conv_kxk[0].conv
        k = first.kernel_size[0]
        weight = torch.zeros_like(first.weight)
        bias = torch.zeros(first.out_channels)
        for branch in layer.conv_kxk:
            w, b = _fold_bn(branch.conv.weight, branch.bn)
            weight, bias = weight + w, bias + b
        if layer.conv_1x1 is not None:
            w, b = _fold_bn(layer.conv_1x1.conv.weight, layer.conv_1x1.bn)
            weight[:, :, k // 2, k // 2] += w[:, :, 0, 0]
            bias = bias + b
        if layer.identity is not None:
            in_per_group = first.in_channels // first.groups
            eye = torch.zeros_like(first.weight)
            for o in range(first.out_channels):
                eye[o, o % in_per_group, k // 2, k // 2] = 1.0
            w, b = _fold_bn(eye, layer.identity)
            weight, bias = weight + w, bias + b
        scale, shift = layer.lab.scale.reshape(()), layer.lab.bias.reshape(())
        conv = torch.nn.Conv2d(first.in_channels, first.out_channels, k, stride=first.stride, padding=first.padding,
                               groups=first.groups, bias=True)
        with torch.no_grad():
            conv.weight.copy_(weight * scale)
            conv.bias.copy_(bias * scale + shift)
        if layer.stride == 2:
            return torch.nn.Sequential(conv)
        return torch.nn.Sequential(conv, torch.nn.Sequential(layer.act.act, _ChannelAffine.build(layer.act.lab, first.out_channels)))


class _ChannelAffine:
    """Factory: ``x * scale + bias`` with ``(1, C, 1, 1)`` constants (the Metis takes per-channel or ``()`` scalars, not
    the ``(1,)`` parameters of ``LearnableAffineBlock``)."""

    @staticmethod
    def build(lab, channels: int):
        import torch

        class Affine(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", lab.scale.detach().reshape(1, 1, 1, 1).expand(1, channels, 1, 1).clone())
                self.register_buffer("bias", lab.bias.detach().reshape(1, 1, 1, 1).expand(1, channels, 1, 1).clone())

            def forward(self, x):
                return x * self.scale + self.bias

        return Affine()


def _se_hardsigmoid(se):
    """``SEModule``'s gate ``clamp(0.2 * y + 0.5, 0, 1)`` as ``hardsigmoid(1.2 * y)`` (exact): the conv absorbs 1.2 and
    the gate becomes a HardSigmoid, which the Metis compiles (Clip(0, 1) is not a supported Clip form)."""
    import torch

    conv = copy_module(se.conv2)
    with torch.no_grad():
        conv.weight.mul_(1.2)
        conv.bias.mul_(1.2)

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.avg_pool, self.conv1, self.conv2 = se.avg_pool, se.conv1, conv
            self.gate = torch.nn.Hardsigmoid()

        def forward(self, x):
            return x * self.gate(self.conv2(torch.relu(self.conv1(self.avg_pool(x)))))

    return Gate()


def copy_module(module):
    import copy

    return copy.deepcopy(module)


def metis_forms(module):
    """Exact Metis-friendly rewrites of the PP-OCR detection network (in place): rep layers folded, ``Act`` affines as
    per-channel constants, ``SEModule`` gates as HardSigmoid."""
    import torch

    from .backbones import Act
    from .det import SEModule

    fuse_rep_layers(module)
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, SEModule):
                setattr(parent, name, _se_hardsigmoid(child))
    # Act follows a conv whose channel count sizes the affine; find it from the preceding sibling.
    for parent in list(module.modules()):
        prev_channels = None
        for name, child in list(parent.named_children()):
            if isinstance(child, Act):
                if prev_channels is None:
                    raise RuntimeError(f"Cannot size the affine of {name}: no preceding convolution.")
                setattr(parent, name, torch.nn.Sequential(child.act, _ChannelAffine.build(child.lab, prev_channels)))
            convs = [m for m in child.modules() if isinstance(m, torch.nn.Conv2d)]
            if convs:
                prev_channels = convs[-1].out_channels
    return module


def fuse_rep_layers(module):
    """Replace every ``LearnableRepLayer`` under ``module`` (in place) by its folded convolution."""
    from .backbones import LearnableRepLayer

    for name, child in list(module.named_children()):
        if isinstance(child, LearnableRepLayer):
            setattr(module, name, _FusedRepLayer.build(child))
        else:
            fuse_rep_layers(child)
    return module


class _DetLogits:
    """Factory for the export graph: the detection network up to the shrink map's logits."""

    @staticmethod
    def build(det, window: Optional[Sequence[float]] = None):
        """Logits graph; with ``window=(low, high)`` the last ConvTranspose emits ``relu6((logit - low) * 6 / (high - low))``."""
        import copy

        import torch

        head = det.head
        if type(head).__name__ != "DBHead":
            raise NotImplementedError(f"Axelera export of the {type(head).__name__} detection head is not wired.")
        last = head.binarize.conv3
        if window is not None:
            low, high = float(window[0]), float(window[1])
            last = copy.deepcopy(last)
            with torch.no_grad():
                last.weight.mul_(6.0 / (high - low))
                last.bias.copy_((last.bias - low) * (6.0 / (high - low)))

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.det = det
                self.last = last

            def forward(self, x):
                branch = self.det.head.binarize
                f = self.det.neck(self.det.backbone(x))
                f = torch.relu(branch.conv_bn1(branch.conv1(f)))
                f = torch.relu(branch.conv_bn2(branch.conv2(f)))
                y = self.last(f)
                return torch.nn.functional.relu6(y) if window is not None else y

        return Net().eval()


def _calibration_images(data: str | Path) -> List[Path]:
    """Images of an OCR dataset (a root, or a YAML per docs/dataset_schema.md) or of a detection dataset YAML.

    A YAML is read as OCR when it names ``images`` or ``labels``, or has neither
    ``train`` nor ``val`` (the minimal OCR YAML is ``path``, ``nc``, ``names``).
    Calibration needs images only: the OCR split's JSONL labels are not required.
    """
    import yaml

    from ...data.ocr_dataset import resolve_ocr_root
    from ...data.utils import get_img_files, load_data_config

    data_path = Path(str(data))
    splits: list = ["train", "val"]
    is_ocr = data_path.is_dir()
    if data_path.suffix.lower() in (".yaml", ".yml") and data_path.is_file():
        cfg = yaml.safe_load(data_path.read_text()) or {}
        is_ocr = "images" in cfg or "labels" in cfg or ("train" not in cfg and "val" not in cfg)
        splits = [cfg.get("train"), "train", cfg.get("val"), "val"]
    if is_ocr:
        root, images_dirname, _ = resolve_ocr_root(data_path)
        for split in splits:
            folder = root / images_dirname / str(split) if split else None
            if folder is not None and folder.is_dir():
                return sorted(p for p in folder.iterdir()
                              if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        raise ValueError(f"No {images_dirname}/<train|val> directory under {root}.")
    config = load_data_config(str(data), autodownload=True)
    root = Path(config.get("path", "."))
    split = config.get("train") or config.get("val")
    return [Path(p) for p in get_img_files(split, prefix=str(root))]


def export_ppocr_det_axelera(
    model,
    output_path: str | Path | None = None,
    *,
    data: str | Path = "coco128.yaml",
    fraction: float = 1.0,
    canvas: Tuple[int, int] | int = DEFAULT_CANVAS,
    quantization_scheme: str = DEFAULT_QUANTIZATION_SCHEME,
    verify: bool = False,
) -> str:
    """Compile a ``LibrePPOCR`` model's detection network for an Axelera Metis at a fixed ``canvas``."""
    import tempfile

    import cv2
    import torch

    from ...export.axelera import export_axelera

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    canvas = (int(canvas), int(canvas)) if isinstance(canvas, int) else (int(canvas[0]), int(canvas[1]))
    if canvas[0] % 32 or canvas[1] % 32:
        raise ValueError(f"canvas sides must be multiples of 32, got {canvas}.")
    size = str(model.size)
    images = _calibration_images(data)
    images = images[: max(1, int(len(images) * float(fraction)))]
    calibration = []
    for image_path in images:
        bgr = cv2.imread(str(image_path))
        if bgr is not None:
            calibration.append(canvas_input(bgr, canvas)[0])
    if not calibration:
        raise ValueError(f"No images to calibrate on in {data!r}.")
    destination = Path(output_path or f"LibrePPOCR{size}-det_axelera_model")
    import copy

    net = _DetLogits.build(metis_forms(copy.deepcopy(model.model.det).eval().float().cpu()), OUTPUT_WINDOW)
    with tempfile.TemporaryDirectory(prefix="ppocr_det_axelera_") as work:
        onnx_path = Path(work) / f"ppocr{size}_det.onnx"
        with torch.no_grad():
            torch.onnx.export(net, torch.zeros(1, 3, *canvas), str(onnx_path), opset_version=17,
                              input_names=["images"], output_names=["shrink_logits"], do_constant_folding=True,
                              dynamo=False)

        class _Reference(torch.nn.Module):
            output_names = ["shrink_logits"]

            def forward(self, x):
                with torch.no_grad():
                    return (net(x.float()),)

        return export_axelera(
            onnx_path=str(onnx_path),
            output_path=str(destination),
            stem=destination.name.removesuffix("_axelera_model") or "model",
            metadata={
                "model_family": "ppocr",
                "size": size,
                "task": "text_detect",
                "imgsz": list(canvas),
                "input_names": ["images"],
                "output_names": ["shrink_logits"],
                "output_layout": PPOCR_DB_LAYOUT,
                "output_window": list(OUTPUT_WINDOW),
                "contract": f"det_resize + ImageNet-normalized BGR, top-left in a zero canvas [1,3,{canvas[0]},{canvas[1]}]"
                " -> DB shrink-map logits windowed to output_window as ReLU6 [1,1,H,W]; un-window, sigmoid and DB box extraction in Python",
            },
            calibration=calibration,
            calibration_info={"data": str(data), "num_images": len(calibration), "fraction": float(fraction),
                              "batch": 1, "preprocess": f"PP-OCR det_resize into a {canvas[0]}x{canvas[1]} canvas"},
            quantization_scheme=quantization_scheme,
            verify=verify,
            reference_model=_Reference(),
            verify_samples=calibration[-8:] if verify else None,
            verify_source="dataset images",
        )


__all__ = ["AxeleraTextDetector", "fuse_rep_layers", "metis_forms", "PPOCR_DB_LAYOUT", "canvas_input", "export_ppocr_det_axelera", "probability_map"]
