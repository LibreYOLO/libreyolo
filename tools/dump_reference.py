"""Dump reference tensors ("oracle") for a model and an image.

Writes, into ``--out``:

- ``input.f32`` + ``input.json``: the preprocessed network input (NCHW
  float32) exactly as LibreYOLO feeds it, plus the letterbox geometry.
- ``<tap>.f32`` + ``<tap>.json`` for every intermediate tap (module path; a
  module returning a tuple gives ``<path>.out<i>``).
- ``output.f32`` + ``output.json``: the raw graph output (YOLO9: the decoded
  ``(1, 4 + nc, N)`` tensor the exported graphs return).
- ``detections.json``: post-NMS detections in original-image pixels.
- ``manifest.json``: taps in execution order with shapes.

Every ``.f32`` is raw little-endian float32 in C order. Its sidecar is
``{"shape", "dtype": "f32", "layout", "min", "max", "mean", "rms",
"p99_abs", "source"}``.

Usage::

    python tools/dump_reference.py --model LibreYOLO9t.pt --image parkour.jpg \\
        --out dumps/LibreYOLO9t/parkour
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

#: YOLO9 taps: every top-level block, every ELAN / downsampling block, and
#: the per-level head towers (pre-sigmoid logits).
YOLO9_TAPS = (
    "backbone.conv0",
    "backbone.conv1",
    "backbone.elan1",
    "backbone.down2",
    "backbone.elan2",
    "backbone.down3",
    "backbone.elan3",
    "backbone.down4",
    "backbone.elan4",
    "backbone.spp",
    "backbone",
    "neck.up1",
    "neck.elan_up1",
    "neck.up2",
    "neck.elan_up2",
    "neck.down1",
    "neck.elan_down1",
    "neck.down2",
    "neck.elan_down2",
    "neck",
    "head.anchor_convs.0",
    "head.class_convs.0",
    "head.anchor_convs.1",
    "head.class_convs.1",
    "head.anchor_convs.2",
    "head.class_convs.2",
    "head",
)

DEFAULT_TAPS: Dict[str, Sequence[str]] = {"yolo9": YOLO9_TAPS}


def tensor_stats(array: np.ndarray) -> Dict[str, float]:
    a = array.astype(np.float64, copy=False).ravel()
    if a.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "rms": 0.0, "p99_abs": 0.0}
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "rms": float(np.sqrt(np.mean(a * a))),
        "p99_abs": float(np.percentile(np.abs(a), 99)),
    }


def write_tensor(out_dir: Path, name: str, tensor, source: str, extra=None) -> Dict:
    array = np.ascontiguousarray(
        tensor.detach().cpu().float().numpy() if torch.is_tensor(tensor) else tensor,
        dtype=np.float32,
    )
    (out_dir / f"{name}.f32").write_bytes(array.astype("<f4").tobytes())
    layout = {4: "NCHW", 3: "NCA"}.get(array.ndim, "C" * array.ndim)
    sidecar = {
        "shape": list(array.shape),
        "dtype": "f32",
        "layout": layout,
        **tensor_stats(array),
        "source": source,
    }
    if extra:
        sidecar.update(extra)
    (out_dir / f"{name}.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def _flatten(name: str, value) -> List[tuple]:
    if torch.is_tensor(value):
        return [(name, value)]
    if isinstance(value, (list, tuple)):
        items = []
        for i, item in enumerate(value):
            items.extend(_flatten(f"{name}.out{i}", item))
        return items
    if isinstance(value, dict) and "predictions" in value:
        return [(name, value["predictions"])]
    return []


def dump_reference(
    model,
    image: str,
    out_dir: str | Path,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    taps: Sequence[str] | None = None,
) -> Dict:
    """Run ``model`` (a LibreYOLO wrapper) on ``image`` and dump the oracle."""
    from libreyolo.export.gguf import _pad_name
    from libreyolo.export.onnx import _get_version
    from libreyolo.preprocess.letterbox import letterbox_geometry

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    family = model._get_model_name()
    if taps is None:
        taps = DEFAULT_TAPS.get(family) or [n for n, _ in model.model.named_children()]
    source = (
        f"libreyolo {_get_version()}, torch {torch.__version__}, cpu fp32, "
        f"{Path(str(getattr(model, 'model_path', '') or '')).name or family}"
    )

    nn_model = model.model
    nn_model.eval()
    input_tensor, _img, original_size, _ratio = model._preprocess(image, input_size=imgsz)
    input_tensor = input_tensor.to(next(nn_model.parameters()).device)
    orig_w, orig_h = original_size
    pad_mode = getattr(model, "letterbox_pad", None)
    ratio, new_h, new_w, pad_left, pad_top = letterbox_geometry(
        orig_h, orig_w, imgsz, imgsz, pad_mode
    )
    write_tensor(
        out_dir,
        "input",
        input_tensor,
        f"{source}, preprocess",
        extra={
            "image": Path(image).name,
            "original_wh": [int(orig_w), int(orig_h)],
            "letterbox": {
                "pad": _pad_name(pad_mode),
                "ratio": ratio,
                "resized_wh": [new_w, new_h],
                "pad_left": pad_left,
                "pad_top": pad_top,
                "pad_value": 114,
            },
        },
    )

    modules = dict(nn_model.named_modules())
    missing = [t for t in taps if t not in modules]
    if missing:
        raise KeyError(f"Unknown tap module paths: {missing}")
    captured: List[tuple] = []
    handles = [
        modules[t].register_forward_hook(
            (lambda path: lambda _m, _i, out: captured.extend(_flatten(path, out)))(t)
        )
        for t in taps
    ]
    try:
        with torch.no_grad():
            output = model._forward(input_tensor)
    finally:
        for h in handles:
            h.remove()

    manifest = {"model": source, "image": Path(image).name, "taps": []}
    for name, tensor in captured:
        side = write_tensor(out_dir, name, tensor, f"{source}, forward hook {name}")
        manifest["taps"].append({"name": name, "shape": side["shape"]})

    raw = output["predictions"] if isinstance(output, dict) else output
    write_tensor(out_dir, "output", raw, f"{source}, graph output")

    post = model._postprocess(
        output,
        conf_thres=conf,
        iou_thres=iou,
        original_size=original_size,
        max_det=max_det,
        input_size=imgsz,
    )
    detections = [
        {
            "x0": float(b[0]),
            "y0": float(b[1]),
            "x1": float(b[2]),
            "y1": float(b[3]),
            "score": float(s),
            "class_id": int(c),
        }
        for b, s, c in zip(post["boxes"], post["scores"], post["classes"])
    ]

    # The public predict() path must agree with the dumped pipeline.
    result = model.predict(image, conf=conf, iou=iou, max_det=max_det, imgsz=imgsz)
    result = result[0] if isinstance(result, list) else result
    public = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0, 4))
    mine = np.array([[d["x0"], d["y0"], d["x1"], d["y1"]] for d in detections]).reshape(-1, 4)
    if public.shape != mine.shape or not np.allclose(public, mine, atol=1e-3):
        raise RuntimeError("dumped detections differ from model.predict()")

    (out_dir / "detections.json").write_text(
        json.dumps(
            {
                "conf": conf,
                "iou": iou,
                "max_det": max_det,
                "multi_label": True,
                "source": f"{source}, libreyolo postprocess",
                "detections": detections,
            },
            indent=2,
        )
        + "\n"
    )
    manifest["num_detections"] = len(detections)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="Weights name or path")
    parser.add_argument("--image", required=True, action="append", help="Image path (repeatable)")
    parser.add_argument("--out", required=True, help="Output dir; one subdir per image stem")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--tap", action="append", help="Extra tap module path (repeatable)")
    args = parser.parse_args(argv)

    from libreyolo import LibreYOLO

    model = LibreYOLO(args.model, device="cpu")
    taps = None
    if args.tap:
        taps = list(DEFAULT_TAPS.get(model._get_model_name(), ())) + args.tap
    for image in args.image:
        out = Path(args.out) / Path(image).stem
        manifest = dump_reference(
            model,
            image,
            out,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            taps=taps,
        )
        print(f"{out}: {len(manifest['taps'])} taps, {manifest['num_detections']} detections")


if __name__ == "__main__":
    main()
