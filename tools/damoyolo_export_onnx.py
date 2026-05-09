"""Export a ported DAMO-YOLO checkpoint to ONNX and verify PyTorch parity.

Output: ONNX model with two outputs ``cls_scores`` (B, anchors, num_classes)
and ``boxes`` (B, anchors, 4 xyxy in model-input pixels). NMS is *not* baked
in — apply it in your inference framework.

Usage:
    python tools/damoyolo_export_onnx.py \\
        --weights downloads/damoyolo_tinynasL20_T.pt \\
        --size t \\
        --output downloads/damoyolo_t.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libreyolo.models.damoyolo.nn import build_damoyolo  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("damoyolo-onnx")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--size", default="t", choices=["t"])
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-verify", action="store_true", help="Skip onnxruntime parity check")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    log.info("Building model size=%s", args.size)
    model = build_damoyolo(size=args.size, num_classes=80)
    ck = torch.load(str(args.weights), map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.switch_to_deploy()

    h, w = args.input_size, args.input_size
    dummy = torch.randn(1, 3, h, w)
    # Warm-up: populate ZeroHead's mlvl_priors cache so the trace is
    # deterministic (priors get baked as constants for fixed input size).
    with torch.no_grad():
        _ = model(dummy)

    log.info("Exporting ONNX → %s", args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # ``dynamo=False`` selects the legacy TorchScript exporter. The new
    # dynamo path (default in torch ≥ 2.5) chokes on the buffer cast in
    # ``Integral.forward`` (`self.project.type_as(x)`).
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["images"],
        output_names=["cls_scores", "boxes"],
        opset_version=args.opset,
        dynamic_axes={
            "images": {0: "batch"},
            "cls_scores": {0: "batch"},
            "boxes": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    log.info("Wrote %.1f MB", args.output.stat().st_size / 1e6)

    if args.no_verify:
        return 0

    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping parity check")
        return 0

    log.info("Parity check: torch vs onnxruntime on a fresh random tensor")
    test_in = torch.randn(1, 3, h, w)
    with torch.no_grad():
        torch_cls, torch_box = model(test_in)
    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    ort_cls, ort_box = sess.run(["cls_scores", "boxes"], {"images": test_in.numpy()})
    cls_diff = float(np.abs(torch_cls.numpy() - ort_cls).max())
    box_diff = float(np.abs(torch_box.numpy() - ort_box).max())
    log.info("  cls abs max diff: %.2e", cls_diff)
    log.info("  box abs max diff: %.2e", box_diff)
    if cls_diff > 1e-4 or box_diff > 1e-3:
        log.error("Parity exceeds tolerance — investigate before trusting the export")
        return 1
    log.info("Parity within tolerance ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
