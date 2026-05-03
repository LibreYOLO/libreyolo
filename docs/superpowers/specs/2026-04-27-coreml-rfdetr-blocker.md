# CoreML export — RF-DETR known limitation

**Issue:** #90 — Add CoreML support
**Date:** 2026-04-27
**Status:** Known limitation, not blocking #90 closure
**Affects:** RF-DETR family only. YOLOX / YOLO9 / RT-DETR work end-to-end with verified numerical parity.

## Summary

`libreyolo/export/coreml.py:export_coreml` includes an RF-DETR ImageNet-normalization preprocess wrapper and is wired to handle the family. **However, conversion itself fails inside `coremltools` due to multiple unsupported ops in the PyTorch → MIL frontend** (coremltools 9.0). The blocker is upstream, not in our preprocessing logic.

## Verification

Tested with the upstream `rfdetr` package's `RFDETRNano` (auto-downloads `rf-detr-nano.pth`, 349 MB) on 2026-04-27 against `coremltools==9.0`, `torch==2.11.0`, Python 3.12, macOS arm64.

The PyTorch model loads and runs correctly (4 detections on `libreyolo/assets/parkour.jpg`). Feeding the raw `LWDETR` `nn.Module` through our `export_coreml(..., model_family="rfdetr")` triggers conversion failures in this order:

| Order | Op | Source location | coremltools status |
|------:|---|---|---|
| 1 | `_upsample_bicubic2d_aa` | DINOv2 backbone, positional-encoding resize | No converter |
| 2 | `upsample_bicubic2d` | Same site, with `antialias=False` patched on `F.interpolate` | No converter |
| 3 | `meshgrid` (multi-dim inputs) | Transformer reference-points construction | Converter rejects with `meshgrid received non-1d tensor` |

There may be more failures past the meshgrid one — conversion stops at the first error. The point is the RF-DETR graph uses several ops the direct PyTorch→CoreML path can't translate.

## Why this is upstream, not ours

* Our preprocess wrapper traces fine. The error is downstream, inside `ct.convert`.
* The same failures reproduce on a `RFDETRNano().model.model` instance constructed directly from the `rfdetr` package — no LibreYOLO code in the path.
* The unsupported ops are inherent to RF-DETR's architecture (DINOv2 backbone + DETR transformer), not specific to how LibreYOLO uses it.

## Side note — pre-existing LibreYOLO RF-DETR wrapper bug

While verifying, I also discovered that `libreyolo/models/rfdetr/nn.py:11` does `from rfdetr.main import Model as RFDETRMainModel`, but the installed `rfdetr` package no longer exposes `rfdetr.main` (the module was refactored upstream). Any call to `LibreYOLO("...rfdetr...pt")` therefore fails with `ModuleNotFoundError: No module named 'rfdetr.main'` regardless of CoreML. This is a separate bug, **out of scope for #90**, but worth filing as its own issue — it currently masks any RF-DETR export failure behind an earlier import failure.

## Workaround paths (not implemented)

In rough order of practicality:

1. **PyTorch → ONNX → CoreML.** ONNX has wider op coverage and the ONNX→CoreML converter handles bicubic + multi-D meshgrid cleanly. Would mean adding a second export path specifically for transformer-heavy families. Lossless. Recommended.
2. **Patch RF-DETR's forward** to use bilinear interpolation + 1-D meshgrid before tracing. Lossy: model accuracy will drift; would likely need a small finetune to recover. Not recommended without numerical evaluation.
3. **Add custom MIL op converters** to coremltools (`bicubic2d`, multi-D `meshgrid`). Real upstream contribution; significant work. Right answer long-term.

## Current behavior on `model.export(format="coreml")` for RF-DETR

If LibreYOLO's RF-DETR loader is fixed (separate issue), the export call will start running, `torch.jit.trace` will succeed, and `coremltools.convert` will raise `NotImplementedError: PyTorch convert function for op '_upsample_bicubic2d_aa' not implemented`. The error message is descriptive enough that a user encountering it will land on this doc when they search.

We could pre-empt this with an explicit `if model_family == "rfdetr": raise NotImplementedError(...)` in `export_coreml`, but that would need to be removed the moment the upstream gap closes. Leaving the underlying coremltools error to propagate is lower-maintenance.

## Recommendation

* Close #90 as done for **YOLOX, YOLO9, RT-DETR** (parity-verified end-to-end).
* Open a follow-up issue: **"RF-DETR CoreML export via ONNX bridge"**. That issue should also fix the `rfdetr.main` import skew in `libreyolo/models/rfdetr/nn.py` first (or document its dependency on a specific upstream `rfdetr` package version).
