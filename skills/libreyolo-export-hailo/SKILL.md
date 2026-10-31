---
name: libreyolo-export-hailo
description: Export LibreYOLO models for Hailo accelerators (Hailo-8/8L, Raspberry Pi AI Kit/AI HAT+). Use when a user asks to run LibreYOLO on Hailo, compile to HEF, or asks whether a given model family can be deployed on Hailo. Covers the "will it compile?" architectural rule (most families cannot), the ONNX-then-DFC two-stage flow, end nodes, calibration, and toolchain blockers.
---

# Export LibreYOLO models to Hailo (HEF)

There is **no `format="hef"`** in LibreYOLO, and there will not be one: the Hailo
Dataflow Compiler (DFC) is a proprietary SDK distributed as a private wheel behind
[Hailo Developer Zone](https://hailo.ai/developer-zone/) registration. It cannot be a
pip dependency or extra. Deployment is a **two-stage flow**:

1. LibreYOLO exports a plain **static ONNX** file (`format="onnx"`).
2. The **user** runs Hailo's DFC to parse, quantize (INT8), and compile to `.hef`.

```text
Libre<Model>.pt  →  ONNX  →  HAR (parse)  →  HAR (optimize/quantize INT8)  →  HEF
                 [libreyolo]              [Hailo DFC — user-installed]
```

The DFC cuts the ONNX graph at user-supplied `end_node_names` (the detection-head
convolutions) and **discards everything downstream**, so LibreYOLO's standard decoded
ONNX is acceptable input — the decode tail is simply ignored by the parser.

## Will it compile? Decide from the architecture, not a model list

Hailo-8/8L is a fixed-function **INT8 CNN accelerator with static shapes**. Whether a
LibreYOLO model can target it is a property of its architecture, not its name. Apply
these rules to any family — including ones added after this skill was written — instead
of maintaining a per-model table.

### Disqualifiers — if the model contains ANY of these, it will NOT compile

- **Attention of any kind.** Self-, cross-, deformable, or windowed attention is not
  supported on Hailo-8/8L. This one rule alone excludes every transformer-based family:
  all DETR-style detectors (RT-DETR and variants, D-FINE, DEIM, RF-DETR), every
  open-vocabulary/text-conditioned detector (Grounding DINO, OMDet-Turbo, OWLv2), every
  ViT backbone (DINOv2, Swin, Depth-Anything, EoMT, and the SAM / MobileSAM encoders),
  and every language/vision-language tower (BERT, CLIP, VLM).
  *(Hailo's own zoo ships a handful of hand-tuned ViT/DETR HEFs. That is bespoke vendor
  work and is not evidence that an arbitrary attention graph compiles — treat attention
  as a hard stop.)*
- **Dynamic shapes or data-dependent control flow.** The DFC compiles one fixed input
  shape and a static graph. Anything whose graph depends on runtime values — variable
  query counts, text prompts, dynamic top-k, `NonZero`, `Gather`/`TopK` with dynamic
  indices, `grid_sample` — cannot compile. (This is a second, independent reason the
  open-vocabulary and DETR families are out.)
- **LayerNorm-/GELU-dominated designs.** BatchNorm folds into convs cleanly; LayerNorm
  support is poor and GELU is not a native activation. A ConvNeXt-style stack is a bad
  fit even though it is nominally "convolutional."
- **Native-resolution image-to-image with large activations.** Restoration/enhancement
  models (e.g. NAFNet) run at full input resolution and blow past practical Hailo SRAM
  budgets. Do not attempt.

### What DOES compile — pure-CNN, fixed-shape graphs

A family is a **candidate** when it is convolution-only (conv, pooling, resize, concat,
elementwise add/mul), uses BatchNorm + ReLU/SiLU, and has a fixed input size. In the
current library that means:

- **CNN single-stage detectors with a conv detection head** — YOLOX and YOLO9 are the
  primary targets. These are also the only two that map onto a **HailoRT NMS
  meta-architecture** (`yolox`, and the `yolov8`-style decoupled head), so Hailo can own
  NMS in the compiled pipeline.
- **Other CNN detectors** (PicoDet, YOLO-NAS, RTMDet, and similar conv-headed detectors)
  compile as graphs, but there is **no matching HailoRT NMS meta-arch** — the HEF emits
  raw head tensors and your application must run decode + NMS on the CPU.
- **CNN image classifiers** (ResNet, MobileNetV4-conv, EfficientNetV2). ResNet is the
  best-supported classify path because Hailo's Model Zoo ships ResNet recipes.
- **Small conv task-heads** (FOMO point heads, L2CS gaze on a ResNet backbone, EC pose)
  are compilable in principle but have no Hailo recipe — application-side postprocess
  only, and unvalidated.

Rule of thumb for a new family: **open the model, look for a transformer block, a
LayerNorm, or a dynamic shape. If you find one, stop and recommend YOLOX / YOLO9 (or a
different runtime — ONNX Runtime, TensorRT, OpenVINO, all first-class `libreyolo export`
targets). If it is plain CNN, it is a candidate worth trying through the DFC.**

### Op-level limits behind the rules

If asked *why*, these are the concrete DFC constraints the rules encode: static shapes
only; no attention; BatchNorm folds but LayerNorm support is poor; no dynamic
`Gather`/`TopK`/`NonZero`/`grid_sample`; INT8 only (accuracy loss must be measured);
HailoRT NMS exists only for known meta-architectures (`yolov8`, `yolox`, `nanodet_v8`,
…), so unknown or NMS-free heads (e.g. `yolo9_e2e`) get raw tensors and need app-side
postprocess.

**Status caveat:** no LibreYOLO family has been validated end-to-end through the DFC to a
running HEF yet. The rules above predict compilability from architecture; parser,
quantization, and accuracy remain unproven until a HEF is compiled and measured. Treat
every "candidate" as requiring recorded validation evidence.

## Step 1 — export static ONNX from LibreYOLO

Hailo needs **batch 1, fixed resolution, no dynamic axes**. The Python API defaults to
`dynamic=True`, so disable it explicitly:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t.pt")
model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
```

CLI equivalent (the CLI already defaults to static shapes, unlike the Python API):

```bash
libreyolo export --model LibreYOLO9t.pt --format onnx --imgsz 640
```

Do not use `half=True` — the DFC ingests fp32 ONNX and does its own INT8 quantization.
Do not use `nms=True` (embedded ONNX NMS) — Hailo either owns NMS via `nms_postprocess`
or the application does; an NMS subgraph is dead weight past the end nodes. The default
opset is fine; if the DFC parser complains, re-export with `opset=11`. Confirm the input
is `[1, 3, H, W]` with static dims before spending compile time.

## Step 2 — compile with the Hailo DFC (user-side)

Prerequisites, all on the **user's** side, none installable from PyPI:

- Linux **x86_64** machine (WSL2 Ubuntu 22.04 works). Compilation cannot run on ARM —
  the Raspberry Pi is a runtime target, never the compile host.
- DFC wheel (`hailo_sdk_client`) from the Hailo Developer Zone (free registration).
- For Hailo-8/8L, pin to the **Hailo Model Zoo v2.x** line for recipes/NMS configs. Do
  not use Model Zoo master/v5.x unless targeting Hailo-10/15.
- A GPU is strongly recommended for the quantization step (hours vs minutes without).

Pick `hw_arch` for the target:

| `hw_arch` | Device |
| --- | --- |
| `hailo8` | Hailo-8 (26 TOPS AI HAT+, M.2/PCIe modules) |
| `hailo8l` | Hailo-8L (Raspberry Pi AI Kit, 13 TOPS AI HAT+) |
| `hailo10h` | AI HAT+ 2 / Hailo-10H (needs matching newer DFC/Model Zoo) |

If unsure, run `hailortcli fw-control identify` on the device.

Compile pipeline (LibreYOLO9 example):

```python
import numpy as np
from hailo_sdk_client import ClientRunner
from PIL import Image
from pathlib import Path

ONNX = "libreyolo9t.onnx"
HW_ARCH = "hailo8"          # hailo8 | hailo8l
IMGSZ = 640
NMS_CONFIG = "yolo9_nms_config.json"  # adapt from the Model Zoo config for the closest YOLO
                                      # variant; it is class-count AND imgsz specific — a
                                      # COCO-80 config is wrong for a fine-tuned 3-class model.

# LibreYOLO9 detection-head end nodes (all sizes), verified against the standard
# LibreYOLO ONNX export. NOTE: LibreYOLO graphs use a "/head/..." prefix, not the
# "model.N" prefix seen in other libraries' docs — configs copied from elsewhere will
# not match. If parsing fails, confirm the names in your own export (netron, or grep the
# graph for "anchor_convs.0.2/Conv"). Exports made before LibreYOLO renamed the YOLO9
# head towers use "/head/cv2.N/cv2.N.2/Conv" (box) and "/head/cv3.N/cv3.N.2/Conv" (class).
END_NODES = [
    "/head/anchor_convs.0/anchor_convs.0.2/Conv", "/head/class_convs.0/class_convs.0.2/Conv",
    "/head/anchor_convs.1/anchor_convs.1.2/Conv", "/head/class_convs.1/class_convs.1.2/Conv",
    "/head/anchor_convs.2/anchor_convs.2.2/Conv", "/head/class_convs.2/class_convs.2.2/Conv",
]

runner = ClientRunner(hw_arch=HW_ARCH)
runner.translate_onnx_model(ONNX, end_node_names=END_NODES)

# Normalization must match LibreYOLO preprocessing (0-255 → 0-1, no mean/std for YOLO9/YOLOX).
# The convN layer names in change_output_activation are assigned by the DFC at parse
# time and are model-specific — read them from the DFC log, do not copy blindly.
runner.load_model_script(
    "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"
    f'nms_postprocess("{NMS_CONFIG}", meta_arch=yolov8, engine=cpu)\n'
)

# Calibration: 64-128 images REPRESENTATIVE OF DEPLOYMENT DATA, resized to IMGSZ.
calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]
calib = np.stack([
    np.asarray(Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)), dtype=np.float32)
    for p in calib_paths
])

runner.optimize(calib)
hef = runner.compile()
Path("libreyolo9t.hef").write_bytes(hef)
```

Path notes by family:

- **YOLO9**: end nodes above; `meta_arch=yolov8` (identical decoupled-head layout).
- **YOLOX**: run `translate_onnx_model(ONNX)` **without** `end_node_names` first — the
  DFC log prints suggested end nodes; re-run with those. Use `meta_arch=yolox`.
- **Other conv detectors (PicoDet / YOLO-NAS / RTMDet) and conv task-heads (FOMO / EC /
  L2CS)**: compile without `nms_postprocess`; the HEF outputs raw head tensors and the
  application owns decode + NMS/argmax.
- **Anything the disqualifier rules exclude**: stop. Do not try to force a transformer
  through the DFC.

Random calibration images will compile but silently destroy accuracy. Keep the compile
log on failure — fixes always hinge on the exact failing layer/op name.

## Step 3 — run on the device

On the target (e.g. Raspberry Pi 5 + AI Kit):

```bash
sudo apt install dkms hailo-all      # Raspberry Pi path; other hosts: HailoRT from Developer Zone
hailortcli fw-control identify       # device sanity check
hailortcli run libreyolo9t.hef       # smoke test / FPS
```

Application inference uses the `hailo_platform` Python API. With `nms_postprocess`
compiled in, the output is `(batch, num_classes, max_dets, 5)` with `[y1, x1, y2, x2,
score]` in model coordinates — scale back to the original image size yourself.
LibreYOLO's `Results` pipeline is not involved at runtime; the HEF is a standalone
artifact.

## Support-claim discipline

Do not tell a user a family "works on Hailo" until all of these exist:

1. A compiled `.hef` from the exact LibreYOLO checkpoint (with DFC / Model Zoo / HailoRT
   versions recorded).
2. Representative calibration documented.
3. Accuracy measured on-device against the fp32 baseline (mAP delta, not just FPS).

Until then the honest phrasing is: "pure-CNN detectors like YOLOX/YOLO9 are compile
candidates; anything with attention, dynamic shapes, or LayerNorm will not run on
Hailo."
