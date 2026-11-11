# Axelera Metis export

> **PREVIEW.** Axelera Metis support is new in 1.8.0. Every size below
> compiles, runs on a Metis device and exports. "validated" means the INT8
> result is within 1.0 point of FP32 on the paired set; "supported" means it
> works with a larger measured drop, which the export logs once as a warning.
> Accuracy never blocks an export; "blocked" is only for families that do not
> compile, do not run or give meaningless output (`docs/export_support.md`).
> Results are from one Metis M.2 Max with axelera-devkit / axelera-rt 1.9.0.

Measured on device, paired INT8 vs FP32 on the same images, each size with the
recipe LibreYOLO applies to it (section 7 has calibration details):

| Family | Size | Task | Dataset (paired) | FP32 | Metis INT8 | Delta | Label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO-NAS | s | detect | COCO val2017, 1000 | 0.4549 | 0.4519 | -0.30 | validated |
| YOLO-NAS | m | detect | COCO val2017, 1000 | 0.4942 | 0.4893 | -0.49 | validated |
| YOLO-NAS | l | detect | COCO val2017, 1000 | 0.5036 | 0.5025 | -0.11 | validated |
| YOLO-NAS-pose | n | pose | COCO val2017 persons, 300 | 0.7089 | 0.6959 | -1.30 | supported |
| YOLO-NAS-pose | s | pose | COCO val2017 persons, 300 | 0.7565 | 0.7344 | -2.21 | supported |
| YOLO-NAS-pose | m | pose | COCO val2017 persons, 300 | 0.7885 | 0.7821 | -0.64 | validated |
| YOLO-NAS-pose | l | pose | COCO val2017 persons, 300 | 0.7922 | 0.7884 | -0.38 | validated |
| YOLO-NAS-R | s | obb | DOTA-v2.0 val, 300 tiles | 0.3844 | 0.3747 | -0.97 | validated |
| YOLO-NAS-R | m | obb | DOTA-v2.0 val, 300 tiles | 0.4141 | 0.4105 | -0.36 | validated |
| YOLO-NAS-R | l | obb | DOTA-v2.0 val, 300 tiles | 0.4327 | 0.4348 | +0.21 | validated |
| YOLO9 | t | detect | COCO val2017, 1000 | 0.3669 | 0.3624 | -0.45 | validated |
| YOLO9 | s | detect | COCO val2017, 1000 | 0.4502 | 0.4450 | -0.52 | validated |
| YOLO9 | m | detect | COCO val2017, 1000 | 0.4913 | 0.4896 | -0.17 | validated |
| YOLO9 | c | detect | COCO val2017, 1000 | 0.5127 | 0.5093 | -0.34 | validated |
| YOLOX | n | detect | COCO val2017, 300 | 0.2609 | 0.2428 | -1.81 | supported |
| YOLOX | t | detect | COCO val2017, 1000 | 0.3238 | 0.3202 | -0.36 | validated |
| YOLOX | s | detect | COCO val2017, 1000 | 0.3981 | 0.3933 | -0.48 | validated |
| YOLOX | m | detect | COCO val2017, 1000 | 0.4582 | 0.4516 | -0.66 | validated |
| YOLOX | l | detect | COCO val2017, 1000 | 0.4849 | 0.4781 | -0.68 | validated |
| YOLOX | x | detect | COCO val2017, 1000 | 0.4943 | 0.4867 | -0.76 | validated |
| YOLOv3 | t | detect | COCO val2017, 1000 | 0.1023 | 0.1006 | -0.17 | validated |
| YOLOv3 | b | detect | COCO val2017, 1000 | 0.3656 | 0.3580 | -0.76 | validated |
| YOLOv3 | spp | detect | COCO val2017, 1000 | 0.4304 | 0.4261 | -0.43 | validated |
| YOLOv4 | t | detect | COCO val2017, 1000 | 0.2021 | 0.1968 | -0.53 | validated |
| YOLOv4 | b | detect | COCO val2017, 1000 | 0.4806 | 0.4654 | -1.52 | supported |
| YOLOv7 | b | detect | COCO val2017, 1000 | 0.2843 | 0.2859 | +0.16 | validated |
| PP-YOLOE | s | detect | COCO val2017, 1000 | 0.4085 | 0.3962 | -1.23 | supported |
| PP-YOLOE | m | detect | COCO val2017, 1000 | 0.4538 | 0.4416 | -1.22 | supported |
| PP-YOLOE | l | detect | COCO val2017, 1000 | 0.4771 | 0.4740 | -0.31 | validated |
| PP-YOLOE | x | detect | COCO val2017, 1000 | 0.4918 | 0.4747 | -1.71 | supported |
| PicoDet | s | detect | COCO val2017, 1000 | 0.2687 | 0.2453 | -2.34 | supported |
| RTMDet | t | detect | COCO val2017, 1000 | 0.3906 | 0.3622 | -2.84 | supported |
| ResNet | 18 | classify (top-1) | ImageNetV2, 1000 | 0.595 | 0.589 | -0.6 | validated |
| ResNet | 34 | classify (top-1) | ImageNetV2, 1000 | 0.657 | 0.647 | -1.0 | validated |
| ResNet | 50 | classify (top-1) | ImageNetV2, 1000 | 0.688 | 0.681 | -0.7 | validated |
| ResNet | 101 | classify (top-1) | ImageNetV2, 1000 | 0.713 | 0.708 | -0.5 | validated |
| MobileNetV4 | s | classify (top-1) | ImageNetV2, 1000 | 0.624 | 0.607 | -1.7 | supported |
| MobileNetV4 | m | classify (top-1) | ImageNetV2, 1000 | 0.682 | 0.684 | +0.2 | validated |
| MobileNetV4 | l | classify (top-1) | ImageNetV2, 1000 | 0.727 | 0.710 | -1.7 | supported |
| EfficientNetV2 | b0 | classify (top-1) | ImageNetV2, 1000 | 0.667 | 0.656 | -1.1 | supported |
| FCN | r50 | semantic (mIoU) | COCO val2017 VOC-21, 300 | 0.6515 | 0.6492 | -0.23 | validated |
| FCN | r101 | semantic (mIoU) | COCO val2017 VOC-21, 300 | 0.6798 | 0.6779 | -0.19 | validated |
| QuickSRNet | m2 | restore 2x (PSNR, dB) | COCO crops from 128 px, 296 | 29.12 | 28.66 | -0.46 | supported |
| MiDaS | s | depth (agreement with FP32) | COCO val2017 at 256 px, 300 | reference | AbsRel 0.069, delta1 0.94 | AbsRel 0.069 (bar 0.02) | supported |
| DeepLabV3 | mv3 | semantic (mIoU) | COCO val2017 VOC-21, 300 | 0.5230 | 0.4853 | -3.77 | supported |

Detection and pose report mAP50-95; ImageNetV2 is used locally under its
licence terms and not redistributed.

`format="axelera"` compiles a LibreYOLO model to an INT8 `.axm` for Axelera's
Metis AIPU, and `LibreYOLO("<stem>_axelera_model")` runs it with the same
`predict` and `val` calls (Python and CLI) as the PyTorch model.

## 1. Scope and status

| Family | Task | Sizes | Tier |
| --- | --- | --- | --- |
| YOLO-NAS | detect | s, m, l | see `docs/export_support.md` |
| YOLO-NAS | pose | n, s, m, l (n and s supported with a measured drop) | see `docs/export_support.md` |
| YOLO-NAS-R | obb | s, m, l | see `docs/export_support.md` (calibration data required) |
| YOLO9 | detect | t, s, m, c | see `docs/export_support.md` |
| ResNet | classify | 18, 34, 50, 101 | see `docs/export_support.md` |
| MobileNetV4 | classify | s, m, l (s and l supported with a measured drop) | see `docs/export_support.md` |
| EfficientNetV2 | classify | b0 (supported with a measured drop; b1 and b2 blocked: do not compile; b3 not measured yet) | see `docs/export_support.md` |
| FCN | semantic | r50, r101 | see `docs/export_support.md` |
| QuickSRNet | restore | m2 (supported with a measured drop; bar 0.3 dB) | see `docs/export_support.md` |
| MiDaS | depth | s (supported with a measured drop; l not measured yet) | see `docs/export_support.md` |
| DeepLabV3 | semantic | mv3 (supported with a measured drop; r50 and r101 blocked: do not compile) | see `docs/export_support.md` |
| YOLOX | detect | n, t, s, m, l, x (n supported with a measured drop) | see `docs/export_support.md` |
| YOLOv3 | detect | t, b, spp | see `docs/export_support.md` |
| YOLOv4 | detect | t, b (b supported with a measured drop) | see `docs/export_support.md` |
| YOLOv7 | detect | b | see `docs/export_support.md` |
| PP-YOLOE | detect | s, m, l, x (s, m and x supported with a measured drop) | see `docs/export_support.md` |
| PicoDet | detect | s (supported with a measured drop; m and l not measured yet) | see `docs/export_support.md` |
| RTMDet | detect | t (supported with a measured drop; s blocked: collapses on a Metis; m, l and x not measured yet) | see `docs/export_support.md` |
| YOLO9-E2E, YOLO9-P2, other families | all | all | not measured yet (section 11) |
| RF-DETR and other transformer families | all | all | out of scope (section 11) |

The tiers in `docs/export_support.md` are generated from
`libreyolo/export/support.py`; `libreyolo formats --family yolonas` prints
them. Validated with Axelera's Voyager SDK 1.9.0 (`axelera-devkit` and
`axelera-rt`) on a Metis M.2 Max (113m, 16 GiB) with flashed firmware 1.9.0,
board controller 7.4 and the `metis` kernel driver 1.7.16.

## 2. Requirements

- Export (compile): Linux x86_64 (Ubuntu 22.04 or 24.04; WSL2 on Windows),
  Python 3.10 to 3.13. No Metis device is needed to compile.
- Inference: a Metis device, Axelera's `metis-dkms` kernel driver (1.7.0 or
  newer for SDK 1.9) and `axelera-rt`. The firmware and driver versions each
  SDK release supports are listed in Axelera's release compatibility matrix.
- PyTorch: the SDK requires `torch>=2.7,<2.15`.
- Check the device with `axdevice` (installed with `axelera-rt`); it lists each
  device with its firmware, board controller and clock. `libreyolo checks`
  lists the Axelera packages and the Metis devices it sees.

## 3. Installation

```bash
pip install "libreyolo[onnx]"   # the onnx extra is needed to export (compile)
# compile (export)
pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-devkit==1.9.0
# run (.axm inference)
pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt==1.9.0
```

Both packages come from Axelera's own package index, not PyPI, and are
proprietary software under Axelera's EULA. LibreYOLO does not install them on
its own: when one is missing, export or load fails with the command above.
Set `LIBREYOLO_AUTOINSTALL=1` to let LibreYOLO run that command for you; it
logs what it installs.

Use a dedicated virtual environment for Axelera work. The SDK 1.9.0 wheels pin
several packages LibreYOLO also uses, and pip downgrades them:

| Package | Pinned to | Pinned by |
| --- | --- | --- |
| click | `==8.1.8` | `axelera-atex-utils` |
| typer | `>=0.12,<0.13` | `axelera-compiler`, `axelera-atex-utils` |
| rich | `<14` | `axelera-atex-utils`, `axelera-package-lib` |
| onnx | `<1.19` | `axelera-onnx2torch` |
| opencv-python | `<5` | `axelera-runtime2` (also with `axelera-rt` alone) |
| pillow | `<13` | `axelera-runtime2` |
| torch | `>=2.7,<2.15` | `axelera-qtoolsv2` |

LibreYOLO works with all of these. Installing other LibreYOLO extras after the
SDK can try to upgrade them again; install the SDK last.

## 4. Export

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
path = model.export(format="axelera")   # weights/LibreYOLONASs-converted_axelera_model
```

```bash
libreyolo export model=LibreYOLONASs.pt format=axelera
libreyolo export model=LibreYOLO9t.pt format=axelera   # weights/LibreYOLO9t_axelera_model
```

| Argument | Axelera behavior |
| --- | --- |
| `imgsz` | Input size, fixed in the compiled model; must be a multiple of 32 (other values raise). Default: the checkpoint's size. |
| `data` | Calibration images (dataset YAML). Default `coco128.yaml` for detect and pose. |
| `fraction` | Share of the calibration split to use. 100 to 400 images are recommended; fewer than 100 logs a warning. |
| `batch` | Calibration loader batch size only. The compiled model takes one image at a time. |
| `device` | Torch device used to trace the model, not the Metis device. |
| `quantize` / `int8` | INT8 is the only precision and is always on. `quantize=16`, `quantize=32` and `half=True` raise. |
| `dynamic` | Always static; `dynamic=True` raises. |
| `opset` | ONNX opset of the intermediate graph, 14 to 17 (default 17). |
| `nms` | Not supported: NMS runs on the host. |
| `output_path` | Where to write the directory. Default `weights/<checkpoint stem>_axelera_model`. |
| `verify` | Writes `<stem>.parity.json`, comparing the quantized model with FP32 (and the device, when one is present). Also `libreyolo export ... verify=true`. |
| `compiler_config` | Python only: a dict of Axelera `CompilerConfig` fields for advanced tuning. Unknown fields raise; fields LibreYOLO owns (output format, core count, output paths) are rejected. |

Calibration uses the model's own preprocessing, one image at a time. For a
fine-tuned model, calibrate with images from its own domain. Compiling takes
minutes to tens of minutes and several GB of RAM (section 7).

The directory is named after the checkpoint file, as for every LibreYOLO
export. Pretrained YOLO-NAS weights are converted from Deci's format on first
use into `LibreYOLONAS<size>-converted.pt`, so their exports are named
`weights/LibreYOLONASs-converted_axelera_model`; a fine-tuned `best.pt`
exports to `weights/best_axelera_model`. `output_path=` sets any other path.

Output:

```text
weights/LibreYOLONASs-converted_axelera_model/
    LibreYOLONASs-converted.axm   compiled model
    compiler_config_final.toml    compiler settings used for the build
    metadata.yaml                 LibreYOLO metadata (section 8)
    LibreYOLONASs-converted.parity.json   only with verify=True
```

Re-exporting to the same path replaces the directory only after the new build
succeeded. Nothing else is written: no ONNX file and no compiler build files.

## 5. Inference and validation

```python
from libreyolo import LibreYOLO

npu = LibreYOLO("weights/LibreYOLONASs-converted_axelera_model")   # or the .axm inside it
results = npu("image.jpg", conf=0.25)
metrics = npu.val(data="coco128.yaml")
for r in npu.predict("video.mp4", stream=True):
    ...
```

```bash
libreyolo predict model=weights/LibreYOLONASs-converted_axelera_model source=image.jpg
libreyolo val model=weights/LibreYOLONASs-converted_axelera_model data=coco128.yaml
```

Object tracking (`track()`) is available on PyTorch models only; exported
models of every format, Axelera included, do not have it yet.

The network runs on the Metis AIPU; preprocessing, the box decode and NMS run
on the host. Results are the same `Results` objects as from the PyTorch model.
The input size is fixed by the export. `device="auto"`, `"axelera"`,
`"metis"`, `"aipu"` and `"cpu"` all run on the AIPU (`"cpu"` refers to the
host-side steps); a CUDA device is ignored with a warning. The directory can be copied to any Linux machine with a Metis device
and `axelera-rt`; `axelera-devkit` is not needed there.

## 6. Migrating from Ultralytics

LibreYOLO follows the same export and inference contract, so the commands map
one to one. The artifacts are not interchangeable between the two libraries:
re-export the model with LibreYOLO.

| Ultralytics | LibreYOLO |
| --- | --- |
| `YOLO("yolo11n.pt").export(format="axelera")` | `LibreYOLO("LibreYOLONASs.pt").export(format="axelera")` |
| `YOLO("yolo11n_axelera_model")` | `LibreYOLO("weights/LibreYOLONASs-converted_axelera_model")` |
| `model("image.jpg")`, `model.val(data=...)` | same |
| `yolo export model=... format=axelera` | `libreyolo export model=... format=axelera` |
| `yolo predict model=..._axelera_model source=...` | `libreyolo predict model=..._axelera_model source=...` |
| output next to the weights | output in `weights/` (as for every LibreYOLO export) |
| SDK downloaded automatically on first use | install command printed; opt in with `LIBREYOLO_AUTOINSTALL=1` |

## 7. Accuracy and performance

Validated 2026-11-08 to 2026-11-10 on a Metis M.2 Max (113m, 16 GiB, firmware 1.9.0, board
controller 7.4, driver 1.7.16) with Voyager SDK 1.9.0, from LibreYOLO's
pretrained checkpoints. Each row compares the FP32 model and the Metis INT8
model on the same images with LibreYOLO's validator (conf 0.001, IoU 0.6).
The image sets are subsets of COCO val2017 (detection: the 300 or 1000 images
named in the row; pose: the last 300 val2017 images with person keypoints)
and of DOTA-v2.0 val tiled to 1024 (300 tiles), so the absolute numbers are
not comparable to published full-set results; the deltas are the point.

| Model | Task | Image set (paired) | Calibration | FP32 | Metis INT8 | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO-NAS-s | detect | COCO val2017, 1000 images | 64 coco128 | 0.4549 | 0.4519 | -0.30 |
| YOLO-NAS-m | detect | COCO val2017, 1000 images | 128 coco128 | 0.4942 | 0.4893 | -0.49 |
| YOLO-NAS-l | detect | COCO val2017, 1000 images | 64 coco128 | 0.5036 | 0.5025 | -0.11 |
| YOLO-NAS-pose-n | pose | COCO val2017 persons, 300 images | 128 coco128 | 0.7089 | 0.6959 | -1.30 (supported) |
| YOLO-NAS-pose-s | pose | COCO val2017 persons, 300 images | 128 coco128 | 0.7565 | 0.7344 | -2.21 (supported; class and keypoint towers bounded) |
| YOLO-NAS-pose-m | pose | COCO val2017 persons, 300 images | 128 coco128 | 0.7885 | 0.7821 | -0.64 |
| YOLO-NAS-pose-l | pose | COCO val2017 persons, 300 images | 128 coco128 | 0.7922 | 0.7884 | -0.38 |
| YOLO-NAS-R-s | obb | DOTA-v2.0 val, 300 tiles | 64 DOTA tiles | 0.3844 | 0.3747 | -0.97 |
| YOLO-NAS-R-m | obb | DOTA-v2.0 val, 300 tiles | 64 DOTA tiles | 0.4141 | 0.4105 | -0.36 |
| YOLO-NAS-R-l | obb | DOTA-v2.0 val, 300 tiles | 64 DOTA tiles | 0.4327 | 0.4348 | +0.21 |
| YOLO9-t | detect | COCO val2017, 1000 images | 64 coco128 | 0.3669 | 0.3624 | -0.45 |
| YOLO9-s | detect | COCO val2017, 1000 images | 64 coco128 | 0.4502 | 0.4450 | -0.52 |
| YOLO9-m | detect | COCO val2017, 1000 images | 64 coco128 | 0.4913 | 0.4896 | -0.17 |
| YOLO9-c | detect | COCO val2017, 1000 images | 64 coco128 | 0.5127 | 0.5093 | -0.34 |
| YOLOX-n | detect | COCO val2017, 300 images | 64 coco128 | 0.2609 | 0.2428 | -1.81 (supported; -1.66 with 128 calibration images) |
| YOLOX-t | detect | COCO val2017, 1000 images | 64 coco128 | 0.3238 | 0.3202 | -0.36 |
| YOLOX-s | detect | COCO val2017, 1000 images | 64 coco128 | 0.3981 | 0.3933 | -0.48 |
| YOLOX-m | detect | COCO val2017, 1000 images | 64 coco128 | 0.4582 | 0.4516 | -0.66 |
| YOLOX-l | detect | COCO val2017, 1000 images | 64 coco128 | 0.4849 | 0.4781 | -0.68 |
| YOLOX-x | detect | COCO val2017, 1000 images | 64 coco128 | 0.4943 | 0.4867 | -0.76 |
| YOLOv3-t | detect | COCO val2017, 1000 images | 64 coco128 | 0.1023 | 0.1006 | -0.17 |
| YOLOv3-b | detect | COCO val2017, 1000 images | 64 coco128 | 0.3656 | 0.3580 | -0.76 |
| YOLOv3-spp | detect | COCO val2017, 1000 images | 64 coco128 | 0.4304 | 0.4261 | -0.43 |
| YOLOv4-t | detect | COCO val2017, 1000 images | 64 coco128 | 0.2021 | 0.1968 | -0.53 |
| YOLOv4-b | detect | COCO val2017, 1000 images | 64 coco128 | 0.4806 | 0.4654 | -1.52 (supported; histogram ranges + [-6, 6] box window) |
| YOLOv7 | detect | COCO val2017, 1000 images | 64 coco128 | 0.2843 | 0.2859 | +0.16 |
| PP-YOLOE-s | detect | COCO val2017, 1000 images | 64 coco128 | 0.4085 | 0.3962 | -1.23 (supported) |
| PP-YOLOE-m | detect | COCO val2017, 1000 images | 64 coco128 | 0.4538 | 0.4416 | -1.22 (supported) |
| PP-YOLOE-l | detect | COCO val2017, 1000 images | 64 coco128 | 0.4771 | 0.4740 | -0.31 |
| PP-YOLOE-x | detect | COCO val2017, 1000 images | 64 coco128 | 0.4918 | 0.4747 | -1.71 (supported) |
| PicoDet-s | detect | COCO val2017, 1000 images | 64 coco128 | 0.2687 | 0.2453 | -2.34 (supported; -1.28 on 300) |
| RTMDet-t | detect | COCO val2017, 1000 images | 64 coco128 | 0.3906 | 0.3622 | -2.84 (supported) |

Classifiers, top-1 on 1000 ImageNetV2 MatchedFrequency images (one per
class, a set with ImageNet's licence terms used locally only), calibrated on
250 other ImageNetV2 images (`data=` a dataset YAML whose train folder holds
them):

| Model | Input | FP32 top-1 | Metis INT8 top-1 | Delta | Compile (28 W host) |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 | 224 | 0.595 | 0.589 | -0.6 | 1.5 min |
| ResNet-34 | 224 | 0.657 | 0.647 | -1.0 | 3.3 min |
| ResNet-50 | 224 | 0.688 | 0.681 | -0.7 | 4.6 min |
| ResNet-101 | 224 | 0.713 | 0.708 | -0.5 | 9.1 min |

Calibrating a classifier on coco128 (the fallback when `data=` is omitted)
cost ResNet-18 1.1 top-1 points instead of 0.6, and the histogram scheme 2.0.

MobileNetV4 needs more than the ResNet recipe: with plain min-max ranges
size s lost 6.2 top-1 points. LibreYOLO equalizes its residual streams per
channel and bounds the stage-2 stream before export (both nearly
FP32-neutral; largest logit change on the calibration images 0.1 on m),
then calibrates with histogram activation ranges
(`hybrid_per_tensor_per_channel`):

| Model | Input | FP32 top-1 | Metis INT8 top-1 | Delta | Compile (28 W host) |
| --- | --- | --- | --- | --- | --- |
| MobileNetV4-m | 224 | 0.682 | 0.684 | +0.2 | 3.1 min |

MobileNetV4-s and -l are supported with a measured drop of 1.7 points each
(s 0.607 vs 0.624 with histogram ranges and no stream transform; l 0.710
vs 0.727 with min-max ranges and the stream transform; LibreYOLO applies
these per-size recipes). The recipe
that passes on m costs s 2.6 and l 3.0 points. Axelera's
`quantize_dw_channel_wise` and `smoothquant_coefficient` options changed
nothing measurable.

Semantic segmentation, FCN-ResNet50/101 (VOC-21 head) at 520x520, mIoU on 300
COCO val2017 images with VOC-21 masks rasterized from COCO's instance
annotations (the 20 VOC classes all exist in COCO; crowd regions, other
objects and pixels claimed by two classes are ignored), calibrated on
coco128. Both columns go through the same exported-model path (stretch to
the canvas, host upsampling of the logits), so the delta is INT8 and the
device only:

| Model | Input | FP32 (ONNX) mIoU | Metis INT8 mIoU | Delta | Compile (28 W host) |
| --- | --- | --- | --- | --- | --- |
| FCN-ResNet50 | 520 | 0.6515 | 0.6492 | -0.23 | 26 min |
| FCN-ResNet101 | 520 | 0.6798 | 0.6779 | -0.19 | 51 min |

Values are mAP50-95 (box, rotated box for obb, keypoint OKS for pose). On
mAP50 the obb deltas are -1.36 (s), -0.11 (m) and -0.43 (l).

YOLOX compiles with the default settings: its box, objectness and class
outputs stay within a few INT8 steps of their useful range on every size,
so no output windows are needed. Graphs with depthwise convolutions
(YOLOX-n) compile with per-channel ranges for the depthwise inputs
(`hybrid_per_tensor_per_channel`): the stem output that feeds YOLOX-n's
first depthwise convolution has channel maxima from 0.02 to 106, and one
per-tensor scale scored 0.011 vs FP32 0.261 on 300 images. Per-channel
depthwise ranges bring it to 0.244 (128 calibration images), still 1.66
points below FP32; the loss is spread over the network, not in the outputs
(windowing the logits or SmoothQuant changed nothing). YOLOX-n is supported
with that drop: -1.81 with the default 64 calibration images, -1.66 with 128.
YOLOX-t is validated.

YOLOv4-b (608x608, Mish backbone) loses 6.8 points with the default
min-max ranges, almost all of it in the box outputs (swapping them back to
FP32 recovers 6.1 of 6.4 points on 300 images). Histogram activation ranges
plus a [-6, 6] box-output window bring it to -1.52 on 1000 images; LibreYOLO
applies that recipe to size b, which is supported with that drop.

RTMDet (CSPNeXt backbone) has depthwise convolutions, so it compiles with
per-channel depthwise ranges; with one per-tensor scale RTMDet-t collapses
(0.009 vs 0.393 on 300 images). The remaining loss sat in one stage-2
pointwise convolution whose input channels span very different ranges
(peaks 14.2 against a median of 2.7 on RTMDet-t, 12.8 against 0.28 on
RTMDet-s), which one per-tensor int8 scale does not survive. The export equalizes every
CSPNeXt pointwise input on its own copy of the model (each channel scaled by
1/s through a 1x1 depthwise convolution, the pointwise weights by s, from
calibration peaks; FP32 outputs unchanged): RTMDet-t goes from -4.44 to
-2.84 on 1000 images and is supported with that measured drop. RTMDet-s
still collapses with it (0.079 vs 0.411 on 300 images; 0.136 without it), so
its export is blocked.

The pose rows include everything LibreYOLO applies to pose heads: ReLU6
output windows, equalization of the keypoint branch from calibration
images, a cap on the class branch's last activation at its 99.5th
calibration percentile, and histogram activation ranges. Without them
pose-s measured 0.0758. Pose m and l are validated; n (-1.30, -1.11 with
person-image calibration) and s are supported with their drop. Size s also
gets a cap on the keypoint branch (-2.21 instead of -3.16), which helps s but
costs n, m and l several points, so only s uses it.

YOLO9 FP32 values come from the same raw-head graph and host decode as the
device run; the PyTorch model gives the same numbers.

Calibration scheme, YOLO-NAS-s detect, device, COCO val2017 1000 images:

| Calibration scheme | mAP50-95 | mAP50 |
| --- | --- | --- |
| histogram (Axelera's default) | 0.442 | 0.610 |
| min-max (LibreYOLO's default) | 0.452 | 0.618 |

The histogram scheme clips rare large values, here the positive class logits
and the peaks of the box distributions, which costs about one point.
LibreYOLO therefore compiles with min-max ranges by default, except YOLO-NAS
pose and MobileNetV4, which keep more accuracy with histogram ranges
(`libreyolo.export.axelera.default_quantization_scheme(family, task)` names
the scheme); pass `compiler_config={"quantization_scheme": ...}` to choose
another.

Throughput, YOLO-NAS-s detect, 640x640, raw model calls on the device
(host quantize, AIPU, dequantize; measured while other jobs loaded the host):

| Submission | Images per second |
| --- | --- |
| one image per call | about 30 |
| `predict(..., batch=8)` path | about 48 |
| batch of 32, optimized runtime pipeline | about 73 |

Host preprocessing (about 17 ms per image) and the host decode plus NMS add
to the end-to-end time of a single `predict()` call. Latency tuning
(`resources_used`, core allocation, an optimized pipeline in the backend) is
future work.

## 8. How it works

The exported graph stops after the detection head's final convolutions. The
Metis AIPU runs the backbone, neck and head convolutions in INT8; LibreYOLO
then decodes the raw head outputs on the host (distribution-to-distance box
decode, grid anchors, sigmoids, keypoint offsets) and runs the usual NMS.
Box coordinates and probabilities never pass through an 8-bit tensor, which
keeps their resolution. The decode lives in `libreyolo/postprocess/yolonas_raw.py`
and is unit-tested against the in-graph decode. YOLO9 follows the same
path: its graph stops after each level's last box and class convolutions
(layout `yolo9_raw_heads_v1`) and the host rebuilds the decoded export's
tensor.

Integer quantization gives every tensor one scale over its calibrated range,
so a few extreme values can leave the useful ones only a handful of steps.
The export bounds such tensors inside the graph, with the scaling folded into
the convolutions and the clamp expressed as a fused ReLU6 or HardTanh:

- raw head outputs whose range dwarfs the useful part are emitted through a
  window (`output_windows` below; YOLO-NAS pose, YOLO9);
- in YOLO9 the class towers' second activation reaches about +540 while
  99.9% of it stays below about 50, so its pre-activation is clamped to
  [-32, 32]. Without the bound YOLO9-t lost 5.7 mAP50-95 on the device.

None of these bounds changes a float32 detection on COCO.

```text
image -> host preprocess -> [Metis: backbone, neck, head convs] -> raw heads
      -> host decode -> NMS -> Results
```

Classifiers (ResNet, MobileNetV4) need no graph cut: the whole network runs
on the AIPU and returns the logits (layout `classify_logits_v1`); the host
applies the softmax, with the same resize, crop and normalization as the
PyTorch model (recorded in `metadata.yaml`).

Dense models (super-resolution, depth, semantic segmentation, HRNet's
person-crop heatmaps) also run whole on the AIPU (layout `graph_outputs_v1`):
the device returns the same tensors as the family's ONNX export, under the
same names, and LibreYOLO's shared parsers turn them into results on the
host. The input is fixed at export: depth and semantic models stretch each
image to the canvas, super-resolution pads smaller images into it and
rejects larger ones. Calibration images go through that same preprocessing.

`metadata.yaml` holds LibreYOLO's export metadata (`schema_version`,
`model_family`, `task`, `names`, `imgsz`, ...) plus:

| Key | Meaning |
| --- | --- |
| `format` | `axelera` |
| `axm` | the `.axm` file name |
| `output_layout` | output contract, for example `yolonas_raw_heads_v1`, `yolox_raw_heads_v1`, `darknet_raw_heads_v1`, `yolo7_raw_heads_v1`, `ppyoloe_raw_heads_v1`, `classify_logits_v1` or `graph_outputs_v1` |
| `output_names` | output names in the order the compiled model returns them |
| `graph_outputs` | output names in graph order (`graph_outputs_v1`) |
| `output_windows` | parts emitted through a ReLU6 window, as `[low, high]` logit bounds (YOLO-NAS pose and YOLO9 exports); the backend maps them back before decoding |
| `strides`, `reg_max`, `grid_cell_offset` | decode constants (`reg_max` only for distribution heads) |
| `crop_pct`, `interpolation` | classifier eval transform |
| `anchors`, `scale_x_y` | anchor boxes in input pixels per level, and Darknet's centre scaling (anchor-based heads) |
| `axelera` | SDK version, quantization scheme, cores, resources, compiler overrides, calibration source and image count |
| `weights_license` | reminder of the weight license |

## 9. Troubleshooting

| Message | Cause | Fix |
| --- | --- | --- |
| `Axelera export requires Axelera's axelera-devkit package` | SDK not installed | run the printed `pip install` command, or set `LIBREYOLO_AUTOINSTALL=1` |
| `Could not run the model on an Axelera Metis device` | no device, driver mismatch, or the device is busy | run `axdevice`; check the driver against Axelera's compatibility matrix |
| `... exported with a fixed 640x640 input` | `imgsz` differs from the export | use the exported size or re-export |
| `... is not a LibreYOLO Axelera export` | directory from another tool | export with LibreYOLO |
| `Axelera compilation failed` | compiler error | the message names any saved error archive; retry with `compiler_config` tuning (section 4) |

## 10. Licensing

LibreYOLO's code is MIT. `axelera-devkit` and `axelera-rt` are proprietary
software under Axelera's EULA: you install them, and LibreYOLO never bundles
or redistributes them, Axelera's documentation or compiler build files.
YOLO-NAS pretrained weights are Deci's, under `LICENSE.YOLONAS.md`
(non-commercial use, no redistribution); models compiled from them carry the
same terms, so LibreYOLO does not host YOLO-NAS `.axm` files and each user
compiles their own.

## 11. Known limits and out of scope

Blocked for structural reasons (accuracy never blocks an export):

- EfficientNetV2-b1 and -b2: do not compile with SDK 1.9.0 (the compiled model has two output buffers for its one output at 224 and 240 px; at 256 px the squeeze-excitation pooling exceeds a reduction limit).
- HRNet (pose): compiles, but the device output drifts from the quantized model stage by stage (keypoint mAP 0.000 on device vs a working CPU-quantized model).
- ZipDepth: after an exact ReduceMean-to-pooling rewrite, quantization fails on the GCNet context MatMul.
- DeepLabV3-ResNet50/101: quantize, but at output stride 8 the ASPP convolutions need the whole 64x64 feature map and lowering finds no L1 tiling.
- RTMDet-s: compiles and runs, but the INT8 output collapses on a Metis (mAP50-95 0.079 vs FP32 0.411 on 300 COCO val2017 images, 0.136 without the pointwise equalization).

Not measured on a Metis yet (nothing is known to fail; the export raises until a size is measured):

- EfficientNetV2-b3, PicoDet-m and -l, RTMDet-m, -l and -x, MiDaS-l.
- YOLO9-E2E and YOLO9-P2.
- Other families.

Other known limits:

- YOLO-NAS-R (obb): validated on 300 DOTA-v2.0 tiles (the paired set prepared for this work), not on a 1000-image set.
- YOLOX-n: supported, mAP50-95 1.81 points below FP32 (1.66 with 128 calibration images; section 7).
- PP-YOLOE-s, -m and -x: supported, mAP50-95 1.2 to 1.7 points below FP32 (section 7). The loss is in the box-distribution outputs and does not come from their output range: windowing them, or histogram ranges, did not help. PP-YOLOE-l is validated.
- YOLOv4-b: supported, mAP50-95 1.52 points below FP32 with its recipe (section 7): its box outputs lose resolution in the Mish backbone; histogram ranges and a box-output window recover most of it (-6.8 to -1.52).
- YOLO-NAS-pose n and s: supported, keypoint mAP50-95 1.3 and 2.2 points below FP32 (section 7); m and l are validated.
- RF-DETR and other transformer families: attention, data-dependent TopK and
  grid sampling are outside what the compiler places on the AIPU, and the host
  can only run operators before or after the accelerated graph.
- MobileNetV4 (classify) s and l: supported, top-1 1.7 points below FP32
  at best (section 7). Per-tensor INT8 of the inverted-residual streams (the
  stage-2 stream of s spans about +-400 while its median channel stays
  within +-14) is the cause. Use m.
- Europa and multi-device scheduling: not covered.
- Large models on M.2 cards can hit the card's power limit.
