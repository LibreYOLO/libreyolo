# Axelera Metis export

> **PREVIEW.** Axelera Metis support is new in 1.8.0. Every size below
> compiles, runs on a Metis device and exports. "validated" means the INT8
> result is within 1.0 point of FP32 on the paired set; "supported" means it
> works with a larger measured drop, which the export logs once as a warning.
> Accuracy never blocks an export; "blocked" is only for families that do not
> compile, do not run or give meaningless output (`docs/export_support.md`).
> Every size runs its whole network on the Metis; the host only prepares the
> input and decodes the outputs (boxes, NMS, masks), plus the light numpy steps
> between compiled parts of SSD, FCOS and Real-ESRGAN (section 7). Exports that
> run part of the network itself on the host ("hybrid") are not supported on
> Metis: they are parked by decision 2026-11-19 on branch
> `axelera-hybrid-parked` (section 11).
> Results are from one Metis M.2 Max with axelera-devkit / axelera-rt 1.9.0.

Measured on device, paired INT8 vs FP32 on the same images, each size with the
recipe LibreYOLO applies to it (section 7 has calibration details). A row
labelled "option" is an alternative setting for a size listed above it, not
a separate size:

| Family | Size | Task | Dataset (paired) | FP32 | Metis INT8 | Delta | Label | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO-NAS | s | detect | COCO val2017, 1000 | 0.4549 | 0.4519 | -0.30 | validated | `yolonas/detect/s/2026-11-08/3` |
| YOLO-NAS | m | detect | COCO val2017, 1000 | 0.4942 | 0.4893 | -0.49 | validated | `yolonas/detect/m/2026-11-08` |
| YOLO-NAS | l | detect | COCO val2017, 1000 | 0.5036 | 0.5025 | -0.11 | validated | `yolonas/detect/l/2026-11-10` |
| YOLO-NAS-pose | n | pose | COCO val2017 persons, 300 | 0.7089 | 0.6959 | -1.30 | supported | `yolonas/pose/n/2026-11-11` |
| YOLO-NAS-pose | s | pose | COCO val2017 persons, 300 | 0.7565 | 0.7344 | -2.21 | supported | `yolonas/pose/s/2026-11-10/16` |
| YOLO-NAS-pose | m | pose | COCO val2017 persons, 300 | 0.7885 | 0.7821 | -0.64 | validated | `yolonas/pose/m/2026-11-11/2` |
| YOLO-NAS-pose | l | pose | COCO val2017 persons, 300 | 0.7922 | 0.7884 | -0.38 | validated | `yolonas/pose/l/2026-11-10/2` |
| HRNet | w32 | pose (256x192 person crops; 14 compiled parts, joined on the host by upsample + concat; slower than the CPU at batch 1 with 4 host threads: 3.69 vs 4.98 img/s) | COCO val2017 persons, 300 (1172 GT-box crops) | 0.7658 | 0.7566 | -0.92 | validated | `hrnet/pose/w32/2026-11-20/2` |
| YOLO-NAS-R | s | obb | DOTA-v2.0 val, 300 tiles | 0.3844 | 0.3747 | -0.97 | validated | `yolonas/obb/s/2026-11-10/2` |
| YOLO-NAS-R | m | obb | DOTA-v2.0 val, 300 tiles | 0.4141 | 0.4105 | -0.36 | validated | `yolonas/obb/m/2026-11-10` |
| YOLO-NAS-R | l | obb | DOTA-v2.0 val, 300 tiles | 0.4327 | 0.4348 | +0.21 | validated | `yolonas/obb/l/2026-11-10` |
| YOLO9 | t | detect | COCO val2017, 1000 | 0.3669 | 0.3624 | -0.45 | validated | `yolo9/detect/t/2026-11-10/7` |
| YOLO9 | s | detect | COCO val2017, 1000 | 0.4502 | 0.4450 | -0.52 | validated | `yolo9/detect/s/2026-11-10/2` |
| YOLO9 | m | detect | COCO val2017, 1000 | 0.4913 | 0.4896 | -0.17 | validated | `yolo9/detect/m/2026-11-10/2` |
| YOLO9 | c | detect | COCO val2017, 1000 | 0.5127 | 0.5093 | -0.34 | validated | `yolo9/detect/c/2026-11-10/2` |
| YOLO9-P2 | s | detect | VisDrone2019-DET val, 448 | 0.1785 | 0.1758 | -0.27 | validated | `yolo9_p2/detect/s/2026-11-15` |
| YOLO9-E2E | t | detect | COCO val2017, 1000 | 0.3329 | 0.3255 | -0.74 | validated | `yolo9_e2e/detect/t/2026-11-13` |
| YOLO9-E2E | s | detect | COCO val2017, 1000 | 0.4154 | 0.4104 | -0.50 | validated | `yolo9_e2e/detect/s/2026-11-13` |
| YOLO9-E2E | m | detect | COCO val2017, 1000 | 0.4695 | 0.4649 | -0.46 | validated | `yolo9_e2e/detect/m/2026-11-13` |
| YOLO9-E2E | c | detect | COCO val2017, 1000 | 0.4900 | 0.4851 | -0.49 | validated | `yolo9_e2e/detect/c/2026-11-13` |
| YOLOX | n | detect | COCO val2017, 300 | 0.2609 | 0.2428 | -1.81 | supported | `yolox/detect/n/2026-11-11/7` |
| YOLOX | t | detect | COCO val2017, 1000 | 0.3238 | 0.3202 | -0.36 | validated | `yolox/detect/t/2026-11-11/2` |
| YOLOX | s | detect | COCO val2017, 1000 | 0.3981 | 0.3933 | -0.48 | validated | `yolox/detect/s/2026-11-11/2` |
| YOLOX | m | detect | COCO val2017, 1000 | 0.4582 | 0.4516 | -0.66 | validated | `yolox/detect/m/2026-11-11/2` |
| YOLOX | l | detect | COCO val2017, 1000 | 0.4849 | 0.4781 | -0.68 | validated | `yolox/detect/l/2026-11-11/2` |
| YOLOX | x | detect | COCO val2017, 1000 | 0.4943 | 0.4867 | -0.76 | validated | `yolox/detect/x/2026-11-11/2` |
| YOLOv1 | b | detect | COCO val2017 (20 VOC classes), 1000 | 0.1638 | 0.1617 | -0.21 | validated | `yolo1/detect/b/2026-11-15/2` |
| YOLOv2 | t | detect | COCO val2017, 1000 | 0.1108 | 0.1097 | -0.11 | validated | `yolo2/detect/t/2026-11-13/2` |
| YOLOv2 | b | detect | COCO val2017, 1000 | 0.2947 | 0.2856 | -0.91 | validated | `yolo2/detect/b/2026-11-13/2` |
| YOLOv3 | t | detect | COCO val2017, 1000 | 0.1023 | 0.1006 | -0.17 | validated | `yolo3/detect/t/2026-11-11/6` |
| YOLOv3 | b | detect | COCO val2017, 1000 | 0.3656 | 0.3580 | -0.76 | validated | `yolo3/detect/b/2026-11-11/6` |
| YOLOv3 | spp | detect | COCO val2017, 1000 | 0.4304 | 0.4261 | -0.43 | validated | `yolo3/detect/spp/2026-11-11/3` |
| YOLOv4 | t | detect | COCO val2017, 1000 | 0.2021 | 0.1968 | -0.53 | validated | `yolo4/detect/t/2026-11-11/5` |
| YOLOv4 | b | detect | COCO val2017, 1000 | 0.4806 | 0.4654 | -1.52 | supported | `yolo4/detect/b/2026-11-11/6` |
| YOLOv7 | b | detect | COCO val2017, 1000 | 0.2843 | 0.2859 | +0.16 | validated | `yolo7/detect/b/2026-11-11/4` |
| PP-YOLOE | s | detect | COCO val2017, 1000 | 0.4085 | 0.3962 | -1.23 | supported | `ppyoloe/detect/s/2026-11-11/3` |
| PP-YOLOE | m | detect | COCO val2017, 1000 | 0.4538 | 0.4416 | -1.22 | supported | `ppyoloe/detect/m/2026-11-11/2` |
| PP-YOLOE | l | detect | COCO val2017, 1000 | 0.4771 | 0.4740 | -0.31 | validated | `ppyoloe/detect/l/2026-11-11/2` |
| PP-YOLOE | x | detect (histogram ranges) | COCO val2017, 1000 | 0.4918 | 0.4810 | -1.08 | supported | `ppyoloe/detect/x/2026-11-19/2` |
| PicoDet | s | detect | COCO val2017, 1000 | 0.2687 | 0.2453 | -2.34 | supported | `picodet/detect/s/2026-11-11/7` |
| PicoDet | m | detect | COCO val2017, 1000 | 0.3449 | 0.3163 | -2.86 | supported | `picodet/detect/m/2026-11-11/4` |
| PicoDet | l | detect | COCO val2017, 1000 | 0.3971 | 0.3801 | -1.70 | supported | `picodet/detect/l/2026-11-11/4` |
| RTMDet | t | detect | COCO val2017, 1000 | 0.3906 | 0.3677 | -2.29 | supported | `rtmdet/detect/t/2026-11-11/8` |
| RTMDet | s | detect (learned INT8 rounding of the pointwise convs) | COCO val2017, 1000 | 0.4121 | 0.3681 | -4.40 | supported | `rtmdet/detect/s/2026-11-17/8` |
| RTMDet | m | detect | COCO val2017, 1000 | 0.4727 | 0.4461 | -2.66 | supported | `rtmdet/detect/m/2026-11-12/2` |
| RTMDet | l | detect | COCO val2017, 1000 | 0.4805 | 0.4618 | -1.87 | supported | `rtmdet/detect/l/2026-11-12/2` |
| RTMDet | x | detect (5 compiled parts, joined on the host by upsample + concat) | COCO val2017, 1000 | 0.5128 | 0.4788 | -3.40 | supported | `rtmdet/detect/x/2026-11-20/4` |
| FCOS | r50 | detect (800x800 canvas; 6 compiled parts, GroupNorm on the host) | COCO val2017, 1000 | 0.3818 | 0.3405 | -4.13 | supported | `fcos/detect/r50/2026-11-15/3` |
| RetinaNet | r50 | detect (1024x1024 canvas, default) | COCO val2017, 1000 | 0.3472 | 0.3410 | -0.62 (canvas -0.49 more; -1.11 vs own resize) | supported | `retinanet/detect/r50/2026-11-14/4` |
| RetinaNet | r50 | detect (800x800 canvas, imgsz=800) | COCO val2017, 1000 | 0.3318 | 0.3296 | -0.22 (canvas -2.03 more; -2.25 vs own resize) | option (imgsz=800) | `retinanet/detect/r50/2026-11-14/3` |
| SSD | 300 | detect (two parts, L2Norm on the host) | COCO val2017, 1000 | 0.2408 | 0.2403 | -0.05 | validated | `ssd/detect/300/2026-11-13/2` |
| EfficientDet | d0 | detect | COCO val2017, 1000 | 0.3363 | 0.3185 | -1.78 | supported | `efficientdet/detect/d0/2026-11-12/3` |
| EfficientDet | d1 | detect (512x512 canvas) | COCO val2017, 1000 | 0.3639 | 0.3475 | -1.64 (canvas -3.10 more) | supported | `efficientdet/detect/d1/2026-11-13/3` |
| EfficientDet | d2 | detect (512x512 canvas) | COCO val2017, 1000 | 0.3744 | 0.3606 | -1.38 (canvas -5.31 more) | supported | `efficientdet/detect/d2/2026-11-14` |
| RTMDet-Ins | t | segment (mask mAP50-95) | COCO val2017, 1000 | 0.3394 | 0.3250 | -1.44 | supported | `rtmdet/segment/t/2026-11-12/2` |
| RTMDet-Ins | s | segment (mask mAP50-95; learned INT8 rounding of the pointwise convs) | COCO val2017, 1000 | 0.3719 | 0.3392 | -3.27 (box -5.02) | supported | `rtmdet/segment/s/2026-11-19/4` |
| RTMDet-Ins | m | segment (mask mAP50-95) | COCO val2017, 1000 | 0.4011 | 0.3932 | -0.79 (box -3.96) | supported | `rtmdet/segment/m/2026-11-12/2` |
| RTMDet-Ins | l | segment (mask mAP50-95) | COCO val2017, 1000 | 0.4163 | 0.4113 | -0.50 (box -5.34) | supported | `rtmdet/segment/l/2026-11-12/2` |
| ResNet | 18 | classify (top-1) | ImageNetV2, 1000 | 0.595 | 0.589 | -0.6 | validated | `resnet/classify/18/2026-11-10/2` |
| ResNet | 34 | classify (top-1) | ImageNetV2, 1000 | 0.657 | 0.647 | -1.0 | validated | `resnet/classify/34/2026-11-10` |
| ResNet | 50 | classify (top-1) | ImageNetV2, 1000 | 0.688 | 0.681 | -0.7 | validated | `resnet/classify/50/2026-11-10` |
| ResNet | 101 | classify (top-1) | ImageNetV2, 1000 | 0.713 | 0.708 | -0.5 | validated | `resnet/classify/101/2026-11-10` |
| MobileNetV4 | s | classify (top-1; learned INT8 rounding of every layer) | ImageNetV2, 1000 | 0.624 | 0.617 | -0.7 | validated | `mobilenetv4/classify/s/2026-11-20/6` |
| MobileNetV4 | m | classify (top-1) | ImageNetV2, 1000 | 0.682 | 0.684 | +0.2 | validated | `mobilenetv4/classify/m/2026-11-11/2` |
| MobileNetV4 | l | classify (top-1) | ImageNetV2, 1000 | 0.727 | 0.710 | -1.7 | supported | `mobilenetv4/classify/l/2026-11-10` |
| EfficientNetV2 | b0 | classify (top-1) | ImageNetV2, 1000 | 0.667 | 0.656 | -1.1 | supported | `efficientnetv2/classify/b0/2026-11-11/5` |
| EfficientNetV2 | b1 | classify (top-1; 240 px; stage-0 skip as identity conv) | ImageNetV2, 1000 | 0.689 | 0.687 | -0.2 | validated | `efficientnetv2/classify/b1/2026-11-17` |
| EfficientNetV2 | b2 | classify (top-1; 260 px; stage-0 skip as identity conv) | ImageNetV2, 1000 | 0.706 | 0.693 | -1.3 | supported | `efficientnetv2/classify/b2/2026-11-17` |
| EfficientNetV2 | b3 | classify (top-1; 300 px; stage-0 skip as identity conv) | ImageNetV2, 1000 | 0.715 | 0.726 | +1.1 | validated | `efficientnetv2/classify/b3/2026-11-17` |
| VGG | 16 | classify (top-1) | ImageNetV2, 1000 | 0.608 | 0.606 | -0.2 | validated | `vgg/classify/16/2026-11-12` |
| VGG | 19 | classify (top-1) | ImageNetV2, 1000 | 0.610 | 0.619 | +0.9 | validated | `vgg/classify/19/2026-11-12` |
| VGG | 16bn | classify (top-1) | ImageNetV2, 1000 | 0.633 | 0.631 | -0.2 | validated | `vgg/classify/16bn/2026-11-12/2` |
| VGG | 19bn | classify (top-1) | ImageNetV2, 1000 | 0.644 | 0.593 | -5.1 | supported | `vgg/classify/19bn/2026-11-12` |
| AlexNet | b | classify (top-1) | ImageNetV2, 1000 | 0.434 | 0.440 | +0.6 | validated | `alexnet/classify/b/2026-11-13` |
| FCN | r50 | semantic (mIoU) | COCO val2017 VOC-21, 300 | 0.6515 | 0.6492 | -0.23 | validated | `fcn/semantic/r50/2026-11-11/2` |
| FCN | r101 | semantic (mIoU) | COCO val2017 VOC-21, 300 | 0.6798 | 0.6779 | -0.19 | validated | `fcn/semantic/r101/2026-11-11` |
| QuickSRNet | m2 | restore 2x (PSNR, dB) | COCO crops from 128 px, 296 | 29.12 | 28.92 | -0.20 | validated | `quicksrnet/restore/m2/2026-11-16/5` |
| Real-ESRGAN | x4 | restore 4x (PSNR, dB) | COCO crops from 64 px, 296 | 22.49 | 22.44 | -0.05 | validated | `realesrgan/restore/x4/2026-11-16` |
| Real-ESRGAN | x4t | restore 4x (PSNR, dB) | COCO crops from 64 px, 296 | 22.49 | 22.30 | -0.19 | validated | `realesrgan/restore/x4t/2026-11-16` |
| Real-ESRGAN | x2 | restore 2x (PSNR, dB; three compiled parts) | COCO crops from 128 px, 296 | 25.07 | 24.73 | -0.34 | supported | `realesrgan/restore/x2/2026-11-16` |
| MiDaS | s | depth (agreement with FP32) | COCO val2017 at 256 px, 300 | reference | AbsRel 0.069, delta1 0.94 | AbsRel 0.069 (bar 0.02) | supported | `midas/depth/s/2026-11-11/4` |
| DexiNed | b | edge (edge-pixel F1 vs FP32 edges at 0.5; 352x352) | COCO val2017 at 352 px, 300 | 1.0000 | 0.9651 | -0.0349 (cosine 0.997) | supported | `dexined/edge/b/2026-11-16` |
| Face recognition | librefacerec-l | embed (embedding cosine vs the FP32 head; head only: aligned 112x112 face in, 512-d embedding out, detection, alignment and L2 normalization on the host; the 25088-input embedding layer as row convolutions) | COCO val2017 aligned faces, 890 (336 images) | 1.0000 | 0.9933 | -0.0067 (pairwise similarity diff 0.0064; all same/different decisions at 0.4 kept) | supported | `facerec/embed/l/2026-11-20/3` |
| DeepLabV3 | mv3 | semantic (mIoU) | COCO val2017 VOC-21, 300 | 0.5230 | 0.4853 | -3.77 | supported | `deeplabv3/semantic/mv3/2026-11-11/6` |
| DeepLabV3 | r50 | semantic (mIoU; ASPP as per-tap 1x1 convolutions plus shifts) | COCO val2017 VOC-21, 300 | 0.5243 | 0.4734 | -5.09 | supported | `deeplabv3/semantic/r50/2026-11-17/2` |
| DeepLabV3 | r101 | semantic (mIoU; ASPP as per-tap 1x1 convolutions plus shifts) | COCO val2017 VOC-21, 300 | 0.5888 | 0.3795 | -20.93 | supported | `deeplabv3/semantic/r101/2026-11-14` |
| UNet | s | semantic (fidelity mIoU vs FP32 labels; 512x1024; histogram ranges) | COCO val2017, 1000 | 1.0000 | 0.8562 | -0.1438 | supported | `unet/semantic/s/2026-11-20` |
| PP-LiteSeg | t50 | semantic (fidelity mIoU vs FP32 labels; 512x1024; UAFM blends as x*sigmoid(z) + skip*sigmoid(-z)) | COCO val2017, 1000 | 1.0000 | 0.8502 | -0.1498 | supported | `ppliteseg/semantic/t50/2026-11-18` |
| PP-LiteSeg | b50 | semantic (fidelity mIoU vs FP32 labels; 512x1024; UAFM blends as x*sigmoid(z) + skip*sigmoid(-z)) | COCO val2017, 1000 | 1.0000 | 0.8571 | -0.1429 | supported | `ppliteseg/semantic/b50/2026-11-18` |
| PP-LiteSeg | t75 | semantic (fidelity mIoU vs FP32 labels; 768x1536; UAFM blends as x*sigmoid(z) + skip*sigmoid(-z)) | COCO val2017, 1000 | 1.0000 | 0.8068 | -0.1932 | supported | `ppliteseg/semantic/t75/2026-11-18` |
| PP-LiteSeg | b75 | semantic (fidelity mIoU vs FP32 labels; 768x1536; UAFM blends as x*sigmoid(z) + skip*sigmoid(-z)) | COCO val2017, 1000 | 1.0000 | 0.8484 | -0.1516 | supported | `ppliteseg/semantic/b75/2026-11-18` |
| PicoSAM3 | pico | segment (raw ROI CNN: roi_image [1,3,96,96] -> mask_logits [1,1,96,96]; mask IoU vs FP32 masks at logit 0 on ground-truth-box ROIs; stem depthwise conv as an exact dense conv) | COCO val2017, 297 images, 2360 ROIs | 1.0000 | 0.9683 | -0.0317 (pixel agreement 0.992, logit cosine 0.9985) | supported | `picosam3/segment/pico/2026-11-19` |

Detection and pose report mAP50-95; ImageNetV2 is used locally under its
licence terms and not redistributed. Evidence is the id of the run's line in
the project's results ledger (family/task/size/date), which holds its
metrics, logs, recipe and code commit.

`format="axelera"` compiles a LibreYOLO model to an INT8 `.axm` for Axelera's
Metis AIPU, and `LibreYOLO("<stem>_axelera_model")` runs it with the same
`predict` and `val` calls (Python and CLI) as the PyTorch model.

## 1. Scope and status

| Family | Task | Sizes | Tier | Evidence (ledger id per size) |
| --- | --- | --- | --- | --- |
| YOLO-NAS | detect | s, m, l | see `docs/export_support.md` | s `yolonas/detect/s/2026-11-08/3`; m `yolonas/detect/m/2026-11-08`; l `yolonas/detect/l/2026-11-10` |
| YOLO-NAS | pose | n, s, m, l (n and s supported with a measured drop) | see `docs/export_support.md` | n `yolonas/pose/n/2026-11-11`; s `yolonas/pose/s/2026-11-10/16`; m `yolonas/pose/m/2026-11-11/2`; l `yolonas/pose/l/2026-11-10/2` |
| HRNet | pose | w32 (14 compiled parts; w48 not measured yet) | see `docs/export_support.md` | w32 `hrnet/pose/w32/2026-11-20/2`, speed `hrnet/pose/w32/2026-11-20/6` |
| YOLO-NAS-R | obb | s, m, l | see `docs/export_support.md` (calibration data required) | s `yolonas/obb/s/2026-11-10/2`; m `yolonas/obb/m/2026-11-10`; l `yolonas/obb/l/2026-11-10` |
| YOLO9 | detect | t, s, m, c | see `docs/export_support.md` | t `yolo9/detect/t/2026-11-10/7`; s `yolo9/detect/s/2026-11-10/2`; m `yolo9/detect/m/2026-11-10/2`; c `yolo9/detect/c/2026-11-10/2` |
| YOLO9-P2 | detect | s (t: no published checkpoint) | see `docs/export_support.md` | s `yolo9_p2/detect/s/2026-11-15` |
| YOLO9-E2E | detect | t, s, m, c | see `docs/export_support.md` | t `yolo9_e2e/detect/t/2026-11-13`; s `yolo9_e2e/detect/s/2026-11-13`; m `yolo9_e2e/detect/m/2026-11-13`; c `yolo9_e2e/detect/c/2026-11-13` |
| ResNet | classify | 18, 34, 50, 101 | see `docs/export_support.md` | 18 `resnet/classify/18/2026-11-10/2`; 34 `resnet/classify/34/2026-11-10`; 50 `resnet/classify/50/2026-11-10`; 101 `resnet/classify/101/2026-11-10` |
| MobileNetV4 | classify | s, m, l (l supported with a measured drop) | see `docs/export_support.md` | s `mobilenetv4/classify/s/2026-11-20/6`, held-out `mobilenetv4/classify/s/2026-11-20/7`; m `mobilenetv4/classify/m/2026-11-11/2`; l `mobilenetv4/classify/l/2026-11-10` |
| EfficientNetV2 | classify | b1, b3 (validated); b0, b2 (supported with measured drops); b1-b3 read their stage-0 residual skip through an exact identity 1x1 conv | see `docs/export_support.md` | b0 `efficientnetv2/classify/b0/2026-11-11/5`; b1 `efficientnetv2/classify/b1/2026-11-17`; b2 `efficientnetv2/classify/b2/2026-11-17`; b3 `efficientnetv2/classify/b3/2026-11-17` |
| VGG | classify | 16, 19, 16bn; 19bn (supported with a measured drop, -5.1) | see `docs/export_support.md` | 16 `vgg/classify/16/2026-11-12`; 19 `vgg/classify/19/2026-11-12`; 16bn `vgg/classify/16bn/2026-11-12/2`; 19bn `vgg/classify/19bn/2026-11-12` |
| AlexNet | classify | b | see `docs/export_support.md` | b `alexnet/classify/b/2026-11-13` |
| FCN | semantic | r50, r101 | see `docs/export_support.md` | r50 `fcn/semantic/r50/2026-11-11/2`; r101 `fcn/semantic/r101/2026-11-11` |
| QuickSRNet | restore | m2 (learned INT8 rounding; held-out half -0.17 dB) | see `docs/export_support.md` | m2 `quicksrnet/restore/m2/2026-11-16/5`, held-out `quicksrnet/restore/m2/2026-11-16/6` |
| Real-ESRGAN | restore | x4 (three compiled parts + a host add), x4t (equalized trunk; held-out half -0.20 dB); x2 (three compiled parts, supported with a measured drop) | see `docs/export_support.md` | x4 `realesrgan/restore/x4/2026-11-16`; x4t `realesrgan/restore/x4t/2026-11-16`, held-out `realesrgan/restore/x4t/2026-11-16/2`; x2 `realesrgan/restore/x2/2026-11-16` |
| MiDaS | depth | s (supported with a measured drop; l blocked: does not quantize, section 11) | see `docs/export_support.md` | s `midas/depth/s/2026-11-11/4` |
| DexiNed | edge | b (supported with a measured fidelity; whole graph; the user supplies the upstream BIPED-trained checkpoint, non-commercial) | see `docs/export_support.md` | b `dexined/edge/b/2026-11-16` |
| Face recognition | embed | librefacerec-l head (supported with a measured fidelity; `LibreFaceEmbedder.export(format="axelera")` compiles the recognition head under the aligned-face contract, the detector stays on the host; `LibreYOLO("librefacerec-l_axelera_model")` runs it) | not in `docs/export_support.md` (the face embedder is outside the export matrix) | l `facerec/embed/l/2026-11-20/3` |
| DeepLabV3 | semantic | mv3, r50, r101 (supported with measured drops: -3.77, -5.09, -20.93) | see `docs/export_support.md` | mv3 `deeplabv3/semantic/mv3/2026-11-11/6`; r50 `deeplabv3/semantic/r50/2026-11-17/2`; r101 `deeplabv3/semantic/r101/2026-11-14` |
| UNet | semantic | s (supported with a measured fidelity; whole graph at the 512x1024 training crop with histogram ranges, compile about 11-13 h) | see `docs/export_support.md` | s `unet/semantic/s/2026-11-20` |
| PP-LiteSeg | semantic | t50, b50, t75, b75 (supported with a measured fidelity; whole graph at 512x1024 (t50, b50) or 768x1536 (t75, b75), the UAFM blends in an exact INT8-friendly form) | see `docs/export_support.md` | t50 `ppliteseg/semantic/t50/2026-11-18`; b50 `ppliteseg/semantic/b50/2026-11-18`; t75 `ppliteseg/semantic/t75/2026-11-18`; b75 `ppliteseg/semantic/b75/2026-11-18` |
| PicoSAM3 | segment | pico (supported with a measured fidelity; the raw 96x96 ROI CNN, the same contract as the ONNX export: cutting the ROI from a box and pasting the mask logits back stay in Python) | see `docs/export_support.md` | pico `picosam3/segment/pico/2026-11-19` |
| YOLOX | detect | n, t, s, m, l, x (n supported with a measured drop) | see `docs/export_support.md` | n `yolox/detect/n/2026-11-11/7`; t `yolox/detect/t/2026-11-11/2`; s `yolox/detect/s/2026-11-11/2`; m `yolox/detect/m/2026-11-11/2`; l `yolox/detect/l/2026-11-11/2`; x `yolox/detect/x/2026-11-11/2` |
| YOLOv1 | detect | b | see `docs/export_support.md` | b `yolo1/detect/b/2026-11-15/2` |
| YOLOv2 | detect | t, b | see `docs/export_support.md` | t `yolo2/detect/t/2026-11-13/2`; b `yolo2/detect/b/2026-11-13/2` |
| YOLOv3 | detect | t, b, spp | see `docs/export_support.md` | t `yolo3/detect/t/2026-11-11/6`; b `yolo3/detect/b/2026-11-11/6`; spp `yolo3/detect/spp/2026-11-11/3` |
| YOLOv4 | detect | t, b (b supported with a measured drop) | see `docs/export_support.md` | t `yolo4/detect/t/2026-11-11/5`; b `yolo4/detect/b/2026-11-11/6` |
| YOLOv7 | detect | b | see `docs/export_support.md` | b `yolo7/detect/b/2026-11-11/4` |
| PP-YOLOE | detect | s, m, l, x (s, m and x supported with a measured drop) | see `docs/export_support.md` | s `ppyoloe/detect/s/2026-11-11/3`; m `ppyoloe/detect/m/2026-11-11/2`; l `ppyoloe/detect/l/2026-11-11/2`; x `ppyoloe/detect/x/2026-11-19/2` |
| PicoDet | detect | s, m, l (supported with a measured drop) | see `docs/export_support.md` | s `picodet/detect/s/2026-11-11/7`; m `picodet/detect/m/2026-11-11/4`; l `picodet/detect/l/2026-11-11/4` |
| RTMDet | detect | t, s, m, l, x (supported with a measured drop; x as five compiled parts) | see `docs/export_support.md` | t `rtmdet/detect/t/2026-11-11/8`; s `rtmdet/detect/s/2026-11-17/8`, held-out `rtmdet/detect/s/2026-11-17/9`; m `rtmdet/detect/m/2026-11-12/2`; l `rtmdet/detect/l/2026-11-12/2`; x `rtmdet/detect/x/2026-11-20/4` |
| RTMDet-Ins | segment | t, s, m, l (supported with a measured drop, box -2.51, -5.02, -3.96 and -5.34; x blocked: does not compile) | see `docs/export_support.md` | t `rtmdet/segment/t/2026-11-12/2`; s `rtmdet/segment/s/2026-11-19/4`; m `rtmdet/segment/m/2026-11-12/2`; l `rtmdet/segment/l/2026-11-12/2`; x blocked `rtmdet/segment/x/2026-11-12` |
| FCOS | detect | r50 (supported with a measured drop; six compiled parts, GroupNorm and the decode on the host) | see `docs/export_support.md` | r50 `fcos/detect/r50/2026-11-15/3` |
| RetinaNet | detect | r50 (supported: 1024x1024 canvas by default, -1.11 end to end; imgsz=800 is about 1.64x less compute at -2.25; r50v2 blocked: GroupNorm heads) | see `docs/export_support.md` | r50 `retinanet/detect/r50/2026-11-14/4`; r50v2 blocked `retinanet/detect/r50v2/2026-11-14` |
| SSD | detect | 300 (two compiled parts, conv4_3 L2Norm on the host) | see `docs/export_support.md` | 300 `ssd/detect/300/2026-11-13/2` |
| EfficientDet | detect | d0, d1, d2 (supported with measured drops; d1 and d2 at a fixed 512x512 canvas; d3 and d4 blocked: do not compile) | see `docs/export_support.md` | d0 `efficientdet/detect/d0/2026-11-12/3`; d1 `efficientdet/detect/d1/2026-11-13/3`; d2 `efficientdet/detect/d2/2026-11-14`; d3 blocked `efficientdet/detect/d3/2026-11-14`; d4 blocked `efficientdet/detect/d4/2026-11-15/4` |
| Other families | all | all | not measured yet (section 11) | - |
| HRNet-w48 (not measured as parts yet), DEKR, ZipDepth, PIDNet, CenterNet, Faster R-CNN, Mask R-CNN, DETR, Deformable DETR, DINO-DETR, RT-DETR, RT-DETRv2, RiO-DETRv2, D-FINE, DEIM, DEIMv2, RT-DETRv4, PP-DocLayoutV3, TinyFormer; RetinaNet-r50v2, EfficientDet-d3/d4 | as measured | none | not supported on Metis: part of the network runs on the host (hybrid parked by decision 2026-11-19; branch axelera-hybrid-parked); section 11 | - |
| RF-DETR and other transformer-backbone families | all | all | out of scope (section 11) | - |

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
| `fraction` | Share of the calibration split to use (its first images). 100 to 400 images are recommended; fewer than 100 logs a warning. The COCO results in section 7 ("64 coco128") use `data="coco128.yaml", fraction=0.5`. |
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

The host steps dominate end-to-end time on this hardware, so a list of
images, a directory or a video file runs 8 images per call by default
(`npu(images)` is `npu(images, batch=8)`; set `batch=N` for another size and
`batch=1` to run image by image). The N images go to the device in one
submission, and their preprocessing, decode and `Results` building run in
parallel on host threads (`OMP_NUM_THREADS`, by default up to 4). With a list
or a directory, the next image or chunk is preprocessed while the device runs
the current one (single-part exports). A single image and live streams
(webcam, RTSP, screen) still run one frame at a time. The outputs are
identical to processing the images one at a time.

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
The ledger id behind each row is in the Evidence column of the summary
table at the top of this page. In the Calibration column, "64 coco128" is the
first 64 images of coco128 (`data="coco128.yaml", fraction=0.5`); the default
`fraction=1.0` uses all 128. Most sizes move little with the count (YOLOX-n:
-1.81 with 64 images, -1.66 with 128). RTMDet-s was sensitive to it until
its learned rounding (see the RTMDet paragraph below); Faster R-CNN-n and -s
still are, in opposite directions (their rows).

| Model | Task | Image set (paired) | Calibration | FP32 | Metis INT8 | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO-NAS-s | detect | COCO val2017, 1000 images | 64 coco128 | 0.4549 | 0.4519 | -0.30 |
| YOLO-NAS-m | detect | COCO val2017, 1000 images | 128 coco128 | 0.4942 | 0.4893 | -0.49 |
| YOLO-NAS-l | detect | COCO val2017, 1000 images | 64 coco128 | 0.5036 | 0.5025 | -0.11 |
| YOLO-NAS-pose-n | pose | COCO val2017 persons, 300 images | 128 coco128 | 0.7089 | 0.6959 | -1.30 (supported) |
| YOLO-NAS-pose-s | pose | COCO val2017 persons, 300 images | 128 coco128 | 0.7565 | 0.7344 | -2.21 (supported; class and keypoint towers bounded) |
| YOLO-NAS-pose-m | pose | COCO val2017 persons, 300 images | 128 coco128 | 0.7885 | 0.7821 | -0.64 |
| YOLO-NAS-pose-l | pose | COCO val2017 persons, 300 images | 128 coco128 | 0.7922 | 0.7884 | -0.38 |
| HRNet-w32 (14 compiled parts) | pose | COCO val2017 persons, 300 images (1172 GT-box crops) | COCO person crops | 0.7658 | 0.7566 | -0.92 |
| YOLO-NAS-R-s | obb | DOTA-v2.0 val, 300 tiles | 64 DOTA tiles | 0.3844 | 0.3747 | -0.97 |
| YOLO-NAS-R-m | obb | DOTA-v2.0 val, 300 tiles | 64 DOTA tiles | 0.4141 | 0.4105 | -0.36 |
| YOLO-NAS-R-l | obb | DOTA-v2.0 val, 300 tiles | 64 DOTA tiles | 0.4327 | 0.4348 | +0.21 |
| YOLO9-t | detect | COCO val2017, 1000 images | 64 coco128 | 0.3669 | 0.3624 | -0.45 |
| YOLO9-s | detect | COCO val2017, 1000 images | 64 coco128 | 0.4502 | 0.4450 | -0.52 |
| YOLO9-m | detect | COCO val2017, 1000 images | 64 coco128 | 0.4913 | 0.4896 | -0.17 |
| YOLO9-c | detect | COCO val2017, 1000 images | 64 coco128 | 0.5127 | 0.5093 | -0.34 |
| YOLO9-P2-s | detect | VisDrone2019-DET val, 448 images | 100 other VisDrone val images | 0.1785 | 0.1758 | -0.27 |
| YOLO9-E2E-t | detect | COCO val2017, 1000 images | 128 coco128 | 0.3329 | 0.3255 | -0.74 |
| YOLO9-E2E-s | detect | COCO val2017, 1000 images | 128 coco128 | 0.4154 | 0.4104 | -0.50 |
| YOLO9-E2E-m | detect | COCO val2017, 1000 images | 128 coco128 | 0.4695 | 0.4649 | -0.46 |
| YOLO9-E2E-c | detect | COCO val2017, 1000 images | 128 coco128 | 0.4900 | 0.4851 | -0.49 |
| YOLOX-n | detect | COCO val2017, 300 images | 64 coco128 | 0.2609 | 0.2428 | -1.81 (supported; -1.66 with 128 calibration images) |
| YOLOX-t | detect | COCO val2017, 1000 images | 64 coco128 | 0.3238 | 0.3202 | -0.36 |
| YOLOX-s | detect | COCO val2017, 1000 images | 64 coco128 | 0.3981 | 0.3933 | -0.48 |
| YOLOX-m | detect | COCO val2017, 1000 images | 64 coco128 | 0.4582 | 0.4516 | -0.66 |
| YOLOX-l | detect | COCO val2017, 1000 images | 64 coco128 | 0.4849 | 0.4781 | -0.68 |
| YOLOX-x | detect | COCO val2017, 1000 images | 64 coco128 | 0.4943 | 0.4867 | -0.76 |
| YOLOv1-b | detect | COCO val2017 (20 VOC classes), 1000 images | 128 coco128 | 0.1638 | 0.1617 | -0.21 |
| YOLOv2-t | detect | COCO val2017, 1000 images | 128 coco128 | 0.1108 | 0.1097 | -0.11 (+0.41 on 300) |
| YOLOv2-b | detect | COCO val2017, 1000 images | 128 coco128 | 0.2947 | 0.2856 | -0.91 (-1.15 on 300) |
| YOLOv3-t | detect | COCO val2017, 1000 images | 64 coco128 | 0.1023 | 0.1006 | -0.17 |
| YOLOv3-b | detect | COCO val2017, 1000 images | 64 coco128 | 0.3656 | 0.3580 | -0.76 |
| YOLOv3-spp | detect | COCO val2017, 1000 images | 64 coco128 | 0.4304 | 0.4261 | -0.43 |
| YOLOv4-t | detect | COCO val2017, 1000 images | 64 coco128 | 0.2021 | 0.1968 | -0.53 |
| YOLOv4-b | detect | COCO val2017, 1000 images | 64 coco128 | 0.4806 | 0.4654 | -1.52 (supported; histogram ranges + [-6, 6] box window) |
| YOLOv7 | detect | COCO val2017, 1000 images | 64 coco128 | 0.2843 | 0.2859 | +0.16 |
| PP-YOLOE-s | detect | COCO val2017, 1000 images | 64 coco128 | 0.4085 | 0.3962 | -1.23 (supported) |
| PP-YOLOE-m | detect | COCO val2017, 1000 images | 64 coco128 | 0.4538 | 0.4416 | -1.22 (supported) |
| PP-YOLOE-l | detect | COCO val2017, 1000 images | 64 coco128 | 0.4771 | 0.4740 | -0.31 |
| PP-YOLOE-x (histogram ranges) | detect | COCO val2017, 1000 images | 64 coco128 | 0.4918 | 0.4810 | -1.08 (supported; -1.23 on 300, was -2.41 with min-max) |
| PicoDet-s | detect | COCO val2017, 1000 images | 64 coco128 | 0.2687 | 0.2453 | -2.34 (supported; -1.28 on 300) |
| PicoDet-m | detect | COCO val2017, 1000 images | 64 coco128 | 0.3449 | 0.3163 | -2.86 (supported) |
| PicoDet-l | detect | COCO val2017, 1000 images | 64 coco128 | 0.3971 | 0.3801 | -1.70 (supported) |
| RTMDet-t | detect | COCO val2017, 1000 images | 64 coco128 | 0.3906 | 0.3677 | -2.29 (supported) |
| RTMDet-s | detect | COCO val2017, 1000 images | 64 coco128 | 0.4121 | 0.3681 | -4.40 (supported; -5.43 on 300, -5.15 on a held-out 1000) |
| RTMDet-m | detect | COCO val2017, 1000 images | 64 coco128 | 0.4727 | 0.4461 | -2.66 (supported; -2.11 on 300) |
| RTMDet-l | detect | COCO val2017, 1000 images | 64 coco128 | 0.4805 | 0.4618 | -1.87 (supported; -1.92 on 300) |
| RTMDet-x (5 compiled parts) | detect | COCO val2017, 1000 images | 64 coco128 | 0.5128 | 0.4788 | -3.40 (supported; -2.92 on 300) |
| RetinaNet-R50 (1024x1024 canvas, default) | detect | COCO val2017, 1000 images | 128 coco128 | 0.3472 | 0.3410 | -0.62 (supported: -1.11 vs own resize) |
| RetinaNet-R50 (800x800 canvas, imgsz=800) | detect | COCO val2017, 1000 images | 64 coco128 | 0.3318 | 0.3296 | -0.22 (supported: canvas -2.03) |
| SSD300 (two parts) | detect | COCO val2017, 1000 images | 128 coco128 | 0.2408 | 0.2403 | -0.05 (+0.09 on 300) |
| EfficientDet-d0 | detect | COCO val2017, 1000 images | 128 coco128 | 0.3363 | 0.3185 | -1.78 (supported) |
| EfficientDet-d1 (512x512 canvas) | detect | COCO val2017, 1000 images | 128 coco128 | 0.3639 | 0.3475 | -1.64 (supported; canvas -3.10 more) |
| EfficientDet-d2 (512x512 canvas) | detect | COCO val2017, 1000 images | 128 coco128 | 0.3744 | 0.3606 | -1.38 (supported; canvas -5.31 more) |
| RTMDet-Ins-t | segment (mask) | COCO val2017, 1000 images | 64 coco128 | 0.3394 | 0.3250 | -1.44 (supported; box -2.51) |
| RTMDet-Ins-s | segment (mask) | COCO val2017, 1000 images | 64 coco128 | 0.3719 | 0.3392 | -3.27 (supported; box -5.02; on 300 mask -3.20, box -5.19) |
| RTMDet-Ins-m | segment (mask) | COCO val2017, 1000 images | 64 coco128 | 0.4011 | 0.3932 | -0.79 (supported: box -3.96; on 300 mask -0.67, box -4.56) |
| RTMDet-Ins-l | segment (mask) | COCO val2017, 1000 images | 64 coco128 | 0.4163 | 0.4113 | -0.50 (supported: box -5.34; on 300 mask -0.58, box -5.75) |
| RTMDet-Ins-x | segment (mask) | COCO val2017, 1000 images | 64 coco128 | - | - | does not compile (blocked) |

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
| VGG-16 | 224 | 0.608 | 0.606 | -0.2 | 7 min |
| VGG-19 | 224 | 0.610 | 0.619 | +0.9 | - |
| VGG-16-BN (histogram ranges) | 224 | 0.633 | 0.631 | -0.2 | - |
| VGG-19-BN | 224 | 0.644 | 0.593 | -5.1 | - |
| AlexNet | 224 | 0.434 | 0.440 | +0.6 | 3 min |

Calibrating a classifier on coco128 (the fallback when `data=` is omitted)
cost ResNet-18 1.1 top-1 points instead of 0.6, and the histogram scheme 2.0.

VGG's first fully connected layer reduces over the whole 7x7x512 map (25088
values), more than the compiler maps in one piece; the export runs it as one
1x7 convolution per row of the map, summed, and the other two layers as 1x1
convolutions (an exact graph pass requested by VGG only; heads that pool to
1x1 first, like ResNet's and MobileNet's, are left unchanged). VGG-16 top-5
is 0.812 vs FP32 0.819. The batch-norm variants fold the norm into the
convolutions, which widens their min-max ranges: VGG-16-BN compiles with
histogram ranges (-2.9 with min-max, -0.2 with histogram).

VGG-19-BN runs whole on the Metis and is supported with a measured drop: top-1
0.593 vs 0.644 FP32 on the 1000 images (-5.1) with min-max ranges, -6.0 with
histogram ranges, although the CPU-quantized model loses only 2.4 and 1.9: the
device output drifts from the CPU-quantized model layer by layer through conv
block 5 (output cosine 0.984 at the fourth pool, 0.785 after the last
convolution), with no single diverging operator (ledger
`vgg/classify/19bn/2026-11-12`). Running block 5 and the classifier on the
host recovers it (-0.2), but that is a hybrid and is parked (section 11).

MobileNetV4 needs more than the ResNet recipe: with plain min-max ranges
size s lost 6.2 top-1 points. LibreYOLO equalizes its residual streams per
channel and bounds the stage-2 stream before export (both nearly
FP32-neutral; largest logit change on the calibration images 0.1 on m),
then calibrates with histogram activation ranges
(`hybrid_per_tensor_per_channel`):

| Model | Input | FP32 top-1 | Metis INT8 top-1 | Delta | Compile (28 W host) |
| --- | --- | --- | --- | --- | --- |
| MobileNetV4-m | 224 | 0.682 | 0.684 | +0.2 | 3.1 min |
| MobileNetV4-s | 224 | 0.624 | 0.617 | -0.7 | 6.3 min |

MobileNetV4-s takes another route: histogram ranges, no stream transform,
and learned INT8 rounding of every layer (per-size recipe
`network_learned_rounding`, 64 calibration images, about 4 minutes at
export). For each convolution and the classifier, layer by layer, the export
chooses whether each weight rounds up or down on the per-channel INT8 grid
the compiler uses, so that the layer's output matches the FP32 network's
output for that layer on calibration images (in the style of AdaRound; the
remaining mean shift goes into the batch norm). With nearest rounding s
scored 0.607 (-1.7); with the learned rounding 0.617 (-0.7), and -0.9 on a
held-out 1000-image ImageNetV2 set (0.603 vs 0.612; nearest rounding
0.593). Matching each layer to
the FP32 network's own layer output matters: matching the FP32 weights on the
rounded network's inputs gained nothing on the CPU-quantized model.

MobileNetV4-l is supported with a measured drop of 1.7 points (0.710 vs
0.727 with min-max ranges and the stream transform; LibreYOLO applies this
per-size recipe; learned rounding of every layer gained nothing on the
CPU-quantized model). The recipe
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

DeepLabV3-ResNet50 and -ResNet101 run whole on the Metis, each dilated ASPP
convolution rewritten as per-tap 1x1 convolutions plus zero-filled shifts (as
dilated convolutions they find no L1 tiling at output stride 8). On the 300
images mIoU is 0.4734 vs 0.5243 FP32 for R50 (-5.09) and 0.3795 vs 0.5888 for
R101 (-20.93): the CPU-quantized model stays within 0.4 mIoU of FP32, and a
device bisect puts the divergence in the rewritten ASPP (device vs
CPU-quantized cosine 0.995 for the backbone alone, 0.53 with the ASPP). Both
are supported with these drops; for R101 check accuracy on your data, or use
R50 or MobileNetV3. Running the ASPP and head on the host recovers them (-0.40
and -0.09), but that is a hybrid and is parked (section 11). Ledgers
`deeplabv3/semantic/r50/2026-11-17/2`, `deeplabv3/semantic/r101/2026-11-14`.

Values are mAP50-95 (box, rotated box for obb, keypoint OKS for pose). On
mAP50 the obb deltas are -1.36 (s), -0.11 (m) and -0.43 (l).

YOLO9-E2E is cut at its one-to-one head (the branch its NMS-free inference
decodes); the host rebuilds the same candidates as its ONNX export and
keeps the top-scoring ones without NMS. The FP32 column is the plain ONNX
export on the same images.

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
calibration peaks; FP32 outputs unchanged). For detection s is balanced,
sqrt(activation peak / weight peak), so the activation and the int8 weight
columns share the range: RTMDet-t goes from -4.44 to -2.29 on 1000 images
(-2.84 with s = activation peak / mean peak) and is supported with that
measured drop. RTMDet-s, whose stage-2 pointwise input spreads 45.9x
(channel peak against the median), collapses without the equalization
(-27.5 on 300 images) and gets worse with the activation-only factor
(-33.2); with the balanced factor it scores -6.75 on 1000 images. Nearly
all of what remains is the INT8 rounding of that one pointwise convolution's
weights, so for RTMDet-s the export also learns each pointwise weight's
rounding direction on the per-channel INT8 grid (in the style of AdaRound,
matching the FP32 output after batch norm and SiLU on calibration
activations, with the mean shift folded into the bias; per-size recipe
`pointwise_learned_rounding`, about 7 minutes at export): -4.40 on the 1000
images, -5.43 on 300 (was -8.06) and -5.15 on a held-out 1000. It also
removes RTMDet-s's sensitivity to the calibration count: on the
CPU-quantized model (300 images) all 128 coco128 images instead of 64 cost
7.6 points without the learned rounding (-16.24 vs -8.65) and 0.6 with it
(-5.57 vs -5.00). RTMDet-Ins uses the balanced factor too: with the
activation-only factor RTMDet-Ins-s collapses (mask 0.080 vs 0.402 on 300
images); with the balanced one it scores -5.80 mask / -7.26 box on 1000,
and RTMDet-Ins-t is unchanged within noise (-1.44 / -2.51 vs -1.40 / -2.64).
RTMDet-Ins-s also takes RTMDet-s's learned rounding (same recipe key):
-3.27 mask / -5.02 box on the 1000 images, and -3.20 / -5.19 on 300 (was
-6.63 / -8.20).

HRNet-w32 (top-down pose on 256x192 person crops) compiles as one graph, but
on the device its output drifts away from the SDK's own quantized model
through stage 4's multi-resolution fusion modules (keypoint mAP50-95 0.0001).
It runs as 14 compiled parts instead: the trunk to stage 3, then one part per
branch of each of the three stage-4 modules, then the heatmap layer. Each
fusion (a sum over the four resolutions, then ReLU) is split: every branch
part emits its contribution toward each resolution (the 1x1 conv and BN for a
coarser branch, the stride-2 conv chain for a finer one), the host upsamples
(nearest) and concatenates them, and the next part starts with a fixed 1x1
conv that sums the copies, then the ReLU. The 32-channel branch is handed over
twice (a narrower part input fails a compiler type check). With the fusion
modules split this way the drift is gone: 0.7566 vs 0.7658 FP32 keypoint
mAP50-95 on 1172 ground-truth person crops from 300 COCO val2017 images
(-0.92, validated). Speed on this NUC: 16.4 img/s batched (8 crops) vs 6.87
for the best FP32 CPU run, both at 4 host threads; at batch 1 the 14 part calls
make it slower than the CPU (3.69 vs 4.98 img/s).

RTMDet-x does not compile as one graph: its PAFPN neck finds no L1 tiling
(LowerTIR still overflows after 4 iterations, whole graph or backbone + neck),
although each of the neck's CSP blocks compiles alone in about a minute. It
runs as five compiled parts: the backbone with the neck's first 1x1 reduce,
then one part per PAFPN block, each carrying its level's output conv and head.
Between parts the host only joins tensors the way the neck does: nearest 2x
upsampling and channel concatenation (`cat([upsample(high), low])` top-down,
`cat([down, lateral])` bottom-up). Each block's two entry 1x1 convs run as one
merged conv, so every part has a single entry op. RTMDet-x scores 0.4788 vs
0.5128 FP32 (-3.40) on 1000 images and -2.92 on 300. Speed on this NUC: 2.79
img/s at batch 1 and 6.79 at batch 8 end to end (host steps 6-14% of the
time), against 0.80 img/s for the best FP32 CPU run (ONNX Runtime, batch 8;
OpenVINO 0.65).

RTMDet-Ins runs the same trunk on the device and returns, per level, the
box, class and dynamic-kernel outputs, plus the stride-8 mask feature
(layout `rtmdet_ins_raw_heads_v1`); the host runs LibreYOLO's own RTMDet-Ins
postprocess (box decode, NMS and the dynamic-kernel mask decode), so an
exported model returns the same masks as native inference would for the
same head outputs. Mask rows use COCO val2017 instance polygons for the same
images as the box rows.

EfficientDet runs whole on the AIPU after exact rewrites: the BiFPN's
fast-normalized fusion divides by a constant (a multiplication by its
reciprocal) and sums a stacked tensor (a chain of additions), and the
squeeze-excite averages over large maps run as 2x2 average pools down to
4x4 before the global average, since one global average over 256x256 does
not compile. The box decode (sigmoid, top-k, anchors) runs on the host in
float. The FP32 reference is identical through the PyTorch model, the plain
ONNX export and the rewritten graph (0.3316 on 300 images each). With
min-max ranges the Metis model scored 0.1593 on those 300 images (-17.2);
per-channel ranges (`hybrid_per_tensor_per_channel`, the EfficientDet
default) bring it to -1.0 there and -1.78 on 1000 images, spread over small
(0.055 vs 0.056), medium (0.299 vs 0.320) and large objects (0.533 vs
0.558).

EfficientDet-d1, -d2 and -d3 do not compile at their native 640, 768 and
896: lowering finds no L1 tiling even with one-row tiles (3.42, 3.15 and
2.52 MB needed against 2.82, 2.54 and 2.15 MB). EfficientDet-d1 and -d2
therefore export at d0's 512x512 canvas: the exporter rebuilds the BiFPN,
whose resampling is sized from the image size, for 512 with the same
weights, and images are resized to 512. On 1000 images the smaller canvas
costs d1 3.10 mAP50-95 points in FP32 (0.3949 at 640 vs 0.3639 at 512; small
objects 0.125 vs 0.072) and INT8 1.64 more (0.3475 on the device), and d2
5.31 (0.4275 at 768 vs 0.3744) and INT8 1.38 more (0.3606). Both stay above
EfficientDet-d0 on the device (0.3185). EfficientDet-d3 does not fit at 512
either (2.58 vs 2.14 MB): its need is set by its 160-channel BiFPN and heads,
not by the canvas. EfficientDet-d4 finds no tiling at 1024 or 512 either,
and prefix cuts put its overflow in the EfficientNet-B4 backbone's
272-channel stage (3.12 MB needed, 2.47 MB available).

Compiler tiling options do not change d3's need (force_oc_tiling, or
resources_used=1.0, which raises only the L2 budget). Each BiFPN node does
compile alone, so d3 runs entirely on the Metis as 54 compiled parts (the
backbone, one part per BiFPN node, one per level's heads; between parts the
host only upsamples and concatenates, then decodes the anchors): 0.3718 vs
0.3853 FP32 mAP50-95 on 300 COCO val2017 images (-1.35). End to end that
chain is slower than FP32 on this CPU at the same 512 input: 1.09 vs 4.64
img/s at batch 1 and 3.50 vs 4.48 at batch 8, so it is parked for speed
(branch `axelera-effdet-parts`; ledgers `efficientdet/detect/d3/2026-11-20`
to `/9`).

EfficientDet-d3 and -d4 are therefore blocked. Running their EfficientNet
backbone (d4: stages 0-4) on the Metis and the rest on the host works (-0.39
and -0.29 against FP32 at 512), but that is a hybrid and is parked (section 11).

SSD300 runs in two compiled parts with one host step between them (layout
`ssd_raw_heads_v1`, metadata `parts`). Its conv4_3 L2Norm divides each
pixel's 512 channels by their norm, which the Metis AIPU has no operator for,
and only the level-0 heads read it. Part 1 (`<stem>.axm`) runs the VGG trunk,
returns conv4_3 unnormalized, and runs the extra blocks and the level 1-5 box
and class convolutions; the host normalizes conv4_3 (`l2norm`, with SSD's
learned per-channel scale); part 2 (`<stem>_part2.axm`) runs the level-0
convolutions, calibrated on the normalized conv4_3 of the FP32 part 1 for the
same calibration images. The host then decodes the 8,732 default boxes as
SSD's own export does. Batched inference batches each part. Keep both `.axm`
files and `metadata.yaml` together.

Multi-part exports in general (metadata `parts`) chain three kinds of step:
compiled parts on the Metis AIPU (one input each; `<stem>.axm`, then
`<stem>_part2.axm`, `<stem>_part3.axm`, ...), host operations in numpy (the
backend's `HOST_OPS`, any number of inputs and outputs; large parameters live
in `<stem>_host.npz`), and FP32 ONNX graphs the host runs on the CPU with
ONNX Runtime (`<stem>_host1.onnx`, ...; those artifacts need `onnxruntime`
installed). A compiled part marked `per_item` takes a stack of items, such as
regions of interest, and runs them one at a time on the device. The exporter
calibrates every compiled part on what it will see: the calibration images
run through the FP32 chain, host steps included, and each part takes its own
input from it; `verify=True` compares each part's quantized model with its
FP32 graph on held-out images carried through the chain the same way
(`<stem>.parity.json`). The backend sums the time spent on the AIPU parts and
on the host steps in `chain_timing` (`images`, `aipu_ms`, `host_ms`); each
multi-part model's support note gives that split as measured.

RetinaNet resizes each image so its shorter side is 800 pixels (longer
side up to 1333), so its input shape changes from image to image. The Metis
graph takes one fixed shape, so the export fits each image into a square
canvas and pads the rest. At 800x800 most images reach the network smaller
than with RetinaNet's own resize: on 1000 COCO val2017 images the PyTorch
model scores 0.3521 with its own resize and 0.3318 on the 800 canvas
(-2.03), and the Metis 0.3296 (-2.25 end to end). R50 therefore exports at
a 1024x1024 canvas by default: FP32 on that canvas scores 0.3472 (0.49
below RetinaNet's own resize), the Metis 0.3410 (-0.62 for INT8, -1.11 end
to end), so it stays supported, not validated. The 1024 canvas has 1.64x the
pixels and costs about 1.64x the device compute (val() on the shared test
machine: 581 vs 341 ms per image, host preprocessing and decode included);
pass imgsz=800 for speed at -2.25. FP32 canvas sweep on 300 images: 800x800
0.3355, 1024x1024 0.3659, 1344x1344 0.3717, own resize 0.3606 (on those 300
the 1024 canvas scores above the own resize, on the 1000 above it is 0.49
below); a landscape 800x1344 canvas (0.3586 on 300) gains less than 1024
and would need a rectangular export.

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
pose, MobileNetV4, DeepLabV3 and EfficientDet, which keep more accuracy with
histogram ranges
(`libreyolo.export.axelera.default_quantization_scheme(family, task)` names
the scheme); pass `compiler_config={"quantization_scheme": ...}` to choose
another.

Throughput, YOLO-NAS-s detect, 640x640, raw model calls on the device
(host quantize, AIPU, dequantize; measured at commit 45bfb9136, 2026-11-11,
while other jobs loaded the host):

| Submission | Images per second |
| --- | --- |
| one image per call | about 30 |
| `predict(..., batch=8)` path | about 48 |
| batch of 32, optimized runtime pipeline | about 73 |

End to end through `predict()` (host preprocessing, the Metis, host decode,
NMS and `Results`; the runtime's default 4 AIPU cores; 64 COCO val2017 images,
ImageNetV2 for ResNet-50; 4 host threads on the reference NUC, an Intel Core
Ultra 5 125H with a Metis M.2 Max), measured at commit f4801b7d4 (2026-11-20),
images per second:

| Model | batch 1 | batch 8 | batch 32 | device only |
| --- | --- | --- | --- | --- |
| ResNet-50 | 69 | 256 | 320 | 1118 |
| YOLO9-s | 18 | 49 | 58 | 300 |
| RTMDet-s | 25 | 71 | 79 | 365 |
| PP-YOLOE-s | 19 | 59 | 63 | 383 |
| YOLO-NAS-s | 21 | 58 | 63 | 365 |
| PP-LiteSeg-t50 (512x1024) | 8 | 17 | - | 327 |

The host side bounds these: the device-only column is the compiled model
alone (the SDK's `axrunmodel`). Lists, directories and videos run at batch 8
by default (section 5). The
end-to-end numbers move with the host CPU and its load.

A size is listed as validated or supported only if it is also faster with the
network on the Metis than on the host CPU alone: its best end-to-end
`predict()` throughput is at least that of the FP32 model on the CPU (ONNX
Runtime or OpenVINO, whichever is faster) at the same host thread budget of 4
threads, on the same images. Every size in section 1 meets this on the
reference NUC; the closest margins were re-measured on 2026-11-20 at commit
801b8eae8 and are PicoDet-s (1.5x) and YOLOX-n (2.0x), while most 640-pixel
detectors run 5 to 35 times faster. Where the CPU reference cannot use
LibreYOLO's ONNX export (RTMDet-Ins) it is the FP32 network forward alone,
without decoding; for PicoSAM3 it is LibreYOLO's own PyTorch `predict()`.

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

Dense models (super-resolution, depth, semantic segmentation) also run whole on the AIPU (layout `graph_outputs_v1`):
the device returns the same tensors as the family's ONNX export, under the
same names, and LibreYOLO's shared parsers turn them into results on the
host. The input is fixed at export: depth and semantic models stretch each
image to the canvas, super-resolution pads smaller images into it and
rejects larger ones. Calibration images go through that same preprocessing.

Real-ESRGAN-x4t (the compact network) is the exception: its output adds a
nearest-upsampled copy of the input to the pixel-shuffled trunk, and a graph
whose output depends directly on its input cannot be split into a host
preamble and an accelerated core. The device runs the convolution trunk up to
its last convolution (layout `restore_shuffle_skip_v1`); the host applies the
pixel shuffle and adds the upsampled input. Before calibration the export
equalizes the trunk's 33 conv+PReLU pairs (cross-layer equalization; PReLU is
positively homogeneous, so FP32 outputs are unchanged). On 296 COCO crops
upscaled 4x from 64 px, PSNR is 22.30 dB on a Metis M.2 Max vs 22.49 dB FP32
(-0.19 dB; SSIM 0.621 vs 0.658), and -0.20 dB on a held-out half of the crops;
without the equalization it was 22.00 dB (-0.49 dB).

Real-ESRGAN-x4 (the RRDB network) does not fit as one graph: its constants overflow the Metis DDR pool. It runs as three compiled parts: `conv_first` and RRDB blocks 0-11 (returning the trunk skip and the block output), blocks 12-22 and `conv_body`, then the upsampling tail; the host adds the skip to the body between parts 2 and 3. On the same 296 crops PSNR is 22.44 dB vs 22.49 dB FP32 (-0.05 dB; SSIM 0.622). The library export reproduces it exactly: 22.4383 dB, SSIM 0.622 (ledger `realesrgan/restore/x4/2026-11-16`; first measured on a prototype of the same parts, `realesrgan/restore/x4/2026-11-15/2`).

Real-ESRGAN-x2 uses the same RRDB body and the same three parts. Its network first folds each 2x2 pixel block into channels (pixel unshuffle); the export writes that as an exact one-hot stride-2 convolution so it runs on the AIPU inside part 1. At a 128x128 canvas (a 64x64 trunk, as x4) the device keeps PSNR at 24.73 dB vs 25.07 dB FP32 on the 296 cocosr crops (-0.34 dB, SSIM 0.717; ledger `realesrgan/restore/x2/2026-11-16`). That is just over the 0.3 dB bar, so x2 is supported, not validated. Its whole graph hit the same DDR-pool limit as x4 (319 MiB of constants, 248 MiB pool; ledger `realesrgan/restore/x2/2026-11-15`).

Hybrid exports, which run part of the network itself on the host (the
RT-DETR and DETR families, Faster and Mask R-CNN, HRNet-w48, DEKR, ZipDepth,
PIDNet, CenterNet, TinyFormer, PP-DocLayoutV3 and some sizes of other
families), are not supported on Metis: they are parked by decision
2026-11-19 on branch `axelera-hybrid-parked` (section 11). The multi-part
mechanism above stays for exports whose host steps are light numpy
operations between compiled parts (SSD, FCOS, Real-ESRGAN).

`metadata.yaml` holds LibreYOLO's export metadata (`schema_version`,
`model_family`, `task`, `names`, `imgsz`, ...) plus:

| Key | Meaning |
| --- | --- |
| `format` | `axelera` |
| `axm` | the `.axm` file name |
| `output_layout` | output contract, for example `yolonas_raw_heads_v1`, `yolox_raw_heads_v1`, `darknet_raw_heads_v1`, `darknet_region_raw_heads_v1`, `yolo7_raw_heads_v1`, `ssd_raw_heads_v1`, `ppyoloe_raw_heads_v1`, `classify_logits_v1` or `graph_outputs_v1` |
| `output_names` | output names in the order the compiled model returns them |
| `graph_outputs` | output names in graph order (`graph_outputs_v1`) |
| `output_windows` | parts emitted through a ReLU6 window, as `[low, high]` logit bounds (YOLO-NAS pose and YOLO9 exports); the backend maps them back before decoding |
| `strides`, `reg_max`, `grid_cell_offset` | decode constants (`reg_max` only for distribution heads) |
| `crop_pct`, `interpolation` | classifier eval transform |
| `anchors`, `scale_x_y` | anchor boxes in input pixels per level (YOLOv2's grid-cell anchors are converted), and Darknet's centre scaling (anchor-based heads) |
| `parts` | multi-part exports only (SSD300): the compiled parts and host steps in order. A compiled part is `{axm, input, outputs}` (`input` is `images` for the first part, which is also `axm`), plus `per_item: true` and `item_shapes` for a part that runs a stack of items one at a time; a host operation is `{host, input or inputs, output or outputs, ...parameters}` with `host` one of the backend's `HOST_OPS` (`l2norm`: `scale` per channel; a parameter `"@npz:<key>"` is an array from `host_arrays`); a host ONNX graph is `{onnx, inputs, outputs}`, run on the CPU by ONNX Runtime. The backend runs them in order and assembles `output_names` from their outputs; artifacts without `parts` are one `.axm`. |
| `host_arrays` | multi-part exports whose host operations take arrays: the `.npz` holding them |
| `hybrid` | hybrid exports only (parked, section 11): `split` (`neck`, `backbone` or `encoder`), what runs on the AIPU and on the host, the canvas level scales, and the neck part's quantization scheme (always per tensor: its input feeds a depthwise convolution, which the compiler cannot take per-channel input ranges for at a part boundary) |
| `class_index_map` | logit columns kept after the softmax (SSD300 and RetinaNet: COCO-91 to the 80 classes) |
| `axelera` | SDK version, quantization scheme, cores, resources, compiler overrides, calibration source and image count (of the first part, for multi-part exports) |
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

Parked (not supported on Metis: part of the network runs on the host; hybrid parked by decision 2026-11-19, branch `axelera-hybrid-parked`, which keeps the code, tests and measurements): HRNet w48 (not measured as compiled parts yet) and DEKR w32 (pose), ZipDepth bnpu/b (depth), PIDNet s/m/l (semantic), CenterNet resdcn18/dla34, Faster R-CNN n/s/m/l, Mask R-CNN r50 (detect, segment), DETR r50/r50dc5/r101/r101dc5, Deformable DETR (5 sizes), DINO-DETR r50/r50s5, RT-DETR (7 sizes), RT-DETRv2 detect (5) and obb (5), RiO-DETRv2 obb (5), D-FINE detect and segment (5 each), DEIM (5), DEIMv2 atto/femto/pico/n, RT-DETRv4 (4), PP-DocLayoutV3 l, TinyFormer s, RTMDet x, RetinaNet r50v2 and EfficientDet d3/d4 (whole graph does not compile or does not run usefully, see the blocked reasons).

Blocked for structural reasons (accuracy never blocks an export):

- L2CS (gaze): its only checkpoint is trained on Gaze360, whose licence forbids redistributing models trained on it; LibreYOLO fetches it from the authors and does not mirror it, so no Axelera export is measured or shipped.
- RTMDet-Ins-x: at 640x640 lowering fails in AssignTilingFactors (ScheduleMemory, LowerTIR): no tiling factor fits L1 (3.99 MB needed, 3.32 MB available; 2 h 08 min, 8.1 GB peak RSS; ledger `rtmdet/segment/x/2026-11-12`). Use RTMDet-Ins-l. Its CSPNeXt-x backbone alone compiles in 24 min, so the overflow is in the neck and heads; a backbone | neck+head split would need the three backbone levels packed into one input (the quantizer takes no multi-input calibration; ledger `rtmdet/segment/x/2026-11-17`).
- MiDaS-l (DPT ViT-L, 384 px): quantization stops at the quantizer's ONNX import ("unexpected input value type") on the ViT encoder (48 LayerNormalization, 24 Softmax, 28 Erf, attention MatMuls; the same graph's 5 Resize nodes also occur in MiDaS-s, which compiles). It peaks at 18 GB of host RAM before failing. Use MiDaS-s.
- ConvNeXt and ConvNeXt V2: per-pixel LayerNorm (and V2's Global Response Normalization) divide by statistics of each pixel or map; quantization stops on the first of them. The channel mean and variance have exact AIPU forms, but the per-pixel reciprocal square root feeding ConvNeXt's GELU has none; a host split at every LayerNorm would need about 23 parts for ConvNeXt-t, so a hybrid export (convolutions on the Metis, LayerNorms on the host) is the route if ConvNeXt is ever needed (ledgers `convnext/classify/t/2026-11-11`, `convnextv2/classify/atto/2026-11-11`).
- EC and TinyFormer decoders: gather-based deformable attention; no quantizer converter for its GreaterOrEqual bounds masks (ledgers `ec/detect/s/2026-11-11`, `tinyformer/detect/s/2026-11-11`), so only a hybrid (decoder on the host) could run them, and hybrids are parked (above).
- GTR (all tasks): its ViT runs on the host, and its multi-scale neck (at most ~48% of GTR-s detect's MACs) follows every convolution with a channel LayerNorm (GTR-s: 30 convolutions, 30 LayerNorm2d), so a hybrid export would need about 30 single-convolution parts (ledger `gtr/detect/s/2026-11-18`).
- TEED: the Smish activation takes a logarithm, which quantization stops on; no trained weights are published (ledger `teed/edge/t/2026-11-11/2`).

Not measured on a Metis yet (nothing is known to fail; the export raises until a size is measured):

- NAFNet: no trained LibreYOLO weights are published, so nothing can be exported or measured. When weights exist, the input skip around the whole graph runs on the host (as for Real-ESRGAN), and the per-pixel LayerNorms have the same reciprocal square root as ConvNeXt's (ledger `nafnet/restore/s/2026-11-11`).
- FOMO (point): the published checkpoints (s, m, l) are architecture-only random weights, so nothing trained can be exported or measured; it is a small MobileNetV2-style CNN with no operator known to block.
- YOLO9-P2-t: no trained checkpoint is published (P2-s is validated on VisDrone; its checkpoint is a non-commercial research preview).
- Other families.

Out of scope (not attempted): models whose backbone is itself a transformer. FeyNobg (BiRefNet, Swin-L) is one of them.
Even a hybrid export (parked, see above) would leave almost nothing on the
Metis with an attention backbone. This covers ViT, DeiT, Swin, DINOv2, CLIP, SigLIP2, PE,
SegFormer, EoMT, Depth Anything (1, 3 and video), the SAM family (SAM,
MobileSAM), the detectors built on such backbones (RF-DETR and LW-DETR on
DINOv2/ViT, Grounding DINO on Swin, OWLv2 on ViT), vision-language models,
generative models and 3D models. Checked against LibreYOLO's own model code
(2026-11-16):

- EC (EdgeCrafter, detect/pose/segment): its backbone is ECViT, a
  DINOv3-style ViT with RoPE attention (embed 192-384, all sizes), with an
  adapter resampling its tokens to three pyramid levels; only the hybrid
  encoder's fusion neck is convolutional.
- OV-DEIM: a DINOv3 ViT backbone with a spatial-tuning adapter (vit_tiny,
  vit_tinyplus, dinov3_vits16 for s, m, l) and a CLIP text transformer.
- OmDet-Turbo: a Swin-T backbone and a CLIP text encoder, loaded through
  transformers; LibreYOLO has no runtime export for it in any format.
- GTR: a ViT backbone whose blocks use gated linear attention, with an
  adapter to multi-scale features; its convolutional neck puts a channel
  LayerNorm after every convolution (section 11, known limits).
- DEIMv2 s, m, l and x: DINOv3 ViT backbones with a spatial-tuning adapter
  (vit_tiny, vit_tinyplus, dinov3_vits16, dinov3_vits16plus). The atto,
  femto, pico and n sizes use HGNetv2; they are hybrids, parked (above).
- DINO-DETR swinl: a Swin-L backbone (shifted-window attention, embed 192,
  depths 2/2/18/2, window 12). The r50 and r50s5 sizes use ResNet-50; they
  are hybrids, parked (above).

Other known limits:

- YOLO-NAS-R (obb): validated on 300 DOTA-v2.0 tiles (the paired set prepared for this work), not on a 1000-image set.
- YOLOX-n: supported, mAP50-95 1.81 points below FP32 (1.66 with 128 calibration images; section 7).
- PP-YOLOE-s, -m and -x: supported, mAP50-95 1.1 to 1.2 points below FP32 (section 7). The loss is in the box-distribution outputs and does not come from their output range: windowing them did not help, and histogram ranges help only PP-YOLOE-x (-1.08, was -1.71), which now uses them. PP-YOLOE-l is validated.
- YOLOv4-b: supported, mAP50-95 1.52 points below FP32 with its recipe (section 7): its box outputs lose resolution in the Mish backbone; histogram ranges and a box-output window recover most of it (-6.8 to -1.52).
- YOLO-NAS-pose n and s: supported, keypoint mAP50-95 1.3 and 2.2 points below FP32 (section 7); m and l are validated.
- RF-DETR and other transformer-backbone families: attention, data-dependent
  TopK and grid sampling are outside what the compiler places on the AIPU.
  RT-DETR, whose backbone is a CNN, could run as a hybrid (backbone and
  fusion neck on the Metis, attention on the host); hybrids are parked (above).
- MobileNetV4 (classify) l: supported, top-1 1.7 points below FP32 at best
  (section 7). Per-tensor INT8 of the inverted-residual streams is the cause
  (the stage-2 stream of s spans about +-400 while its median channel stays
  within +-14; s gets within 1 point with learned rounding). Use m or s.
- Europa and multi-device scheduling: not covered.
- Large models on M.2 cards can hit the card's power limit.
