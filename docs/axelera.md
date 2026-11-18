# Axelera Metis export

> **PREVIEW.** Axelera Metis support is new in 1.8.0. Every size below
> compiles, runs on a Metis device and exports. "validated" means the INT8
> result is within 1.0 point of FP32 on the paired set; "supported" means it
> works with a larger measured drop, which the export logs once as a warning.
> Accuracy never blocks an export; "blocked" is only for families that do not
> compile, do not run or give meaningless output (`docs/export_support.md`).
> "hybrid" means the convolutional part runs on the Metis and the attention
> part (for DeepLabV3-R50/R101 and VGG-19-BN, the layers that drift on the
> device) on the host CPU (section 8).
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
| HRNet | w32 | pose (hybrid: stem to stage 3 on the Metis; stage 4 and the heatmap layer on the host; 256x192 person crops from GT boxes) | COCO val2017 persons, 300 (1172 crops) | 0.7658 | 0.7631 | -0.27 | hybrid | `hrnet/pose/w32/2026-11-17/2` |
| HRNet | w48 | pose (hybrid: stem to stage 3 on the Metis; stage 4 and the heatmap layer on the host; 384x288 person crops from GT boxes; calibration 597 COCO person crops, fraction 1.0) | COCO val2017 persons, 300 (1172 crops) | 0.7855 | 0.7833 | -0.22 | hybrid | `hrnet/pose/w48/2026-11-18/2` |
| DEKR | w32 | pose (hybrid: HRNet-w32 stem to stage 3 on the Metis; stage 4, heatmap and offset heads on the host; 640x640) | COCO val2017 persons, 300 | 0.6523 | 0.6426 | -0.97 | hybrid | `dekr/pose/w32/2026-11-18/2` |
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
| PP-YOLOE | x | detect | COCO val2017, 1000 | 0.4918 | 0.4747 | -1.71 | supported | `ppyoloe/detect/x/2026-11-11/2` |
| PicoDet | s | detect | COCO val2017, 1000 | 0.2687 | 0.2453 | -2.34 | supported | `picodet/detect/s/2026-11-11/7` |
| PicoDet | m | detect | COCO val2017, 1000 | 0.3449 | 0.3163 | -2.86 | supported | `picodet/detect/m/2026-11-11/4` |
| PicoDet | l | detect | COCO val2017, 1000 | 0.3971 | 0.3801 | -1.70 | supported | `picodet/detect/l/2026-11-11/4` |
| RTMDet | t | detect | COCO val2017, 1000 | 0.3906 | 0.3677 | -2.29 | supported | `rtmdet/detect/t/2026-11-11/8` |
| RTMDet | s | detect (learned INT8 rounding of the pointwise convs) | COCO val2017, 1000 | 0.4121 | 0.3681 | -4.40 | supported | `rtmdet/detect/s/2026-11-17/8` |
| RTMDet | m | detect | COCO val2017, 1000 | 0.4727 | 0.4461 | -2.66 | supported | `rtmdet/detect/m/2026-11-12/2` |
| RTMDet | l | detect | COCO val2017, 1000 | 0.4805 | 0.4618 | -1.87 | supported | `rtmdet/detect/l/2026-11-12/2` |
| RTMDet | x | detect (hybrid: CSPNeXt-x backbone on the Metis; PAFPN neck, heads and decode on the host) | COCO val2017, 300 | 0.5319 | 0.5328 | +0.09 | hybrid | `rtmdet/detect/x/2026-11-18/2` |
| FCOS | r50 | detect (800x800 canvas; 6 compiled parts, GroupNorm on the host) | COCO val2017, 1000 | 0.3818 | 0.3405 | -4.13 | supported | `fcos/detect/r50/2026-11-15/3` |
| RetinaNet | r50 | detect (1024x1024 canvas, default) | COCO val2017, 1000 | 0.3472 | 0.3410 | -0.62 (canvas -0.49 more; -1.11 vs own resize) | supported | `retinanet/detect/r50/2026-11-14/4` |
| RetinaNet | r50 | detect (800x800 canvas, imgsz=800) | COCO val2017, 1000 | 0.3318 | 0.3296 | -0.22 (canvas -2.03 more; -2.25 vs own resize) | option (imgsz=800) | `retinanet/detect/r50/2026-11-14/3` |
| RetinaNet | r50v2 | detect (hybrid: ResNet-50 body on the Metis; FPN, GroupNorm heads and decode on the host; 800x800 canvas) | COCO val2017, 300 | 0.3962 | 0.3722 | -2.40 | hybrid | `retinanet/detect/r50v2/2026-11-17` |
| CenterNet | resdcn18 | detect (hybrid: ResNet-18 body on the Metis; deformable up-path, heads and peak decode on the host; 512x512) | COCO val2017, 300 | 0.2824 | 0.2604 | -2.20 | hybrid | `centernet/detect/resdcn18/2026-11-17/2` |
| CenterNet | dla34 | detect (hybrid: DLA-34 base on the Metis; deformable aggregation, heads and peak decode on the host; 512x512) | COCO val2017, 300 | 0.3702 | 0.3575 | -1.27 | hybrid | `centernet/detect/dla34/2026-11-17/2` |
| SSD | 300 | detect (two parts, L2Norm on the host) | COCO val2017, 1000 | 0.2408 | 0.2403 | -0.05 | validated | `ssd/detect/300/2026-11-13/2` |
| RT-DETR | r18 | detect (hybrid: backbone and neck on the Metis, attention on the host) | COCO val2017, 1000 | 0.4535 | 0.4528 | -0.07 | hybrid | `rtdetr/detect/r18/2026-11-15/2` |
| RT-DETR | r34 | detect (hybrid: backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.4985 | 0.4982 | -0.03 | hybrid | `rtdetr/detect/r34/2026-11-15` |
| RT-DETR | r50 | detect (hybrid: backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5413 | 0.5368 | -0.45 | hybrid | `rtdetr/detect/r50/2026-11-15` |
| RT-DETR | r50m | detect (hybrid: backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5139 | 0.5072 | -0.67 | hybrid | `rtdetr/detect/r50m/2026-11-15` |
| RT-DETR | r101 | detect (hybrid: backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5518 | 0.5502 | -0.16 | hybrid | `rtdetr/detect/r101/2026-11-15` |
| RT-DETR | l | detect (hybrid: backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5402 | 0.5376 | -0.26 | hybrid | `rtdetr/detect/l/2026-11-15` |
| RT-DETR | x | detect (hybrid: backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5587 | 0.5573 | -0.14 | hybrid | `rtdetr/detect/x/2026-11-15` |
| RT-DETRv2 | r18 | detect (hybrid: backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.4965 | 0.4840 | -1.25 | hybrid | `rtdetrv2/detect/r18/2026-11-15` |
| RT-DETRv2 | r34 | detect (hybrid: backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5050 | 0.4847 | -2.03 | hybrid | `rtdetrv2/detect/r34/2026-11-15` |
| RT-DETRv2 | r50 | detect (hybrid: backbone and neck on the Metis, attention on the host; neck part with histogram ranges) | COCO val2017, 300 | 0.5494 | 0.5322 | -1.72 | hybrid | `rtdetrv2/detect/r50/2026-11-16` |
| RT-DETRv2 | r50m | detect (hybrid: backbone and neck on the Metis, attention on the host; neck part with histogram ranges) | COCO val2017, 300 | 0.5321 | 0.5065 | -2.56 | hybrid | `rtdetrv2/detect/r50m/2026-11-16/2` |
| RT-DETRv2 | r101 | detect (hybrid: backbone and neck on the Metis, attention on the host; neck part with histogram ranges) | COCO val2017, 300 | 0.5522 | 0.5510 | -0.12 | hybrid | `rtdetrv2/detect/r101/2026-11-16` |
| RT-DETRv2 | n | obb (hybrid: HGNetv2 backbone and neck on the Metis, attention and rotated-box decoder on the host; 1024x1024) | DOTA-v1.0 classes of DOTA val, 300 (`dota2val300-rio`) | 0.3804 | 0.3648 | -1.56 | hybrid | `rtdetrv2/obb/n/2026-11-17` |
| RT-DETRv2 | s | obb (hybrid: HGNetv2 backbone and neck on the Metis, attention and rotated-box decoder on the host; 1024x1024) | DOTA-v1.0 classes of DOTA val, 300 (`dota2val300-rio`) | 0.5923 | 0.5793 | -1.30 | hybrid | `rtdetrv2/obb/s/2026-11-17` |
| RT-DETRv2 | m | obb (hybrid: HGNetv2 backbone and neck on the Metis, attention and rotated-box decoder on the host; 1024x1024) | DOTA-v1.0 classes of DOTA val, 300 (`dota2val300-rio`) | 0.5918 | 0.5915 | -0.03 | hybrid | `rtdetrv2/obb/m/2026-11-18` |
| RT-DETRv2 | l | obb (hybrid: HGNetv2 backbone and neck on the Metis, attention and rotated-box decoder on the host; 1024x1024) | DOTA-v1.0 classes of DOTA val, 300 (`dota2val300-rio`) | 0.6223 | 0.6213 | -0.10 | hybrid | `rtdetrv2/obb/l/2026-11-18` |
| RT-DETRv2 | x | obb (hybrid: HGNetv2 backbone and neck on the Metis, attention and rotated-box decoder on the host; 1024x1024) | DOTA-v1.0 classes of DOTA val, 300 (`dota2val300-rio`) | 0.6506 | 0.6458 | -0.48 | hybrid | `rtdetrv2/obb/x/2026-11-18` |
| RiO-DETRv2 | n | obb (hybrid as RT-DETRv2-obb; neck part with histogram ranges; 1024x1024) | DOTA-v1.0 classes of DOTA val, 300 (`dota2val300-rio`) | 0.5581 | 0.4912 | -6.69 | hybrid | `riodetrv2/obb/n/2026-11-17` |
| RiO-DETRv2 | s | obb (hybrid as RT-DETRv2-obb; neck part with histogram ranges; 1024x1024) | DOTA-v1.0 classes of DOTA val, 300 (`dota2val300-rio`) | 0.5958 | 0.5859 | -0.99 | hybrid | `riodetrv2/obb/s/2026-11-17` |
| RiO-DETRv2 | m | obb (hybrid as RT-DETRv2-obb; neck part with histogram ranges; 1024x1024) | DOTA-v1.0 classes of DOTA val, 300 (`dota2val300-rio`) | 0.6323 | 0.6325 | +0.02 | hybrid | `riodetrv2/obb/m/2026-11-17` |
| RiO-DETRv2 | l | obb (hybrid as RT-DETRv2-obb; neck part with histogram ranges; 1024x1024) | DOTA-v1.0 classes of DOTA val, 300 (`dota2val300-rio`) | 0.6239 | 0.6169 | -0.70 | hybrid | `riodetrv2/obb/l/2026-11-17` |
| RiO-DETRv2 | x | obb (hybrid as RT-DETRv2-obb; neck part with histogram ranges; 1024x1024) | DOTA-v1.0 classes of DOTA val, 300 (`dota2val300-rio`) | 0.6964 | 0.6955 | -0.09 | hybrid | `riodetrv2/obb/x/2026-11-17` |
| D-FINE | n | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.4248 | 0.4209 | -0.39 | hybrid | `dfine/detect/n/2026-11-15` |
| D-FINE | s | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5155 | 0.4989 | -1.66 | hybrid | `dfine/detect/s/2026-11-16` |
| D-FINE | m | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5653 | 0.5572 | -0.81 | hybrid | `dfine/detect/m/2026-11-16` |
| D-FINE | l | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5862 | 0.5739 | -1.23 | hybrid | `dfine/detect/l/2026-11-16` |
| D-FINE | x | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.6098 | 0.6039 | -0.59 | hybrid | `dfine/detect/x/2026-11-16/2` |
| D-FINE | n | segment (hybrid: HGNetv2 backbone, neck and mask convolutions on the Metis; attention, decoder and mask GroupNorms on the host) | COCO val2017 (masks), 300 | 0.3536 | 0.3485 | -0.51 | hybrid | `dfine/segment/n/2026-11-17/3` |
| D-FINE | s | segment (hybrid: HGNetv2 backbone, neck and mask convolutions on the Metis; attention, decoder and mask GroupNorms on the host) | COCO val2017 (masks), 300 | 0.4051 | 0.3035 | -10.16 | hybrid | `dfine/segment/s/2026-11-17/4` |
| D-FINE | m | segment (hybrid: HGNetv2 backbone, neck and mask convolutions on the Metis; attention, decoder and mask GroupNorms on the host) | COCO val2017 (masks), 300 | 0.4263 | 0.4257 | -0.06 | hybrid | `dfine/segment/m/2026-11-17` |
| D-FINE | l | segment (hybrid: HGNetv2 backbone, neck and mask convolutions on the Metis; attention, decoder and mask GroupNorms on the host) | COCO val2017 (masks), 300 | 0.4431 | 0.4388 | -0.43 | hybrid | `dfine/segment/l/2026-11-17` |
| D-FINE | x | segment (hybrid: HGNetv2 backbone, neck and mask convolutions on the Metis; attention, decoder and mask GroupNorms on the host) | COCO val2017 (masks), 300 | 0.4512 | 0.4500 | -0.12 | hybrid | `dfine/segment/x/2026-11-17` |
| DEIM | n | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.4238 | 0.4213 | -0.25 | hybrid | `deim/detect/n/2026-11-15` |
| DEIM | s | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.4980 | 0.4429 | -5.51 | hybrid | `deim/detect/s/2026-11-16` |
| DEIM | m | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5358 | 0.5390 | +0.32 | hybrid | `deim/detect/m/2026-11-16` |
| DEIM | l | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5614 | 0.5592 | -0.22 | hybrid | `deim/detect/l/2026-11-16` |
| DEIM | x | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5783 | 0.5766 | -0.17 | hybrid | `deim/detect/x/2026-11-16` |
| RT-DETRv4 | s | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5090 | 0.4984 | -1.06 | hybrid | `rtdetrv4/detect/s/2026-11-16` |
| RT-DETRv4 | m | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5546 | 0.5466 | -0.80 | hybrid | `rtdetrv4/detect/m/2026-11-16` |
| RT-DETRv4 | l | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5699 | 0.5679 | -0.20 | hybrid | `rtdetrv4/detect/l/2026-11-16` |
| RT-DETRv4 | x | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.5782 | 0.5731 | -0.51 | hybrid | `rtdetrv4/detect/x/2026-11-16` |
| DEIMv2 | atto | detect (hybrid: whole convolutional model up to the decoder on the Metis; 320x320) | COCO val2017, 300 | 0.2299 | 0.1951 | -3.48 | hybrid | `deimv2/detect/atto/2026-11-16` |
| DEIMv2 | femto | detect (hybrid: whole convolutional model up to the decoder on the Metis; 416x416) | COCO val2017, 300 | 0.3078 | 0.2801 | -2.77 | hybrid | `deimv2/detect/femto/2026-11-16` |
| DEIMv2 | pico | detect (hybrid: whole convolutional model up to the decoder on the Metis) | COCO val2017, 300 | 0.3852 | 0.3697 | -1.55 | hybrid | `deimv2/detect/pico/2026-11-16` |
| DEIMv2 | n | detect (hybrid: HGNetv2 backbone and neck on the Metis, attention on the host) | COCO val2017, 300 | 0.4247 | 0.4188 | -0.59 | hybrid | `deimv2/detect/n/2026-11-16` |
| DETR | r50 | detect (hybrid: ResNet backbone on the Metis, transformer on the host; 800x800) | COCO val2017, 300 | 0.3879 | 0.3852 | -0.27 | hybrid | `detr/detect/r50/2026-11-15` |
| DETR | r50dc5 | detect (hybrid: ResNet backbone on the Metis, transformer on the host; 800x800) | COCO val2017, 300 | 0.4172 | 0.4125 | -0.47 | hybrid | `detr/detect/r50dc5/2026-11-16` |
| DETR | r101 | detect (hybrid: ResNet backbone on the Metis, transformer on the host; 800x800) | COCO val2017, 300 | 0.4019 | 0.3732 | -2.87 | hybrid | `detr/detect/r101/2026-11-16` |
| DETR | r101dc5 | detect (hybrid: ResNet backbone on the Metis, transformer on the host; 800x800) | COCO val2017, 300 | 0.4232 | 0.4172 | -0.60 | hybrid | `detr/detect/r101dc5/2026-11-16` |
| Deformable DETR | r50 | detect (hybrid: ResNet backbone on the Metis, GroupNorm projections and deformable transformer on the host; 800x800) | COCO val2017, 300 | 0.4176 | 0.4171 | -0.05 | hybrid | `deformable_detr/detect/r50/2026-11-16` |
| Deformable DETR | r50ss | detect (hybrid as r50; single-scale; 800x800) | COCO val2017, 300 | 0.3704 | 0.3682 | -0.22 | hybrid | `deformable_detr/detect/r50ss/2026-11-17` |
| Deformable DETR | r50ssdc5 | detect (hybrid as r50; single-scale, dilated C5; 800x800) | COCO val2017, 300 | 0.3920 | 0.3900 | -0.20 | hybrid | `deformable_detr/detect/r50ssdc5/2026-11-17` |
| Deformable DETR | r50refine | detect (hybrid as r50; iterative box refinement; 800x800) | COCO val2017, 300 | 0.4559 | 0.4526 | -0.33 | hybrid | `deformable_detr/detect/r50refine/2026-11-17` |
| Deformable DETR | r50twostage | detect (hybrid as r50; box refinement, two-stage; 800x800) | COCO val2017, 300 | 0.4503 | 0.4504 | +0.01 | hybrid | `deformable_detr/detect/r50twostage/2026-11-17` |
| DINO-DETR | r50 | detect (hybrid: ResNet backbone on the Metis; GroupNorm projections, deformable transformer and two-stage query selection on the host; 800x800) | COCO val2017, 300 | 0.4714 | 0.4710 | -0.04 | hybrid | `dinodetr/detect/r50/2026-11-17` |
| DINO-DETR | r50s5 | detect (hybrid: 4-level ResNet backbone on the Metis; GroupNorm projections, 5-level deformable transformer and two-stage query selection on the host; 800x800; host-bound, 0.07 img/s) | COCO val2017, 300 | 0.4917 | 0.4891 | -0.26 | hybrid | `dinodetr/detect/r50s5/2026-11-17` |
| Faster R-CNN | n | detect (hybrid: backbone and FPN on the Metis, RPN and RoI heads on the host; 320x320 canvas) | COCO val2017, 300 | 0.2005 | 0.1896 | -1.09 (canvas -3.30 more) | hybrid | `faster_rcnn/detect/n/2026-11-15` |
| Faster R-CNN | s | detect (hybrid as n; 800x800 canvas) | COCO val2017, 300 | 0.2907 | 0.2867 | -0.40 | hybrid | `faster_rcnn/detect/s/2026-11-16` |
| Faster R-CNN | m | detect (hybrid as n; ResNet-50-FPN; 800x800 canvas) | COCO val2017, 300 | 0.3472 | 0.3456 | -0.16 (canvas -3.12 more) | hybrid | `faster_rcnn/detect/m/2026-11-16` |
| Faster R-CNN | l | detect (hybrid as n; ResNet-50-FPN v2; 800x800 canvas) | COCO val2017, 300 | 0.4468 | 0.4489 | +0.21 | hybrid | `faster_rcnn/detect/l/2026-11-16` |
| Mask R-CNN | r50 | detect (hybrid: the segment export's Metis part, the host stops before the mask branch; 800x800 canvas) | COCO val2017, 300 | 0.4476 | 0.4481 | +0.05 | hybrid | `mask_rcnn/detect/r50/2026-11-16` |
| PP-DocLayoutV3 | l | detect (document layout; matched-box F1 vs FP32 boxes at IoU 0.5; hybrid: HGNetV2 on the Metis, encoder, decoder and reading-order head on the host) | DocLayNet val pages, 300 | 1.0000 | 0.9795 | -0.0205 (score r 0.947) | hybrid | `ppdoclayoutv3/detect/l/2026-11-16` |
| EfficientDet | d0 | detect | COCO val2017, 1000 | 0.3363 | 0.3185 | -1.78 | supported | `efficientdet/detect/d0/2026-11-12/3` |
| EfficientDet | d1 | detect (512x512 canvas) | COCO val2017, 1000 | 0.3639 | 0.3475 | -1.64 (canvas -3.10 more) | supported | `efficientdet/detect/d1/2026-11-13/3` |
| EfficientDet | d2 | detect (512x512 canvas) | COCO val2017, 1000 | 0.3744 | 0.3606 | -1.38 (canvas -5.31 more) | supported | `efficientdet/detect/d2/2026-11-14` |
| RTMDet-Ins | t | segment (mask mAP50-95) | COCO val2017, 1000 | 0.3394 | 0.3250 | -1.44 | supported | `rtmdet/segment/t/2026-11-12/2` |
| RTMDet-Ins | s | segment (mask mAP50-95) | COCO val2017, 1000 | 0.3719 | 0.3139 | -5.80 | supported | `rtmdet/segment/s/2026-11-12/4` |
| RTMDet-Ins | m | segment (mask mAP50-95) | COCO val2017, 1000 | 0.4011 | 0.3932 | -0.79 (box -3.96) | supported | `rtmdet/segment/m/2026-11-12/2` |
| RTMDet-Ins | l | segment (mask mAP50-95) | COCO val2017, 1000 | 0.4163 | 0.4113 | -0.50 (box -5.34) | supported | `rtmdet/segment/l/2026-11-12/2` |
| Mask R-CNN | r50 | segment (mask mAP50-95; hybrid: backbone and FPN on the Metis, RPN, box and mask heads on the host; 800x800 canvas) | COCO val2017, 300 | 0.4045 | 0.4039 | -0.06 (box +0.02) | hybrid | `mask_rcnn/segment/r50/2026-11-16` |
| ResNet | 18 | classify (top-1) | ImageNetV2, 1000 | 0.595 | 0.589 | -0.6 | validated | `resnet/classify/18/2026-11-10/2` |
| ResNet | 34 | classify (top-1) | ImageNetV2, 1000 | 0.657 | 0.647 | -1.0 | validated | `resnet/classify/34/2026-11-10` |
| ResNet | 50 | classify (top-1) | ImageNetV2, 1000 | 0.688 | 0.681 | -0.7 | validated | `resnet/classify/50/2026-11-10` |
| ResNet | 101 | classify (top-1) | ImageNetV2, 1000 | 0.713 | 0.708 | -0.5 | validated | `resnet/classify/101/2026-11-10` |
| MobileNetV4 | s | classify (top-1) | ImageNetV2, 1000 | 0.624 | 0.607 | -1.7 | supported | `mobilenetv4/classify/s/2026-11-11/3` |
| MobileNetV4 | m | classify (top-1) | ImageNetV2, 1000 | 0.682 | 0.684 | +0.2 | validated | `mobilenetv4/classify/m/2026-11-11/2` |
| MobileNetV4 | l | classify (top-1) | ImageNetV2, 1000 | 0.727 | 0.710 | -1.7 | supported | `mobilenetv4/classify/l/2026-11-10` |
| EfficientNetV2 | b0 | classify (top-1) | ImageNetV2, 1000 | 0.667 | 0.656 | -1.1 | supported | `efficientnetv2/classify/b0/2026-11-11/5` |
| EfficientNetV2 | b1 | classify (top-1; 240 px; stage-0 skip as identity conv) | ImageNetV2, 1000 | 0.689 | 0.687 | -0.2 | validated | `efficientnetv2/classify/b1/2026-11-17` |
| EfficientNetV2 | b2 | classify (top-1; 260 px; stage-0 skip as identity conv) | ImageNetV2, 1000 | 0.706 | 0.693 | -1.3 | supported | `efficientnetv2/classify/b2/2026-11-17` |
| EfficientNetV2 | b3 | classify (top-1; 300 px; stage-0 skip as identity conv) | ImageNetV2, 1000 | 0.715 | 0.726 | +1.1 | validated | `efficientnetv2/classify/b3/2026-11-17` |
| VGG | 16 | classify (top-1) | ImageNetV2, 1000 | 0.608 | 0.606 | -0.2 | validated | `vgg/classify/16/2026-11-12` |
| VGG | 19 | classify (top-1) | ImageNetV2, 1000 | 0.610 | 0.619 | +0.9 | validated | `vgg/classify/19/2026-11-12` |
| VGG | 16bn | classify (top-1) | ImageNetV2, 1000 | 0.633 | 0.631 | -0.2 | validated | `vgg/classify/16bn/2026-11-12/2` |
| VGG | 19bn | classify (top-1; conv blocks 1-4 on Metis, block 5 + classifier on the host) | ImageNetV2, 1000 | 0.644 | 0.642 | -0.2 | hybrid | `vgg/classify/19bn/2026-11-17/3` |
| AlexNet | b | classify (top-1) | ImageNetV2, 1000 | 0.434 | 0.440 | +0.6 | validated | `alexnet/classify/b/2026-11-13` |
| FCN | r50 | semantic (mIoU) | COCO val2017 VOC-21, 300 | 0.6515 | 0.6492 | -0.23 | validated | `fcn/semantic/r50/2026-11-11/2` |
| FCN | r101 | semantic (mIoU) | COCO val2017 VOC-21, 300 | 0.6798 | 0.6779 | -0.19 | validated | `fcn/semantic/r101/2026-11-11` |
| QuickSRNet | m2 | restore 2x (PSNR, dB) | COCO crops from 128 px, 296 | 29.12 | 28.92 | -0.20 | validated | `quicksrnet/restore/m2/2026-11-16/5` |
| Real-ESRGAN | x4 | restore 4x (PSNR, dB) | COCO crops from 64 px, 296 | 22.49 | 22.44 | -0.05 | validated | `realesrgan/restore/x4/2026-11-16` |
| Real-ESRGAN | x4t | restore 4x (PSNR, dB) | COCO crops from 64 px, 296 | 22.49 | 22.30 | -0.19 | validated | `realesrgan/restore/x4t/2026-11-16` |
| Real-ESRGAN | x2 | restore 2x (PSNR, dB; three compiled parts) | COCO crops from 128 px, 296 | 25.07 | 24.73 | -0.34 | supported | `realesrgan/restore/x2/2026-11-16` |
| MiDaS | s | depth (agreement with FP32) | COCO val2017 at 256 px, 300 | reference | AbsRel 0.069, delta1 0.94 | AbsRel 0.069 (bar 0.02) | supported | `midas/depth/s/2026-11-11/4` |
| ZipDepth | bnpu | depth (hybrid: encoder through stage 3 block 4 on the Metis; the rest of stage 3 with the GCNet block, stage 4 and the decoder on the host; agreement with FP32) | COCO val2017 at 384 px, 300 | reference | AbsRel 0.030, delta1 0.989 | AbsRel 0.030 | hybrid | `zipdepth/depth/bnpu/2026-11-18/4` |
| ZipDepth | b | depth (hybrid: as bnpu, encoder through stage 3 block 4 on the Metis; the rest on the host; agreement with FP32; calibration 128 coco128 images, fraction 1.0) | COCO val2017 at 384 px, 300 | reference | AbsRel 0.029, delta1 0.990 | AbsRel 0.029 | hybrid | `zipdepth/depth/b/2026-11-18/2` |
| DexiNed | b | edge (edge-pixel F1 vs FP32 edges at 0.5; 352x352) | COCO val2017 at 352 px, 300 | 1.0000 | 0.9651 | -0.0349 (cosine 0.997) | supported | `dexined/edge/b/2026-11-16` |
| DeepLabV3 | mv3 | semantic (mIoU) | COCO val2017 VOC-21, 300 | 0.5230 | 0.4853 | -3.77 | supported | `deeplabv3/semantic/mv3/2026-11-11/6` |
| DeepLabV3 | r50 | semantic (mIoU; backbone on Metis, ASPP + head on the host) | COCO val2017 VOC-21, 300 | 0.5243 | 0.5203 | -0.40 | hybrid | `deeplabv3/semantic/r50/2026-11-17/3` |
| DeepLabV3 | r101 | semantic (mIoU; backbone on Metis, ASPP + head on the host) | COCO val2017 VOC-21, 300 | 0.5888 | 0.5879 | -0.09 | hybrid | `deeplabv3/semantic/r101/2026-11-17/4` |
| UNet | s | semantic (fidelity mIoU vs FP32 labels; 512x1024) | COCO val2017, 1000 | 1.0000 | 0.7767 | -0.2233 | supported | `unet/semantic/s/2026-11-16` |
| PP-LiteSeg | t50 | semantic (fidelity mIoU vs FP32 labels; 512x1024; UAFM blends as x*sigmoid(z) + skip*sigmoid(-z)) | COCO val2017, 1000 | 1.0000 | 0.8502 | -0.1498 | supported | `ppliteseg/semantic/t50/2026-11-18` |
| PP-LiteSeg | b50 | semantic (fidelity mIoU vs FP32 labels; 512x1024; UAFM blends as x*sigmoid(z) + skip*sigmoid(-z)) | COCO val2017, 1000 | 1.0000 | 0.8571 | -0.1429 | supported | `ppliteseg/semantic/b50/2026-11-18` |
| PP-LiteSeg | t75 | semantic (fidelity mIoU vs FP32 labels; 768x1536; UAFM blends as x*sigmoid(z) + skip*sigmoid(-z)) | COCO val2017, 1000 | 1.0000 | 0.8068 | -0.1932 | supported | `ppliteseg/semantic/t75/2026-11-18` |
| PP-LiteSeg | b75 | semantic (fidelity mIoU vs FP32 labels; 768x1536; UAFM blends as x*sigmoid(z) + skip*sigmoid(-z)) | COCO val2017, 1000 | 1.0000 | 0.8484 | -0.1516 | supported | `ppliteseg/semantic/b75/2026-11-18` |
| PIDNet | s | semantic (fidelity mIoU vs FP32 labels; hybrid: branches through layer 5 on the Metis, PAPPM, branch fusion and head on the host; 1024x1024) | COCO val2017, 300 | 1.0000 | 0.8773 | -0.1227 | hybrid | `pidnet/semantic/s/2026-11-18/3` |
| PIDNet | m | semantic (fidelity mIoU vs FP32 labels; hybrid as s; 1024x1024; calibration 32 coco128, fraction=0.25: at the default fraction=1.0 the SDK calibrates m at about 65 s per image on CPU, over 2 h) | COCO val2017, 96 | 1.0000 | 0.8401 | -0.1599 | hybrid | `pidnet/semantic/m/2026-11-18/2` |
| PIDNet | l | semantic (fidelity mIoU vs FP32 labels; hybrid as s, DAPPM and Bag fusion on the host; 1024x1024; calibration 32 coco128, fraction=0.25) | COCO val2017, 100 | 1.0000 | 0.8288 | -0.1712 | hybrid | `pidnet/semantic/l/2026-11-18` |

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
| HRNet | pose | w32 (hybrid at the 256x192 person crop: stem to stage 3 on the Metis; stage 4, where the device drifts, and the heatmap layer on the host; w48 the same split at its 384x288 crop) | see `docs/export_support.md` | w32 `hrnet/pose/w32/2026-11-17/2`, w48 `hrnet/pose/w48/2026-11-18/2` |
| DEKR | pose | w32 (hybrid at 640x640: the HRNet trunk to stage 3 on the Metis; stage 4 and the heads on the host) | see `docs/export_support.md` | w32 `dekr/pose/w32/2026-11-18/2` |
| YOLO-NAS-R | obb | s, m, l | see `docs/export_support.md` (calibration data required) | s `yolonas/obb/s/2026-11-10/2`; m `yolonas/obb/m/2026-11-10`; l `yolonas/obb/l/2026-11-10` |
| YOLO9 | detect | t, s, m, c | see `docs/export_support.md` | t `yolo9/detect/t/2026-11-10/7`; s `yolo9/detect/s/2026-11-10/2`; m `yolo9/detect/m/2026-11-10/2`; c `yolo9/detect/c/2026-11-10/2` |
| YOLO9-P2 | detect | s (t: no published checkpoint) | see `docs/export_support.md` | s `yolo9_p2/detect/s/2026-11-15` |
| YOLO9-E2E | detect | t, s, m, c | see `docs/export_support.md` | t `yolo9_e2e/detect/t/2026-11-13`; s `yolo9_e2e/detect/s/2026-11-13`; m `yolo9_e2e/detect/m/2026-11-13`; c `yolo9_e2e/detect/c/2026-11-13` |
| ResNet | classify | 18, 34, 50, 101 | see `docs/export_support.md` | 18 `resnet/classify/18/2026-11-10/2`; 34 `resnet/classify/34/2026-11-10`; 50 `resnet/classify/50/2026-11-10`; 101 `resnet/classify/101/2026-11-10` |
| MobileNetV4 | classify | s, m, l (s and l supported with a measured drop) | see `docs/export_support.md` | s `mobilenetv4/classify/s/2026-11-11/3`; m `mobilenetv4/classify/m/2026-11-11/2`; l `mobilenetv4/classify/l/2026-11-10` |
| EfficientNetV2 | classify | b1, b3 (validated); b0, b2 (supported with measured drops); b1-b3 read their stage-0 residual skip through an exact identity 1x1 conv | see `docs/export_support.md` | b0 `efficientnetv2/classify/b0/2026-11-11/5`; b1 `efficientnetv2/classify/b1/2026-11-17`; b2 `efficientnetv2/classify/b2/2026-11-17`; b3 `efficientnetv2/classify/b3/2026-11-17` |
| VGG | classify | 16, 19, 16bn; 19bn (hybrid: conv blocks 1-4 on Metis, block 5 + classifier on the host; held-out ImageNetV2 +0.2) | see `docs/export_support.md` | 16 `vgg/classify/16/2026-11-12`; 19 `vgg/classify/19/2026-11-12`; 16bn `vgg/classify/16bn/2026-11-12/2`; 19bn `vgg/classify/19bn/2026-11-17/3`, held-out `vgg/classify/19bn/2026-11-17/4` |
| AlexNet | classify | b | see `docs/export_support.md` | b `alexnet/classify/b/2026-11-13` |
| FCN | semantic | r50, r101 | see `docs/export_support.md` | r50 `fcn/semantic/r50/2026-11-11/2`; r101 `fcn/semantic/r101/2026-11-11` |
| QuickSRNet | restore | m2 (learned INT8 rounding; held-out half -0.17 dB) | see `docs/export_support.md` | m2 `quicksrnet/restore/m2/2026-11-16/5`, held-out `quicksrnet/restore/m2/2026-11-16/6` |
| Real-ESRGAN | restore | x4 (three compiled parts + a host add), x4t (equalized trunk; held-out half -0.20 dB); x2 (three compiled parts, supported with a measured drop) | see `docs/export_support.md` | x4 `realesrgan/restore/x4/2026-11-16`; x4t `realesrgan/restore/x4t/2026-11-16`, held-out `realesrgan/restore/x4t/2026-11-16/2`; x2 `realesrgan/restore/x2/2026-11-16` |
| MiDaS | depth | s (supported with a measured drop; l blocked: does not quantize, section 11) | see `docs/export_support.md` | s `midas/depth/s/2026-11-11/4` |
| ZipDepth | depth | bnpu (hybrid at 384x384: the encoder through stage 3 block 4 on the Metis; the GCNet context block, whose softmax has no AIPU form, and the rest on the host; b the same cut) | see `docs/export_support.md` | bnpu `zipdepth/depth/bnpu/2026-11-18/4`, b `zipdepth/depth/b/2026-11-18/2` |
| DexiNed | edge | b (supported with a measured fidelity; whole graph; the user supplies the upstream BIPED-trained checkpoint, non-commercial) | see `docs/export_support.md` | b `dexined/edge/b/2026-11-16` |
| DeepLabV3 | semantic | mv3 (supported with a measured drop); r50, r101 (hybrid: backbone on Metis, ASPP + head on the host) | see `docs/export_support.md` | mv3 `deeplabv3/semantic/mv3/2026-11-11/6`; r50 `deeplabv3/semantic/r50/2026-11-17/3`; r101 `deeplabv3/semantic/r101/2026-11-17/4` |
| UNet | semantic | s (supported with a measured fidelity; whole graph at the 512x1024 training crop, compile about 12.7 h) | see `docs/export_support.md` | s `unet/semantic/s/2026-11-16` |
| PP-LiteSeg | semantic | t50, b50, t75, b75 (supported with a measured fidelity; whole graph at 512x1024 (t50, b50) or 768x1536 (t75, b75), the UAFM blends in an exact INT8-friendly form) | see `docs/export_support.md` | t50 `ppliteseg/semantic/t50/2026-11-18`; b50 `ppliteseg/semantic/b50/2026-11-18`; t75 `ppliteseg/semantic/t75/2026-11-18`; b75 `ppliteseg/semantic/b75/2026-11-18` |
| PIDNet | semantic | s, m, l (hybrid at 1024x1024: the stem and the P, I and D branches through layer 5 on the Metis, the PagFM fusions in an exact INT8-friendly form; PAPPM, the branch fusion and the head on the host; m and l calibrated with fraction=0.25) | see `docs/export_support.md` | s `pidnet/semantic/s/2026-11-18/3`; m `pidnet/semantic/m/2026-11-18/2`; l `pidnet/semantic/l/2026-11-18` |
| YOLOX | detect | n, t, s, m, l, x (n supported with a measured drop) | see `docs/export_support.md` | n `yolox/detect/n/2026-11-11/7`; t `yolox/detect/t/2026-11-11/2`; s `yolox/detect/s/2026-11-11/2`; m `yolox/detect/m/2026-11-11/2`; l `yolox/detect/l/2026-11-11/2`; x `yolox/detect/x/2026-11-11/2` |
| YOLOv1 | detect | b | see `docs/export_support.md` | b `yolo1/detect/b/2026-11-15/2` |
| YOLOv2 | detect | t, b | see `docs/export_support.md` | t `yolo2/detect/t/2026-11-13/2`; b `yolo2/detect/b/2026-11-13/2` |
| YOLOv3 | detect | t, b, spp | see `docs/export_support.md` | t `yolo3/detect/t/2026-11-11/6`; b `yolo3/detect/b/2026-11-11/6`; spp `yolo3/detect/spp/2026-11-11/3` |
| YOLOv4 | detect | t, b (b supported with a measured drop) | see `docs/export_support.md` | t `yolo4/detect/t/2026-11-11/5`; b `yolo4/detect/b/2026-11-11/6` |
| YOLOv7 | detect | b | see `docs/export_support.md` | b `yolo7/detect/b/2026-11-11/4` |
| PP-YOLOE | detect | s, m, l, x (s, m and x supported with a measured drop) | see `docs/export_support.md` | s `ppyoloe/detect/s/2026-11-11/3`; m `ppyoloe/detect/m/2026-11-11/2`; l `ppyoloe/detect/l/2026-11-11/2`; x `ppyoloe/detect/x/2026-11-11/2` |
| PicoDet | detect | s, m, l (supported with a measured drop) | see `docs/export_support.md` | s `picodet/detect/s/2026-11-11/7`; m `picodet/detect/m/2026-11-11/4`; l `picodet/detect/l/2026-11-11/4` |
| RTMDet | detect | t, s, m, l (supported with a measured drop; x hybrid: CSPNeXt-x backbone on the Metis, neck and heads on the host, +0.09) | see `docs/export_support.md` | t `rtmdet/detect/t/2026-11-11/8`; s `rtmdet/detect/s/2026-11-17/8`, held-out `rtmdet/detect/s/2026-11-17/9`; m `rtmdet/detect/m/2026-11-12/2`; l `rtmdet/detect/l/2026-11-12/2`; x `rtmdet/detect/x/2026-11-18/2` |
| RTMDet-Ins | segment | t, s, m, l (supported with a measured drop, box -2.51, -7.26, -3.96 and -5.34; x blocked: does not compile) | see `docs/export_support.md` | t `rtmdet/segment/t/2026-11-12/2`; s `rtmdet/segment/s/2026-11-12/4`; m `rtmdet/segment/m/2026-11-12/2`; l `rtmdet/segment/l/2026-11-12/2`; x blocked `rtmdet/segment/x/2026-11-12` |
| Mask R-CNN | detect, segment | r50 (hybrid at a fixed 800x800 canvas, as Faster R-CNN; for segment the host adds the mask head and mask paste) | see `docs/export_support.md` | r50 segment `mask_rcnn/segment/r50/2026-11-16`; detect `mask_rcnn/detect/r50/2026-11-16` |
| FCOS | detect | r50 (supported with a measured drop; six compiled parts, GroupNorm and the decode on the host) | see `docs/export_support.md` | r50 `fcos/detect/r50/2026-11-15/3` |
| RetinaNet | detect | r50 (supported: 1024x1024 canvas by default, -1.11 end to end; imgsz=800 is about 1.64x less compute at -2.25; r50v2 hybrid: ResNet-50 body on the Metis, FPN and GroupNorm heads on the host, -2.40) | see `docs/export_support.md` | r50 `retinanet/detect/r50/2026-11-14/4`; r50v2 `retinanet/detect/r50v2/2026-11-17` |
| CenterNet | detect | resdcn18, dla34 (hybrid at 512x512: the body without deformable convolutions on the Metis; the deformable up-path, heads and peak decode on the host) | see `docs/export_support.md` | resdcn18 `centernet/detect/resdcn18/2026-11-17/2`; dla34 `centernet/detect/dla34/2026-11-17/2` |
| SSD | detect | 300 (two compiled parts, conv4_3 L2Norm on the host) | see `docs/export_support.md` | 300 `ssd/detect/300/2026-11-13/2` |
| EfficientDet | detect | d0, d1, d2 (supported with measured drops; d1 and d2 at a fixed 512x512 canvas; d3 and d4 blocked: do not compile) | see `docs/export_support.md` | d0 `efficientdet/detect/d0/2026-11-12/3`; d1 `efficientdet/detect/d1/2026-11-13/3`; d2 `efficientdet/detect/d2/2026-11-14`; d3 blocked `efficientdet/detect/d3/2026-11-14` |
| Other families | all | all | not measured yet (section 11) | - |
| RT-DETR | detect | r18, r34, r50, r50m, r101, l, x (hybrid: CNN backbone and fusion neck on the Metis, attention encoder layer and decoder on the host) | see `docs/export_support.md` | r18 `rtdetr/detect/r18/2026-11-15/2` (300: `rtdetr/detect/r18/2026-11-15`); r34 `rtdetr/detect/r34/2026-11-15`; r50 `rtdetr/detect/r50/2026-11-15`; r50m `rtdetr/detect/r50m/2026-11-15`; r101 `rtdetr/detect/r101/2026-11-15`; l `rtdetr/detect/l/2026-11-15`; x `rtdetr/detect/x/2026-11-15` |
| RT-DETRv2 | detect | r18, r34, r50, r50m, r101 (hybrid, as RT-DETR: v1 backbone and fusion neck on the Metis, v2 decoder on the host) | see `docs/export_support.md` | r18 `rtdetrv2/detect/r18/2026-11-15`; r34 `rtdetrv2/detect/r34/2026-11-15`; r50 `rtdetrv2/detect/r50/2026-11-16`; r50m `rtdetrv2/detect/r50m/2026-11-16/2`; r101 `rtdetrv2/detect/r101/2026-11-16` |
| RT-DETRv2 | obb | n, s, m, l, x (hybrid, as RT-DETRv2 detect: HGNetv2 backbone and fusion neck on the Metis; attention encoder layer and rotated-box decoder on the host; calibrated on DOTA) | see `docs/export_support.md` | n `rtdetrv2/obb/n/2026-11-17`, s `rtdetrv2/obb/s/2026-11-17`, m `rtdetrv2/obb/m/2026-11-18`, l `rtdetrv2/obb/l/2026-11-18`, x `rtdetrv2/obb/x/2026-11-18` |
| RiO-DETRv2 | obb | n, s, m, l, x (hybrid, as RT-DETRv2-obb: HGNetv2 backbone and fusion neck on the Metis, the neck part with histogram ranges; attention encoder layer and rotated-box decoder on the host; DOTA-v1.0 weights, calibrated on DOTA) | see `docs/export_support.md` | n `riodetrv2/obb/n/2026-11-17`; s `riodetrv2/obb/s/2026-11-17`; m `riodetrv2/obb/m/2026-11-17`; l `riodetrv2/obb/l/2026-11-17`; x `riodetrv2/obb/x/2026-11-17` |
| D-FINE | detect | n, s, m, l, x (hybrid, as RT-DETR: HGNetv2 backbone and fusion neck on the Metis, attention encoder layer and decoder on the host) | see `docs/export_support.md` | n `dfine/detect/n/2026-11-15`; s `dfine/detect/s/2026-11-16`; m `dfine/detect/m/2026-11-16`; l `dfine/detect/l/2026-11-16`; x `dfine/detect/x/2026-11-16/2` |
| D-FINE | segment | n, s, m, l, x (hybrid, as D-FINE detect: HGNetv2 backbone and fusion neck on the Metis; mask decoder convolutions on the Metis; attention encoder layer, decoder and mask GroupNorms on the host) | see `docs/export_support.md` | n `dfine/segment/n/2026-11-17/3`, s `dfine/segment/s/2026-11-17/4`, m `dfine/segment/m/2026-11-17`, l `dfine/segment/l/2026-11-17`, x `dfine/segment/x/2026-11-17` |
| DEIM | detect | n, s, m, l, x (hybrid, as D-FINE; s with a measured drop of 5.51) | see `docs/export_support.md` | n `deim/detect/n/2026-11-15`; s `deim/detect/s/2026-11-16`; m `deim/detect/m/2026-11-16`; l `deim/detect/l/2026-11-16`; x `deim/detect/x/2026-11-16` |
| RT-DETRv4 | detect | s, m, l, x (hybrid, as D-FINE) | see `docs/export_support.md` | s `rtdetrv4/detect/s/2026-11-16`; m `rtdetrv4/detect/m/2026-11-16`; l `rtdetrv4/detect/l/2026-11-16`; x `rtdetrv4/detect/x/2026-11-16` |
| DEIMv2 | detect | atto, femto, pico (hybrid: backbone, convolutional encoder and decoder input projections on the Metis, decoder on the host), n (hybrid, as D-FINE); s, m, l and x are out of scope (DINOv3 ViT backbones) | see `docs/export_support.md` | atto `deimv2/detect/atto/2026-11-16`; femto `deimv2/detect/femto/2026-11-16`; pico `deimv2/detect/pico/2026-11-16`; n `deimv2/detect/n/2026-11-16` |
| DETR | detect | r50, r50dc5, r101, r101dc5 (hybrid: ResNet backbone and input projection on the Metis, transformer encoder, decoder and heads on the host) | see `docs/export_support.md` | r50 `detr/detect/r50/2026-11-15`; r50dc5 `detr/detect/r50dc5/2026-11-16`; r101 `detr/detect/r101/2026-11-16`; r101dc5 `detr/detect/r101dc5/2026-11-16` |
| PP-DocLayoutV3 | detect | l (hybrid: the HGNetV2 backbone and encoder input projections on the Metis; the attention encoder, mask-enhanced decoder and reading-order head on the host; measured as fidelity to FP32 on DocLayNet pages) | see `docs/export_support.md` | l `ppdoclayoutv3/detect/l/2026-11-16` |
| Deformable DETR | detect | r50, r50ss, r50ssdc5, r50refine, r50twostage (hybrid: ResNet-50 body on the Metis; input projections with GroupNorm, deformable transformer and heads on the host) | see `docs/export_support.md` | r50 `deformable_detr/detect/r50/2026-11-16`; r50ss `deformable_detr/detect/r50ss/2026-11-17`; r50ssdc5 `deformable_detr/detect/r50ssdc5/2026-11-17`; r50refine `deformable_detr/detect/r50refine/2026-11-17`; r50twostage `deformable_detr/detect/r50twostage/2026-11-17` |
| DINO-DETR | detect | r50, r50s5 (hybrid, as Deformable DETR: ResNet-50 body on the Metis; input projections with GroupNorm, deformable transformer and heads on the host; swinl out of scope, Swin-L transformer backbone) | see `docs/export_support.md` | r50 `dinodetr/detect/r50/2026-11-17`, r50s5 `dinodetr/detect/r50s5/2026-11-17` |
| Faster R-CNN | detect | n, s, m, l (hybrid at a fixed canvas, 320x320 for n and 800x800 for s, m and l: backbone and FPN on the Metis, RPN, RoIAlign, box head and NMS on the host) | see `docs/export_support.md` | n `faster_rcnn/detect/n/2026-11-15`; s `faster_rcnn/detect/s/2026-11-16`; m `faster_rcnn/detect/m/2026-11-16`; l `faster_rcnn/detect/l/2026-11-16` |
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
| PP-YOLOE-x | detect | COCO val2017, 1000 images | 64 coco128 | 0.4918 | 0.4747 | -1.71 (supported) |
| PicoDet-s | detect | COCO val2017, 1000 images | 64 coco128 | 0.2687 | 0.2453 | -2.34 (supported; -1.28 on 300) |
| PicoDet-m | detect | COCO val2017, 1000 images | 64 coco128 | 0.3449 | 0.3163 | -2.86 (supported) |
| PicoDet-l | detect | COCO val2017, 1000 images | 64 coco128 | 0.3971 | 0.3801 | -1.70 (supported) |
| RTMDet-t | detect | COCO val2017, 1000 images | 64 coco128 | 0.3906 | 0.3677 | -2.29 (supported) |
| RTMDet-s | detect | COCO val2017, 1000 images | 64 coco128 | 0.4121 | 0.3681 | -4.40 (supported; -5.43 on 300, -5.15 on a held-out 1000) |
| RTMDet-m | detect | COCO val2017, 1000 images | 64 coco128 | 0.4727 | 0.4461 | -2.66 (supported; -2.11 on 300) |
| RTMDet-l | detect | COCO val2017, 1000 images | 64 coco128 | 0.4805 | 0.4618 | -1.87 (supported; -1.92 on 300) |
| RTMDet-x | detect | COCO val2017, 1000 images | 64 coco128 | 0.5128 | - | does not compile (blocked) |
| RetinaNet-R50 (1024x1024 canvas, default) | detect | COCO val2017, 1000 images | 128 coco128 | 0.3472 | 0.3410 | -0.62 (supported: -1.11 vs own resize) |
| RetinaNet-R50 (800x800 canvas, imgsz=800) | detect | COCO val2017, 1000 images | 64 coco128 | 0.3318 | 0.3296 | -0.22 (supported: canvas -2.03) |
| SSD300 (two parts) | detect | COCO val2017, 1000 images | 128 coco128 | 0.2408 | 0.2403 | -0.05 (+0.09 on 300) |
| RT-DETR-r18 (hybrid) | detect | COCO val2017, 1000 images | 64 coco128 | 0.4535 | 0.4528 | -0.07 (hybrid; -0.12 on 300) |
| RT-DETR-r34 (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.4985 | 0.4982 | -0.03 (hybrid) |
| RT-DETR-r50 (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5413 | 0.5368 | -0.45 (hybrid) |
| RT-DETR-r50m (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5139 | 0.5072 | -0.67 (hybrid) |
| RT-DETR-r101 (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5518 | 0.5502 | -0.16 (hybrid) |
| RT-DETR-l (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5402 | 0.5376 | -0.26 (hybrid; HGNetv2 backbone part per channel, neck part per tensor) |
| RT-DETR-x (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5587 | 0.5573 | -0.14 (hybrid; HGNetv2 backbone part per channel, neck part per tensor) |
| RT-DETRv2-r18 (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.4965 | 0.4840 | -1.25 (hybrid) |
| RT-DETRv2-r34 (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5050 | 0.4847 | -2.03 (hybrid) |
| RT-DETRv2-r50 (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5494 | 0.5322 | -1.72 (hybrid; neck part histogram ranges, -3.79 per tensor min-max) |
| RT-DETRv2-r50m (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5321 | 0.5065 | -2.56 (hybrid; neck part histogram ranges, -3.67 per tensor min-max) |
| RT-DETRv2-r101 (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5522 | 0.5510 | -0.12 (hybrid; neck part histogram ranges) |
| RT-DETRv2-n-obb (hybrid, 1024x1024) | obb (mAP50-95 OBB) | DOTA val, 300 images, DOTA-v1.0 classes | 64 DOTA | 0.3804 | 0.3648 | -1.56 (hybrid) |
| RT-DETRv2-s-obb (hybrid, 1024x1024) | obb (mAP50-95 OBB) | DOTA val, 300 images, DOTA-v1.0 classes | 64 DOTA | 0.5923 | 0.5793 | -1.30 (hybrid) |
| RT-DETRv2-m-obb (hybrid, 1024x1024) | obb (mAP50-95 OBB) | DOTA val, 300 images, DOTA-v1.0 classes | 64 DOTA | 0.5918 | 0.5915 | -0.03 (hybrid) |
| RT-DETRv2-l-obb (hybrid, 1024x1024) | obb (mAP50-95 OBB) | DOTA val, 300 images, DOTA-v1.0 classes | 64 DOTA | 0.6223 | 0.6213 | -0.10 (hybrid) |
| RT-DETRv2-x-obb (hybrid, 1024x1024) | obb (mAP50-95 OBB) | DOTA val, 300 images, DOTA-v1.0 classes | 64 DOTA | 0.6506 | 0.6458 | -0.48 (hybrid) |
| RiO-DETRv2-n-obb (hybrid, 1024x1024) | obb (mAP50-95 OBB) | DOTA val, 300 images, DOTA-v1.0 classes | 64 DOTA | 0.5581 | 0.4912 | -6.69 (hybrid) |
| RiO-DETRv2-s-obb (hybrid, 1024x1024) | obb (mAP50-95 OBB) | DOTA val, 300 images, DOTA-v1.0 classes | 64 DOTA | 0.5958 | 0.5859 | -0.99 (hybrid) |
| RiO-DETRv2-m-obb (hybrid, 1024x1024) | obb (mAP50-95 OBB) | DOTA val, 300 images, DOTA-v1.0 classes | 64 DOTA | 0.6323 | 0.6325 | +0.02 (hybrid) |
| RiO-DETRv2-l-obb (hybrid, 1024x1024) | obb (mAP50-95 OBB) | DOTA val, 300 images, DOTA-v1.0 classes | 64 DOTA | 0.6239 | 0.6169 | -0.70 (hybrid) |
| RiO-DETRv2-x-obb (hybrid, 1024x1024) | obb (mAP50-95 OBB) | DOTA val, 300 images, DOTA-v1.0 classes | 64 DOTA | 0.6964 | 0.6955 | -0.09 (hybrid) |
| D-FINE-n (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.4248 | 0.4209 | -0.39 (hybrid; backbone part per channel, neck part per tensor) |
| D-FINE-s (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5155 | 0.4989 | -1.66 (hybrid; backbone part per channel, neck part per tensor) |
| D-FINE-m (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5653 | 0.5572 | -0.81 (hybrid; backbone part per channel, neck part per tensor) |
| D-FINE-l (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5862 | 0.5739 | -1.23 (hybrid; backbone part per channel, neck part per tensor) |
| D-FINE-x (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.6098 | 0.6039 | -0.59 (hybrid; backbone part per channel, neck part per tensor) |
| D-FINE-n-seg (hybrid) | segment (mask mAP50-95; box 0.4091 -> 0.4085) | COCO val2017, 300 images | 64 coco128 | 0.3536 | 0.3485 | -0.51 (hybrid) |
| D-FINE-s-seg (hybrid) | segment (mask mAP50-95; box 0.4709 -> 0.3622) | COCO val2017, 300 images | 64 coco128 | 0.4051 | 0.3035 | -10.16 (hybrid; the backbone's stride-32 output drifts, cosine 0.75) |
| D-FINE-m-seg (hybrid) | segment (mask mAP50-95; box 0.5014 -> 0.5016) | COCO val2017, 300 images | 64 coco128 | 0.4263 | 0.4257 | -0.06 (hybrid) |
| D-FINE-l-seg (hybrid) | segment (mask mAP50-95; box 0.5221 -> 0.5156) | COCO val2017, 300 images | 64 coco128 | 0.4431 | 0.4388 | -0.43 (hybrid) |
| D-FINE-x-seg (hybrid) | segment (mask mAP50-95; box 0.5214 -> 0.5177) | COCO val2017, 300 images | 64 coco128 | 0.4512 | 0.4500 | -0.12 (hybrid) |
| DEIM-n (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.4238 | 0.4213 | -0.25 (hybrid; backbone part per channel, neck part per tensor) |
| DEIM-s (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.4980 | 0.4429 | -5.51 (hybrid; the loss is in the backbone part's stride-32 output, no scheme recovers it) |
| DEIM-m (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5358 | 0.5390 | +0.32 (hybrid; backbone part per channel, neck part per tensor) |
| DEIM-l (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5614 | 0.5592 | -0.22 (hybrid; backbone part per channel, neck part per tensor) |
| DEIM-x (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5783 | 0.5766 | -0.17 (hybrid; backbone part per channel, neck part per tensor) |
| RT-DETRv4-s (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5090 | 0.4984 | -1.06 (hybrid; backbone part per channel, neck part per tensor) |
| RT-DETRv4-m (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5546 | 0.5466 | -0.80 (hybrid; backbone part per channel, neck part per tensor) |
| RT-DETRv4-l (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5699 | 0.5679 | -0.20 (hybrid; backbone part per channel, neck part per tensor) |
| RT-DETRv4-x (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.5782 | 0.5731 | -0.51 (hybrid; backbone part per channel, neck part per tensor) |
| DEIMv2-atto (hybrid, 320x320) | detect | COCO val2017, 300 images | 64 coco128 | 0.2299 | 0.1951 | -3.48 (hybrid; one compiled part, decoder on the host) |
| DEIMv2-femto (hybrid, 416x416) | detect | COCO val2017, 300 images | 64 coco128 | 0.3078 | 0.2801 | -2.77 (hybrid; one compiled part, decoder on the host) |
| DEIMv2-pico (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.3852 | 0.3697 | -1.55 (hybrid; one compiled part, decoder on the host) |
| DEIMv2-n (hybrid) | detect | COCO val2017, 300 images | 64 coco128 | 0.4247 | 0.4188 | -0.59 (hybrid; backbone part per channel, neck part per tensor) |
| DETR-r50 (hybrid, 800x800) | detect | COCO val2017, 300 images | 64 coco128 | 0.3879 | 0.3852 | -0.27 (hybrid) |
| DETR-r50dc5 (hybrid, 800x800) | detect | COCO val2017, 300 images | 64 coco128 | 0.4172 | 0.4125 | -0.47 (hybrid) |
| DETR-r101 (hybrid, 800x800) | detect | COCO val2017, 300 images | 64 coco128 | 0.4019 | 0.3732 | -2.87 (hybrid) |
| DETR-r101dc5 (hybrid, 800x800) | detect | COCO val2017, 300 images | 64 coco128 | 0.4232 | 0.4172 | -0.60 (hybrid) |
| Deformable DETR-r50 (hybrid, 800x800) | detect | COCO val2017, 300 images | 64 coco128 | 0.4176 | 0.4171 | -0.05 (hybrid) |
| Deformable DETR-r50ss (hybrid, 800x800) | detect | COCO val2017, 300 images | 64 coco128 | 0.3704 | 0.3682 | -0.22 (hybrid) |
| Deformable DETR-r50ssdc5 (hybrid, 800x800) | detect | COCO val2017, 300 images | 64 coco128 | 0.3920 | 0.3900 | -0.20 (hybrid) |
| Deformable DETR-r50refine (hybrid, 800x800) | detect | COCO val2017, 300 images | 64 coco128 | 0.4559 | 0.4526 | -0.33 (hybrid) |
| Deformable DETR-r50twostage (hybrid, 800x800) | detect | COCO val2017, 300 images | 64 coco128 | 0.4503 | 0.4504 | +0.01 (hybrid) |
| DINO-DETR-r50 (hybrid, 800x800) | detect | COCO val2017, 300 images | 64 coco128 | 0.4714 | 0.4710 | -0.04 (hybrid) |
| DINO-DETR-r50s5 (hybrid, 800x800) | detect | COCO val2017, 300 images | 64 coco128 | 0.4917 | 0.4891 | -0.26 (hybrid) |
| Faster R-CNN-n (hybrid, 320x320 canvas) | detect | COCO val2017, 300 images | 64 coco128 | 0.2005 | 0.1896 | -1.09 (hybrid; canvas -3.30 vs the native 0.2335; measured with the first 64 coco128 images, fraction=0.5: with all 128, the default, the CPU-quantized model scores 2.2 points higher on 100 of these images) |
| Faster R-CNN-s (hybrid, 800x800 canvas) | detect | COCO val2017, 300 images | 64 coco128 | 0.2907 | 0.2867 | -0.40 (hybrid; FP32 at the same canvas; measured with the first 64 coco128 images, fraction=0.5: with all 128, the default, the CPU-quantized model gives -2.32 on these images instead of -0.33) |
| Faster R-CNN-m (hybrid, 800x800 canvas) | detect | COCO val2017, 300 images | 64 coco128 | 0.3472 | 0.3456 | -0.16 (hybrid; canvas -3.12 vs the native 0.3784) |
| Faster R-CNN-l (hybrid, 800x800 canvas) | detect | COCO val2017, 300 images | 64 coco128 | 0.4468 | 0.4489 | +0.21 (hybrid) |
| Mask R-CNN-r50 (hybrid, 800x800 canvas) | segment | COCO val2017, 300 images | 64 coco128 | 0.4045 | 0.4039 | -0.06 mask, +0.02 box (hybrid; FP32 at the same canvas) |
| Mask R-CNN-r50 (hybrid, 800x800 canvas) | detect | COCO val2017, 300 images | 64 coco128 | 0.4476 | 0.4481 | +0.05 (hybrid; same Metis part as segment) |
| PP-DocLayoutV3-l (hybrid, 800x800) | detect | DocLayNet v1.2 val, 300 pages | 64 other DocLayNet pages | 1.0000 | 0.9795 | matched-box F1 vs FP32 (fidelity; score correlation 0.947, matched IoU 0.991) |
| EfficientDet-d0 | detect | COCO val2017, 1000 images | 128 coco128 | 0.3363 | 0.3185 | -1.78 (supported) |
| EfficientDet-d1 (512x512 canvas) | detect | COCO val2017, 1000 images | 128 coco128 | 0.3639 | 0.3475 | -1.64 (supported; canvas -3.10 more) |
| EfficientDet-d2 (512x512 canvas) | detect | COCO val2017, 1000 images | 128 coco128 | 0.3744 | 0.3606 | -1.38 (supported; canvas -5.31 more) |
| RTMDet-Ins-t | segment (mask) | COCO val2017, 1000 images | 64 coco128 | 0.3394 | 0.3250 | -1.44 (supported; box -2.51) |
| RTMDet-Ins-s | segment (mask) | COCO val2017, 1000 images | 64 coco128 | 0.3719 | 0.3139 | -5.80 (supported; box -7.26) |
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
| VGG-19-BN (hybrid: block 5 + classifier on the host) | 224 | 0.644 | 0.642 | -0.2 | 2.4 min |
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

VGG-19-BN runs hybrid: conv blocks 1-4 (through the fourth max pool) run on
the Metis AIPU with min-max ranges, and conv block 5, the average pool and the
classifier run on the host as FP32 ONNX. Top-1 is 0.642 vs 0.644 FP32 on the
1000 images (-0.2) and 0.608 vs 0.606 on a held-out 1000-image ImageNetV2 set
(+0.2). Run whole on the Metis it loses 5.1 points with min-max ranges and 6.0
with histogram ranges, although the CPU-quantized model loses only 2.4 and 1.9:
the device output drifts from the CPU-quantized model layer by layer through
block 5 (output cosine 0.984 at the fourth pool, 0.785 after the last
convolution), with no single diverging operator. Without ``hybrid_split`` in
its per-size recipe the whole-graph form is exported (ledger
`vgg/classify/19bn/2026-11-12`).

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

DeepLabV3-ResNet50 and -ResNet101 run hybrid: the ResNet backbone through
layer4 runs on the Metis AIPU, and the ASPP (plain dilated convolutions), the
head and the logit upsampling run on the host as FP32 ONNX. On the 300 images
mIoU is 0.5203 vs 0.5243 FP32 for R50 (-0.40) and 0.5879 vs 0.5888 for R101
(-0.09). Run whole on the Metis, with each dilated ASPP convolution rewritten as
per-tap 1x1 convolutions plus zero-filled shifts (they find no L1 tiling at
output stride 8 as dilated convolutions), they lose 5.09 (R50) and 20.93 (R101):
the CPU-quantized model stays within 0.4 mIoU of FP32, and a device bisect puts
the divergence in the rewritten ASPP (device vs CPU-quantized cosine 0.995 for
the backbone alone, 0.53 with the ASPP). Without ``hybrid_split`` in their
per-size recipe the whole-graph form is exported (ledger
`deeplabv3/semantic/r50/2026-11-17/2`, `deeplabv3/semantic/r101/2026-11-14`).

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
not by the canvas.

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

RT-DETR-r18 (hybrid, section 8) runs 83% of its multiply-accumulates on the
Metis (25.7 of 30.8 GMACs: the ResNet backbone and the fusion neck), but the
host steps take most of the time: per image, 82 ms on the AIPU parts and
302 ms in the two host ONNX graphs at batch 1 (27 and 200 ms at batch 8),
2.5 img/s end to end at batch 1 and 4.1 img/s at batch 8. RT-DETR-r34
runs 87% of its MACs on the Metis (40.8 of 46.8 G) at 2.8 and 4.6 img/s
(AIPU 75 ms, host 274 ms per image at batch 1); RT-DETR-r50 88% (60.9 of
68.9 G) at 1.5 and 1.7 img/s (AIPU 102 ms, host 567 ms); RT-DETR-r50m 90% (45.2 of
50.3 G) at 2.0 and 2.3 img/s (AIPU 102 ms, host 395 ms); RT-DETR-r101 93% (121.7
of 130.3 G) at 1.4 and 1.5 img/s (AIPU 138 ms, host 578 ms); RT-DETR-l (HGNetv2) 86%
(47.9 of 55.8 G) at 2.2 and 2.2 img/s (AIPU 89 ms, host 363 ms); RT-DETR-x 93%
(109.4 of 118.0 G) at 1.5 and 2.4 img/s (AIPU 149 ms, host 485 ms).
RT-DETRv2-r18 (same split) runs 84% on the Metis at 1.7 and 2.3 img/s;
RT-DETRv2-r34 at 3.4 and 4.2 img/s, RT-DETRv2-r50 at 3.1 and 4.0 img/s, RT-DETRv2-r101 at 1.4 and 2.6 img/s, RT-DETRv2-r50m at 1.9 and 4.0 img/s. RT-DETRv2's neck part compiles with
histogram ranges: on r50 per-tensor min-max left its finest memory map at a
0.68 held-out cosine and the export 3.79 points below FP32, histogram 0.91 and
1.72 points; r18 and r34 keep their measured min-max neck until re-measured. D-FINE-n runs 83% of its MACs on the Metis
(3.1 of 3.7 G) at 3.7 and 7.0 img/s, D-FINE-s 80% at 5.2 and 6.4 img/s, D-FINE-m 89% at 3.4 and 3.5 img/s, D-FINE-l 91% at 1.7 and 2.0 img/s, D-FINE-x 96% at 2.8 and 3.3 img/s; DEIM-n (the same
network shape) at 4.7 and 8.4 img/s, DEIM-m 89% at 3.8 and 4.5 img/s, DEIM-l 91% at 2.2 and 4.0 img/s, DEIM-x 96% at 2.7 and 3.6 img/s. RT-DETRv4-s (the same split)
80% at 4.6 and 5.6 img/s, RT-DETRv4-m 89% at 2.8 and 3.1 img/s, RT-DETRv4-l 91% at
2.6 and 3.1 img/s, RT-DETRv4-x 96% at 2.6 and 3.2 img/s.
DEIMv2-n runs at 10.0 and 12.1 img/s; DEIMv2 atto, femto and pico, whose encoder
has no attention, run everything but the decoder on the Metis (95%, 93% and 96%
of their MACs) at 42.7, 12.4 and 7.9 img/s at batch 1.
DETR-r50 (backbone only, 800x800) runs 87% of its MACs on the Metis (52.5 of
60.2 G) at 2.0 and 2.3 img/s, but only 36 ms of its 459 ms chain at batch 1:
its six-layer encoder over all 625 positions and the decoder run on the host.
The dilated sizes (r50dc5, r101dc5) feed the host transformer a stride-16 map
(2,500 positions) and run at 0.6-0.9 img/s; DETR-r101 runs at 4.0 img/s.
D-FINE-seg's mask decoder normalizes with GroupNorm (no AIPU form), so its two
convolutions are compiled parts of their own and the GroupNorms run on the host
between them: D-FINE-n-seg runs 81% of its MACs on the Metis at 4.1 img/s (4.3
at batch 8; 3.0 / 3.6 with the whole mask decoder on the host), D-FINE-s-seg
85% at 2.3 img/s (2.8; 1.8 / 1.9); m, l and x run 89-93% of their MACs on the
Metis at 1.6-1.7 img/s (1.8-2.8 at batch 8). D-FINE-s-seg loses 10 pt: its
segmentation-trained backbone quantizes worse than D-FINE-s detect's.
Deformable DETR-r50 runs 43% of its MACs on the Metis but is host-bound: its
input projections normalize with GroupNorm (no AIPU form) and its deformable
encoder samples four levels, so it runs at 0.3 img/s.
DINO-DETR-r50 takes the same cut (29% of its MACs on the Metis); its 900-query
two-stage decoder makes it slower still, 0.2 img/s. DINO-DETR-r50s5 adds a
stride-4 level, so its host transformer is 467 G of the 519 G MACs (10% on the
Metis) and it runs at 0.07 img/s (15 s per image on the host).
Faster R-CNN-n runs 81% of its MACs on the Metis (0.58 of 0.72 G) at 5.1 and
6.5 img/s; its RPN and box head take 167 of the 182 ms chain. Those were
measured with 4 host threads on a busy 18-core NUC (load average about 20),
so they are a floor; the host share depends on the CPU.

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

Hybrid exports (support tier `hybrid`, RT-DETR) run only the convolutional
parts on the Metis. RT-DETR's attention encoder layer, its query selection
(TopK over every feature position) and its decoder's deformable attention
(sampling at learned offsets, softmax, LayerNorm) have no AIPU form, so the
artifact chains four steps (`parts` in `metadata.yaml`):

```text
image -> [Metis: ResNet backbone, encoder input projections] -> 3 feature maps
      -> [host ONNX: attention encoder on the stride-32 map, levels stacked into one canvas]
      -> [Metis: fusion neck, decoder input projections] -> 3 memory maps
      -> [host ONNX: query selection, decoder] -> pred_logits, pred_boxes -> Results
```

DEIMv2's small sizes (atto, femto, pico) have an encoder without attention
(projections, pooling and CSP blocks), so their hybrid export is one compiled
part, the backbone, the whole encoder and the decoder's input projections, and
the decoder on the host (`hybrid` split `encoder`).

Faster R-CNN is two-stage: proposals are chosen at run time and RoIAlign crops
them, so only the first stage's convolutions compile. The Metis part takes the
fixed square canvas (the image stretched to the model's `min_size`, RGB in
[0, 1]) and runs the input normalization, the backbone and the FPN; the host
graph adds the FPN's extra pooled level and runs the region proposal network,
RoIAlign, the box head and NMS, ending at the ONNX export's `boxes`, `scores`
and `labels`. The stretch costs accuracy against the native aspect-preserving
resize, so section 7 gives the INT8 drop against FP32 at the same canvas and
the canvas cost separately.

Mask R-CNN-r50 shares Faster R-CNN-l's ResNet-50-FPN v2 and runs the same Metis part. Its host graph adds the mask branch: RoIAlign at 14x14 on the kept boxes, four convolutions, the mask predictor, and the paste of each soft mask onto the canvas; the backend resizes the masks to the image with the boxes. On 300 COCO val2017 images at the 800x800 canvas the device gives mask mAP50-95 0.4039 vs 0.4045 FP32 (box 0.4323 vs 0.4321; ledger `mask_rcnn/segment/r50/2026-11-16`). The detect task compiles the same Metis part (identical graph and calibration) and its host graph stops before the mask branch: box mAP50-95 0.4481 vs 0.4476 FP32 at the same canvas (ledger `mask_rcnn/detect/r50/2026-11-16`).

PP-DocLayoutV3-l (document layout) splits after its HGNetV2-L backbone: the Metis part returns the stride-4 stage (the mask branch's input) and the encoder's input projections at strides 8, 16 and 32; the host runs the attention encoder, the mask-enhanced decoder and the reading-order head. With no document-layout labels on the measuring machine it is measured as fidelity: on 300 DocLayNet v1.2 validation pages (calibrated on 64 other pages) the device keeps 97.9% matched-box F1 against FP32 at IoU 0.5 (score correlation 0.947; ledger `ppdoclayoutv3/detect/l/2026-11-16`).

DETR (the original, single-level detector) has no convolutional neck: its
hybrid export is two steps, the ResNet backbone and the 1x1 input projection
on the Metis, then the sine positions, the transformer and the prediction
heads on the host (`hybrid` split `backbone`).

A compiled part takes one input, so the host stacks the three levels on
channels at stride 8 (coarser levels nearest-upsampled, each scaled by the
power of two that brings its calibration peak nearest the largest level's,
so all share the canvas's one INT8 scale well);
the neck part slices them back and undoes the upsampling and the scale with
constant 1x1 convolutions, exactly (strided ones are dense with the scale on
the diagonal: the compiler rejects a strided 1x1 depthwise convolution there). The host steps run in FP32 through ONNX
Runtime (`onnxruntime` must be installed; `OMP_NUM_THREADS` caps its
threads) and end at the ONNX export's `pred_logits` and `pred_boxes`
(layout `graph_outputs_v1`), which the shared parser decodes. The backend's
`chain_timing` holds the time spent in the AIPU parts and in the host steps;
section 7 quotes the measured split.

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
| `hybrid` | hybrid exports only: `split` (`neck`, `backbone` or `encoder`), what runs on the AIPU and on the host, the canvas level scales, and the neck part's quantization scheme (always per tensor: its input feeds a depthwise convolution, which the compiler cannot take per-channel input ranges for at a part boundary) |
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

Blocked for structural reasons (accuracy never blocks an export):

- L2CS (gaze): its only checkpoint is trained on Gaze360, whose licence forbids redistributing models trained on it; LibreYOLO fetches it from the authors and does not mirror it, so no Axelera export is measured or shipped.
- EfficientDet-d3: lowering finds no L1 tiling at its native 896 (2.52 MB needed, 2.15 MB available) nor at a 512 canvas (2.58 vs 2.14 MB); its 160-channel BiFPN and heads set the need (ledgers `efficientdet/detect/d3/2026-11-13/2`, `efficientdet/detect/d3/2026-11-14`). Use d2.
- EfficientDet-d4: lowering finds no L1 tiling at its native 1024 (2.69 MB needed, 0.94 MB available) nor at a 512 canvas (2.10 vs 1.22 MB). Prefix cuts at 512 put the overflow in the EfficientNet-B4 backbone: stages 0-4 compile, the 272-channel stage (blocks.5) needs 3.12 MB vs 2.47 MB (ledgers `efficientdet/detect/d4/2026-11-15` to `/4`).
- RTMDet-Ins-x: at 640x640 lowering fails in AssignTilingFactors (ScheduleMemory, LowerTIR): no tiling factor fits L1 (3.99 MB needed, 3.32 MB available; 2 h 08 min, 8.1 GB peak RSS; ledger `rtmdet/segment/x/2026-11-12`). Use RTMDet-Ins-l. Its CSPNeXt-x backbone alone compiles in 24 min, so the overflow is in the neck and heads; a backbone | neck+head split would need the three backbone levels packed into one input (the quantizer takes no multi-input calibration; ledger `rtmdet/segment/x/2026-11-17`).
- MiDaS-l (DPT ViT-L, 384 px): quantization stops at the quantizer's ONNX import ("unexpected input value type") on the ViT encoder (48 LayerNormalization, 24 Softmax, 28 Erf, attention MatMuls; the same graph's 5 Resize nodes also occur in MiDaS-s, which compiles). It peaks at 18 GB of host RAM before failing. Use MiDaS-s.
- ConvNeXt and ConvNeXt V2: per-pixel LayerNorm (and V2's Global Response Normalization) divide by statistics of each pixel or map; quantization stops on the first of them. The channel mean and variance have exact AIPU forms, but the per-pixel reciprocal square root feeding ConvNeXt's GELU has none; a host split at every LayerNorm would need about 23 parts for ConvNeXt-t, so a hybrid export (convolutions on the Metis, LayerNorms on the host) is the route if ConvNeXt is ever needed (ledgers `convnext/classify/t/2026-11-11`, `convnextv2/classify/atto/2026-11-11`).
- EC and TinyFormer: gather-based deformable attention in the decoder; no quantizer converter for its GreaterOrEqual bounds masks (ledgers `ec/detect/s/2026-11-11`, `tinyformer/detect/s/2026-11-11`).
- TEED: the Smish activation takes a logarithm, which quantization stops on; no trained weights are published (ledger `teed/edge/t/2026-11-11/2`).

Not measured on a Metis yet (nothing is known to fail; the export raises until a size is measured):

- NAFNet: no trained LibreYOLO weights are published, so nothing can be exported or measured. When weights exist, the input skip around the whole graph runs on the host (as for Real-ESRGAN), and the per-pixel LayerNorms have the same reciprocal square root as ConvNeXt's (ledger `nafnet/restore/s/2026-11-11`).
- FOMO (point): the published checkpoints (s, m, l) are architecture-only random weights, so nothing trained can be exported or measured; it is a small MobileNetV2-style CNN with no operator known to block.
- YOLO9-P2-t: no trained checkpoint is published (P2-s is validated on VisDrone; its checkpoint is a non-commercial research preview).
- Other families.

Out of scope (not attempted): models whose backbone is itself a transformer. FeyNobg (BiRefNet, Swin-L) is one of them.
A hybrid export runs a convolutional backbone on the Metis and the attention
layers on the host CPU; with an attention backbone almost nothing would be
left on the Metis. This covers ViT, DeiT, Swin, DINOv2, CLIP, SigLIP2, PE,
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
  adapter to multi-scale features.
- DEIMv2 s, m, l and x: DINOv3 ViT backbones with a spatial-tuning adapter
  (vit_tiny, vit_tinyplus, dinov3_vits16, dinov3_vits16plus). The atto,
  femto, pico and n sizes use HGNetv2 and export as hybrid (section 7).
- DINO-DETR swinl: a Swin-L backbone (shifted-window attention, embed 192,
  depths 2/2/18/2, window 12). The r50 and r50s5 sizes use ResNet-50 and are
  hybrid candidates.

Other known limits:

- YOLO-NAS-R (obb): validated on 300 DOTA-v2.0 tiles (the paired set prepared for this work), not on a 1000-image set.
- YOLOX-n: supported, mAP50-95 1.81 points below FP32 (1.66 with 128 calibration images; section 7).
- PP-YOLOE-s, -m and -x: supported, mAP50-95 1.2 to 1.7 points below FP32 (section 7). The loss is in the box-distribution outputs and does not come from their output range: windowing them, or histogram ranges, did not help. PP-YOLOE-l is validated.
- YOLOv4-b: supported, mAP50-95 1.52 points below FP32 with its recipe (section 7): its box outputs lose resolution in the Mish backbone; histogram ranges and a box-output window recover most of it (-6.8 to -1.52).
- YOLO-NAS-pose n and s: supported, keypoint mAP50-95 1.3 and 2.2 points below FP32 (section 7); m and l are validated.
- RF-DETR and other transformer-backbone families: attention, data-dependent
  TopK and grid sampling are outside what the compiler places on the AIPU.
  RT-DETR, whose backbone is a CNN, exports as hybrid instead (section 8):
  its backbone and fusion neck run on the Metis, its attention encoder layer
  and decoder on the host.
- MobileNetV4 (classify) s and l: supported, top-1 1.7 points below FP32
  at best (section 7). Per-tensor INT8 of the inverted-residual streams (the
  stage-2 stream of s spans about +-400 while its median channel stays
  within +-14) is the cause. Use m.
- Europa and multi-device scheduling: not covered.
- Large models on M.2 cards can hit the card's power limit.
