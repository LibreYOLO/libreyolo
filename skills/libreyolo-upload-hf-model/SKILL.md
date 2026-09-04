---
name: libreyolo-upload-hf-model
description: Prepare and upload a LibreYOLO weight repo to the HuggingFace LibreYOLO org. Use when publishing new weights (new family, new size, or new task like -seg). Covers filename, README, LICENSE, NOTICE, and collection membership.
---

# Upload a LibreYOLO weight repo to HuggingFace

Use this skill when publishing model weights to `https://huggingface.co/LibreYOLO/<repo>`.

Scope: **weight-only repos** (one `.pt`, one canonical filename, matching a family defined in `libreyolo/models/<family>/model.py`). Not product repos (`face-*`, `libreyolo-web`) — those are bespoke and out of scope.

## The 5-file contract

Every weight repo contains exactly these 5 files. No more, no less.

```
<repo>/
├── .gitattributes       # LFS rules (copy from any existing LibreYOLO weight repo, 1519 bytes)
├── README.md            # YAML frontmatter + Source / Modifications / License
├── LICENSE              # upstream license text, verbatim
├── NOTICE               # attribution block (required for Apache-2.0 upstreams)
└── Libre<Family><size>[-<task>].pt   # the canonical weight file
```

Do **not** upload:

- Lowercase or legacy filenames (`libreyolo9s.pt`, `rf-detr-nano.pth`).
- Raw upstream checkpoints alongside the converted weight.
- Both `.pt` and `.pth` of the same weights.

## Canonical filename

Derived from code, not invented:

```
name = FILENAME_PREFIX + size + ("-" + task if task else "")
file = name + ".pt"
```

`FILENAME_PREFIX` per family — read from `libreyolo/models/<family>/model.py`:

| Family | Prefix | Example |
|---|---|---|
| YOLOX | `LibreYOLOX` | `LibreYOLOXs.pt` |
| YOLO1 | `LibreYOLO1` | `LibreYOLO1b.pt` (public-domain Darknet, VOC-20; `t` weights lost upstream) |
| YOLO2 | `LibreYOLO2` | `LibreYOLO2b.pt` (public-domain Darknet) |
| YOLO3 | `LibreYOLO3` | `LibreYOLO3b.pt`, `LibreYOLO3spp.pt` (public-domain Darknet) |
| YOLO4 | `LibreYOLO4` | `LibreYOLO4b.pt` (public-domain Darknet) |
| YOLO7 | `LibreYOLO7` | `LibreYOLO7b.pt` (MIT MultimediaTechLab/YOLO) |
| YOLO9 | `LibreYOLO9` | `LibreYOLO9m.pt` |
| YOLO9-P2 | `LibreYOLO9P2` | `LibreYOLO9P2s.pt`, `LibreYOLO9P2s-visdrone.pt` (dataset-variant suffix) |
| RFDETR | `LibreRFDETR` | `LibreRFDETRn.pt`, `LibreRFDETRn-seg.pt`, `LibreRFDETRx-pose.pt` |
| DETR | `LibreDETR` | `LibreDETRr50.pt` (original DETR; Apache-2.0 code + weights; inference-only) |
| LWDETR | `LibreLWDETR` | `LibreLWDETRt.pt` (LW-DETR, RF-DETR's ancestor; Apache-2.0 code + weights; inference-only) |
| DOMEDETR | `LibreDOMEDETR` | `LibreDOMEDETRs-aitod.pt`, `LibreDOMEDETRs-visdrone.pt` (dataset-specific tiny-object weights; academic-research-only, see below) |
| FasterRCNN | `LibreFasterRCNN` | `LibreFasterRCNNn.pt` (modernized torchvision Faster R-CNN; BSD-3-Clause implied for weights, with the pretrained-model caveat on every card; inference-only) |
| RetinaNet | `LibreRetinaNet` | `LibreRetinaNetr50.pt` (torchvision RetinaNet; BSD-3-Clause implied for weights, with the pretrained-model caveat on every card; inference-only) |
| SSD | `LibreSSD` | `LibreSSD300.pt` (torchvision SSD300 VGG16; BSD-3-Clause implied for the checkpoint, Oxford VGG feature-weight lineage CC BY 4.0; inference-only) |
| MaskRCNN | `LibreMaskRCNN` | `LibreMaskRCNNr50.pt` (torchvision Mask R-CNN v2; segment default plus detect; BSD-3-Clause implied for weights, with the pretrained-model caveat on the card; inference-only) |
| FCN | `LibreFCN` | `LibreFCNr50.pt` (torchvision's ResNet FCN, not the original VGG FCN-8s; BSD-3-Clause implied for weights, with the pretrained-model caveat on every card; semantic inference-only) |
| CenterNet | `LibreCenterNet` | `LibreCenterNetresdcn18.pt`, `LibreCenterNetdla34.pt` (official CenterNet COCO detectors; MIT implied for weights; inference-only) |
| FCOS | `LibreFCOS` | `LibreFCOSr50.pt` (torchvision FCOS ResNet-50 FPN; BSD-3-Clause implied for weights, with the pretrained-model caveat on every card; inference-only) |
| DeepLabv3 | `LibreDeepLabv3` | `LibreDeepLabv3r50-sem.pt` (torchvision semantic family; BSD-3-Clause implied for weights, with the pretrained-model caveat on every card; inference-only) |
| EfficientDet | `LibreEfficientDet` | `LibreEfficientDetd0.pt` (D0-D4; Apache-2.0 project release weights with no separate asset license; inference-only) |
| Deformable DETR | `LibreDeformableDETR` | `LibreDeformableDETRr50.pt` (original Apache-2.0 family; inference-only) |
| DINO-DETR | `LibreDINODETR` | `LibreDINODETRr50.pt` (IDEA DINO detector; Apache-2.0 implied for weights; inference-only) |
| RTDETR | `LibreRTDETR` | `LibreRTDETRr50.pt` |
| RTDETRv2 | `LibreRTDETRv2` | `LibreRTDETRv2r50.pt` |
| RTDETRv4 | `LibreRTDETRv4` | `LibreRTDETRv4s.pt` |
| RTMDet | `LibreRTMDet` | `LibreRTMDets.pt` |
| YOLONAS | `LibreYOLONAS` | `LibreYOLONASs.pt` |
| HRNet | `LibreHRNet` | `LibreHRNetw32-pose.pt`, `LibreHRNetw48-pose.pt` |
| MobileNetV4 | `LibreMobileNetV4` | `LibreMobileNetV4s-cls.pt` |
| ConvNeXt | `LibreConvNeXt` | `LibreConvNeXtt-cls.pt` |
| DeiT | `LibreDeiT` | `LibreDeiTt-cls.pt` (plain 224px classifier; Apache-2.0; inference-only) |
| EfficientNetV2 | `LibreEfficientNetV2` | `LibreEfficientNetV2b0-cls.pt` |
| ResNet | `LibreResNet` | `LibreResNet50-cls.pt` |
| ViT | `LibreViT` | `LibreViTti-cls.pt` (classic patch-16 AugReg classifier; Apache-2.0 code + weights; inference-only) |
| AlexNet | `LibreAlexNet` | `LibreAlexNetb-cls.pt` (torchvision museum classifier; BSD-3-Clause implied for the checkpoint) |
| VGG | `LibreVGG` | `LibreVGG16-cls.pt` |
| Swin | `LibreSwin` | `LibreSwint-cls.pt` (Swin V1; MIT weights; inference-only) |
| CLIP | `LibreCLIP` | `LibreCLIPb32-cls.pt` (zero-shot, open-vocab classify) |
| SigLIP2 | `LibreSigLIP2` | `LibreSigLIP2b16-cls.pt` (zero-shot, open-vocab classify) |
| PE | `LibrePE` | `LibrePEb16-cls.pt` (Perception Encoder Core; zero-shot classify + image/text/video embed) |
| V-JEPA 2 | `LibreVJEPA2` | `LibreVJEPA2l256-embed.pt` (video clip embedding), `LibreVJEPA2l256-cls-ssv2.pt` (attentive-probe video classify) |
| NAFNet | `LibreNAFNet` | `LibreNAFNets-restore.pt` (restore-only; `-sidd` variant = SIDD denoise) |
| BEN2 | `LibreBEN2` | `LibreBEN2b-matte.pt` (matte / background-removal; MIT code + weights; inference-only) |
| BiRefNet | `LibreBiRefNet` | `LibreBiRefNetl-matte.pt` (matte / background-removal; `l` is MIT, `t`/lite has no explicit weights-license tag) |
| FeyNobg | `LibreFeyNobg` | `LibreFeyNobgl-matte.pt` (matte / background-removal; Apache-2.0 code+weights; also ships `-fp8`/`-nvfp4` pre-quantized repos, see below) |
| RealESRGAN | `LibreRealESRGAN` | `LibreRealESRGANx4-restore.pt` (super-resolution; sizes `x4`/`x2`/`x4t`) |
| QuickSRNet | `LibreQuickSRNet` | `LibreQuickSRNetm2-restore.pt` (Medium 2x super-resolution; BSD-3-Clause) |
| SwinIR | `LibreSwinIR` | `LibreSwinIRm-restore.pt` (4x super-resolution; sizes `s`/`m`/`l`; Apache-2.0) |
| PPOCR | `LibrePPOCR` | `LibrePPOCRt-ocr.pt` (PP-OCRv5 text det+rec; sizes `t`/`l`; Apache-2.0) |
| PIDNet | `LibrePIDNet` | `LibrePIDNets-sem.pt` (semantic-only) |
| PPLiteSeg | `LibrePPLiteSeg` | `LibrePPLiteSegt50-sem.pt` (semantic-only; sizes `t50`/`b50`/`t75`/`b75`, natively rectangular. Weights are **non-commercial** — Cityscapes dataset terms, see below) |
| UNet | `LibreUNet` | `LibreUNets-sem.pt` (semantic-only; size `s`, natively 512x1024. Weights are **non-commercial** — Cityscapes dataset terms, same hosting path as PP-LiteSeg) |
| LingBotVision | `LibreLingBotVision` | `LibreLingBotVisions-sem.pt` (semantic-only; Apache-2.0 backbone + LibreYOLO-trained ADE20K head) |
| SegFormer | `LibreSegformer` | `LibreSegformerb0-sem.pt` (semantic-only; ADE20K. Weights are **non-commercial** — NVIDIA Source Code License, see below) |
| EoMT | `LibreEoMT` | `LibreEoMTl-sem.pt` (semantic), `LibreEoMTl-seg.pt` (COCO instance), `LibreEoMTs-panoptic.pt` (COCO panoptic) |
| DINOv2 | `LibreDINOv2` | `LibreDINOv2n.pt` (semantic default), `LibreDINOv2n-cls.pt` |
| DepthAnythingV2 | `LibreDepthAnythingV2` | `LibreDepthAnythingV2s-depth.pt` (only `s` is Apache; b/l/g are CC-BY-NC, see below) |
| DepthAnything3 | `LibreDepthAnything3` | `LibreDepthAnything3l-depth.pt` (DA3MONO-LARGE; Apache-2.0) |
| ZipDepth | `LibreZipDepth` | `LibreZipDepthb-depth.pt` (MIT code + weights; `bnpu` is the NPU-decoder checkpoint) |
| FOMO | `LibreFOMO` | `LibreFOMOs-point.pt` (no weights hosted yet; license-gate first) |

Never-upload families: **L2CS** (Gaze360 terms forbid redistribution) and any
weight whose upstream/training-data license fails the gate in
`skills/libreyolo-license-audit`. **Redistributable is the only bar for
hosting weights**: non-commercial but redistributable weights (CC-BY-NC, the
NVIDIA Source Code License) are hosted — ship the upstream license verbatim,
tag the card correctly, and lead with a non-commercial banner (SegFormer
precedent); downstream users are responsible for complying with the weight
license. Only weights whose terms forbid redistribution (L2CS) or whose
license is unknown stay unhosted. The
open-vocabulary and SAM/VLM tiers ship HF *snapshot directories*
(`LibreGroundingDINOt`, `LibreOWLv2b16`, ...), not single `.pt` files; their
repos mirror upstream snapshot layout plus card, so the 5-file contract below
does not apply verbatim (follow the existing mirror repos).

**Ask the user** if: the size code isn't obvious, the family isn't one of the above, or the filename doesn't match what the loader at `libreyolo/models/base/model.py:get_download_url` builds. Do not guess.

## Canonical filename whitelist

Before creating the HF repo or uploading, **verify that the `.pt` filename you are about to ship appears verbatim in this list**. If it doesn't, stop and ask the user — either a new family/size/task is being introduced (skill should be updated) or the name is wrong.

Authoritative list of all valid weight filenames (matches the schema enforced by `BaseModel._filename_regex` and the family table in `docs/nomenclature.md`):

```
LibreYOLOXn.pt, LibreYOLOXt.pt, LibreYOLOXs.pt, LibreYOLOXm.pt,
LibreYOLOXl.pt, LibreYOLOXx.pt,

LibreYOLO1t.pt, LibreYOLO1b.pt,

LibreYOLO2t.pt, LibreYOLO2b.pt,

LibreYOLO3t.pt, LibreYOLO3b.pt, LibreYOLO3spp.pt,

LibreYOLO4t.pt, LibreYOLO4b.pt,

LibreYOLO7b.pt,

LibreYOLO9t.pt, LibreYOLO9s.pt, LibreYOLO9m.pt, LibreYOLO9c.pt,

LibreYOLO9E2Et.pt, LibreYOLO9E2Es.pt, LibreYOLO9E2Em.pt,
LibreYOLO9E2Ec.pt,

LibreYOLO9P2t.pt, LibreYOLO9P2s.pt,
LibreYOLO9P2t-visdrone.pt, LibreYOLO9P2s-visdrone.pt,

LibreYOLONASs.pt, LibreYOLONASm.pt, LibreYOLONASl.pt,
LibreYOLONASn-pose.pt, LibreYOLONASs-pose.pt,
LibreYOLONASm-pose.pt, LibreYOLONASl-pose.pt,

LibreHRNetw32-pose.pt, LibreHRNetw48-pose.pt,

LibreDFINEn.pt, LibreDFINEs.pt, LibreDFINEm.pt, LibreDFINEl.pt,
LibreDFINEx.pt,
LibreDFINEn-seg.pt, LibreDFINEs-seg.pt, LibreDFINEm-seg.pt,
LibreDFINEl-seg.pt, LibreDFINEx-seg.pt,

LibreDEIMn.pt, LibreDEIMs.pt, LibreDEIMm.pt, LibreDEIMl.pt,
LibreDEIMx.pt,

LibreDEIMv2atto.pt, LibreDEIMv2femto.pt, LibreDEIMv2pico.pt,
LibreDEIMv2n.pt, LibreDEIMv2s.pt, LibreDEIMv2m.pt,
LibreDEIMv2l.pt, LibreDEIMv2x.pt,

LibreTinyFormers.pt, LibreTinyFormerm.pt, LibreTinyFormerl.pt,
LibreTinyFormerx.pt, LibreTinyFormerxl.pt,
LibreTinyFormers-visdrone.pt, LibreTinyFormerm-visdrone.pt,
LibreTinyFormerl-visdrone.pt, LibreTinyFormerx-visdrone.pt,
LibreTinyFormerx-obj2coco.pt, LibreTinyFormerxl-obj2coco.pt,

LibrePICODETs.pt, LibrePICODETm.pt, LibrePICODETl.pt,

LibreRTDETRr18.pt, LibreRTDETRr34.pt, LibreRTDETRr50.pt,
LibreRTDETRr50m.pt, LibreRTDETRr101.pt, LibreRTDETRl.pt,
LibreRTDETRx.pt,

LibreRTDETRv2r18.pt, LibreRTDETRv2r34.pt, LibreRTDETRv2r50.pt,
LibreRTDETRv2r50m.pt, LibreRTDETRv2r101.pt,
LibreRTDETRv2n-obb.pt, LibreRTDETRv2s-obb.pt,
LibreRTDETRv2m-obb.pt, LibreRTDETRv2l-obb.pt,
LibreRTDETRv2x-obb.pt,

LibreRTDETRv4s.pt, LibreRTDETRv4m.pt, LibreRTDETRv4l.pt,
LibreRTDETRv4x.pt,

LibreRTMDett.pt, LibreRTMDets.pt, LibreRTMDetm.pt,
LibreRTMDetl.pt, LibreRTMDetx.pt,
LibreRTMDett-seg.pt, LibreRTMDets-seg.pt, LibreRTMDetm-seg.pt,
LibreRTMDetl-seg.pt, LibreRTMDetx-seg.pt,

LibreRFDETRn.pt, LibreRFDETRs.pt, LibreRFDETRm.pt,
LibreRFDETRl.pt, LibreRFDETRn-seg.pt, LibreRFDETRs-seg.pt,
LibreRFDETRm-seg.pt, LibreRFDETRl-seg.pt, LibreRFDETRx-pose.pt,
LibreRFDETRn-obb.pt, LibreRFDETRs-obb.pt, LibreRFDETRm-obb.pt,
LibreRFDETRl-obb.pt,

LibreDETRr50.pt, LibreDETRr50dc5.pt,
LibreDETRr101.pt, LibreDETRr101dc5.pt,

LibreLWDETRt.pt, LibreLWDETRs.pt, LibreLWDETRm.pt,
LibreLWDETRl.pt, LibreLWDETRx.pt,

LibreDOMEDETRs-aitod.pt, LibreDOMEDETRm-aitod.pt,
LibreDOMEDETRl-aitod.pt,
LibreDOMEDETRs-visdrone.pt, LibreDOMEDETRm-visdrone.pt,
LibreDOMEDETRl-visdrone.pt,

LibreFasterRCNNn.pt, LibreFasterRCNNs.pt,
LibreFasterRCNNm.pt, LibreFasterRCNNl.pt,

LibreRetinaNetr50.pt, LibreRetinaNetr50v2.pt,
LibreMaskRCNNr50.pt,
LibreFCNr50.pt, LibreFCNr101.pt,
LibreCenterNetresdcn18.pt, LibreCenterNetdla34.pt,
LibreFCOSr50.pt,
LibreDeepLabv3r50-sem.pt, LibreDeepLabv3r101-sem.pt,
LibreDeepLabv3mv3-sem.pt,
LibreEfficientDetd0.pt, LibreEfficientDetd1.pt,
LibreEfficientDetd2.pt, LibreEfficientDetd3.pt,
LibreEfficientDetd4.pt,

LibreDeformableDETRr50ss.pt, LibreDeformableDETRr50ssdc5.pt,
LibreDeformableDETRr50.pt, LibreDeformableDETRr50refine.pt,
LibreDeformableDETRr50twostage.pt,

LibreDINODETRr50.pt, LibreDINODETRr50s5.pt,
LibreDINODETRswinl.pt,

LibreECs.pt, LibreECm.pt, LibreECl.pt, LibreECx.pt,
LibreECs-pose.pt, LibreECm-pose.pt, LibreECl-pose.pt,
LibreECx-pose.pt, LibreECs-seg.pt, LibreECm-seg.pt,
LibreECl-seg.pt, LibreECx-seg.pt,

LibreMobileNetV4s-cls.pt, LibreMobileNetV4m-cls.pt,
LibreMobileNetV4l-cls.pt,

LibreConvNeXtt-cls.pt, LibreConvNeXts-cls.pt, LibreConvNeXtb-cls.pt,

LibreDeiTt-cls.pt, LibreDeiTs-cls.pt, LibreDeiTb-cls.pt,

LibreEfficientNetV2b0-cls.pt, LibreEfficientNetV2b1-cls.pt,
LibreEfficientNetV2b2-cls.pt, LibreEfficientNetV2b3-cls.pt,

LibreResNet18-cls.pt, LibreResNet34-cls.pt,
LibreResNet50-cls.pt, LibreResNet101-cls.pt,

LibreViTti-cls.pt, LibreViTs-cls.pt,
LibreViTb-cls.pt, LibreViTl-cls.pt,
LibreAlexNetb-cls.pt,
LibreVGG16-cls.pt, LibreVGG19-cls.pt,
LibreVGG16bn-cls.pt, LibreVGG19bn-cls.pt,
LibreSwint-cls.pt, LibreSwins-cls.pt,
LibreSwinb-cls.pt, LibreSwinl-cls.pt,

LibreCLIPb32-cls.pt, LibreCLIPb16-cls.pt, LibreCLIPl14-cls.pt,

LibreSigLIP2b16-cls.pt, LibreSigLIP2so400m-cls.pt,

LibrePEt16-cls.pt, LibrePEs16-cls.pt, LibrePEb16-cls.pt,
LibrePEl14-cls.pt, LibrePEg14-cls.pt,
LibreVJEPA2l256-embed.pt, LibreVJEPA2h256-embed.pt,
LibreVJEPA2g256-embed.pt, LibreVJEPA2g384-embed.pt,
LibreVJEPA2l256-cls-ssv2.pt, LibreVJEPA2l256-cls-diving48.pt,
LibreVJEPA2g384-cls-ssv2.pt, LibreVJEPA2g384-cls-diving48.pt,

LibreNAFNets-restore.pt, LibreNAFNetl-restore.pt,
LibreNAFNetl-restore-sidd.pt,

LibreRealESRGANx4-restore.pt, LibreRealESRGANx2-restore.pt,
LibreRealESRGANx4t-restore.pt,

LibreQuickSRNetm2-restore.pt,

LibreSwinIRs-restore.pt, LibreSwinIRm-restore.pt,
LibreSwinIRl-restore.pt,

LibrePPOCRt-ocr.pt, LibrePPOCRl-ocr.pt,

LibreBEN2b-matte.pt,

LibreBiRefNett-matte.pt, LibreBiRefNetl-matte.pt,

LibreFeyNobgl-matte.pt, LibreFeyNobgl-matte-fp16.pt,
LibreFeyNobgl-matte-fp8.pt,

LibrePIDNets-sem.pt, LibrePIDNetm-sem.pt, LibrePIDNetl-sem.pt,

LibrePPLiteSegt50-sem.pt, LibrePPLiteSegb50-sem.pt,
LibrePPLiteSegt75-sem.pt, LibrePPLiteSegb75-sem.pt,
LibreUNets-sem.pt,

LibreLingBotVisions-sem.pt, LibreLingBotVisionb-sem.pt,
LibreLingBotVisionl-sem.pt,

LibreSegformerb0-sem.pt, LibreSegformerb1-sem.pt, LibreSegformerb2-sem.pt,
LibreSegformerb3-sem.pt, LibreSegformerb4-sem.pt, LibreSegformerb5-sem.pt,

LibreEoMTl-sem.pt,

LibreEoMTl-seg.pt, LibreEoMTl-seg-1280.pt,

LibreEoMTs-panoptic.pt, LibreEoMTb-panoptic.pt,
LibreEoMTl-panoptic.pt,

LibreDINOv2n.pt, LibreDINOv2s.pt, LibreDINOv2m.pt, LibreDINOv2l.pt,
LibreDINOv2n-cls.pt, LibreDINOv2s-cls.pt, LibreDINOv2m-cls.pt,
LibreDINOv2l-cls.pt,

LibreDepthAnythingV2s-depth.pt, LibreDepthAnythingV2b-depth.pt,
LibreDepthAnythingV2l-depth.pt, LibreDepthAnythingV2g-depth.pt,

LibreDepthAnything3l-depth.pt,

LibreMoGe2s-normal.pt, LibreMoGe2b-normal.pt, LibreMoGe2l-normal.pt,

LibreZipDepthb-depth.pt, LibreZipDepthbnpu-depth.pt,

LibreFOMOs-point.pt, LibreFOMOm-point.pt, LibreFOMOl-point.pt
```

License caveats inside the list: BiRefNet `l` (general) weights are MIT-tagged
and hosted; BiRefNet `t` (lite) weights have no explicit license tag on the
upstream HF repo (MIT badge in the card body only), so hosting the lite weights
is a maintainer decision, not a default (`weights/upload_birefnet_hf.py` guards
it behind `--confirm-lite-license`). DepthAnythingV2 `b`/`l`/`g` are CC-BY-NC
(redistributable, therefore hostable — license verbatim + non-commercial
banner on the card); `-visdrone` variants are a research preview
under VisDrone's CC BY-NC-SA (repo `LibreYOLO/LibreYOLO9P2s-visdrone`, with
the license stated loudly on the card); FOMO weights have no cleared hosting
license yet. **LibreSegformer b0-b5 are NON-COMMERCIAL**: they derive from
NVIDIA's ADE20K checkpoints under the NVIDIA Source Code License, which permits
redistribution (a verbatim `LICENSE` copy and the attribution notices must
travel with the weights) but limits *use* to research or evaluation, virally
through derivative works. Their cards use `license: other` +
`license_name: nvidia-source-code-license-segformer` + `license_link`, lead with
a non-commercial banner, and the loader prints the restriction before every
auto-download. Never tag them `apache-2.0` because the *code* is Apache.
**LibrePPLiteSeg t50/b50/t75/b75 are NON-COMMERCIAL** for the same shape of
reason, but the restriction comes from the *training data*, not the code: the
checkpoints are Cityscapes-trained, and the
[Cityscapes license](https://www.cityscapes-dataset.com/license/) allows
distributing abstract derivative models while restricting the dataset and its
derivatives to non-commercial use. Their cards use `license: other` +
`license_name: cityscapes-non-commercial` + `license_link`, lead with a
non-commercial banner, and the loader prints the restriction before every
auto-download. The `LICENSE` file carries the Apache-2.0 (super-gradients) and
MIT (STDC-Seg) code texts *plus* the Cityscapes weight terms, because the code
and the weights are genuinely different surfaces. Never tag them `apache-2.0`.
**LibreUNets-sem is NON-COMMERCIAL** on the same Cityscapes-training-data
grounds as PP-LiteSeg: mmsegmentation's official UNet-S5-D16 Cityscapes
checkpoint, converted in place. Same card tags, same banner, same
`LICENSE` split (Apache-2.0 mmseg code text plus Cityscapes weight terms).
Faster R-CNN's four torchvision checkpoints and FCOS's torchvision checkpoint
have no per-object license file; the maintainer approved BSD-3-Clause rehosting
on the releasing-project **implied** basis. Every card and NOTICE must say that
the grant is implied, must reproduce torchvision's pretrained-model caveat,
and must not call the checkpoint license publisher-confirmed.
SSD300 follows the same implied BSD-3-Clause checkpoint rule. Its card and
NOTICE must additionally attribute Karen Simonyan and Andrew Zisserman's
Oxford VGG-16 feature-weight lineage under CC BY 4.0, link the Oxford source
and CC license, and state the torchvision training plus LibreYOLO metadata
changes. Do not describe CC BY 4.0 as the license for torchvision's SSD code.
A name being *valid* does not make it *hostable*; run the gate.

**LibreDOMEDETR s/m/l AI-TOD-V2 and VisDrone weights are ACADEMIC-RESEARCH-ONLY.**
The upstream Hugging Face card has no license metadata or LICENSE file and
simultaneously says the project is Apache-2.0 and the weights are "for academic
research purposes only". The maintainer approved rehosting by treating the
Apache statement as the redistribution basis while preserving the stricter
academic-use sentence. Cards use `license: other` +
`license_name: dome-detr-academic-research-only` + a revision-pinned upstream
`license_link`, lead with an academic-only banner, and disclose that this is a
LibreYOLO interpretation rather than upstream clarification. Every mirror's
LICENSE and NOTICE reproduce the complete context. The loader warns before
every auto-download. Never tag these repos `apache-2.0` or imply commercial
clearance.

The `-visdrone` suffix is a `WEIGHT_VARIANTS` dataset variant (grammar in
`docs/nomenclature.md`): only families that declare `WEIGHT_VARIANTS` in
their `model.py` may carry one, and plain COCO-default weights never do.
TinyFormer declares `("visdrone", "obj2coco")`; its `l`/`x`/`xl` checkpoints
embed finetuned DINOv3 towers, so those repos ship Meta's DINOv3 License
Agreement alongside the Apache-2.0 code license (same split as the DEIMv2
DINO sizes), while `s`/`m` use the DEIMv2-distilled ViT-Tiny towers.

Pre-quantized variant repos (`-fp8`, `-nvfp4`; FeyNobg only today): built by
`weights/upload_feynobg_hf.py --recipe <r>` from a finalized quantized
checkpoint (`docs/quantization.md`). They follow the same 5-file contract but
are **not** auto-download names: the loader never fetches them, users pass the
downloaded `.pt` path as the weights argument. Their README YAML must carry
`base_model: <upstream-hf-repo>` + `base_model_relation: quantized` so the repo
appears in the upstream model's "Quantizations" sidebar on Hugging Face.

Classification (`-cls`) repos use `pipeline_tag: image-classification`,
`datasets: imagenet-1k`, and **omit the Benchmarks section** (Vision Analysis
tracks detection only). Record each classifier's actual upstream and license;
do not assume they are all timm-derived or Apache-2.0. AlexNet is the
torchvision BSD-3-Clause graph, and its checkpoint uses the explicitly
disclosed implied-BSD basis plus the torchvision/ImageNet terms caveat.
Torchvision VGG likewise uses BSD-3-Clause implied by the releasing project
and must carry the upstream pretrained-model caveat. Native classifier parity
remains a `max_abs_diff == 0` gate.

LibreCLIP is the zero-shot, open-vocabulary classifier (CLIP). Its HF cards use
`pipeline_tag: zero-shot-image-classification`, **must document the LAION-2B
data-provenance note** (see `libreyolo/models/clip/NOTICE.md`), and omit the VA
Benchmarks section (zero-shot, not a trained-on-COCO detector).

LibreSigLIP2 is the SigLIP 2 zero-shot, open-vocabulary classifier. Its HF cards
use `pipeline_tag: zero-shot-image-classification`, `license: apache-2.0`
(weights derive from the Apache-2.0 `google/siglip2-*` release; state the
upstream repo and commit pin), note the vendored SentencePiece tokenizer, and
omit the VA Benchmarks section. Conversion is a metadata wrap
(`weights/convert_siglip2_weights.py`); learned parameters are unchanged.

LibrePE is the Perception Encoder Core zero-shot classifier and embedder. Its HF
cards use `pipeline_tag: zero-shot-image-classification`, `license: apache-2.0`
(weights derive from the Apache-2.0 OpenCLIP-compatible `timm/PE-Core-*`
repositories — state the repo **and** the exact revision pin), and omit the VA
Benchmarks section. The cards must say the artifact is a **converted
OpenCLIP-compatible** PE Core checkpoint, not an unmodified official
`facebook/PE-Core-*` package checkpoint, and must not repeat any claim from the
official `perception_models` repository, whose `LICENSE.PE` (Apache-2.0)
conflicts with its `setup.py` package license. Conversion is a metadata wrap
(`weights/convert_pe_weights.py`); learned parameters are unchanged. Note the
`g14` memory profile (1.88B-parameter vision tower) so users are not surprised.

Common rule violations to reject before upload:

- Wrong casing on the size — `LibreDEIMv2Atto.pt` (PascalCase). Sizes are always lowercase.
- Old class-name forms in the file — `LibreYOLORTDETR*.pt` or `LibreECDET*.pt`. The library renamed those.
- Detect carrying an explicit `-det` suffix — `LibreECs-det.pt`. Detect is implicit, no suffix.
- Lowercase prefix — `librerfdetrn.pt` (Hugging Face is case-insensitive on lookup but the file inside the repo must use the canonical case).

If a candidate filename isn't in the list and isn't a brand-new family being introduced now, **stop and ask** rather than uploading.

## README template

```markdown
---
license: <apache-2.0 | mit | ...>
library_name: libreyolo
tags:
  - object-detection
  - <family-tag>          # yolox | yolov9 | rf-detr | rt-detr | yolo-nas
---

# <RepoName>

<One sentence: what architecture, what size, repackaged for LibreYOLO.>

## Source

Derived from [<upstream-org>/<upstream-repo>](https://github.com/<upstream-org>/<upstream-repo>)
at <tag-or-commit>.
Copyright (c) <years> <upstream-authors>. Licensed under the <License> License.

<If a backbone has its own upstream, add a second paragraph for it.>

## Modifications

State-dict key remapping only. Learned parameters are unchanged.
See `weights/convert_<family>_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## Benchmarks

Independent, verified accuracy and speed benchmarks for this model:
[visionanalysis.org/model/<va-slug>](https://www.visionanalysis.org/model/<va-slug>)

## License

<Apache License 2.0 | MIT License>. See the [`LICENSE`](./LICENSE)
and [`NOTICE`](./NOTICE) files in this repository.
```

## Vision Analysis benchmark link (`<va-slug>`)

Detect weight repos link to the model's page on the benchmark site
[visionanalysis.org](https://www.visionanalysis.org). The URL is deterministic —
derive it from `(FAMILY, size)`, never search for it or guess:

```
https://www.visionanalysis.org/model/<va-slug>
```

1. Map the family id: `yolo9` → `yolov9`. Every other family id is used as-is
   (`yolox`, `rfdetr`, `rtdetr`, `rtdetrv2`, `rtdetrv4`, `dfine`, `deim`,
   `deimv2`, `picodet`, `yolonas`, `ec`, `fcos`).
2. Map the size — YOLOX only: `n` → `nano`, `t` → `tiny`. All other sizes are
   used as-is (including `r50`-style RT-DETR codes and DEIMv2's
   `atto`/`femto`/`pico`).
3. Join: `yolov9` concatenates with no separator; every other family joins
   with a hyphen.

| Weight file | `<va-slug>` |
|---|---|
| `LibreYOLOXn.pt` | `yolox-nano` |
| `LibreYOLOXs.pt` | `yolox-s` |
| `LibreYOLO9s.pt` | `yolov9s` |
| `LibreDFINEm.pt` | `dfine-m` |
| `LibreRTDETRr50.pt` | `rtdetr-r50` |
| `LibreDEIMv2atto.pt` | `deimv2-atto` |

Rules:

- **Detect repos only.** Vision Analysis tracks detection; omit the Benchmarks
  section from `-seg` / `-pose` / `-cls` / `-obb` and gaze repos.
- **No slug exists** for `detr`, `lwdetr`, `deformable_detr`, `dinodetr`,
  `yolo9_e2e`, `yolo9_p2`, `l2cs`, RTMDet, the VLM / SAM / open-vocab tiers,
  or the Darknet-lineage families (`yolo1`, `yolo2`, `yolo3`, `yolo4`) and
  `yolo7` — omit the Benchmarks section and tell the user.
  Semantic / depth / restore / point repos also omit it (detection only).
- **The page may lag the upload.** Model pages are generated from
  `website/src/data/metadata/models.json` in
  [LibreYOLO/vision-analysis](https://github.com/LibreYOLO/vision-analysis);
  a page goes live on the next site deploy after the model is added there,
  with or without benchmark runs. The derived URL never changes, so include
  the link at upload time regardless — but check the slug exists in
  `models.json` and, if it doesn't, tell the user to add the model entry so
  the page resolves.

## LICENSE + NOTICE

- **LICENSE**: copy the upstream `LICENSE` file **verbatim**. Do not synthesize, do not template.
- **NOTICE**: required when upstream is Apache-2.0. Short attribution block:

```
Libre<Family> weights
---------------------

This product contains weights derived from <Upstream>
(https://github.com/<upstream-org>/<upstream-repo>).
Copyright (c) <years> <upstream-authors>.
Licensed under the Apache License, Version 2.0.

<Second paragraph if there's a separately-licensed backbone.>
```

For MIT upstreams (e.g. YOLOv9): NOTICE is not legally required. For consistency with existing YOLOX/RFDETR/RTDETR repos, include one anyway.

## Collection membership

After the repo is uploaded, add it to a collection:

| Repo type | Collection |
|---|---|
| Detection weights | `LibreYOLO/libreyolo-models-698875bf2b5f695708415169` |
| Classification weights | `LibreYOLO/libreyolo-classification-6a4164414d64a10aa8576885` |
| Pose weights | `LibreYOLO/libreyolo-models-698875bf2b5f695708415169` |
| RF-DETR segmentation | `LibreYOLO/rf-detr-instance-segmentation-69bde2744d6c285366a69603` |
| New seg family (e.g. YOLOX-seg) | **Ask the user** — create a new collection or extend existing |
| New detection family with no siblings yet | Add to `LibreYOLO Models` |

Add via HF UI or `huggingface_hub.add_collection_item(collection_slug, item_id=<repo>, item_type="model")`.

## Upload workflow

1. Build the 5 files locally in a clean directory. For detect repos, derive
   the Vision Analysis `<va-slug>` (section above) and fill in the README
   Benchmarks link.
2. Verify canonical filename matches `BaseModel.get_download_url()` output for this family + size.
3. **Cross-check the filename against the whitelist above.** If it isn't in the list, halt and ask the user — don't paper over it with a manual override.
4. Validate the `.pt` against the current LibreYOLO checkpoint metadata schema before upload. The source of truth is `docs/checkpoint_schema.md` and the helpers in `libreyolo/utils/serialization.py`; do not duplicate the schema in this skill. A simple load smoke test is not enough.
5. Create the HF repo (skip if it exists): `huggingface-cli repo create LibreYOLO/<RepoName> --type model`.
6. Upload: `huggingface-cli upload LibreYOLO/<RepoName> <local-dir> . --commit-message "Initial upload"`.
7. Smoke test the autodownload path on a fresh machine / cleared cache (and no
   staged copy under `weights/`): `LibreYOLO("<RepoName>.pt")` must download
   from the new repo and load. There is no `from_pretrained` API; the bare
   canonical filename is the download trigger.
8. Add to the matching collection.

One commit per file if iterating — easier to revert than a batch commit.

## Ask the user when

- The upstream release / commit pin isn't known (reproducibility needs it in README).
- The family isn't in the code yet (the skill can't derive canonical filename).
- A file with the same name already exists on the target repo (overwrite is destructive).
- The repo is a new task type and no collection fits.
- The upstream has a non-standard license (neither Apache-2.0 nor MIT).
- The model's `<va-slug>` is missing from vision-analysis `models.json` (the
  benchmark link will 404 until the model entry is added and deployed).

## Common traps

- Relying only on a load smoke test (`LibreYOLO("<RepoName>.pt")`); it can pass even when required checkpoint metadata is missing or stale. Run `validate_checkpoint_metadata` (step 4) too.
- Uploading both `.pt` and `.pth` of the same weights (wastes HF storage, no canonical filename).
- Copying a lowercase filename from an old release — the loader only fetches the `FILENAME_PREFIX`-cased `.pt`.
- Writing `license: mit` in README YAML for a repo whose weights derive from an Apache-2.0 upstream — MIT re-licensing is not legal without explicit permission.
- Forgetting `.gitattributes` — weights upload as raw blobs instead of LFS and the repo becomes huge.
- Adding to the wrong collection (seg → detection collection).
