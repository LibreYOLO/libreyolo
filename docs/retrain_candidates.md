# Models to retrain from scratch

Families where the code is permissively licensed and already ported, the
published weights are not permissive, and open training data exists. Where a
candidate depends on an ImageNet-pretrained backbone upstream, that
dependency is called out, because ImageNet is not open; see the open
questions. Retraining
them gives users MIT weights. This list backs item 2 of
[the sponsorship program](../SPONSORS.md). Surveyed 2026-09-12 from
`weights/LICENSE_NOTICE.txt`, `THIRD_PARTY_NOTICES.txt`, the per-family
`NOTICE` files, and the license tags on the Hugging Face LibreYOLO org.

Cost classes assume about 19 minutes per COCO epoch for a medium detector on
an RTX 4090. Small: under a GPU-day. Medium: several GPU-days to a multi-GPU
week. None of these needs a cluster.

## Candidates, in priority order

| # | Family | Task | Code | Current weights | Training data | Cost | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | YOLO-NAS n/s/m/l | detect | Apache-2.0 | Deci non-commercial, not redistributed, fetched from Deci's CDN | COCO 2017 | medium | Most used family with the worst weight terms. Trainer exists. Deci pretrained on Objects365 then COCO; a COCO-only run will land below Deci's numbers. |
| 2 | YOLO-NAS-pose n/s/m/l | pose | Apache-2.0 | Deci non-commercial | COCO-Pose | medium | Trainer exists. One of two trainable pose families. |
| 3 | PP-YOLOE s/m/l/x | detect | Apache-2.0 | No per-artifact license grant, linked from the source CDN | COCO. The upstream CSPResNet backbone is ImageNet-pretrained, see open questions | medium | Trainer exists. Turns an unclear grant into MIT if the backbone question is settled. |
| 4 | SegFormer b0-b5 | semantic seg | Apache-2.0 | NVIDIA Source Code License, non-commercial | Head on COCO-Stuff. Upstream pretrains the encoder on ImageNet-1k, see open questions | medium | Trainer exists. The encoder pretrain is the expensive part; b0-b2 are small. |
| 5 | PP-LiteSeg t50/b50/t75/b75 | semantic seg | Apache-2.0 + MIT | Non-commercial through Cityscapes terms | COCO-Stuff. The STDC backbone is MIT code but ImageNet-pretrained upstream, see open questions | small | Trainer exists. Must not be trained on Cityscapes again. |
| 6 | U-Net s | semantic seg | Apache-2.0 | Non-commercial through Cityscapes terms | COCO-Stuff | small | Cheapest retrain on the list. No backbone pretrain needed. |
| 7 | PIDNet s/m/l | semantic seg | MIT | Tagged MIT but trained on Cityscapes | COCO-Stuff. Upstream uses an ImageNet-pretrained backbone, see open questions | small | Tainted data, not a license problem. Blocker: no trainer in the library yet. |
| 8 | YOLO-NAS-R s/m/l | oriented boxes | Apache-2.0 | Deci non-commercial | DOTA is academic-only, so a clean OBB dataset must be chosen first | medium | Trainer exists. RF-DETR OBB used a CC BY 4.0 Roboflow set; the same could work here. |
| 9 | DDColor t/l | colorization | Apache-2.0 | Tagged Apache-2.0 but trained on ImageNet under research terms | COCO or Open Images | medium | Tainted data. No trainer. Low priority. |

## Excluded, and why

- **Already permissive weights.** YOLOX, YOLOv9 and variants, YOLOv7, RF-DETR,
  D-FINE, DEIM, RT-DETR, RTMDet, PicoDet, LW-DETR, DETR, Deformable DETR, EoMT,
  DINOv2, the classification backbones, CLIP and SigLIP2, SAM and SAM2,
  MoGe-2, the restoration families, BEN2, BiRefNet, PP-OCR, FOMO. Nothing to
  fix.
- **Code license problem.** DEIMv2 s/m/l/x, TinyFormer, OV-DEIM and PAGE use a
  DINOv3 backbone under Meta's non-OSI license, and its pretraining data is
  not available. Swapping the backbone would make a different model.
- **No usable training data.** VisDrone families (CC BY-NC-SA), Dome-DETR,
  Depth Anything V2 (distilled from a non-commercial teacher), MiDaS (twelve
  mixed datasets), TEED and DexiNed (BIPED non-commercial), ViTMatte
  (Composition-1k), HVI-CIDNet (LOLv2 has no license), LaMa (Places365), L2CS
  gaze (Gaze360 forbids redistribution), face embeddings and re-identification
  (undisclosed or research-only face data), the 3D detectors (nuScenes),
  V-JEPA probes (Kinetics), and every VLM snapshot. These categories are out
  of scope until a clean dataset exists.
- **Inference-only with permissive weights.** The torchvision families,
  CenterNet, DINO-DETR, EfficientDet, HRNet, DEKR. Their weight licenses are
  implied by the releasing project rather than confirmed, which is acceptable.
  A from-scratch run would only add provenance.

## Open questions

- **ImageNet.** ImageNet-1k is licensed for non-commercial research and
  education, so an ImageNet-pretrained backbone carries the same taint this
  list is trying to remove. Candidates 3, 4, 5 and 7 use one upstream. For a
  clean result the backbone must be trained without ImageNet: from scratch on
  the target dataset, the way YOLO detectors are trained on COCO, or
  pretrained on an openly licensed dataset such as Objects365 or COCO. This
  raises the cost of those four candidates. Candidates 1, 2 and 6 have no
  ImageNet dependency.
- **ADE20K.** Its annotations are BSD-3 but its images are restricted to
  non-commercial research. The library already hosts ADE20K-trained heads
  without a caveat. Either accept ADE20K for semantic retrains or use
  COCO-Stuff, whose annotations are CC BY 4.0. This list assumes COCO-Stuff.
- **DOTA** prohibits commercial use, which rules it out for candidate 8.
- **Objects365** annotations are CC BY 4.0 with images under Flickr terms, the
  same posture as COCO. Usable for a YOLO-NAS pretraining stage if budget
  allows.
- Only a few families carry a `weight_license` field in checkpoint metadata.
  YOLO-NAS and SegFormer signal restrictions through download notices only,
  so a retrained checkpoint for those needs the field added.
