# Changelog

All notable changes to LibreYOLO are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases
before 1.4.0 are documented in the
[GitHub Releases](https://github.com/LibreYOLO/libreyolo/releases) only.

## [Unreleased]

### Added

- **U-Net semantic family.** `LibreUNets-sem.pt` is the mmseg UNet-S5-D16
  + FCN-head graph (same-padded 2D, not the 2015 Caffe valid-convolution
  U-Net), Cityscapes 19-class. Whole-frame inference and validation at the
  upstream 1024x2048 evaluation canvas; training samples 512x1024 crops from
  a 0.5-2.0 rescale of the source frame (`CE + 0.4 aux CE`).
  `weights/parity_unet.py` proves bit-identical logits against the pinned
  mmseg implementation and identical class maps through `mmseg.apis`. The
  converted Cityscapes checkpoint is NON-COMMERCIAL; train from scratch for
  unrestricted weights.

- **LibreGround** sibling factory: screenshot + instruction →
  `Results.points`. Shipped adapters are Florence-2-base (MIT), ShowUI-2B
  (MIT weights; Apache-2.0 code/base), and Qwen3-VL-2B (Apache-2.0).
  Inference-only. Install `libreyolo[vlm]` or `libreyolo[ground]`.

### Changed

- **YOLOv9 letterbox is now stamped on the checkpoint, not flipped
  globally.** Unmarked ≤1.5 weights keep top-left pad (same predict/val
  boxes as today). Newly converted official MultimediaTechLab checkpoints
  stamp `letterbox_pad: center`. A 1.6 upgrade does not silently move
  boxes on existing user fine-tunes. YOLO-NAS is unchanged (its official
  pipeline already pads bottom-right).
- **YOLOv9 training recipe closer to MultimediaTechLab/YOLO.**
  `max_labels` default is 300 (was 100). LinearLR SGD momentum warms
  `0.8 → 0.937` over the existing 3-epoch warmup. New stock-detect
  fine-tunes attach the PGI auxiliary head (`aux_weight=0.25`); resume of
  a single-head 1.5 checkpoint stays single-head. Inference and export
  remain single-head either way. Val NMS defaults are **not** changed
  (`0.001` / `0.6` / `300`); reported mAP stays comparable across 1.5
  and 1.6 for the same unmarked weights.

### Fixed

- **RF-DETR, DINOv2 and VLM training no longer overwrite the previous run
  (#833).** Through the Python API, `train()` now increments the run
  directory like every other family (`exist_ok=False` by default) instead of
  writing into the same folder. The default location moves from
  `runs/train` to `runs/train/rfdetr_exp` and `runs/train/dinov2_exp`;
  the DINOv2 CLI default name is now `dinov2_exp`. `resume=True` keeps the
  original run directory. DINOv2 `resume=` previously did nothing and now
  restores the checkpoint before continuing.
- **Guardless Python multi-GPU launch (#817).**
  `model.train(device=[0, 1])` and `device="0,1"` now coordinate local DDP
  ranks without re-importing and repeating an unguarded user script. Guarded
  entry points remain supported, but coordinator ranks do not replay arbitrary
  guarded top-level side effects. Explicit `torchrun` launches are unchanged.
- **QAT training-state guards (#768).** Training a quantized model now
  disables EMA and SyncBatchNorm before setup and logs the changes, preventing
  those features from interfering with fake-quant observer and scale state.
- **CUDA MSDA discovery (#781).** Eager DETR-family CUDA calls that no
  accelerated provider accepts log one hint for `libreyolo[hub-kernels]`;
  `LIBREYOLO_HUB_KERNELS=0` keeps the portable path and silences the hint.
- **MSDA slot follow-ups from #784.** `maybe_ms_deform_attn` walks eligible
  providers when the preferred Hub kernel returns None, so Triton still
  runs after a Hub reject or disable. Triton skips the `spatial_shapes`
  `.item()` checks while CUDA-graph capturing, launches on `value.device`,
  and disables itself after the first compile/launch failure instead of
  retrying every call.

### Added

- **Four restoration and guided-matting specialists.**
  `LibreDDColor{t,l}-restore.pt` adds automatic colorization;
  `LibreHVICIDNett-restore.pt` adds adjustable low-light enhancement;
  `LibreLaMab-restore.pt` adds mask-guided inpainting through the pinned
  OpenCV Zoo ONNX graph; and `LibreViTMattes-matte.pt` adds trimap-guided
  alpha matting. All use existing `restore` or `matte` result contracts,
  include paired validation, strict hash-pinned conversion, and
  family-specific pinned parity harnesses. `mask=` and `trimap=` are
  available in Python and the CLI for one-image guided prediction. Checkpoint
  and training-data terms are documented separately from source-code terms;
  ViTMatte's published
  Composition-1k checkpoint is non-commercial. LaMa runtime requires
  `libreyolo[onnx]`.

- **QuickSRNet Medium 2x super-resolution.** `LibreQuickSRNetm2-restore.pt`
  adds compact native-resolution RGB upscaling, paired PSNR/SSIM validation,
  dynamic-spatial ONNX and fixed-canvas TorchScript export, exact parity with
  the pinned BSD-3-Clause upstream, and an auto-downloaded hosted checkpoint.

- **Hosted Dome-DETR checkpoints.** The six official AI-TOD-V2 and VisDrone
  checkpoints now auto-download from LibreYOLO under the upstream card's
  academic-research-only restriction. Every download warns that the weights
  are not covered by LibreYOLO's MIT license. Each mirror uses `license: other`
  and records the complete upstream ambiguity: the card also claims
  Apache-2.0 but has neither license metadata nor a weight-repository LICENSE
  file. The learned tensors are unchanged; conversion adds metadata only.

- **BEN2 background removal.** `LibreBEN2b-matte.pt` adds the MIT-licensed
  BEN2 Base model at fixed 1024 resolution with batched prediction, matte
  validation, transparent cutout saving, strict safetensors conversion, and
  fixed-resolution ONNX/TorchScript export.

- **Opt-in training helpers (#768).** `class_balanced=True` enables
  repeat-factor sampling for long-tailed detection datasets;
  `average_best=N` writes the uniform mean of the N best validation
  checkpoints; `export_check=True` verifies ONNX export before epoch 1; and
  `precise_bn=N` recomputes BatchNorm statistics from N training images before
  final validation. All four default off.

- **Gemma 4 and Moondream in LibreVLM.** `LibreVLM("gemma-4")` loads
  Gemma 4 E4B (Apache-2.0; `e2b` also aliased) and parses the official
  y-first `box_2d` 0-1000 JSON. `LibreVLM("moondream")` loads Moondream 2
  (Apache-2.0) through its native `detect`/`point`/`query` skills, one
  class per call; `moondream-3` is the BSL 1.1 preview and logs a license
  notice. Both are inference-only. Gemma 3 stays out: gated license, no
  native boxes. Snapshots are mirrored at `LibreYOLO/LibreGemma4e2b`,
  `LibreYOLO/LibreGemma4e4b`, `LibreYOLO/LibreMoondream2`, and
  `LibreYOLO/LibreMoondream3`. Moondream 3 is BSL 1.1 (redistributable;
  competing paid hosted API forbidden); the card ships the license
  verbatim and a banner.

- **EC pose on the `ms_deform_attn` slot.** The pose decoder's pre-split
  `(bs*heads, c, hw)` value tuple now adapts onto the classic slot layout
  used by detect EC and the other DETR-lineage families. Export and any
  shape that cannot reshape keep the portable `grid_sample` path.

- **In-tree Triton multi-scale deformable attention.** The `ms_deform_attn`
  slot now has a `triton` provider that needs no `kernels` extra: CUDA
  fp32/fp16/bf16 inference, same bilinear equation as the portable
  `grid_sample` path. Inputs that require grad fall through, so training
  keeps a backward. Hub stays preferred when `libreyolo[hub-kernels]` is
  installed. Opt out with `LIBREYOLO_TRITON_MSDA=0`. On an RTX 5070 Ti
  the kernel is ~4x the portable core on RF-DETR-s shapes and ~12x on
  Deformable-DETR / RT-DETR multi-level shapes; RF-DETR-n end-to-end
  moves about 7% because the windowed DINOv2 backbone dominates.

- **`min_samples` epoch-length floor for tiny datasets.** New opt-in training
  knob (`model.train(..., min_samples=640)` /
  `libreyolo train --min-samples 640`, issue #768): when the training dataset
  has fewer images than `min_samples`, each epoch draws `min_samples` samples
  with replacement instead of one short pass over the dataset, so per-epoch
  overhead (dataloader spin-up, validation, checkpointing) stops dominating
  wall-clock time on ~100-image datasets. Under DDP the draw is sharded across
  ranks (rounded up to a multiple of the world size) and reshuffled each epoch;
  dataloader workers are clamped to the dataset length while the floor is
  active. Default `min_samples=0` keeps training byte-identical to before.
  Honored by the families that build their train loader through the shared
  detection dataloader; families with hand-built loaders (DEIM/DEIMv2/D-FINE
  lineage, RF-DETR, FOMO, EC pose, YOLO-NAS pose, V-JEPA 2) and the
  classify/semantic/depth/restore task loaders ignore it for now.

- **North Micro Vision VLM detector.** New `northmicrovision` LibreVLM family
  (issue #758): `LibreVLM("north-micro-vision")` loads Cohere Labs'
  North-Micro-Vision-Instruct (2.4B, Apache-2.0, native-resolution) as an
  open-vocabulary detector. The model answers single-class grounding queries
  with unlabeled `[[x1, y1, x2, y2], ...]` boxes on a 0-1000 scale and does not
  follow the tier's labeled-JSON ask, so the family runs one "Locate every
  <label> ..." generation per vocabulary entry and assigns labels itself
  (`predict()` cost scales with the vocabulary size; an absent but plausible
  label can hallucinate a box). Requires transformers >= 5.16.0 (native
  `CohereCompass` architecture, no remote code); the family fails fast with an
  install hint on older versions instead of downloading ~5 GB first. `chat()`
  is supported.
- **LFM2.5-VL-3B.** The `lfm2-vl` family gains the largest LFM2.5-VL size:
  `LibreVLM("lfm2-vl-3b")` (LFM Open License v1.0, same one-time notice as the
  smaller sizes). Unlike the 450M/1.6B, the 3B emits boxes on a 0-1000 scale
  regardless of the prompt's stated convention, and drifts between `bbox` and
  Qwen-style `bbox_2d` keys, both verified on known boxes; the family now
  carries a per-size coordinate divisor and the shared parser falls back
  across the known box-key aliases.

- **TinyFormer detection.** New trainable `tinyformer` family:
  `LibreYOLO("LibreTinyFormers.pt")` loads the DEIMv2-derived YOLO-DETR hybrid
  from "TinyFormer: Preserving Tiny Objects in YOLO-DETR Hybrid Real-time
  Detectors" (arXiv:2605.25046), specialised for tiny objects via the Spatial
  Semantic Adapter and the 4-scale Parallel Bi-fusion neck (the decoder keeps
  3 levels at strides 8/16/32). Sizes `s`/`m`/`l`/`x`/`xl` (51.5 to 60.6 COCO
  AP), all from the release's PBM checkpoints; official VisDrone (`-visdrone`,
  nc=10) and Objects365-to-COCO (`-obj2coco`) variants convert with the same
  script. Ported from the Apache-2.0 official release
  (mmpmmpmmpjosh/TinyFormer); every size reproduces the upstream engine
  exactly (`max_abs_diff == 0` on `pred_logits`/`pred_boxes` for all 5
  released checkpoints, `weights/parity_tinyformer.py`). Reuses the vendored
  DEIMv2 engine (decoder, DINOv3/ViT-Tiny towers; a `dinov3_vitb16` config was
  added for `xl`), the DEIMv2 Dense O2O-style trainer with TinyFormer's
  released recipes, and the DEIM postprocess. ONNX and OpenVINO export and
  reload with detection parity; NCNN is blocked like every DETR family. The
  `l`/`x`/`xl` weights embed finetuned DINOv3 towers and inherit Meta's DINOv3
  License Agreement; `s`/`m` use the DEIMv2-distilled ViT-Tiny towers
  (themselves DINOv3-distilled), so the hosted weight repos carry the
  Apache-2.0 + DINOv3 dual license for every size, as the DEIMv2 DINO sizes do.

- **DEKR bottom-up pose.** New inference-only `dekr` family:
  `LibreYOLO("LibreDEKRw32-pose.pt")` returns multi-person COCO-17 pose on the
  original image canvas without running a person detector first, the first
  genuinely bottom-up pose family in the library. The model emits a dense
  keypoint-plus-centre heatmap and a dense per-keypoint offset field at stride
  4; centres, offsets, joint scores and pose NMS are decoded outside the graph.
  Ported from the Apache-2.0 SuperGradients DEKR-W32-NO-DC implementation at
  pinned commit `63de22c404d5740f34f7706c302b37fce3c8fe5d` (that file also
  carries an upstream Microsoft MIT notice). The port reproduces the pinned
  upstream model exactly (`max_abs_diff == 0` for heatmap logits and offsets on
  zeros, seeded noise, a real preprocessed image, batch 2 and a rectangular
  input), and the decoder reproduces the pinned upstream decoder exactly across
  14 cases including zero people, border candidates, overlapping centres,
  more-than-`max_num_people` peaks and mixed per-item counts in a batch.
  ONNX, TorchScript, TensorRT and OpenVINO export and reload. DEKR predicts no
  boxes: `Results.boxes` is a documented LibreYOLO adapter fitted to the
  confident decoded joints, not an upstream regression head. Weights stay
  hosted by the source provider and are linked rather than redistributed, as
  for YOLO-NAS. Training is not implemented and `model.train(...)` says so.
- **PP-LiteSeg semantic segmentation.** New `ppliteseg` family:
  `LibreYOLO("LibrePPLiteSegt50-sem.pt")` returns `Results.semantic_mask` on the
  original image canvas for Cityscapes' 19 classes. Four released sizes, all
  natively rectangular: `t50`/`b50` at 512x1024 and `t75`/`b75` at 768x1536,
  where `t`/`b` selects the STDC1/STDC2 backbone and `50`/`75` is the source
  validation scale, not a width multiplier. Architecture, compound loss, and
  training recipe adapted from the Apache-2.0 SuperGradients implementation at
  pinned commit `63de22c404d5740f34f7706c302b37fce3c8fe5d`, with STDC backbone
  lineage from the MIT STDC-Seg repository; the port reproduces the pinned
  upstream exactly (`max_abs_diff == 0` for the main logits on all four
  checkpoints and for all three auxiliary logits on `t50` and `b75`, see
  `weights/parity_ppliteseg.py`). ONNX, TorchScript, TensorRT, and OpenVINO
  export and reload at each size's native rectangle with backend results
  matching native inference. **The released checkpoints are trained on
  Cityscapes and are NON-COMMERCIAL** under the Cityscapes dataset terms; that
  restriction covers the pretrained weights only, never LibreYOLO's MIT code or
  the architecture, and weights trained from scratch carry none of it.
- **PP-LiteSeg training.** `model.train(data=...)` runs the source Cityscapes
  recipe through the public API: the main head plus all three auxiliary heads,
  Dice + cross-entropy + edge-attention loss (edge kernel 5) honoring ignore
  index 255 with no extra ignore class, SGD with two explicit parameter groups
  (STDC backbone at `lr0`, everything else at 10x) and zero weight decay on
  BatchNorm and bias parameters, polynomial decay after 10 warmup epochs, EMA at
  0.9999, and full precision. Best-checkpoint selection uses `metrics/mIoU`.
  Fine-tuning on another dataset rebuilds the main and every auxiliary
  classifier together. `imgsz` defaults to the size's source *train crop*
  (512x1024 for `t50`/`b50`, 768x768 for `t75`/`b75`) while validation runs at
  the native rectangle, keeping the two geometries distinct as the recipe
  intends.
- **Rectangular semantic datasets.** `SemanticDataset` accepts an explicit
  `(height, width)` canvas alongside the existing scalar, which stays exactly
  equivalent to the equal pair. It also gains the `rescale_crop` sampler (random
  absolute rescale, ignore-pad, random crop) used by the PP-LiteSeg and STDC
  recipes, and an optional family-local `photometric` transform that replaces
  the shared HSV-gain jitter. `PolyLRScheduler` is available to every trainer.
- **PP-YOLOE detection family.** `LibreYOLO("LibrePPYOLOEs.pt")` runs the
  anchor-free PP-YOLOE detector in sizes `s`/`m`/`l`/`x` at 640 on COCO's 80
  classes: CSPResNet backbone, CSP-PAN neck, and an Efficient Task-aligned
  head with Efficient Squeeze-and-Excitation attention. Ported from the
  Apache-2.0 SuperGradients implementation at pinned commit
  `63de22c404d5740f34f7706c302b37fce3c8fe5d`; all four sizes reproduce the
  pinned upstream model exactly (`max_abs_diff == 0.0` on decoded boxes,
  class scores and the training-form raw tuple, see
  `weights/parity_ppyoloe.py`). Preprocessing is the source's stretch resize
  with ImageNet mean/std on the 0-255 scale, and postprocessing keeps the
  source selection sequence with no invented objectness term. ONNX,
  TorchScript, TensorRT and OpenVINO export, reload and match native
  detections. Pretrained weights stay hosted by the source provider and are
  linked, SHA256-verified and loaded through the restricted loader rather
  than mirrored by LibreYOLO.
- **PP-YOLOE training.** `model.train(data=...)` fine-tunes PP-YOLOE on
  standard YOLO detection datasets through the public API, reusing the
  in-tree `PPYoloELoss` (ATSS, task-aligned assignment, GIoU, DFL) and
  adding the source recipe's two-stage assigner schedule: ATSS first, then
  the task-aligned assigner, with the switch point scaled to the requested
  epoch budget or pinned via `static_assigner_epochs`. Augmentation follows
  the source recipe (affine, mixup, HSV, horizontal flip, random
  90-degree rotation, random RGB-to-BGR swap). Fine-tuning on a different
  class count rebuilds only the class-prediction convolutions.
- **YOLO-NAS-R oriented boxes.** The existing `yolonas` family gains an `obb`
  task: `LibreYOLO("LibreYOLONASs-obb.pt")` returns `Results.obb` on the
  original image canvas, in sizes `s`/`m`/`l` at 1024 on DOTA2's 18 classes.
  Ported from the Apache-2.0 SuperGradients pull request 2014 at pinned commit
  `69141b55c1161d939939a270523a7eca5a645f72`; the rotated head reproduces the
  pinned upstream head exactly (`max_abs_diff == 0` for all three sizes, see
  `weights/parity_yolonas_obb.py`). ONNX, TorchScript, TensorRT and OpenVINO
  export and reload with backend results matching native inference.
- **YOLO-NAS-R training.** `model.train(data=...)` trains the rotated head on
  standard YOLO OBB datasets through the public API, with the loss ported from
  the same pinned upstream commit: varifocal classification, probabilistic-IoU
  regression on `cxcywhr`, and width/height DFL, under a rotated task-aligned
  assigner. Upstream's DOTA recipe supplies the defaults (AdamW 5e-5, weight
  decay 3.5e-6, cosine to 0.1, EMA 0.9997, assigner top-k 12, loss weights
  2.5 / 2.0 / 0.5). Best-checkpoint selection uses `metrics/mAP50-95(OBB)`, not
  the axis-aligned proxy. `LibreYOLONAS.load_detect_weights_for_obb()` seeds a
  rotated model from a YOLO-NAS detection checkpoint, mirroring upstream's
  `strict_load: key_matching` start. Augmentation is flips plus HSV only:
  mosaic, mixup and affine do not have a correct rotated-label mapping and are
  not offered rather than silently corrupting angles.
  The pretrained weights stay under Deci's separate non-redistributable
  YOLO-NAS-R license and are downloaded from Deci's CDN and SHA256-verified
  rather than mirrored — the same treatment YOLO-NAS detect and pose get.
- Shared rotated-box postprocessing helpers in `libreyolo/postprocess/obb_ops.py`
  (corner conversion, axis-aligned proxy, exact rotated NMS, long-side
  canonicalization), extracted from the YOLO9 OBB postprocessor so OBB families
  share one implementation.
- **Perception Encoder (PE) Core** (`pe` / `LibrePE`) — Meta's dual-tower
  vision-language encoder, in all five released sizes (`t16`, `s16`, `b16`,
  `l14`, `g14`). Zero-shot `classify` through `set_classes(...)` and `embed`
  for images, text and **whole videos**, all in one normalized space:

  ```python
  model = LibreYOLO("LibrePEb16-cls.pt", task="embed")
  model.predict("photo.jpg")          # (1, D) image row
  model.embed_text(["a dog"])         # (1, D) text row
  model.predict("clip.mp4")           # (1, D) row for the whole clip
  ```

  PE is the first family to embed a finite video as a *single* vector rather
  than one result per frame: frames are sampled uniformly, encoded
  independently, averaged, then L2-normalized once. This is opt-in per family
  via the new `BaseModel.VIDEO_EMBED_MODE` (default `"frames"`), so CLIP,
  SigLIP2, DINOv2 and every other family keep their existing frame-by-frame
  behavior. Live cameras, network streams and screen captures are rejected for
  whole-clip embedding rather than buffered without bound. `predict()` accepts
  `clip_frames=` (default 8) to change how many frames are sampled.

  Inference-only by design: PE Core has no closed-set supervised head, so
  `train()` rejects immediately with an explanation instead of silently routing
  to the generic classification trainer.

  Exports: ONNX and TorchScript, for image embedding, frozen-class
  classification, and fixed-frame video embedding.

  The towers are a native `torch` re-implementation adapted from timm v1.0.28
  (Apache-2.0) and OpenCLIP v3.2.0 (MIT); neither is imported at runtime.
  Image, text, zero-shot-logit and video outputs are bit-identical
  (`max_abs_diff == 0.0`) to `open_clip_torch==3.2.0` on float32 CPU for all
  five sizes.
- **V-JEPA 2** (`vjepa2`), a self-supervised video encoder, under the generic
  `embed` task plus `classify` for its released attentive probes. Sizes
  `l256`, `h256`, `g256` and `g384`; canonical checkpoints
  `LibreVJEPA2<size>-embed.pt` and `LibreVJEPA2<size>-cls-<ssv2|diving48>.pt`.
  The encoder is a native pure-PyTorch port, so inference and export do not
  depend on `transformers` at runtime.

  This is the first family to consume a *clip* rather than a frame. A finite
  video produces one result per sampled clip, with one clip by default;
  every other family keeps its per-frame video result cardinality. An image
  is accepted as an explicitly documented single-frame input, which is a
  static appearance representation and not a motion one.

  The public embedding is a LibreYOLO pooling contract, defined as the
  arithmetic mean of the final encoder tokens followed by L2 normalization.
  Upstream designates no global retrieval vector, and no retrieval benchmark
  is claimed for it. The native spatiotemporal token grid is available
  separately through `model.embed_tokens(...)` as `(B, T', H', W', D)`.

  Encoder token and pooled-vector parity against unmodified
  `transformers==5.1.0` is exact (`max_abs_diff == 0.0`) on float32 CPU for
  all four sizes, as is single-view logit parity for three of the four
  released probes. The `g384` Diving48 probe differs by one float32 ULP
  (9.54e-07 absolute, 1.1e-07 relative); it is documented rather than papered
  over, and the parity script still asserts `== 0.0`. ONNX and TorchScript
  ship fixed-frame 5D graphs with dynamic batch.

  A frozen attentive-probe trainer ships behind the normal
  `model.train(data="video_dataset.yaml")` entry point. The encoder is frozen
  and kept in eval mode; only the three-layer attentive pooler and the linear
  classifier are optimized, and best checkpoints are selected by top-1
  accuracy. Datasets are entirely user-supplied: manifests are validated in
  full before the first epoch, and no video corpus is ever downloaded.

  Self-supervised encoder pretraining, predictor pretraining and full encoder
  fine-tuning all reject with actionable messages before a dataset is built.

### Changed

- **D-FINE, DEIM, RT-DETRv4 and YOLO-NAS now train with AMP by default**,
  matching the recipes their upstream projects publish. With those four
  flipped, every G0 and G1 detection family defaults to `amp=True` with
  `amp_dtype="float16"` rather than silently training in FP32. D-FINE, DEIM
  and RT-DETRv4 upstream pass `--use-amp` in every documented training
  command; the D-FINE decoder's clamp to ±65504 is the FP16-range guard that
  makes that safe, not evidence that FP32 is required, which is what the
  previous default assumed. YOLO-NAS upstream trains its released COCO recipe
  with `mixed_precision: True`. Dome-DETR stays FP32 and now pins `amp=False`
  explicitly instead of inheriting from D-FINE, because its `dist_train.sh` is
  the one upstream launcher that omits `--use-amp`; PP-YOLOE and YOLO-NAS OBB
  keep their existing FP32 defaults. Pass `amp=False` to restore the previous
  behavior on any family, or `amp_dtype="bfloat16"` for BF16 on Ampere and
  newer.

## [1.5.0] - 2026-08-09

The largest release so far: 28 new model families, four new tasks (`edge`,
`normal`, `embed`, `mesh`), five new export formats, and torch-free ONNX
inference. 600 commits, 88 merged pull requests. Two changes move numbers you
may be comparing against v1.4.0, and one argument was removed: read **Changed**
and **Removed** before upgrading.

### Added

#### Tasks

- Four new canonical tasks, taking the task vocabulary from 13 to 17 entries
  with nothing removed: `edge`, `normal`, `embed` and `mesh`. Each ships an
  original-canvas result payload, visualization, dataset schema and filename
  suffix (`-edge`, `-normal`, `-embed`, `-mesh`). New public result types
  `EdgeMap`, `NormalMap`, `Embeddings`, `Identities` and `Meshes`; new
  validators `EdgeValidator` (ODS/OIS) and `NormalValidator` (angular error).
  Contracts in `docs/adr/`: `0013-mesh-task-contract`,
  `0014-normal-task-contract`, `0017-embed-task-contract` and
  `0018-edge-task-contract`. The embed and edge contracts first shipped as a
  second 0013 and a second 0015; they were renumbered after the release.

- `embed`: L2-normalized image and region embeddings. Ships `LibreFaceEmbedder`
  (ONNX Runtime only, `libreyolo[onnx]`), a `Gallery` / `FaceGallery` API, and
  three CLI commands: `libreyolo enroll` (build a gallery from a
  folder-per-person tree), `libreyolo compare` and its alias `libreyolo verify`
  (two-image verification by cosine similarity). `libreyolo predict` gains
  `--gallery` and `--gallery-threshold` for 1:N identification.
  `docs/adr/0015-embed-generalization` widens the task from face-only identity
  vectors to any region embedding, with `face-recognition` and `reid` as
  aliases.

- `mesh`: body mesh recovery. `LibreSAM3DBody` is available as an optional
  model gated on the `sam_3d_body` dependency; it is deliberately not exported
  from the top-level package.

#### Model families

28 new user-facing classes, exported under 29 names (`LibreModus` is an alias
of `LibreMODUS`). All are inference-only unless stated.

- **Dome-DETR** (`domedetr`), a **trainable** tiny-object detect family for
  aerial, drone and remote-sensing imagery, ported from
  [Dome-DETR](https://github.com/RicePasteM/Dome-DETR) (ACM Multimedia 2025).
  D-FINE plus DeFE (a density head), MWAS (encoder attention restricted to
  occupied windows) and PAQI (query count set by local density instead of a
  fixed 300). Sizes s/m/l at 800x800. `weights/parity_domedetr.py` reports
  `max_abs_diff == 0.0` against the six upstream checkpoints (a manual script
  that requires the upstream weights, not a CI test).

  Training is wired against upstream's full objective: the D-FINE losses plus
  DeFE density and count supervision, padded queries masked out of the
  classification terms, and per-image denoising attention masks. Upstream's
  published 160-epoch `MultiStepLR` schedule has not been reproduced, so the
  paper's AP numbers are unverified.

  Three things to know. Its advantage narrows as objects grow, so it sits
  beside D-FINE rather than replacing it. There is no COCO checkpoint upstream,
  only AI-TOD-V2 (9 classes) and VisDrone (12), so canonical filenames always
  carry a dataset suffix (`LibreDOMEDETRs-visdrone.pt`) and class names come
  from checkpoint metadata. And export is unsupported: PAQI's per-image query
  count makes a traced graph valid only for the image it was traced on, so
  `export()` raises rather than emitting one that silently returns wrong
  results. Weights are not rehosted; the upstream card claims Apache-2.0 while
  also restricting use to academic research, so they are linked upstream
  pending clarification, following the YOLO-NAS precedent. Convert them with
  `weights/convert_domedetr_weights.py`.

- **RT-DETRv2 oriented-object detection**, a second task on an existing family:
  inference for the official DOTA 1.0 `n`, `s`, `m`, `l` and `x` checkpoints,
  with strict local conversion, weight auto-download, aspect-preserving
  preprocessing, native `Results.obb` output, validation, and validated ONNX
  and TorchScript export. Checkpoints are distinguished from detect weights by
  `is_obb_state_dict`.

- **Detection**: `LibreDETR` (r50/r50dc5/r101/r101dc5 @800, bit-exact against
  the pinned facebookresearch/detr source; DC5 is recorded in the LibreYOLO
  checkpoint's `size` metadata because it changes the runtime graph without
  changing any tensor shape, so `size` is required when constructing directly),
  `LibreDeformableDETR` (r50ss/r50ssdc5/r50/r50refine/r50twostage @800, the
  portable `grid_sample` path bit-exact against upstream's pure-PyTorch
  reference), `LibreDINODETR` (r50/r50s5/swinl @800), `LibreLWDETR`
  (t/s/m/l/x @640, the architecture RF-DETR was forked from, so LibreYOLO now
  ships both the ancestor and its descendant), `LibreFasterRCNN` (n/s/m/l =
  MobileNetV3-Large 320-FPN, MobileNetV3-Large FPN, ResNet-50 FPN v1 and v2),
  `LibreMaskRCNN` (r50 @800, instance segmentation by default and detection
  from the same checkpoint), `LibreFCOS` (r50 @800), `LibreRetinaNet`
  (r50/r50v2 @800), `LibreSSD` (size `300` @300), `LibreCenterNet`
  (resdcn18/dla34 @512, legacy DCNv2 replaced by torchvision deformable
  convolution), `LibreEfficientDet` (d0-d4 at 512/640/768/896/1024).

- **Classification** at 224px: `LibreViT` (ti/s/b/l, bit-exact against the
  pinned timm AugReg checkpoints, AugReg2 for base), `LibreSwin` (t/s/b/l),
  `LibreDeiT` (t/s/b, museum tier; DeiT III and 384px variants out of scope),
  `LibreVGG` (16/19/16bn/19bn), `LibreAlexNet` (b). All use fused
  scaled-dot-product attention where applicable and respond to
  `libreyolo.kernels.attention.set_fused_attention`.

- **Semantic segmentation**: `LibreDeepLabv3` (r50/r101/mv3 @520) and
  `LibreFCN` (r50/r101 @520), both 21-class COCO-trained with VOC labels and
  bit-exact against pytorch/vision v0.26.0; and **`LibreLingBotVision`**
  (s/b/l/g @512), a **trainable** ViT family pairing self-supervised
  boundary-centric backbones with a 1x1 dense head on the linear-probing
  protocol. Sizes s/b/l are published; `g` is the 1.1B teacher and has no
  LibreYOLO-hosted checkpoint, so requesting it raises with the reason.

- **Pose**: `LibreHRNet`, top-down COCO-17 with W32 at 256x192 and W48 at
  384x288 person crops. Native heatmaps, affine crop geometry, flip testing and
  decoding are exact against the pinned MIT upstream. Full-image inference
  composes a person detector, or accepts explicit boxes or ready-made crops.

- **Depth**: `LibreMiDaS` (v2.1 Small @256, DPT-Large @384), relative inverse
  depth with no metric unit. Official release assets are downloaded from
  upstream and checksum-verified rather than rehosted, pending ADR 0006
  training-data clearance.

- **Normals**: **`LibreMoGe2`** (s/b/l @518), DINOv2 patch grid with letterbox
  preprocessing. Weights come from `Ruicheng/moge-2-*` at pinned revisions.

- **Edge**: `LibreDexiNed` (b @352) and `LibreTEED` (t @352) with native
  MIT-licensed architectures and local checkpoint converters. Upstream
  BIPED-trained checkpoints are not bundled, mirrored or auto-downloaded
  because the dataset terms are non-commercial.

- **Matte**: `LibreFeyNobg` (l @1024), BiRefNet architecture with stage 3
  deepened to 24 blocks, code and weights Apache-2.0.

- **Analysis**: `LibreMODUS` 14B-A7B, four tasks from one checkpoint
  (`detect`, `depth`, `normal`, `edge`) plus a phrase-grounding mode and
  image-conditioned `any2any()` chaining. The Apache-2.0 code port loads the
  upstream custom-term checkpoint directly or from a local directory and never
  mirrors it; BF16 plus a local-only weight-only FP8 cache.

- **Embeddings**: `LibreFaceEmbedder`, ONNX Runtime only, routed via
  `librefacerec-*` filenames. Not a `BaseModel` subclass.

- `libreyolo/models/registry.py` introduces `MODEL_GROUPS` as a public
  organizing concept (g0 flagship, g1/g2 trainable, g3 inference-only,
  g4 museum, s sibling APIs). A registered family missing from it fails
  `tests/unit/test_model_registry.py`.

#### Torch-free ONNX inference

- `import libreyolo` no longer pulls torch. Model and results names resolve
  through a lazy `__getattr__` with a matching `__dir__`; `__all__` went from
  101 to 142 names with zero removals. A new top-level `libreyolo/preprocess/`
  package provides numpy-native preprocessing for deim, deimv2, dfine, ec,
  rfdetr, rtdetr, yolo9, yolonas and yolox, so an ONNX Runtime install needs no
  torch at all. Answers discussion #711 (#737).

#### Export

- Five new formats: **`rknn`** (Rockchip, with `--name` for the target platform
  and `--verify` for PC-simulator versus ONNX Runtime parity, plus public
  `export_rknn`, `run_rknn_simulator`, `compare_rknn_outputs` and
  `verify_rknn_simulator_parity`), **`mnn`**, **`paddle`**, **`executorch`**
  and **`coreai`** (Apple Core AI, macOS only).
- `deepstream=True` on ONNX export writes DeepStream sidecar configs alongside
  the graph. It is an ONNX-only option rather than a `format=` key, and is
  mutually exclusive with `nms=True`.
- New required checkpoint sidecars: `<program>.pte.json` for ExecuTorch and
  `<model>.mnn.json` for MNN. HRNet exports carry `pose_input: "person_crop"`.

#### Serving and backends

- `TritonBackend` plus `create_triton_config` for NVIDIA Triton Inference
  Server (`docs/triton.md`), and `PaddleBackend` for Paddle Inference.

#### Training

- CUDA graph capture of the **training** step (`train(..., cuda_graph=True)` /
  `--cuda-graph`) extended from 2 families to **24**, across five tasks: detect
  (`yolo9`, `yolo9_p2`, `yolo9_e2e`, `yolox`, `yolo7`, `yolonas`, `picodet`,
  `rtmdet`, `rfdetr`, `dfine`, `deim`, `deimv2`, `rtdetr`, `rtdetrv2`,
  `rtdetrv4`, `ec`), classify (`resnet`, `convnext`, `mobilenetv4`,
  `efficientnetv2`), semantic (`segformer`, `lingbotvision`), point (`fomo`)
  and restore (`nafnet`). Measured on an RTX 5070 Ti under AMP: 3.63x (FOMO),
  2.74x (MobileNetV4), 1.99x (YOLO9-t), 1.04-1.26x for the rest; the win tracks
  how much of a step is network rather than loss, and is largest at small batch
  sizes. End to end on a real 20-epoch YOLO9-t fine-tune (406 images,
  dataloader and validation included): 428 s to 368 s, 1.16x, with identical
  mAP50-95 and per-epoch losses.

  The encoder-decoder detectors capture backbone plus encoder only, because
  their decoder sizes its denoising queries from the batch's ground-truth
  count, so its token count is not static. Most families are bit-identical to
  eager; the three documented exceptions (families whose own eager training is
  not reproducible, RTMDet's cross-level shared head convolutions, and networks
  with stochastic depth inside the captured region) are measured and
  tolerance-gated per family in
  `tests/e2e/test_cuda_graph_training_families.py`. The suite asserts capture
  actually engages per family rather than only that the run completes, which is
  what caught the DETR-line trainers routing through the eager `on_forward`
  instead of `_forward_train` while the specs were being written. See
  `docs/training_cuda_graphs.md`.

- Training from scratch (`pretrained=False`) now works for **every** g0/g1/g2
  family through a seeded `_from_scratch()` random init; previously only
  yolo9, rfdetr and dfine had a scratch path and the other families silently
  loaded the pretrained checkpoint anyway. RF-DETR now honors
  `pretrained=false`, which it previously ignored (#730).
- Rectangular (non-square) training input, e.g. `imgsz=480x640`, for the CNN
  detect families (`yolo9`, `yolo9_e2e`, `yolo9_p2`, `yolox`, `yolo7`,
  `rtmdet`, `picodet`). Transformer families and non-detect tasks raise a clear
  error, and both dimensions must divide the family stride (#649, #658).
- `amp_dtype` (`"float16"` default, `"bfloat16"`) on `TrainConfig`,
  `ValidationConfig`, and `--amp-dtype` on `train`, `val` and `profile`.
  GradScaler is skipped for bf16.
- Configurable validation caps during training: `max_det` (default 300) and
  `eval_max_det` (default `None`, preserving pycocotools maxDets behavior),
  which decouples the COCO evaluator cap from NMS.
- `val_loss=True` (still opt-in, `False` by default) extended from the `g0`
  flagships to every trainable family that can support it: detect (`yolo9`,
  `yolo9_p2`, `yolo9_e2e`, `yolonas`, `rtdetr`, `rtdetrv2`, `rtdetrv4`,
  `dfine`, `deim`, `deimv2`, `ec`, `rtmdet`, `picodet`, `yolox`, `yolo7`,
  `rfdetr`, `domedetr`), classify (`resnet`, `convnext`, `mobilenetv4`,
  `efficientnetv2`), semantic (`segformer`, `lingbotvision`, `dinov2`) and
  restore (`nafnet`). Detect, classify, semantic and restore only: segment,
  pose, obb and the dense tasks are not covered and raise. Components stay
  weighted so they sum to the reported total; denoising terms are never
  included because validation forwards without ground truth. `val_loss` moved
  from `YOLO9Config`/`RFDETRConfig` to `TrainConfig`, so a family that has not
  implemented it raises a clear error instead of ignoring the flag. FOMO
  already computed a validation loss unconditionally and now also publishes it
  under the shared `metrics/loss` keys so `libreyolo monitor` overlays it like
  every other family (`metrics/val_loss` is kept).
- Built-in Comet, ClearML, Neptune (`neptune-scale`) and DVC/DVCLive training
  loggers, with the same canonical metrics and failure-isolation contract as
  the existing TensorBoard, MLflow and Weights & Biases integrations. Neptune
  is excluded from the `all` extra because `neptune-scale` needs protobuf<7.
- A post-resize image cache point: families consuming the dataset's
  deterministic resize now cache the resized frame, which skips both decode and
  resize and is roughly an order of magnitude smaller. Caching also applies in
  the per-epoch validation loop.
- Autobatch and DDP spawn honor rectangular `imgsz` and the chosen AMP dtype
  when probing batch size.
- Training profiler coverage extended to the full trainable registry;
  `libreyolo profile infer` verified per family for g0/g1 plus g2 detect and
  point.

#### Inference

- CUDA graph capture for inference (`predict(..., cuda_graph=True)`), new in
  this release. 39 families are verified to capture and replay bit-identically
  against two probe inputs in the shared parity matrix
  (`tests/e2e/test_cuda_graph_families.py`), spanning detect, segment, pose,
  point, classify, semantic, depth, restore and matte; PP-OCR, SAM and
  SenseNova are enabled through family-specific paths with their own tests.
  Families that cannot be captured whole are split at a verified seam (Depth
  Anything 3's sky step, BiRefNet's deformable decoder, PP-OCR's recognition
  stage, SAM's prompt path, SenseNova's autoregressive generation) and the
  remainder runs eagerly with identical numbers. See `docs/cuda_graphs.md` for
  the verification protocol and why unverified families raise instead of
  falling back silently.
- Live and remote predict sources: webcams by index, RTSP and HTTP streams,
  screen capture, and `s3://` and `gs://` URLs, with `--stream`,
  `--stream-buffer`, `--vid-stride` and `--show`. Screen capture adds `mss` as
  a required dependency. See `docs/predict_sources.md`.

#### Kernels and quantization

- Library-wide kernel registry at `libreyolo/kernels/` (lifted from
  `libreyolo/quant/kernels/`), organized by purpose: `quant/simulate/`
  (fake-quant Triton, any device, STE backward), `quant/execute/`
  (finalized-only fp8 GEMM path and unpack kernels), and `attention/`.
  `LIBREYOLO_KERNELS` replaces `LIBREYOLO_QUANT_KERNELS` (still honored), and
  `libreyolo.quant.kernels` remains a working alias. See `docs/kernels.md`.
- Optional accelerated multi-scale deformable attention (`ms_deform_attn`
  slot): with `pip install libreyolo[hub-kernels]`, every
  Deformable-DETR-lineage family runs the compiled Apache-2.0 CUDA kernel from
  `kernels-community/deformable-detr` (forward and backward) instead of the
  portable `grid_sample` path: RF-DETR, LibreDeformableDETR, LibreDINO-DETR,
  LW-DETR, Grounding DINO, RT-DETR, RT-DETRv2, D-FINE (and RT-DETRv4), DEIM
  (and DEIMv2), EC and OV-DEIM. Installing the extra is the opt-in;
  `LIBREYOLO_HUB_KERNELS=0` disables it. Eager CUDA fp32 only; exports always
  keep the portable path; load or runtime failures fall back with one warning.
  The Hub artifact is pinned to an audited commit revision and a CUDA-only
  parity test (`test_hub_matches_portable_on_cuda`) gates revision bumps.
  Shapes the slot cannot express (a per-level sampling point count, or
  `method='discrete'`) also fall back. See **Changed** for the effect on
  RF-DETR outputs.
- Fused scaled-dot-product attention across the transformer families, using
  stock torch and no optional dependency. SegFormer, Depth Anything (and
  MoGe-2), BERT, Grounding DINO, SwinIR, PP-OCR, LibreViT and LibreDeiT use it
  by default. Families pinned to a byte-exact parity bar (Swin,
  LibreDINO-DETR's Swin backbone, BiRefNet, FeyNoBG, OWLv2, LW-DETR, SigLIP 2,
  ZipDepth, MobileSAM) keep manual attention by default and opt in with
  `libreyolo.kernels.attention.set_fused_attention(model)`, which trades
  byte-exact agreement with upstream for the fused kernel. Export graphs keep
  the primitive-op equation: the fallback triggers on ONNX export and on
  `torch.jit.trace`, so the TorchScript, CoreML and NCNN exporters record the
  same graph as before. Informal single-op timings on an RTX 5070 Ti under fp16
  autocast: 1.8x on Swin window attention (1.278 to 0.721 ms, 512 windows x 49
  x 384) and 3.7x on OWLv2 vision attention (6.483 to 1.735 ms, 3600 tokens x
  1024). No benchmark harness ships with these numbers; see `docs/kernels.md`
  for the shapes.
- Quantization support for the birefnet and feynobg families
  (fp16/bf16/fp8/int8/w4a16/w4a8/nvfp4/mxfp4; int2 is rejected because these
  families are inference-only and cannot heal). Pre-quantized fp16 and fp8
  LibreFeyNobg checkpoints are published on the LibreYOLO Hugging Face org,
  loadable by passing the downloaded `.pt` as the weights argument. An nvfp4
  variant was built, measured and withdrawn: no kernel path beats fp16 on these
  GEMM shapes and 4-bit noise can flip foreground selection on ambiguous
  scenes.
- Native fp8 execution tier: finalized fp8 `QuantLinear` runs on the fp8 tensor
  cores via `torch._scaled_mm` (Ada/Hopper/Blackwell); optional Triton kernels
  fuse activation conversion and the per-channel scale/bias epilogue, while
  validation-selected FeyNobg Swin stage-0 Linears use manifest-recorded
  tensorwise weight scales for a fully fused cuBLASLt epilogue. Finalized fp8
  `QuantConv2d` convolves in fp16 on cached dequantized weights, and
  fp16-remainder checkpoints get float32 I/O root hooks. On
  LibreFeyNobg/RTX 5070 Ti, fp8 is 123.1 versus 129.3 ms for batch-1 graphed
  predict and 515.4 versus 535.3 ms at batch 4, with a 275 versus 531 MB file.
- CUDA graph capture for the birefnet and feynobg families via encoder-only
  capture (the deformable decoder replays wrong under capture and stays eager;
  graphed output is bit-identical to eager). `GraphRunner` warms up on the
  capture stream so lazily-allocated cuBLASLt/cuDNN workspaces stop
  invalidating capture, and quant modules cache the calibration flag as a host
  bool (the per-forward `.item()` sync also invalidated capture).

#### Packaging, docs and tooling

- One new required runtime dependency: `mss>=9.0.1` (screen capture).
- 14 new optional extras: `stream`, `triton`, `executorch`, `coreai`, `midas`,
  `fast-eval`, `hub-kernels`, `modus`, `paddle`, `mnn`, `comet`, `clearml`,
  `neptune`, `dvclive` (with `dvc` as an alias). The `all` aggregate gained
  nine of them; `executorch`, `coreai` and `neptune` are excluded over torch-pin
  and protobuf conflicts, and `fast-eval` and `hub-kernels` are deliberately
  opt-in. A new `[tool.uv]` conflicts table declares `neptune` mutually
  exclusive with `tflite`, `litert` and `all`.
- 38 new documentation pages: 10 top-level guides (`cuda_graphs`, `deepstream`,
  `facial_recognition`, `kernels`, `libremodus`, `paddle`, `predict_sources`,
  `rknn`, `training_cuda_graphs`, `triton`), 22 provenance pages and 6 ADRs.
- Two new skills, `libreyolo-write-model-prd` and `run-rf100vl-benchmark`;
  14 existing skills updated.
- `libreyolo checks` now reports `paddlepaddle`, `x2paddle` and `mnn`.

### Changed

- **The faster-coco-eval COCO metrics backend is now the default**
  (`faster_coco_eval=True` on `ValidationConfig` and `TrainConfig`; CLI gains
  `--faster-coco-eval/--no-faster-coco-eval`, and `LIBREYOLO_FASTER_COCO_EVAL=0`
  still forces pycocotools). Decision based on measured parity across all 100
  RF100-VL test splits: 1381 of 1400 metric values bit-identical to
  pycocotools, maximum deviation 2.22e-16, headline deltas exactly 0, with
  15.6x faster evaluation overall and 56x on detection-dense datasets.
  pycocotools remains the automatic fallback when faster-coco-eval is not
  installed. The backend actually used is logged at INFO, surfaced as
  `model.last_eval_backend` after `val()` and as `COCOEvaluator.last_backend`,
  and included as `eval_backend` in the CLI JSON payload. Install with
  `pip install libreyolo[fast-eval]`.

- **YOLOX BatchNorm eps now survives the class-count rebuild.** eps=1e-3 and
  momentum=0.03 (the official YOLOX values) are applied by `LibreYOLOXModel` at
  construction rather than as a post-hoc fixup in the wrapper, so they survive
  the `_rebuild_for_new_classes()` that `train()` performs when the dataset `nc`
  differs from the checkpoint. Previously such a fine-tune trained and reported
  in-training validation at torch's default eps=1e-5 but was reloaded for
  inference at 1e-3: same tensors, different normalization. Regular-conv sizes
  barely move. Depthwise `n` has per-channel running_var small enough for eps to
  dominate: on RF100-VL `ball` the same nano checkpoint scores 0.566 mAP50-95
  evaluated at its trained eps and 0.151 after a stock reload. **Checkpoints
  trained before this fix carry eps=1e-5 semantics** and must be evaluated with
  BN eps overridden to 1e-5 (or have `sqrt((var+1e-3)/(var+1e-5))` folded into
  the BN weights) to report faithful numbers.

- **D-FINE training applies upstream's per-size multi-scale recipe** instead of
  a hardcoded `base_size_repeat=3`: n trains at fixed size, s uses 20, m 6,
  l 4, x 3. Only x matched before, so n/s/m/l now converge differently. The new
  `DFINEConfig.base_size_repeat` overrides the per-size default. DEIM still
  uses the hardcoded 3 (#675).

- **Rectangular-input results changed because they were wrong before.** Box
  coordinates, RTMDet instance-segmentation mask resizing, YOLO-NAS box and
  keypoint rescaling, and validator ground-truth rescaling now all use per-axis
  height and width instead of a single scalar. Square `imgsz` is bit-unchanged;
  any previously-run rectangular inference or validation was mis-scaled.
  YOLO-NAS now rejects rectangular `imgsz` outright rather than silently
  producing wrong numbers, and its internal resize is clamped to the canvas so
  boxes and pose keypoints scale correctly below the family resize constant
  (#649).

- **Metrics dictionaries gained keys.** `COCOEvaluator.compute()` and `val()`
  return `max_det`, `ar_max_det` and `AR_max_det`, and AR100 is recomputed
  rather than read from `coco_eval.stats[8]`. FOMO validation gained
  `metrics/loss` and `metrics/loss/ce` as aliases of the existing
  `metrics/val_loss`. Values at default settings are unchanged, but anything
  iterating metric keys (loggers, monitor overlays, CSV headers) sees new
  columns. Setting `eval_max_det` to anything other than 100 bypasses
  `summarize()` and recomputes all 12 stats at the requested cap, which moves
  every COCO number relative to v1.4.0.

- **`YOLO9.train(seed=N)` seeds before the class-head rebuild** rather than
  after, so a seeded run against a dataset whose class count differs from the
  checkpoint's starts from a different head initialization than v1.4.0.

- **With `libreyolo[hub-kernels]` installed on CUDA, RF-DETR and the other
  deformable-attention families now actually use the native MS-deform-attn
  kernel.** v1.4.0 gated it behind a condition RF-DETR never took, so the
  kernel never ran. Forward outputs now come from the compiled binary rather
  than the portable PyTorch path, so predictions and metrics can shift at float
  tolerance. Stock installs are unaffected.

- **`libreyolo predict` source handling.** Sources are classified before path
  existence is checked, so webcam indices and RTSP/stream URLs no longer fail
  as "Source not found"; remote prefixes now include `s3://` and `gs://`; and
  live sources implicitly enable streaming, which changes the JSON output shape
  to one record per frame. The CLI also filters all predict kwargs against the
  model's `__call__` signature, so an option a family does not accept is now
  dropped rather than raising `TypeError`.

- **`save_json=True`** (opt-in, off by default) now writes the COCO results
  JSON even when zero detections were accumulated, and writes it before
  `loadRes` mutates the result dicts, so on-disk entries no longer carry the
  `id`/`area`/`segmentation` fields pycocotools injected. The logged path is
  absolute.

- **ONNX pose export output naming.** `rfdetr-pose` (3 output tensors) and
  `yolonas-pose` (4) were previously misread as segmentation by an output-count
  heuristic, so the exported graph could carry segmentation output names.
  Re-exporting those two models now yields differently named outputs; existing
  `.onnx` files on disk are untouched.

- **On a torch-free install**, results containers hold numpy arrays instead of
  `torch.Tensor` and NMS runs through a numpy implementation, so `.boxes.data`
  returns a different type and tie-breaking may not match torchvision exactly.
  With torch installed, behavior is byte-for-byte unchanged.

- **`ValidationConfig` and `TrainConfig` validate more at construction.**
  `TrainConfig` gained a `__post_init__` where v1.4.0 had none, and both now
  raise `ValueError` for invalid `max_det`, `eval_max_det`, `edge_max_dist`,
  `edge_thresholds` and malformed `imgsz`. `TrainConfig.imgsz` widened to
  `int | tuple | list | str`; `ValidationConfig.imgsz` to `int | tuple` only.
  `ValidationConfig` serialization gained an `edge_thresholds` key, which
  breaks strict `ValidationConfig(**dump)` round-trips from a v1.4.0 dump.
  `libreyolo val` now autocasts with the configured `amp_dtype` instead of a
  bare fp16 autocast (the default preserves v1.4.0 behavior).

- **`ClassifyValidator`** gained an explicit `__init__` with a keyword-only
  `loss_adapter` and `ValidationLossMixin` in its MRO, and `_num_classes` is
  now the model's head width rather than the folder count, so confusion
  matrices sized from it change shape on an ImageNet subset.
  `BaseValidator._run_validation` calls a new `_update_batch_metrics` hook on
  the raw forward output; third-party validator subclasses inherit that call
  site.

- **CLI weight-name resolution** for families that require a task suffix now
  yields the suffixed filename (`segformer-b0` resolves to
  `LibreSegformerb0-sem.pt`), which fixes 404s but changes the mapping for any
  caller that hardcoded the old unsuffixed name.

- **The LICENSE no longer claims published weights are MIT or Apache-2.0.** It
  now states that weights licenses vary, that some are non-commercial or
  otherwise restricted, and that the repository's MIT License does not extend
  to them. Choosing a model means choosing its license.

- The pytest marker `experimental_backend` was renamed `extended_backend`, and
  `general_nightly` now means "curated native inference checks in the default
  nightly" rather than "broad nightly inference checks across all model
  families".

### Removed

These require a code change when upgrading from v1.4.0. No public model class
or function was removed: `__all__` grew from 101 to 142 names with zero
removals.

- **`allow_experimental=True` is gone from every `.train()` gate**, together
  with the `ddp_aware(experimental_key=...)` mechanism. EC, RTMDet, PicoDet and
  FOMO training and export previously required it. **Migration: delete the
  argument.** There is no deprecation shim; a call that still passes it raises
  `TypeError`.
- **`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` is gone.** Subclasses that
  declared the frozenset no longer emit a preview-checkpoint warning.
  `get_download_notice()` survives as a hook and is still overridden by midas,
  segformer and yolo9_p2; only the base implementation returns `None`.
- **The export support tier `"experimental"` is gone.** `Tier` is now
  `Literal["validated", "available", "blocked"]` and `BaseExporter` no longer
  emits a `RuntimeWarning` for it. **Migration: code branching on the string
  `"experimental"` should use `"available"`.**
- **`pretrained=false` combined with `resume` now raises**
  `ValueError("pretrained=False cannot be combined with resume.")` instead of
  proceeding incoherently.
- **CLI `--imgsz` changed from int to str** on `predict`, `train` and `val` so
  it can accept `480x640` (the comma form still works, and `train`'s default is
  now the string `"640"`). Typing `--imgsz 640` on the command line is
  unaffected, as is `model.predict(imgsz=640)`; only code calling the CLI
  command functions directly in Python needs to pass a string. `export --imgsz`
  was already a string and `profile` is unchanged.
- **Checkpoint schema:** `imgsz` is no longer square-only. Rectangular
  checkpoints keep a scalar `imgsz = max(imgsz_h, imgsz_w)` for legacy readers
  and dual-write `imgsz_h`/`imgsz_w`; readers that understand the rectangular
  fields should prefer them.

### Fixed

#### Data loss and corruption

- Auto-converting an upstream checkpoint wrote the new `.pt` in place, so a
  crash, kill or full disk mid-save left a permanently unloadable truncated
  file. The write is now staged and atomically renamed (#740).
- Auto-conversion rewrote the checkpoint through a 0600 temp file, so a
  group- or world-readable shared weights file silently became owner-only after
  the first conversion. The original file mode is preserved (#740).
- Interrupted weight downloads could resume against a changed remote file and
  silently produce corrupt weights, and two concurrent processes could clobber
  the same download. Downloads now record an ETag/Last-Modified validator for
  `If-Range` resume and take a cross-process file lock (#644).
- If quantization calibration failed part-way, the model was left half-swapped
  with partially written observer ranges. Failure now rolls back to the
  original float modules (#656).

#### Silently wrong results

- Non-strict checkpoint loads (families with `_strict_loading() == False`, for
  example YOLOX) accepted a partially matching checkpoint in silence and then
  predicted with freshly initialized tensors for the dropped keys. Loads now
  warn with the counts and first names of missing and unexpected keys; healthy
  loads stay silent and full key lists are available at DEBUG. YOLO-NAS's
  custom loader bypassed that logging and now routes through it (#707).
- `LibreDINOv2` would claim it could load a LingBot-Vision checkpoint (their
  state dicts look alike) and silently load the wrong model. Rejection is now
  bidirectional, using the RoPE buffer as discriminator (#642).
- PicoDet passed the canvas size with height and width swapped, so boxes were
  clipped against the wrong axis; RTMDet segmentation resized masks to a square
  derived from width only, distorting masks and inflating memory at wide aspect
  ratios (#649).
- Faster R-CNN ONNX exports placed boxes wrong on non-square images: export
  forced a fixed square input while the graph does its own aspect-preserving
  resize, and the backend applied a single wrong inverse scale (#685).
- The Triton backend derived output ordering from `config.pbtxt` rather than
  server metadata, so when the two disagreed results were silently mapped to
  the wrong tensors (#721).
- SenseNova CUDA-graph inference returned wrong results when two different
  image packings produced the same token count, reusing stale position ids and
  `cu_seqlens`. Mismatched packings now run eagerly (#714).
- An exported surface-normal / MoGe-2 model run on an image whose aspect ratio
  differed from the fixed export canvas silently stretched the image and
  produced wrong normal directions. It now raises and points at the `.pt`
  model (#660).
- LW-DETR could return fewer detections than `max_det` and drop real ones,
  because top-K ran over all 91 head columns and unmapped COCO ids were
  discarded afterwards. Unmapped columns are now sliced out before top-K in
  both native and exported decode (#676).
- EfficientDet fine-tuned checkpoints with a different class count failed to
  load because the head was rebuilt at the wrong width, and 80-class custom
  models were wrongly remapped through the sparse COCO-90 label table (#709).
- 3D body-mesh renders looked like a flat silhouette because normals mixed
  pixel coordinates with metres, shading all triangles identically (#659).
- Quantizing matte models (BiRefNet family) produced broken results:
  training-only supervision heads left uncalibrated observers in the manifest,
  and deformable-conv weight containers were never quantized while
  finalization stripped the weights they read (#657).
- `dequantize_model` crashed with `StopIteration` on a bias-free quantized
  Linear, and a calibration batch that raised left all observers permanently
  active so every later forward kept widening quantization ranges (#656).
- RF-DETR validation-loss components were logged unweighted, so they did not
  sum to the reported total (#672).

#### Training reliability

- CUDA graph capture no longer races DataLoader pin-memory threads: training
  and inference/validation capture run with
  `capture_error_mode="thread_local"`, so a `cudaHostAlloc` from a pin-memory
  thread staging the next batch can no longer invalidate the capture and poison
  that thread. Previously the run died with "AcceleratorError ... in pin memory
  thread"; observed twice on an RF100-VL campaign (#681).
- CUDA-graph training warm-up ran extra forward passes that advanced BatchNorm
  running stats beyond one eager step, so validation, EMA and saved checkpoints
  drifted from an equivalent eager run. Buffers are snapshotted and restored
  around capture (#671).
- Graphed validation could silently lose an epoch's validation and ran 2.4x
  slower than eager, because capture raced pin-memory threads mid-loop and was
  retried with full warmup on every later batch after failing. Capture now
  happens once before the loop, shape misses run eager, and one failure latches
  permanent eager fallback (#677).
- CUDA graphs were captured and replayed on the wrong device, so a model on
  `cuda:1` silently fell back to eager, and a cached graph kept pointing at
  freed memory after quantize/dequantize/device changes (#645).
- A thread race could permanently leave an internal subclass installed as
  `torch.cuda.CUDAGraph` for the rest of the process when two captures
  overlapped. The swap is now lock-serialized (#681).
- CUDA-graph training could die with an out-of-memory error while snapshotting
  BatchNorm buffers just before capture; that allocation now sits inside the
  capture guard so the run falls back to eager instead of crashing (#716).
- Training on a YOLO-format dataset crashed when `imgsz` was a rectangular
  tuple, and `imgsz` input was never sanitized (#658).
- `LibreDINOv2.train(device=...)` crashed with "got multiple values for keyword
  argument 'device'" (#724).
- D-FINE, DEIM and EC-Pose GO-LSD training did roughly 1,200 GPU-to-CPU syncs
  per step building the matched-index union, stalling the GPU. One batched
  transfer now does the same work with identical numerics (#726).
- SwinIR kept its channel mean on CPU and rebuilt its attention mask on CPU on
  every call, forcing device transfers and repeated CPU work per inference
  (#714).

#### Weights and model resolution

- D-FINE, DEIM, DEIMv2 and EC refused to auto-download weights when given a
  plain weight name, raising `FileNotFoundError` instead of fetching. A
  genuinely failing download reports the real reason again (#738).
- CLI names for CLIP, SigLIP2, DepthAnything, ZipDepth and FOMO resolved to
  weight filenames missing their task suffix, so auto-download pointed at
  nonexistent files and 404'd; DINOv2 got no CLI aliases because lazy
  registration ran too late (#643). The same 404 class affected segformer,
  pidnet, eomt and lingbotvision default-task names (#642).
- The CLI's model-name lookup only saw eagerly registered classes plus RF-DETR,
  so lazily registered families could not be resolved by name (#736).
- Dome-DETR's weight-download hook answered for other families' filenames, so
  auto-downloading `LibreTEEDt-edge.pt` and `LibreDexiNedb-edge.pt` failed with
  a Dome-DETR error. A bare `LibreDOMEDETRs.pt` retried a nonexistent repo three
  times and ended in a generic "file not found"; it now fails immediately and
  names the valid suffixed alternatives (#729).
- Requesting `lingbotvision-g`, which has no published checkpoint, produced an
  HTTP failure against a repository that does not exist. It now raises with the
  reason and names the published sizes.

#### Export

- TorchScript, CoreML and NCNN exports embedded
  `aten::scaled_dot_product_attention` because the manual-attention fallback
  only checked for ONNX export. Traced exports now use the primitive-op
  attention path they were validated with (#713).
- DeepStream detection configs pointed `model-engine-file` at `<onnx>.engine`
  while DeepStream-Yolo writes `model_b<N>_gpu0_<mode>.engine`, so every launch
  rebuilt the TensorRT engine from scratch instead of reusing the cache (#728).
- Further DeepStream contract bugs: `deepstream=True` was silently accepted for
  non-ONNX formats; raw-tensor tasks (pose, depth, restoration) had their
  outputs collapsed into one tensor so apps could not decode them;
  aspect-ratio and padding keys were missing for classify; YOLO-NAS pose got
  detection's RGB and centered-pad preprocessing; the unsupported-task error
  listed a stale hardcoded task list; and a config was emitted for
  `depth_anything3`, which has no export implementation at all (#728).
- An RKNN export that failed parity verification still left the `.rknn` model
  and metadata at the final path, so automation could deploy a model that
  failed accuracy checks. The "requires Linux x86_64" guard also passed on
  Linux ARM. Re-exporting left stale `.parity.json` sidecars, and publish was
  not fully transactional (#723).
- Mask R-CNN ONNX export from a CUDA model crashed on a device mismatch because
  mask-pasting helpers created CPU tensors (#709).
- The Hub-hosted deformable-attention kernel failed to load under current
  `kernels` releases, so the accelerated path was unavailable (#724), and
  RF-DETR never reached the kernel slot at all because of a condition it always
  fails (#712).

#### CLI, predict and tasks

- `predict(..., classes=[0])` crashed with `ModuleNotFoundError` on a
  torch-free install because the class filter and the mask and keypoint
  wrapping still called torch (#737).
- `libreyolo predict` forwarded CLI options that a backend or model does not
  accept, raising `TypeError` (#721).
- Predicting on a list of in-memory images with `LibreEnsemble` gave every
  result the same save name, so `save=True` overwrote all outputs with the last
  image (#717).
- HRNet pose prediction did not accept a list of images, and the model
  inventory emitted tuple image sizes that broke JSON round-tripping (#709).
- MiDaS depth on extreme aspect-ratio images could resize a side to zero pixels
  and crash (#709).
- RT-DETRv2 `train()` raised a confusing `AttributeError` when `task` was unset
  instead of the intended OBB-unsupported error; predict and val raised
  `AttributeError` picking a preprocessor when the backend had no `task`
  attribute; and the OBB train guard hid `train()`'s real signature and
  docstring from help and introspection (#733).
- Embedding and gallery bugs: enroll wrongly enrolled every detected face
  instead of the most prominent one, breaking multi-face reference photos;
  `threshold=` was silently ignored on non-embed models; matching wrongly
  required a gallery fingerprint; and zero-row embedding payloads broke the
  dimension check (#661).
- Face recognition: enrolling into an existing gallery through a different
  model silently appended vectors that could never match, and a video source
  fell through to the image path and errored in `ImageLoader`. Passing
  `face_boxes` with a video source raised `RuntimeError` on every frame
  (#654).
- Mesh: passing `person_boxes` with a video source silently dropped the boxes
  and then raised a nonsensical error asking for `person_boxes`; a model
  constructed with `device="cpu"` on a GPU machine was silently forced onto the
  GPU and then crashed; `track()` produced garbage instead of refusing; a
  missing upstream `.faces` attribute raised `AttributeError`; and body-only
  inference crashed with a `KeyError` because hand parameters were assumed
  always present (#659).
- `imgsz` handling was duplicated and inconsistent across CLI, dataset,
  training and export. One shared normalizer now backs every entry point, the
  CLI rejects garbage values with a clear error instead of a raw traceback,
  `val` accepts `HxW`, rectangular training is rejected where it does not work,
  and tiled inference no longer silently accepts a rectangular `imgsz` it
  cannot use (#649, #658).
- Validating a classification model against a WNID/synset ImageFolder that is a
  subset of ImageNet-1k no longer raises; labels map through the canonical timm
  synset index.
- Depth Anything V2's `train()` told users to fine-tune upstream and convert,
  which does not actually work. The error now says training is unsupported and
  points to the pretrained checkpoints and the converter (#641).

### Contributors

- **juni3227**, first-time contributor: rectangular input resolution for the
  convolution-based detectors, and a fix for training with YOLO-format datasets
  at rectangular model sizes (#649, #658).
- **Xuban Ceccon**: everything else.

Thanks to the people whose issues closed in this window: juni3227 (#631, #652),
Octoslav (#542), kaiwen0901 (#484), JPABotermans (#455).

### Stats

600 commits, 902 files changed, +125,701 / -3,315 lines, 88 merged pull
requests (#639-#740), 8 issues closed. 174 new test modules (279 to 453,
+62%). 38 new documentation pages.

```
pip install --upgrade libreyolo
```

## [1.4.0] - 2026-07-24

LibreYOLO v1.4.0: 15 new model families, 3 new tasks (panoptic, matte, OCR), a quantization stack, two new trackers, and a multi-GPU training correctness overhaul.

### Added

- New model families:
  - LibreSegformer (SegFormer), semantic segmentation, sizes b0-b5 at 512px (b5 640px); code Apache-2.0, converted NVIDIA ADE20K weights non-commercial with a pre-download license notice (#589)
  - LibreSwinIR, x4 super-resolution (restore task), sizes s/m/l, Apache-2.0 code and weights (#571)
  - LibreRealESRGAN, super-resolution, sizes x4/x2 (RRDBNet) and x4t (compact SRVGG) (#549)
  - LibreBiRefNet, background removal with the new matte task, sizes t/l at 1024px; the t (lite) weights are not yet rehosted pending license confirmation (#549)
  - LibreZipDepth, depth, sizes b and bnpu (NPU-friendly decoder), 384px, MIT (#562)
  - LibreDepthAnything3, depth, size l at 504px, Apache-2.0; separate family from LibreDepthAnythingV2 (#577)
  - LibrePPOCR (PP-OCRv5), text detection + recognition with the new ocr task, sizes t/l at 960px, inference and validation only (#575, #587)
  - LibreSigLIP2, open-vocabulary zero-shot classification, sizes b16/so400m, native torch, inference only (#546)
  - LibreYOLO1, a YOLOv1 museum family, detect, sizes t/b, VOC 20 classes, fixed 448px; pretrained weights ship for b only (the tiny-yolov1 weights are lost upstream) (#549)
  - LibreSAM3 (SAM 3), promptable segmentation, size large at 1008px, transformers-backed; weights gated on Hugging Face under the Meta custom SAM License (#576)
  - LibreEdgeTAM, promptable segmentation, size edge at 1024px, image inference only, Apache-2.0 (#602)
  - LibrePicoSAM3, native 96px promptable ROI segmentation, ONNX-only export (#585)
  - LibreOMDetTurbo, open-vocabulary detection, size t, transformers-backed (#600)
  - LibreOVDEIM, open-vocabulary detection, sizes s/m/l, native NMS-free port via LibreOpenVocab("ov-deim"); code Apache-2.0, weights CC BY-NC 4.0, licensing confirmed by the upstream author (#607)
  - LibreSenseNovaVision, 7B unified multimodal checkpoint serving 7 tasks; weights CC BY-NC 4.0 non-commercial; not yet in __all__, the model inventory, CLI or UI (#618)
- Three new tasks: panoptic, matte, ocr, with result types PanopticSegmentation, Matte, OCRRegions and validators PanopticValidator (+ PanopticQuality), MatteValidator, OCRValidator (#557, #560, #549, #575)
- EoMT instance segmentation and panoptic: sizes s/b/l, 640px, new 1280 weight variant, panoptic checkpoints for s/b/l (#553, #557, #560)
- RTMDet-Ins instance segmentation, inference and validation, sizes t/s/m/l/x (training not implemented) (#572)
- D-FINE segmentation with published seg weights and automatic detect-to-segment transfer in CLI train (#537)
- NAFNet SIDD denoise weight variant (LibreNAFNetl-restore-sidd, l size only) (#549)
- Quantization subsystem: libreyolo quantize CLI and model.quantize()/quant_info()/dequantize()/save(); recipes fp16/bf16/fp8/int8/w4a16/w4a8/nvfp4/mxfp4/int2 (research); QAT/QAD via train() on quantized checkpoints; supported families yolo9 and rfdetr; in-tree Triton kernels with a pluggable registry and LIBREYOLO_QUANT_KERNELS override (#619, #623)
- BoT-SORT tracker (model.track(tracker="botsort"), BoTSortTracker/BoTSortConfig exported top-level) (#621)
- Deep OC-SORT ReID tracker with an OSNet-AIN embedder auto-downloaded from LibreYOLO/LibreReID-osnet; custom embedder callables supported (#580)
- YOLOv7 training (SimOTA loss); the family was inference-only in v1.3.1 (#538)
- LoRA fine-tuning extended to D-FINE, DEIM, DEIMv2, RT-DETR v1/v2/v4, EC and ConvNeXt; adapters merged on export (#622)
- DINOv2 foundation-teacher distillation (distill_model="dinov2", feat_mse loss, distill_normalize knob; yolo9 backbones) (#534)
- Test-time augmentation for semantic (PIDNet, SegFormer, EoMT, DINOv2) and panoptic (EoMT) segmentation (#601, #608)
- Multi-class keypoint training for YOLO-NAS pose (#530)
- Augmentations: classification auto_augment/erasing/mixup/cutmix, copy-paste for segmentation, perspective and flipud, rot90 for OBB, vflip+rot90 for restore, HSV jitter for semantic (#532)
- Declarative augmentation spec (libreyolo/data/augment/spec.py): a per-family used/mosaic-gated/ignored matrix for every TrainConfig augmentation knob, pinned to the real pipelines by tests; the CLI now warns for every family when an explicitly-set training parameter is ignored (previously RF-DETR only), and training warns when mixup_prob is set with mosaic_prob=0 in the mosaic-gated pipelines (#635)
- Spawn-path multi-GPU training for ResNet, ConvNeXt, EfficientNetV2, MobileNetV4 and NAFNet (#567)
- Canonical export-support matrix with validated/available/blocked tiers, docs page and ADR 0011 (#578, #587)
- TFLite inference backend (LibreYOLO("model.tflite") via ai-edge-litert, Python >= 3.12) (#587)
- "litert" export alias for tflite and libreyolo[litert] extra (#563)
- Semantic, depth and point export unblocked (PIDNet, FOMO, ZipDepth, Depth Anything V2 under a fixed-resolution batch-1 depth contract) (#562, #578, #587)
- CLI: enriched libreyolo models, libreyolo formats --family/--task, libreyolo info export_support, libreyolo predict --json ocr array (#578, #587, #575)
- UI support for gaze, panoptic and open-vocabulary models; non-downloadable models greyed out (#579)
- New optional extras: sensenova, siglip2, siglip2-convert, litert; timm added to sam and openvocab extras (#618, #546, #563, #602, #600)

### Changed

- D-FINE and RT-DETRv4 now evaluate and predict at sizes other than the native 640; v1.3.1 crashed at any non-native imgsz. Known residual: rectangular sizes with the same token count as the native size still reuse a wrong-aspect embedding. DEIM, DEIMv2, EC, RT-DETR and RT-DETRv2 get the same dynamic eval-size support via per-shape regenerated embeddings/anchors (#541, #630)
- PicoDet fine-tune defaults: lr0 0.1 -> 0.01, warmup_lr_start 0.01 -> 0.001; the old default destroyed COCO-pretrained weights (coco128 fine-tune 0.40 -> 0.14 before vs 0.40 -> 0.49 after) (#568)
- DEIM fine-tune defaults: lr0 4e-4 -> 1e-4, min_lr_ratio 0.5 -> 0.05; RT-DETRv4 inherits the min_lr_ratio change; pass the old values to reproduce the upstream COCO recipe (#622)
- DEIMv2-n flat_epochs 7800 -> 78 (iteration count misplaced as epochs; LR schedule shape changes) (#622)
- AdamW no longer applies weight decay to BatchNorm/bias parameter groups in the base trainer (#568)
- Semantic segmentation training applies HSV jitter by default (trained mIoU moves for PIDNet, DINOv2-semantic, RF-DETR-semantic) (#532)
- Restore training adds coupled vertical flip and rot90 (NAFNet training results move) (#532)
- SyncBatchNorm defaults on under multi-GPU DDP for YOLO9, YOLOX, YOLOv7, YOLO-NAS, PicoDet, RTMDet and FOMO (#531, #538, #567)
- DDP now shards correctly for DEIM, D-FINE and YOLO-NAS-pose (previously every rank trained the full dataset at the full batch), and loss normalizers are globally all-reduced to match single-GPU gradients (#605, #567)
- New DDP hard errors: non-divisible global batch, batch < 1 after AutoBatch, and non-sharding custom loaders all raise at setup (#605)
- model.train(profile=True) keeps training after the profiled window; profile_then_stop=True restores the old stop behavior (#590)
- Semantic and panoptic val/predict accept augment=True (previously raised) (#601, #608)
- YOLO-NAS multi-class pose checkpoints load with their real class count and return real class ids (previously forced to single-class person) (#530)
- Export gated by the support matrix: blocked combinations raise up front;
  callable combinations proceed without blanket warnings, with validation
  context recorded in the generated support documentation (#578, #587)
- RF-DETR imgsz validated early in predict/val/export with suggested valid sizes (#551)
- EC training config augmentation defaults zeroed to match the trainer's actual pass-through path (executed training unchanged) (#551)
- libreyolo models --json schema changed (task-suffixed cli_names, new keys); libreyolo formats/info JSON gained keys (#578)
- Checkpoints using the new task strings or finalized quant state are not loadable by v1.3.1 (#619, #575, #557)

### Removed

- libreyolo/models/omdet_turbo/ native graph (unreachable dead code at v1.3.1); replaced by the transformers-based LibreOMDetTurbo (#600)
- broadcast_ema_buffers internal helper (unused) (#567)
- No public API deprecations were added or removed; pre-existing deprecated aliases still warn

### Fixed

- libreyolo[openvocab] now installs ftfy and regex, required by OV-DEIM's CLIP prompt tokenizer at predict time; a clean openvocab-only install previously failed on the first prediction (#636)
- Results and LibreEoMT keep full v1.3 positional-argument compatibility: new v1.4 parameters (panoptic/matte/ocr/restore_scale; num_queries) moved after the complete v1.3 signatures, with compatibility tests (#636)
- Export never mutates the live model before the request is accepted: LoRA adapters are folded (and finalized int8 models re-prepared) only after format lookup, option preflight, and parameter resolution; quantized format='pt' export folds adapters on the checkpoint copy, leaving the live model trainable (#636)
- RTMDet fine-tune collapse from missing head init (~196,000x loss shock on re-heading; nc=1 rebuild 0.26 -> 0.709 mAP50-95) (#568)
- PicoDet/RTMDet AMP training crashes/NaN on CUDA (fp32 loss under AMP, SimOTA BCE outside autocast) (#568, #551)
- Pose validation under DDP: per-rank file clobbering and collective deadlock (#605)
- YOLO9 multi-GPU convergence degradation from per-rank BatchNorm stats (#531)
- DDP loss under-scaling on sparse batches for PicoDet/RTMDet (#567)
- Segmentation training RAM exhaustion on COCO-scale datasets with multiple workers (uint8 variable-length masks) (#529)
- profile=True silently truncating runs and corrupting resume state; stale stop flag on trainer reuse (#590)
- Atomic weight downloads (.part staging, Content-Length verification) (#587)
- Gaze face detection on OpenCV 5 (YuNet detector added) (#587)
- Depth models crashing on video input (#562)
- libreyolo models advertising names the factory refused to load (#600)
- OMDet-Turbo ignoring iou= (#600)
- RF-DETR loud warning on class-count head reinit; rfdetr accepted by the distillation config (#628)
- Security: NAFNet arch peek switched to torch.load(weights_only=True) (#550, #559)
- YOLOv1 decode guards survive python -O; batched predict loops per image (#550)
- D-FINE seg parity with ONNX/TensorRT; seg training auto-downloads published weights; tiled inference rejects segment models loudly (#537)
- YOLOv7 training color-space and fp16 overflow bugs (fine-tune mAP50 0.0 -> 0.92) (#538)
- YOLOX SimOTA crash when no anchor matches any ground truth (#538)
- SegFormer bit-exact reference parity, re-head init, hsv_prob actually applied (#589)
- Panoptic Quality crowd-region under-counting and bounds guards (#560)
- EoMT panoptic checkpoints validate instead of crashing val() (#553)
- Quantization: calibration device mismatch; NVFP4/MXFP4 scale buffers protected from fp16 cast (#619, #623)
- Deep OC-SORT respects the requested device (#587)
- Concurrent SwinIR tiling race (ContextVar) (#587)
- OV-DEIM device-switch crash on cached text features (#607)
- DINOv2 distillation teacher border cropping at non-14-multiple sizes (#534)
- OCR validator optimal one-to-one assignment (#575)
- Depth Anything 3 per-image sky quantile (#581)
- RTMDet-Ins box clamping (#581)
- Video writer sized from output frames; matte overlays on video (#549)
- SenseNova 16 GiB loading and task-state leaks (#618)
- SAM 3 text-prompt threshold semantics (#576)
- Classification square_resize + augment now raises; SigLIP2 fp32 softmax (#546)
- NCNN export on Windows tolerates the PNNX auxiliary-loader failure (#587)
- Export/inventory metadata corrections (AST-based classification, duplicate-entry errors, SwinIR weights published) (#587, #582, #581)

### Release stats

316 commits, 66 merged pull requests, 560 files changed, +97,139 / -3,130 lines, 58 new test modules. Contributors: datarocks0 (SegFormer fixes via #589), imagra93 (#553, #589, #601, #608), Xuban Ceccon (maintainer, 301 of 316 commits).
