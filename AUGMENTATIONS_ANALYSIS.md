# Augmentation Landscape Analysis (dev @ e0e6356)

> **STATUS UPDATE (2026-06-10): the refactor described below has been executed**
> on branch `353-unified-augmentations` in this worktree. All augmentation
> logic now lives in `libreyolo/data/augment/` ({color, boxes, geometry,
> mosaic, segments, yolox, yolo9, yolonas, detr, rfdetr, pose}.py); model
> transforms files are re-export shims or parameter-only recipe subclasses.
> Safety net: `tests/unit/test_augment_parity.py` — 28 golden fixtures
> generated from the pre-refactor code, all byte-identical after every
> migration step; plus three adversarial review passes (RNG/behavior parity
> with 440+ randomized differential comparisons against the originals,
> import/back-compat surface, design). Known deliberate behavior changes:
> (1) YOLO-NAS/EC pose images with more instances than max_labels used to
> crash a DataLoader worker, now truncate correctly; (2) degenerate
> aspect-ratio letterbox no longer crashes cv2.resize. §1–§8 below describe
> the pre-refactor state and remain the rationale record.

Deep analysis of all train-time augmentation code in LibreYOLO, as groundwork for a
unified `libreyolo/augment` module. Companion to the upcoming review of the junior's
refactor PR.

---

## 1. Inventory: where augmentation code lives today

~3,500 lines of transform/augmentation code spread across **10 files in 7 different
directories**, implementing **7 distinct pipelines**:

| # | Pipeline | File(s) | Lines | Used by |
|---|----------|---------|-------|---------|
| 1 | YOLOX-style (shared) | `training/augment.py` | 472 | yolox, rtmdet, damoyolo, picodet |
| 2 | YOLO9 fork of #1 | `models/yolo9/transforms.py` | 578 | yolo9, yolo9_e2e, **rtdetr, rtdetrv2** |
| 3 | YOLONAS affine variant | `models/yolonas/transforms.py` | 228 | yolonas |
| 4 | torchvision-v2 DETR | `models/deim/transforms.py`, `models/dfine/transforms.py` | 308+301 | deim, deimv2, dfine, rtdetrv4, ec-det |
| 5 | RF-DETR custom | `models/rfdetr/seg_transforms.py` | 508 | rfdetr det/seg/obb, ec-seg |
| 6 | Pose (×4 copies) | `models/{yolo9,yolonas,rfdetr,ec}/pose_transforms.py` | 200+329+252+302 | each model's pose task |
| 7 | Classification | `data/classify_dataset.py:172-196` | ~25 | classify task (plain torchvision) |

The cross-model imports are the tell: **rtdetr imports yolo9's transforms, deimv2
imports deim's, rtdetrv4 inherits deim's trainer, ec imports dfine's AND rfdetr's
transforms.** A shared augmentation library already exists de facto — it just lives in
five wrong places under five wrong names.

## 2. Duplication map (quantified)

### Detection
- `augment_hsv()` — **byte-identical** in `training/augment.py:48` and
  `models/yolo9/transforms.py:16`. Also imported from augment.py by yolonas and 3 of 4
  pose files (so the function is shared *sometimes*, copied *other times*).
- `preproc()` (letterbox) — identical except yolo9 adds BGR→RGB + `/255`
  (`augment.py:165` vs `yolo9/transforms.py:30`).
- `mirror()` — identical (`augment.py:156` vs `yolo9/transforms.py:53`).
- 2×2 mosaic placement geometry — written twice (`augment.py:258-276` as
  `get_mosaic_coordinate`, re-inlined in `yolo9/transforms.py:462-478`).
- `MosaicMixupDataset` vs `YOLO9MosaicMixupDataset` — same skeleton (mosaic center
  sampling, 4-tile loop, clip, close_mosaic), divergent details (affine step, mixup
  style beta-blend vs paste, segments/OBB support).
- **DEIM vs D-FINE `transforms.py`: ~98% identical copy-paste.** `TrainTransform`,
  `PassThroughDataset`, `MultiScaleCollate`, `_generate_scales()` all duplicated with
  only the class-name prefix changed (DEIM adds a `sanitize_min_size` param and one
  attribute). This is the most flagrant pair.
- Polygon/segment helpers (`_flip_segments_lr`, `_transform_segments`,
  `_rasterize_segments`, `_filter_segments`) — duplicated between
  `yolo9/transforms.py:62-143` and `rfdetr/seg_transforms.py:95-260`.

### Pose (worst offender per-capita)
Across the four `pose_transforms.py` files:
- `_build_target()` — **100% identical ×4** (area filter, visible-kpt filter, padded slab).
- Horizontal flip + `flip_idx` left/right swap — **identical logic ×4**.
- Out-of-bounds keypoint visibility kill — **identical ×4**.
- `_random_affine()` with keypoint warping — **~95% identical ×3** (yolo9/yolonas/ec;
  differs only in border value constant and interpolation param plumbing).
- `_brightness_contrast()` — **identical ×2** (yolonas, ec).
- `_finalize_image()` — **identical ×2** (yolonas, ec), inlined variants in the other two.

### Rough total
Of ~3,500 transform lines, on the order of **1,200–1,500 lines are copies or
near-copies** of code that exists elsewhere in the repo.

## 3. What actually differs between models (the real variability axes)

Stripping away the duplication, pipelines differ only along five axes — all
parameterizable, none requiring per-model op code:

1. **Op recipe + hyperparams**: which ops run, with what probs (mosaic=1.0 for yolox vs
   0 for DETRs; DETRs add PhotometricDistort/ZoomOut/IoUCrop instead).
2. **Output label format**: `[cls,cx,cy,w,h]` pixel (yolox, deim/dfine, rfdetr) vs
   `[cls,x1,y1,x2,y2]` normalized (yolo9, yolonas, rtdetr); padding class 0 vs −1.
3. **Image finalization**: BGR raw 0-255 (yolox) vs RGB /255 (yolo9) vs RGB+ImageNet
   norm (ec, rfdetr, deimv2-dino).
4. **Extra target types**: segments (yolo9-seg, rfdetr-seg), keypoints (pose ×4),
   OBB angles (yolo9, rfdetr).
5. **Schedule**: `no_aug_epochs` (YOLO-style close_mosaic) vs `aug_stop_epoch_ratio`
   (DETR-style strong-aug stop) vs multi-scale collate toggling — same concept,
   two mechanisms.

There is also one genuine **backend split**: pipelines 1–3, 5, 6 are numpy/cv2;
pipeline 4 is `torchvision.transforms.v2` with `tv_tensors`. This is the only
structurally hard part of unification (see §6).

## 4. Config plumbing is broken/dishonest today

- `TrainConfig` (`training/config.py:86-94`) defines mosaic/mixup/hsv/flip/degrees/
  translate/shear/scales globally; **the CLI exposes all of them for every model**
  (`cli/commands/train.py:224-236`) with YOLOX defaults in the help text.
- But DETR families **silently ignore** mosaic/mixup/degrees/translate/shear — their
  PassThrough datasets never read them. `libreyolo train --model dfine --mosaic 1.0`
  does nothing, with no warning.
- DETR strong ops (PhotometricDistort p, ZoomOut fill, IoUCrop p) are **hardcoded** in
  `create_transforms()` — not user-tunable at all.
- Honor matrix: only `flip_prob` and `no_aug_epochs` are honored by (nearly) every
  model. `hsv_prob` honored by YOLO-grid family only. Geometric params honored only
  where mosaic runs.
- This is exactly why "users can play with augmentations creatively" (goal #4) is
  impossible today: there is no contract between config and behavior.

## 5. Coverage gaps (models with missing augs — goal #2 evidence)

| Model/task | Today | Likely missing (upstream recipe / obvious) |
|---|---|---|
| picodet | flip only | MinIoURandomCrop, PhotoMetricDistortion — **documented as deferred** in `picodet/trainer.py:8-9` |
| rtdetr / rtdetrv2 | flip + HSV (borrowed yolo9 transform) | upstream RT-DETR uses PhotometricDistort/ZoomOut/IoUCrop — the exact pipeline sitting in `deim/transforms.py` next door |
| ec-seg | flip only | color jitter, scale jitter |
| rfdetr-pose | flip + optional crop | HSV/color, affine (its det sibling at least has crop+multi-scale) |
| yolo9 | mosaic+HSV+flip | mixup defaults off; no affine outside mosaic; no multi-scale |
| pose (all 4) | flip/HSV/affine | no mosaic anywhere (shared MosaicMixupDataset is box-only) |
| OBB (yolo9, rfdetr) | flip only | mosaic **force-disabled** in `trainer.py:528` pending corner-aware augs |
| classify | RandomResizedCrop+flip | RandAugment / mixup / cutmix / erasing |

Pattern confirmed: models lack augmentations not by design decision but because each
would have to re-implement them locally. Central library + per-model recipe fixes the
incentive structure.

## 6. Assessment of "full refactor is impossible"

**The claim is wrong for ops, half-right for one infrastructure seam.**

Wrong because:
- Every op in the repo is pure pixel+geometry math over (image, boxes, segments?,
  keypoints?, angle?). Nothing reads model internals. The existing cross-model imports
  (rtdetr→yolo9, ec→dfine+rfdetr, deimv2→deim, rtdetrv4→deim) *prove* the ops are
  model-agnostic — models already share them, just via spaghetti imports.
- Output format / normalization / padding differences are **adapter concerns** — a
  ~30-line declarative finalize step per family, not a reason to keep 300-line
  transform files per model.
- The pose quadruplication is the cleanest counterexample: 4 files, ~1,080 lines,
  where the *only* true deltas are (border value, interpolation, letterbox-vs-stretch,
  ImageNet-norm flag) — all constructor params.

Half-right because:
- The numpy/cv2 vs torchvision-v2 backend split is real. The tv2 ops
  (RandomPhotometricDistort, RandomZoomOut, RandomIoUCrop, SanitizeBoundingBoxes) come
  from torchvision; reimplementing them in cv2 risks fidelity drift, and converting
  cv2 pipelines to tv2 changes RNG streams and uint8/float behavior everywhere.
  **Resolution**: a canonical `Sample` dataclass (image HWC uint8 BGR, boxes xyxy
  pixel, optional segments/keypoints/angles) + thin wrappers that bridge a Sample
  through tv2 ops. The library is unified at the *interface* level; two backends
  coexist behind one op protocol. No augmentation needs to stay inside a model dir.
- Behavioral parity is the real risk, not feasibility: identical-looking refactors
  shift RNG call order and silently change training. Mitigation: golden tests — seeded
  RNG, fixed inputs, assert byte-equal outputs old-vs-new per pipeline, written
  *before* moving code.

## 7. Proposed target architecture

```
libreyolo/augment/
  __init__.py        # public API: build_pipeline(recipe, overrides), op registry
  sample.py          # canonical Sample (image, boxes, cls, segments?, kpts?, angles?)
  ops/
    color.py         # hsv_jitter, brightness_contrast, photometric_distort(tv2 bridge)
    geometry.py      # hflip, vflip, random_affine, letterbox, resize, zoom_out, iou_crop
    mosaic.py        # mosaic4, mixup (beta + paste variants) — target-type aware
    multiscale.py    # multi-scale collate (single impl)
  targets/
    boxes.py         # box transforms under each geometry op
    segments.py      # polygon transform/flip/rasterize (single copy)
    keypoints.py     # flip_idx swap, affine warp, OOB-vis kill, build_target
    obb.py           # angle normalization under flip/affine
  finalize.py        # layout adapters: (bgr|rgb, /255|imagenet|raw, cxcywh|xyxy,
                     #  pixel|normalized, pad value/class) — declarative per family
  schedule.py        # close_mosaic / stop-strong-augs / multi-scale-off, one mechanism
  recipes.py         # YOLOX_RECIPE, YOLO9_RECIPE, DETR_RECIPE, POSE_RECIPE, ... 
```

Model trainers shrink to: `create_transforms()` returns
`build_pipeline(recipe=DETR_RECIPE, finalize=Finalize(norm="imagenet", fmt="cxcywh_px"), **config_overrides)`.

Config layer: replace the flat always-exposed knobs with a per-recipe schema so the CLI
only surfaces (and honors) knobs the chosen model actually uses — fixes §4 and enables
goal #4 (users compose/override pipelines: `augmentations=[...]` in `model.train`).

### Migration order (each step shippable, golden-tested)
1. Golden parity tests for all 7 pipelines (seeded, byte-equal).
2. Extract pose helpers (×4 → 1) — biggest win, lowest risk, zero format questions.
3. Merge DEIM/D-FINE transforms (98% twins) into one module; point rtdetrv4/deimv2/ec at it.
4. Unify segment helpers (yolo9 vs rfdetr copies).
5. Merge yolo9/transforms.py back into the shared YOLOX core via finalize adapters
   (the fork only exists for RGB+/255+normalized-xyxy output).
6. Introduce `Sample` + op protocol; rehome rfdetr custom pipeline.
7. Config schema rework + public `augmentations=` API.
8. Then the gap-filling (picodet crop/photometric, rtdetr→DETR recipe, pose mosaic,
   OBB-aware mosaic) becomes one-line recipe edits instead of per-model projects.

## 8. Review of PR #357 ("refactor augment primitives", branch 353-refacto-augmentations, CLOSED)

+1292/−1364, 17 files. What it did:

1. **Created `libreyolo/data/augment/`** with four modules:
   - `yolo.py` — the low-level primitives moved verbatim from `training/augment.py`
     (hsv, mirror, preproc, affine, mosaic-coordinate, box conversions).
   - `yolox.py` — `TrainTransform`/`ValTransform`/`MosaicMixupDataset` moved verbatim.
   - `detr.py` — **real deduplication**: the DEIM/D-FINE 98% twins merged into
     `DETRTrainTransform`/`DETRPassThroughDataset`/`DETRMultiScaleCollate`;
     `deim/transforms.py` and `dfine/transforms.py` collapse to ~35-line subclass shims.
   - `segments.py` — yolo9 + rfdetr polygon helpers merged into one set, made
     dense-mask (`DenseMaskRing`) aware.
2. `training/augment.py` becomes a pure re-export shim (back-compat preserved).
3. `yolo9/transforms.py` drops its copies of hsv/mirror/segment helpers/mosaic
   geometry, imports shared ones (~100 lines deleted).
4. yolox/rtmdet/damoyolo/picodet/yolonas trainers: import-path-only changes.
5. Tests: re-export identity assertions + small geometry unit tests
   (`tests/unit/test_shared_augment.py`).

### What it did NOT do (vs goals 1–4)

- **Pose quadruplication untouched** — the single worst duplication (~1,080 lines,
  `_build_target` ×4, `_random_affine` ×3, flip+flip_idx ×4) got only an import-path
  tweak in yolonas. The "full refactor is impossible" claim was never actually tested
  against the easiest, highest-yield target.
- **The two mosaic dataset wrappers remain forked**: `YOLO9MosaicMixupDataset`
  (~480 lines) still duplicates `MosaicMixupDataset`'s skeleton; only the 16-line
  coordinate helper was shared. The fork exists solely for output format
  (RGB /255 normalized-xyxy) — exactly what a finalize adapter would absorb.
- **The stance is codified in the docstring** (`data/augment/yolo.py`): "Model families
  should compose them in their own models/<family>/transforms.py recipe instead of
  moving recipe logic into this shared package." Per-model *recipes* are the right
  layering — but what stayed per-model isn't recipe declarations, it's duplicated
  300–500-line pipeline classes. Wrong boundary.
- **No config work**: dead CLI knobs, silent ignoring, hardcoded DETR strong-op params
  all unchanged. Goal #4 (user-composable augmentations) not advanced.
- **No parity/golden tests** — only identity re-exports and toy geometry checks.
- **rfdetr/seg pipeline, classification, multi-scale unification**: untouched.

### Bugs it introduced (Codex flagged both, P2)

Merging the polygon-only (yolo9) and dense-mask-aware (rfdetr) segment helpers into
one dense-mask-aware set broke the yolo9 paths: `transform_segments` shifts/scales only
polygon coords, but `rasterize_segments` now *prefers* the attached `dense_mask`, which
is stale (still in original-image coordinates) after mosaic tile placement and after
letterbox. Segmentation targets silently misalign with the image — a training-corrupting
bug, invisible without visual/golden tests. This is the concrete cost of refactoring
without parity tests.

### Verdict

Good mechanical step 1 (~30% of the value): right package location, right shim
strategy, the DETR twin-merge is exactly correct. But it stopped before every hard or
high-yield part, shipped a correctness regression in the seg path, and the
"impossible" rationale is contradicted by its own DETR merge — the same move applied
to pose and to the yolo9/yolox wrapper fork works fine. Salvage the structure
(`data/augment/` package, shims, `detr.py`), redo on the §7 architecture in the §6
migration order, goldens first.
