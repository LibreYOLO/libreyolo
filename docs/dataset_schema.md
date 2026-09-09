# Dataset Schema

This is the dataset-file contract for canonical tasks in `libreyolo/tasks.py`.

Clean-room rule: use public dataset-format docs and YAML examples only. Do not
use third-party source code, tests, or converters.

## Common YAML

Applies to `detect`, `segment`, `pose`, and `obb`.

- `path`: optional dataset root.
- `train`: required for training.
- `val`: required for validation.
- `test`: optional.
- `names`: required list or integer-keyed class mapping.
- `nc`: optional; must match `names` when present.
- `download`: optional; Python download scripts require explicit opt-in.
- `annotations`: optional mapping of split names to native COCO JSON files for
  detection, instance segmentation, and OBB.

`train`, `val`, and `test` may be image directories, image-list `.txt` files,
or lists of those values. Label paths follow:

```text
images/.../image.jpg -> labels/.../image.txt
```

For native COCO JSON detection/instance-segmentation/OBB datasets, `annotations`
maps a split to the JSON file and the split path gives the image root:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

When `names` is present, native COCO JSON category names must match the YAML
class names; those names define the model label IDs. Without `names`, COCO
category IDs are sorted and mapped densely to `0..N-1`.

Do not require `task` in dataset YAML. Explicit model/task selection wins.

Common label rules:

- one `.txt` label file per image;
- missing or empty label file means no objects;
- `class_id` is an integer in `0..nc-1`;
- coordinates are finite normalized floats in `[0, 1]`;
- coordinates are relative to original image width and height;
- rows contain no confidence or track id.

## detect

Canonical row, exactly 5 fields:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h` is a normalized axis-aligned box. `w` and `h` must be positive.

## segment

Polygon row:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N >= 3`. Coordinate count after `class_id` must be even. The polygon must be
non-degenerate.

A 5-field detection row is also accepted and represents a rectangular segment.

## semantic

Semantic segmentation pairs each image with a dense single-channel mask
(lossless format, typically PNG) instead of a `.txt` label file:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

Mask rules:

- single channel; palette-mode PNGs are read as palette indices;
- each pixel value is a class ID in `0..nc-1`;
- pixel value `255` means ignore and is excluded from loss and metrics;
- mask resolution must equal the paired image resolution.

YAML adds two optional keys on top of the common contract:

- `masks_dir`: mask directory name substituted for `images` in each image
  path (default `masks`).
- `label_mapping`: `{source_id: train_id}` remap applied to mask pixel
  values at load time; unmapped source values become ignore. Train IDs must
  fall in `0..nc-1`.

When `masks_dir` is omitted, masks are rasterized at load time from YOLO
`segment` polygon labels resolved through the standard
`images -> labels` convention, and a `background` class is appended after
the object classes (`nc` grows by one).

Canonical loader: `libreyolo.data.SemanticDataset`.

## panoptic

Panoptic segmentation pairs each image with a dense segment-id map and a
per-image list describing each segment. LibreYOLO adopts the **COCO-panoptic
format** verbatim (`Panoptic Segmentation`, Kirillov et al., CVPR 2019); no
LibreYOLO-specific panoptic format exists, and none is needed.

### Segment-id PNG

One RGB PNG per image, same resolution as the image, where each pixel's colour
encodes the id of the segment it belongs to:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Every pixel belongs to exactly one segment; segments never overlap. Segment id
`0` (RGB black) is **VOID**: unlabeled pixels, excluded from the metric.

### Annotations JSON

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg", ...}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1, "supercategory": "person"}]
}
```

- `annotations[].file_name` names the segment-id PNG inside `panoptic_dir`.
- `segments_info[].id` matches a value in the PNG.
- `iscrowd` marks group regions: they are never false negatives, and a
  prediction mostly covering one is not a false positive.
- **thing-vs-stuff is a per-category property.** `isthing` lives on
  `categories`, never on `segments_info`. The prediction payload
  (`libreyolo.utils.results.PanopticSegmentation`) may denormalize `isthing`
  onto each predicted segment for convenience; the category metadata stays the
  source of truth.

### Class ids

COCO-panoptic `category_id`s are the dataset's raw ids and are typically
non-contiguous (COCO runs 1..200 with gaps). LibreYOLO models predict
contiguous `0..nc-1`. Raw ids are remapped through the YAML `names` **by
category name**, the same rule the native COCO-JSON detect loader follows: when
`names` is present, it defines the label ids. A JSON category absent from
`names` is an error, not a silent drop, because it would otherwise score as a
permanent false negative.

### YAML

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017   # the segment-id PNGs
names: {0: person, 1: bicycle, ..., 132: rug-merged}
```

`annotations` and `panoptic_dir` accept either a single path or a per-split
mapping.

### Validation

Panoptic Quality (`PQ = SQ x RQ`), computed at the ground-truth resolution and
averaged over the categories that appear, then split into `PQ_things` /
`PQ_stuff`. Matching is unique: a predicted and a ground-truth segment of the
same category match iff IoU > 0.5. See
`libreyolo/validation/panoptic_quality.py`.

Canonical loader: `libreyolo.data.PanopticDataset`.

## depth

Depth estimation pairs each image with a dense single-channel depth map instead
of a `.txt` label file:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

Depth rules:

- single channel PNG/TIF or `.npy`;
- map resolution must equal the paired image resolution;
- values are plain depth in a dataset-consistent unit;
- `0`, negative, NaN, and inf mark invalid pixels and are excluded from loss
  and metrics.

YAML adds two optional keys on top of the common contract:

- `depths_dir`: depth directory name substituted for `images` in each image
  path (default `depths`).
- `depth_stem_suffix`: optional suffix appended to the image stem before
  depth extension lookup. When omitted, both same-stem files and the common
  `_depth` suffix are tried.
- `depth_mask_suffix`: optional suffix appended to the resolved depth stem to
  find a validity mask (default `_mask`). If the mask exists, mask values
  `<= 0`, NaN, and inf invalidate the corresponding depth pixels.
- `depth_scale`: divisor for integer-typed depth maps (default `256.0`, the
  common 16-bit PNG convention where stored value / 256 is the depth).

Float `.npy` maps are used as-is and do not apply `depth_scale`.

Canonical loader: `libreyolo.data.DepthDataset`.

## edge

Edge detection pairs each RGB image with a same-stem, single-channel lossless
map and an optional validity mask:

```text
images/val/scene.jpg -> edges/val/scene.png
                      -> masks/val/scene.png    # optional
```

Edge-map rules:

- the map is single-channel PNG/TIF (not an RGB visualization);
- map resolution must equal the paired image resolution;
- integer maps are divided by their dtype maximum; float maps must already be
  finite and in `[0, 1]`;
- `0` means non-edge and `1` means edge;
- optional mask pixels are valid when nonzero;
- resize uses nearest-neighbor interpolation for targets and masks;
- padded pixels are invalid and do not contribute to validation.

YAML keys:

- `edges_dir`: edge-map directory substituted for `images` (default `edges`);
- `edge_stem_suffix`: optional suffix appended to image stems;
- `edge_extension`: lossless target extension (default `.png`);
- `edge_invert`: set `true` when source maps store black edges over white;
- `masks_dir`: optional validity-mask directory (default `masks`).

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Validation thins continuous predictions with four-direction gradient
non-maximum suppression. It reports ODS and OIS F-measures over a configurable
threshold sweep. Predicted and ground-truth pixels are matched one-to-one
within `edge_max_dist * image_diagonal`; the default normalized tolerance is
`0.0075`.

Canonical loader: `libreyolo.data.EdgeDataset`.

The loader is format-only: it does not download or redistribute benchmark
data. BIPED/BIPEDv2 and Berkeley's BSDS data are published for non-commercial
use, with additional terms on their official download pages. Users must obtain
those datasets from their publishers and comply with the applicable terms;
this generic schema does not relicense them:

- BIPED/BIPEDv2: <https://www.kaggle.com/datasets/xavysp/biped>
- BSDS500: <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html>

## normal

Surface-normal estimation pairs each image with a same-stem three-channel
16-bit PNG, plus an optional same-stem validity mask:

```text
images/val/room.jpg -> normals/val/room.png
                     -> masks/val/room.png    # optional
```

Normal-map rules:

- the PNG is exactly three-channel `uint16`, with channels stored as RGB;
- map resolution must equal the paired image resolution;
- decode with `n = png / 65535 * 2 - 1`, then renormalize each vector;
- decoded vectors use LibreYOLO's OpenCV camera frame (`+x` right, `+y` down,
  `+z` into the scene) and face the camera;
- the optional mask is a single-channel PNG where nonzero means valid;
- when no mask exists, every finite, nonzero decoded vector is valid;
- invalid and padded target pixels are represented internally by `(0, 0, 0)`;
- resizing interpolates the three vector components bilinearly and then
  renormalizes; validity masks use nearest-neighbor interpolation;
- a horizontal flip also negates the x component.

YAML adds two optional keys on top of the common split contract:

- `normals_dir`: normal-map directory name substituted for `images`
  (default `normals`).
- `masks_dir`: optional validity-mask directory name substituted for `images`
  (default `masks`). Missing same-stem mask files mean that sample has no
  explicit mask.

```yaml
path: normal-dataset
train: images/train
val: images/val
normals_dir: normals
masks_dir: masks
nc: 1
names: {0: normal}
```

Validation reports mean and median angular error in degrees and the percentage
of valid pixels within 11.25, 22.5, and 30 degrees. Canonical loader:
`libreyolo.data.NormalDataset`.

## restore

Image restoration pairs each degraded input image with a clean RGB target image
instead of a `.txt` label file:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

Restore rules:

- input and target images are RGB-compatible image files;
- input and target resolution must match exactly;
- validation keeps native resolution and pads only enough to stack a batch;
- metrics are computed on the original image canvas;
- training applies coupled crop, horizontal flip, vertical flip, and 90-degree
  rotation to the input/target pair.

YAML adds these optional keys on top of the common split contract:

- `input_dir`: degraded-input directory name used in split paths
  (default `inputs`).
- `target_dir`: clean-target directory name substituted for `input_dir`
  (default `targets`).
- `target_stem_suffix`: optional suffix appended to the input image stem before
  target extension lookup.
- `target_stem_suffixes`: list form of `target_stem_suffix`.
- `degradation`: optional metadata label such as `deblur` or `denoise`.
- `dataset`: optional dataset/provenance label such as `GoPro`.
- `mask_dir`: same-stem binary-mask directory required by LibreLaMa validation.
  It substitutes for the `input_dir` path component while preserving the
  split, subdirectories, and file stem. Each mask must share its source
  image's canvas; zero means preserve and every nonzero pixel means fill.

The class-like YAML fields are schema placeholders: use `nc: 1` and
`names: {0: image}`. Restore models expose `Results.restored`, not detections.

Canonical loader: `libreyolo.data.RestoreDataset`.

## matte

Background removal / dichotomous segmentation pairs each RGB image with a
single-channel ground-truth alpha matte (0 = background, 255 = foreground)
sharing the same stem:

```text
images/subject.jpg -> mattes/subject.png
```

Two layouts are accepted:

- **Directory**: a root containing `images/` and a matte directory, auto-detected
  among `mattes/`, `matte/`, `gt/`, `masks/`, `mask/`, `alpha/`. Pass the root as
  `data=`.
- **YAML**: `path` (root), plus per-split `val_images` / `val_mattes` (and
  optional `train_images` / `train_mattes` for a future fine-tune), each a
  directory relative to `path` or absolute.

Matte rules:

- the matte is grayscale; values are read as alpha in `[0, 1]` (`/255`);
- a matte is resized to the prediction canvas with bilinear interpolation when
  the shapes differ;
- metrics are MAE and S-measure (Fan et al., ICCV 2017), computed on the
  original image canvas; best-checkpoint fitness is S-measure.
- trimap-guided families may accept `trimap_dir=` at validation time, with one
  `0/128/255` guide per image stem. `LibreViTMatte.val(...,
  trimap_dir=path)` uses those guides. When the directory is omitted,
  `trimap_radius=15` is the default: alpha values >= 0.5 form a binary
  foreground, which is eroded and dilated by that fixed radius to derive the
  known-foreground, unknown, and known-background regions.

The class-like YAML fields are schema placeholders: use `nc: 1` and
`names: {0: matte}`. Matte models expose `Results.matte`, not detections.

Validation is inference-only in v1 (matte training/fine-tuning is a documented
follow-up). Canonical pair resolver: `libreyolo.data.matte_dataset.resolve_matte_pairs`.

## ocr

OCR pairs each image with located text regions and their transcripts. Labels
are one JSONL file per split, one JSON object per image:

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

OCR rules:

- `polygon` is a 4-point quad `[[x, y] x 4]` in absolute pixel coordinates,
  ordered top-left, top-right, bottom-right, bottom-left;
- regions with unreadable text use `"text": "###"` (the ICDAR don't-care
  convention): they are excluded from recognition scoring, and predictions
  overlapping them are ignored (not penalized) in detection matching;
- metrics are detection hmean (IoU > 0.5 one-to-one polygon matching),
  end-to-end F1 (IoU > 0.5 AND exact transcript after NFKC normalization and
  whitespace removal; case-sensitive), and 1-NED (normalized edit distance)
  on matched pairs; best-checkpoint fitness is end-to-end F1.

Two layouts are accepted:

- **Directory**: a root containing `images/<split>/` and `labels/<split>.jsonl`.
  Pass the root as `data=`.
- **YAML**: `path` (root), plus optional `images` / `labels` directory names.

The class-like YAML fields are schema placeholders: use `nc: 1` and
`names: {0: text}`. OCR models expose `Results.ocr`, not detections.

Validation is inference-only in v1 (OCR training is out of scope). Canonical
sample resolver: `libreyolo.data.ocr_dataset.resolve_ocr_samples`.

## pose

YAML adds:

- `kpt_shape`: required, `[K, 2]` or `[K, 3]`;
- `flip_idx`: optional integer permutation of `0..K-1`.

Label row:

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Field count is exactly `5 + K * D`, where `D` is the second `kpt_shape` value.
Keypoint `x y` values are normalized. Visibility `v`, when present, is `0`,
`1`, or `2`.

## obb

Row, exactly 9 fields:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

The four points are normalized image coordinates in `[0, 1]` and form a
non-degenerate oriented rectangle. No angle is stored in the label file.

The canonical parser is strict by default and rejects out-of-range
coordinates. Dataset and validation ingestion may clip coordinates to `[0, 1]`
for otherwise valid crop-boundary labels, then still reject degenerate boxes.

Parsing is task-aware: 9 fields mean `obb` only in `obb` mode; in `segment`
mode they may be a 4-point polygon.

Canonical row parser: `libreyolo.data.parse_yolo_obb_label_line`.

Internal OBB geometry: parse normalized corners and convert them to canonical
`xywhr`. The angle is in radians and represents rotation of the width side
around the box center. Model families may adapt that canonical geometry to
their own training tensors, but public results should expose OBB detections as
`xywhr, conf, cls` rows.

YOLO9 OBB currently uses a family-private training adapter that stores targets
as `class, x1, y1, x2, y2, angle`, where `xyxy` is a horizontal proxy box for
assignment and DFL, and `angle` is trained with a separate periodic loss. Do
not treat that proxy tensor as the general OBB contract for other families.

Native COCO JSON OBB loading accepts annotations in this priority order:

- `obb: [x1, y1, x2, y2, x3, y3, x4, y4]` pixel-space corners;
- `obb: [cx, cy, w, h, angle]`, with `angle` in radians;
- COCO `segmentation` polygon/RLE, refit to a minimum-area rectangle;
- COCO `bbox: [x, y, w, h]`, interpreted as an axis-aligned rectangle and
  canonicalized to LibreYOLO `xywhr`.

Mosaic and mixup are disabled for OBB training until corner-aware OBB
augmentation is implemented.

## classify

Classification uses an ImageFolder-style directory tree, not label files:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/` is required for training and defines the class-to-index mapping by
sorted folder name. `val/` is required for validation. `test/` may be present
but is not used by the default train/val commands. Non-training splits must
contain the same class folder names as the expected train/checkpoint class set.
Supported image extensions are defined in
`libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`.

## gaze

No LibreYOLO training or validation dataset-file contract is implemented for
`gaze`.

## point

`point` is currently a model-output task, not a canonical dataset-label schema.
Point model families may adapt existing labels internally, for example by
deriving object centers from YOLO box rows, but a point-only text label format
is not defined in this document yet.

## 3D detection (inference-only integration)

`detect3d` currently has no LibreYOLO dataset loader, trainer, or validator.
Ordinary 2D YOLO labels do not encode 3D calibration or cuboids and must not
be treated as 3D ground truth. `LibreWildDet3D.val()` raises explicitly;
evaluation uses the upstream benchmark configurations and data terms.
Inference takes an RGB image, original-image 3x3 camera calibration, prompts,
and optionally a nonnegative depth array in metres with shape `(H, W)`.
No benchmark images or dataset auto-download routes are bundled.
