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

`train`, `val`, and `test` may be image directories, image-list `.txt` files,
or lists of those values. Label paths follow:

```text
images/.../image.jpg -> labels/.../image.txt
```

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

YOLO9 OBB currently accepts YOLO OBB `.txt` labels only. COCO JSON OBB loading
is not implemented. Mosaic and mixup are disabled for OBB training until
corner-aware OBB augmentation is implemented.

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
