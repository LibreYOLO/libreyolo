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

No LibreYOLO dataset-file contract is implemented for `classify`.

## gaze

No LibreYOLO training or validation dataset-file contract is implemented for
`gaze`.
