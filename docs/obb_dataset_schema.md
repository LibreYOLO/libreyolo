# OBB Dataset Schema

Clean-room rule: use public dataset-format docs and YAML examples only. Do not
use third-party source code, tests, or converters.

YAML: same YOLO dataset shape as detection. `train`, `val`, and `names` are
required; `path`, `test`, `nc`, and `download` are optional. Splits may be
image directories, image-list `.txt` files, or lists of those. Labels follow
`images/.../x.jpg -> labels/.../x.txt`. Do not require `task: obb`; explicit
model/task selection wins.

Label row, exactly 9 fields:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

`class_id` is `0..nc-1`. Coordinates are finite normalized floats in `[0, 1]`,
relative to original image width/height. The four points form a non-degenerate
oriented rectangle. Rows contain no confidence, track id, or angle.
Missing/empty label files mean no objects.

Parsing is task-aware: 9 fields mean OBB only in `obb` mode; in `segment` mode
they may be a 4-point polygon.

Canonical row parser: `libreyolo.data.parse_yolo_obb_label_line`.

Internal geometry: keep corners `(N, 4, 2)` canonical; derive `xyxy` for box
utilities and `xywhr` only at model, metric, or result boundaries. `xywhr`
angles use radians.
