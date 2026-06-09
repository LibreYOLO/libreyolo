# LibreDoctor — dataset health checks

**Status:** Implemented (issue #371)
**Command:** `libreyolo doctor <data.yaml>`
**Package:** `libreyolo/doctor/`

## Motivation

Most "my model trains badly" reports trace back to the dataset, not the model:
empty label files nobody noticed, a class with 4 instances, boxes exported in
pixel coordinates instead of normalized, val images copied from train.
LibreDoctor is an opt-in preflight that scans a dataset and reports problems
*before* the user spends GPU hours. It never mutates the dataset and is never
run implicitly by `train`.

Commercial dataset platforms ship a hosted "health check" for this; LibreDoctor
is the local, free, scriptable equivalent — including a JSON output mode and
exit codes so it can run as a CI gate ("dataset CI").

## Non-goals (v1)

- **Detection only.** v1 strictly checks YOLO-format *detection* datasets
  (`class cx cy w h` labels). Pose/segment/obb/classify come later; the
  snapshot/check architecture is task-extensible so they are additive. A
  format guard (below) keeps non-detect datasets from producing false errors.
- No fixing/rewriting of labels or images (report only).
- No model-assisted checks (embedding similarity, label quality via a trained
  model) — possible v2, requires inference.
- No new dependencies. Everything below is implementable with the existing
  base deps (numpy, Pillow, opencv, scipy, pyyaml, tqdm).

## CLI

```
libreyolo doctor coco8.yaml                 # human report to stdout, progress to stderr
libreyolo doctor coco8.yaml --json          # machine-readable findings to stdout
libreyolo doctor coco8.yaml imgsz=640       # used to convert "tiny object" checks to on-target pixels
libreyolo doctor coco8.yaml --fast          # skip image-content checks (no decode/hash)
libreyolo doctor coco8.yaml --skip images,labels.tiny_object
libreyolo doctor coco8.yaml --only balance  # restrict to one check family
libreyolo doctor coco8.yaml --strict        # warnings also fail the exit code
libreyolo doctor coco8.yaml --download      # allow URL dataset download (never scripts)
```

- Registered like every other subcommand (`KeyValueCommand`, `OutputHandler`).
- Exit codes follow the CLI-wide convention: `0` clean or info-only, `1`
  errors found (`--strict`: warnings too), `2` usage error (unknown check
  selector, missing argument), `3` dataset could not be scanned (missing
  YAML) or is not a detection dataset (format guard).

Python API mirror:

```python
from libreyolo import doctor          # or: from libreyolo.doctor import diagnose
report = doctor.diagnose("data.yaml", imgsz=640)
report.errors, report.warnings, report.infos   # list[Finding]
report.summary()                                # dict for programmatic use
```

## Severity model

| Severity | Meaning | Examples |
|---|---|---|
| ERROR | Training will crash or silently learn garbage | class id >= nc, unreadable image, coords > 1 |
| WARNING | Likely hurts results; user should look | 2-px objects, 95% background, train/val leakage |
| INFO | Statistics worth knowing, no action implied | class histogram, image size distribution |

Every finding carries: `check_id`, `severity`, `message`, `split`,
`paths` (offending files, capped per finding), `count`, and optional
`details` (numbers behind the message). The human renderer groups by severity
and truncates file lists; `--json` emits everything.

## Checks (v1)

### Config & structure (no image decoding)

| id | severity | what |
|---|---|---|
| `config.missing_names` | ERROR | `names` missing or empty |
| `config.nc_names_mismatch` | ERROR | `nc` != len(`names`) |
| `config.missing_split` | ERROR/WARN | `train` missing (E); `val` missing (W) |
| `config.path_not_found` | ERROR | split dirs / list files don't exist, a configured directory contains no images (every entry of a list split is validated individually), or the split resolves to zero images (e.g. an empty .txt list) |
| `config.duplicate_names` | WARNING | two class ids share a name |
| (format guard) | EXIT 3 | `kpt_shape` present, or most label lines are consistently not 5-field detect rows (pose/segment/obb shapes) — exits with "supports detection datasets only" instead of flooding false syntax errors; inconsistent garbage still reports as `labels.syntax` |
| `files.orphan_label` | WARNING | label file with no matching image in any split |
| `files.missing_label` | INFO | image with no label file (counted as background) |
| `files.missing_image` | ERROR | image listed in a `.txt` split but missing on disk |
| `files.unsupported_ext` | WARNING | files in image dirs with non-image extensions |
| `files.case_collision` | WARNING | `a.jpg` and `a.JPG` style stem collisions |

### Label content (parse every label file, detect format)

| id | severity | what |
|---|---|---|
| `labels.syntax` | ERROR | fewer than 5 fields, non-numeric, NaN/inf |
| `labels.polygon_line` | INFO | >5-field rows: accepted exactly as training accepts them (polygon extent becomes the box), but usually a segmentation export |
| `labels.class_out_of_range` | ERROR | class id not in `[0, nc)` |
| `labels.coords_out_of_range` | ERROR | any coord outside `[0, 1]` — classic "pixel coords exported" symptom; box partially out (center+size spills past edge) is WARNING |
| `labels.degenerate_box` | ERROR | width or height <= 0 |
| `labels.tiny_object` | WARNING | box smaller than ~3 px on either side *at the requested imgsz* (default 640) |
| `labels.huge_box` | WARNING | box covers > 95% of image area |
| `labels.extreme_aspect` | WARNING | box aspect ratio > 50:1 in pixels (sliver annotations); without image dims (`--fast`) the normalized ratio is used with a doubled threshold to avoid false positives |
| `labels.duplicate_box` | WARNING | same class, IoU > 0.95 within one image (double annotation) |
| `labels.crowded_image` | INFO | images with anomalously many objects (> p99 × 3) |
| `labels.identical_files` | WARNING | many label files byte-identical to each other (copy-paste exports) |

Line parsing matches `YOLODataset._load_label()` semantics — any line with
>= 5 numeric fields is accepted (5 = box; more = polygon reduced to its
bounding box), so a dataset that trains never gets false syntax errors from
doctor. Doctor is stricter only in *reporting*: polygon rows surface as
`labels.polygon_line` INFO and non-finite values are flagged even though
training would silently consume them. When other tasks land (v2), their
checks reuse the existing parsers (`parse_yolo_pose_label_line`,
`parse_yolo_obb_label_line`, segment ring parsing) and the task is inferred
where unambiguous (`kpt_shape` → pose) with a `task=` override for the
obb/segment ambiguity (a 9-field line is valid in both).

### Class balance

| id | severity | what |
|---|---|---|
| `balance.class_zero_instances` | WARNING | class in `names` with 0 train instances |
| `balance.class_few_instances` | WARNING | class below threshold (default 10 instances or images) |
| `balance.imbalance` | INFO/WARN | max/min instance ratio; WARN past 100:1, silent below 1.5:1 |
| `balance.split_coverage` | WARNING | class present in val but absent from train (or inverse) |
| `balance.background_ratio` | INFO/WARN | % images with no labels; WARN past 50%, silent at zero |
| `balance.split_skew` | INFO | per-class distribution train vs val diverges sharply |

### Image content (skipped by `--fast`; threaded, tqdm progress)

| id | severity | what |
|---|---|---|
| `images.corrupt` | ERROR | PIL cannot open/verify; zero-byte files |
| `images.exif_orientation` | WARNING | EXIF rotation flag set — labels were likely drawn on the rotated view, loaders may differ |
| `images.odd_mode` | WARNING | grayscale/RGBA/CMYK mixed into an RGB dataset |
| `images.tiny_or_extreme` | WARNING | images < 32 px a side, or aspect ratio > 20:1 |
| `images.uniform` | WARNING | all-black/all-white/constant images (failed downloads) |
| `images.exact_duplicates` | WARNING | identical content hash within a split |
| `images.near_duplicates` | INFO | perceptual-hash (dHash, ~15 lines of numpy) collisions within a split |
| `splits.leakage_exact` | ERROR | same content hash in train and val |
| `splits.leakage_near` | WARNING | near-duplicate (dHash distance <= 4) across train/val |

The single decode pass computes everything at once (verify, size, mode, EXIF,
content hash, dHash on a 9×8 thumbnail), so the cost is one read per image.

## Package layout

```
libreyolo/doctor/
├── __init__.py        # public API: diagnose(), Report, Finding, DoctorConfig
├── config.py          # DoctorConfig thresholds + DoctorError hierarchy
├── report.py          # Finding, Severity, Report; console + JSON renderers
├── snapshot.py        # one-pass dataset scan -> DatasetSnapshot; format guard
├── runner.py          # diagnose() orchestration
└── checks/
    ├── __init__.py        # registry, @register, --skip/--only resolution
    ├── dataset_config.py  # config.* checks (named to avoid clashing with config.py)
    ├── files.py           # files.* checks
    ├── labels.py          # labels.* checks, matches libreyolo/data parsing
    ├── balance.py         # balance.* checks
    └── images.py          # images.* + splits.* (decode pass, duplicates, leakage)
```

Design rule: the dataset is scanned **once** into a `DatasetSnapshot`; every
check is a pure function `check(snapshot, config) -> list[Finding]` registered
by id. This keeps checks independently testable with synthetic snapshots and
makes `--skip`/`--only` trivial.

Failure honesty: a check that crashes is reported as an ERROR finding under
its own id ("results are incomplete") so a broken run can never exit 0, and a
`--skip`/`--only`/`--fast` combination that selects zero checks is a usage
error (exit 2) instead of a false-green "No problems found".

## Where utilities live (the general pattern)

LibreDoctor is the second "product utility" after `libreyolo/ui`. The
convention it establishes, for future tools (serve/API, MCP, labeling, ...):

1. One subpackage per utility under `libreyolo/<name>/`, one CLI subcommand
   `libreyolo <name>`. No separate repos, no plugins.
2. The base install must not get heavier: utilities lazy-import everything
   beyond base deps, and anything heavy becomes a `pyproject` extra
   (`pip install libreyolo[serve]`). Doctor needs no extra.
3. Utilities consume the library through its public surface
   (`libreyolo/data`, `libreyolo/tasks.py`) and never reach into model
   internals; core never imports from utilities. Import direction is one-way.
4. Each utility documents itself in `docs/<name>_design.md` and tests live in
   `tests/unit/doctor/` etc.

## Testing

- `tests/unit/doctor/` with `@pytest.mark.unit`: synthetic datasets in
  `tmp_path` (tiny PIL images + handwritten label files), one test module per
  check family. Every check gets a positive (fires) and negative (clean) case.
- One e2e smoke: `libreyolo doctor coco8.yaml --json` parses and exits 0-or-1
  on the bundled coco8.

## Future (v2+, explicitly out of scope now)

- Other tasks: pose (keypoint/visibility/`flip_idx` checks), segment
  (degenerate polygons), obb, classify (folder-layout checks). Task inferred
  from the dataset where unambiguous.
- `--fix` mode for mechanical repairs (drop malformed lines, clamp coords) —
  writes a corrected copy, never in place.
- Embedding-based near-duplicate / leakage detection (needs inference).
- Blur/brightness outlier detection (cv2 Laplacian variance — cheap, but cut
  from v1 to keep the surface reviewable).
- `train --doctor` preflight flag once the standalone tool has settled.
