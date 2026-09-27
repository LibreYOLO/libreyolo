---
name: use-libreyolo
description: >-
  Use LibreYOLO as a computer vision library: run inference, train, validate,
  export, and track with object-detection / segmentation (and pose, classify,
  gaze, OBB, semantic, depth, point, restore) models on your own images and
  video. This is the
  guide for *using* the `libreyolo` pip package — not for contributing to or
  developing it. Use whenever someone wants to detect, segment, or track with a
  YOLO9 or RF-DETR model, train on a YOLO-format dataset, measure mAP, run
  inference on an exported model, or export to ONNX / TensorRT / OpenVINO /
  CoreML / NCNN / TFLite. Covers both the `libreyolo` CLI and the
  `from libreyolo import LibreYOLO` Python API.
---

# Use LibreYOLO

LibreYOLO is an MIT-licensed CV library. Its API follows the **YOLO
standard**, which means two things you can rely on:

1. **CLI and Python mirror each other** — same verbs (`predict`, `train`,
   `val`, `export`), same argument names. Use whichever the user prefers.
2. The CLI is **self-describing**. Never guess a flag — ask the binary (see
   *Exact options* below). This is also why this skill stays short: it teaches
   the shape, the tool supplies the details for the installed version.

Flagship models: **YOLO9** (CNN) and **RF-DETR** (transformer). Weights
auto-download on first use — pass a name like `LibreYOLO9t.pt` / `LibreRFDETRn.pt`,
or a path to the user's own `.pt`.

## Setup

```bash
pip install libreyolo
libreyolo checks      # verify install, CUDA/MPS, and optional export backends
```

The base install is lightweight. Some features need **optional extras** —
install them as `libreyolo[extra]` (or `libreyolo[all]`). Available extras:
`onnx`, `rfdetr`, `eomt`, `tensorrt`, `openvino`, `ncnn`, `tflite` (alias
`litert`; LiteRT is TensorFlow Lite's new name), `coreml`,
`tracking`, `gaze`, `rtdetr`, `vlm`, `ground`, `sam`, `openvocab`, `clip`, `label`,
`plots`, `lora`, `tensorboard`, `mlflow`, `wandb`, `all`. `libreyolo checks`
reports which are present.

## The four verbs

Arguments take either YOLO-style `key=value` **or** `--key value`. Examples use
`key=value`. Tip: `save=true` writes annotated outputs under `runs/` — the
fastest way to eyeball results while experimenting.

**Predict — run inference**
```bash
libreyolo predict model=LibreYOLO9t.pt source=path/to/img_or_dir conf=0.25 save=true
```
```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE
model = LibreYOLO("LibreYOLO9t.pt")
results = model(SAMPLE_IMAGE, save=True)   # equivalently: model.predict(source=...)
```
NumPy image arrays are read as BGR, the OpenCV order; pass
`color_format="rgb"` for an RGB array such as `np.asarray(pil_image)`.

**Train — needs a YOLO-format dataset YAML**
```bash
libreyolo train model=LibreYOLO9t.pt data=coco8.yaml epochs=100 imgsz=640 batch=16 device=0
```
```python
model.train(data="coco8.yaml", epochs=100, imgsz=640)
```
> RF-DETR's `train()` also accepts its native spellings `batch_size`, `lr`
> and `output_dir`; the standard `batch`, `lr0`, `project` and `name` work
> too. Confirm with `libreyolo train --help-json` for the loaded model.

**Validate — mAP on a split**
```bash
libreyolo val model=runs/train/yolo9_exp/weights/best.pt data=coco8.yaml save_json=true save_plots=true
```

**Export — onnx · torchscript · tensorrt · openvino · ncnn · tflite · coreml**
```bash
libreyolo export model=runs/train/yolo9_exp/weights/best.pt format=onnx half=true
```
Run `libreyolo formats` for each format's extension and FP16/INT8 support.

## Reading results

`predict` returns a single `Results` for a single image, or a `list` of
`Results` for multiple inputs (a directory, a list, a 4-D batched array, or
video frames); `track` yields one `Results` per frame. Index the
list, not a single `Results` — indexing a `Results` selects one detection.
Read them programmatically rather than re-parsing saved files:

```python
r = model("img.jpg")          # one Results (single image); use model([...])  / a dir for a list
len(r)              # number of detections
r.boxes.xyxy        # (N, 4) boxes; also .xywh, .conf, .cls, .id (tracking)
r.masks             # segmentation masks (segment task)
r.keypoints         # pose keypoints
r.probs / r.obb / r.gaze   # classify / oriented-box / gaze tasks
r.names             # class-id → label map
```
For scripting from the CLI, add `--json` to get machine-readable results on
stdout.

## Monitoring a training run

Every `train` run writes live monitoring files into its `save_dir`. To check
on a run, read `status.json` (a few tokens) instead of tailing logs:

```bash
cat runs/train/yolo9_exp/status.json   # state (running/completed/failed), epoch,
                                       # progress, eta_seconds, latest/best metrics,
                                       # and on failure the error message
```

The run's console output is tee'd to `train.log`, and `metrics.jsonl` holds
the full per-epoch history. For a human, `libreyolo monitor [run_or_root]`
serves a read-only browser dashboard (live charts, log, val images) over
those files — it works on live, finished, or crashed runs, and one server
handles every run under the root (`?run=` in the URL selects one).

## Beyond the four verbs

- **Inference on an exported model** — the same constructor loads an exported
  file and runs through the matching backend, so export isn't a dead end:
  ```python
  model = LibreYOLO("best.onnx")     # also .torchscript, .engine, OpenVINO, CoreML
  model("img.jpg", save=True)        # same predict API as a .pt
  ```
- **Object tracking** — `model.track(...)` assigns IDs across video frames.
  Four trackers: **ByteTrack** (`tracker="bytetrack"`, default) and
  **OC-SORT** are motion-only; **BoT-SORT** (`tracker="botsort"`) adds
  camera-motion compensation and an improved width/height motion model;
  **Deep OC-SORT** (`tracker="deepocsort"`) adds appearance ReID (OSNet
  embeddings, auto-downloaded) so IDs survive occlusions and crossings. IDs
  come back on `r.boxes.id`. Needs `libreyolo[tracking]`.
- **Tiled inference for large images** — `predict(..., tiling=True,
  overlap_ratio=0.2)` slices high-resolution images so small objects aren't
  lost, then merges detections.
- **Video & streaming** — point `source` at a video file, or pass
  `stream=True` to get a per-frame generator (`r.frame_idx` per result);
  `vid_stride=N` samples every Nth frame.
- **Ensembling** — `LibreEnsemble` combines multiple detectors;
  `ExternalDetector` folds in a non-LibreYOLO model.

## Supported tasks

`detect` (suffixless default), `segment`, `semantic`, `pose`, `classify`,
`gaze`, `obb`, `point`, `depth`, `restore`, `matte`, `ocr`. Detection — plus
**RF-DETR segmentation** — is the heavily-tested core; other task/family
combinations vary in validation coverage, so check the README compatibility table before
relying on one. Task outputs land on matching `Results` fields
(`r.semantic_mask`, `r.depth_map`, `r.restored`, `r.points`, `r.matte`, …).
Matte adds `r.cutout()` (RGBA) and a transparent-PNG `r.save()`.

OCR reads located text (zh/zh-TW/en/ja/pinyin with one model):

```python
model = LibreYOLO("LibrePPOCRl-ocr.pt")   # t = CPU tier, l = quality tier
r = model("receipt.jpg")
for poly, text, conf in zip(r.ocr.polygons, r.ocr.texts, r.ocr.conf):
    print(text, float(conf))              # regions come in reading order
```

## Models

`libreyolo models` lists every family with its sizes and exact names — treat it
as the source of truth. By tier:

- **Flagship:** YOLO9 (CNN), RF-DETR (transformer) — detection + segmentation
  (RF-DETR also pose + OBB).
- **Other detectors:** YOLOX, YOLO9-E2E, YOLO9-P2 (stride-4 small-object),
  YOLO-NAS, D-FINE, DEIM, DEIMv2, RT-DETR / v2 / v4, PicoDet, RTMDet, EC,
  and the classic lineage: YOLO1/2/3/4 (inference-only; YOLO1 is the original
  2016 VOC model, fixed 448) and YOLO7 (also trainable; SimOTA
  recipe).
- **Specialized:** L2CS (gaze), DepthAnything3 (recommended depth quality
  default), DepthAnythingV2 and ZipDepth (depth alternatives), FOMO (point),
  NAFNet (restore: deblur/denoise; denoise ships as
  `LibreYOLO("LibreNAFNetl-restore-sidd.pt")`), RealESRGAN (restore:
  super-resolution, `x4`/`x2`/`x4t`; `r.restored` is `r.restore_scale` x the
  input; big images via `predict(..., tile=512)`), BiRefNet (matte: background
  removal, sizes t/l, fixed 1024), FeyNobg (matte: background removal, size
  l, fixed 1024, the quality flagship; also ships fp8/nvfp4 quantized
  checkpoints on HF, pass the downloaded .pt as the weights argument),
  PPOCR (ocr: text detection + recognition,
  sizes t/l), EoMT + PIDNet + DINOv2 (semantic).
- **Classifiers** (ImageNet-1k, native timm ports — predict logits are
  bit-identical to timm): MobileNetV4 (s/m/l), ConvNeXt (t/s/b),
  EfficientNetV2 (b0–b3), ResNet (18/34/50/101). Names carry the `-cls`
  suffix, e.g. `model = LibreYOLO("LibreResNet50-cls.pt")`. Fine-tune on an
  ImageFolder root (or a known name/`.zip` URL) with `model.train(data=...)`.
- **Zero-shot / promptable tiers** (need `[openvocab]` / `[sam]` / `[clip]` / `[siglip2]`
  / `[vlm]`): `LibreOpenVocab` (text-vocabulary detection), `LibreGround`
  (screenshot + instruction → click `Results.points`), `LibreSAM` /
  `LibreSAM2` / `LibreSAM3` / `LibreMobileSAM` (point/box-prompted masks;
  SAM 3 also accepts concept `text=` prompts), `LibreCLIP` / `LibreSigLIP2`
  (zero-shot classify), and the `LibreVLM` family (vision-language
  detection). For the exact model aliases in each tier, use `libreyolo
  models` and the dedicated guide `skills/use-libreyolo-zero-shot/`.

## The UI

```bash
libreyolo ui          # drag/drop/paste images in the browser, pick a model, see results
```
A local web app with almost no extra dependencies — the easiest way to try most
models without writing any code. Great for quick experimentation.

## Other commands worth knowing

- `libreyolo label [data=<dataset-or-folder>]` — browser labelling tool
  (boxes/masks/classes) that writes YOLO-format labels; `libreyolo[label]`
  adds SAM click-to-mask assist.
- `libreyolo doctor <dataset.yaml>` — dataset sanity checks (corrupt images,
  label mismatches, leakage, tiny objects) before you burn GPU hours on a
  bad dataset.
- `libreyolo profile run|infer ...` — throughput/latency profiling; see
  `skills/libreyolo-profiling/`.

## Exact, version-correct options

The CLI is the source of truth for the installed version. Prefer these over
recalling flags from memory:

```bash
libreyolo --help                 # list every command
libreyolo train --help-json      # full argument schema for one command, as JSON
libreyolo models                 # list model families, sizes, and names
libreyolo formats                # list export formats and their capabilities
libreyolo info model=...         # resolved family / size / task / device / classes
libreyolo metadata path=...      # raw metadata embedded in a checkpoint
libreyolo predict ... --json     # machine-readable results to stdout
libreyolo ... --quiet            # suppress progress output (good for scripting)
```

In Python the same kwargs apply; `help(LibreYOLO.train)` and `model.info()`
describe the loaded model.

## Notes

- **Datasets** are standard YOLO format, so existing YOLO dataset YAMLs
  (e.g. `coco8.yaml`) work unchanged.
- **Outputs** land under `runs/` (`runs/detect`, `runs/train`, `runs/val`).
- **Stuck or an import/CUDA error?** Run `libreyolo checks` first — it diagnoses
  the environment and export-backend problems before you debug anything else.
- **Deeper guides** (concepts, dataset format, per-task details) live at
  <https://www.libreyolo.com/docs> — but for exact flags and what the installed
  version supports, the binary above is authoritative.
- **Hit a bug, crash, missing weights, or plain friction?** If something broke
  (and you've ruled out user error), or if a task took many turns of trial and
  error that better docs, errors, or defaults would have prevented, offer to
  report it upstream with the `libreyolo-report-issue` skill: it drafts an
  anonymized issue and gives the user a one-click pre-filled GitHub link, so
  it gets improved for everyone.
