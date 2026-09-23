---
name: libreyolo-port-model
description: >-
  Port a model into LibreYOLO. Executable guide — finds your closest existing
  family to clone, gives paste-ready templates for the model class, conversion
  script, runtime auto-convert recognizer, and trainer subclass, then walks
  the port as a sequence of self-contained commits. Covers detection, pose,
  and segmentation in depth; routes classify / semantic / depth / restore /
  point / gaze ports to their merged exemplar families. Use this for any new
  family (RTMO, MobileDet, PP-YOLOE, a new HF model).
---

# Port a model into LibreYOLO

This skill is **executable**. It assumes you've been asked to port a model and
your job is to ship it. Reference material (architectural patterns, ABC
contracts, validators) is in the back half; the front half is the path you
follow to write code.

## 0. Read this first — orientation

You have an upstream model. Six questions get you pointed at the right
scaffold:

1. **Tier**: checkpoint-driven (a `BaseModel` factory family — everything below), or prompt-driven (promptable segmentation, open-vocab detection, VLM, GUI grounding)? Prompt-driven models join a **sibling factory** (`LibreSAM` / `LibreOpenVocab` / `LibreVLM` / `LibreGround`), not the `BaseModel` registry — see §4.1.
2. **Architecture**: per-anchor head with NMS (YOLO-grid), set-prediction with Hungarian matching (DETR), or one-to-one head with top-K and no NMS (NMS-free YOLO-grid)?
3. **Tasks shipped**: detect / pose / segment (this skill's templates), or classify / semantic / depth / restore / point / gaze (clone the merged exemplar family — §10 lists them)?
4. **Backbone source**: standard PyTorch, vendored separately-licensed (e.g. DINOv3), or loaded from an optional dependency (e.g. transformers — RF-DETR's DINOv2 backbone)?
5. **License**: code (upstream `LICENSE` file) and weights (HF model card YAML). Both must be permissive (MIT / Apache-2.0 / BSD) for the family to live in core.
6. **Implementation and evidence target**: inference-only, training with recorded validation evidence, or training with e2e parity?

The answers route you through §1 (license gate) → §3 (pick scaffold) → §4
(per-family ledger entry to clone) → §5 (commit sequence) → §6 (paste-ready
templates). Reference sections fill in the contract details.

**Two concrete starts**:

- **RTMO (one-stage pose)** → YOLO-grid pose, Apache-2.0 (MMPose-based). **Closest scaffolds: YOLOX** for the CSPDarknet backbone, **RTMDet** (`models/rtmdet/`) as the worked example of the mm-series route — mm-series upstream and key remap in `models/rtmdet/convert.py`. (RTMDet itself used to be the hypothetical example here; it has since shipped following exactly this skill.)
- **A random model on HF** → Run §1 (license check) first. If permissive, find architecture in §4 ledger; if no row matches, fall back to §3 decision tree.

## 1. License check — both code and weights

> **Two absolute rules. No exceptions, no "just to check", no exploratory read.**
>
> 1. **Never read AGPL code — ever.** Do not open, clone, `curl`, `cat`, or
>    otherwise view the source of an AGPL-licensed project, not even to
>    "understand the architecture" or "check a shape". Once you determine a
>    repo is AGPL (or dual-licensed with AGPL as the open option), stop and
>    port from a permissive source, the paper, or a clean-room spec instead.
>    Reading it contaminates the port. This holds even when the AGPL repo is
>    the most convenient reference.
> 2. **Never point at another repo's assets — especially an AGPL repo's — for
>    datasets or weights.** Do not reference, hotlink, download-at-runtime, or
>    document a URL into a third-party (and above all an AGPL) repository for
>    dataset files, label files, or checkpoints. Ship dataset config that
>    resolves to a permissively-licensed source we control or that the user
>    supplies; if the only copy lives in an AGPL/non-redistributable repo,
>    treat it as unavailable and surface it as a licensing decision.
>
> If either rule would block the port, that is the correct outcome — raise it,
> don't route around it.

LibreYOLO core is **MIT** and stays MIT-compatible. Check both before writing
any code.

```bash
curl -sL https://raw.githubusercontent.com/<org>/<repo>/<branch>/LICENSE | head -3
```

| Upstream license | Code in core? | Rehost weights under `LibreYOLO/`? | Action |
|---|---|---|---|
| MIT / Apache-2.0 / BSD | ✅ | ✅ | normal integration; ship NOTICE per `libreyolo-upload-hf-model` skill |
| GPL-3.0 / AGPL-3.0 | ⚠️ plugin only | ❌ | `libreyolo-<family>` separate package |
| Custom / non-redistributable | case-by-case | ❌ usually | link to upstream CDN like YOLO-NAS / Deci |

For weights, check the HF model card YAML at the top:

```yaml
license: apache-2.0    # ← permissive
license: gpl-3.0       # ← copyleft
license: cc-by-nc-4.0  # ← non-commercial: still hostable — redistributable is
                       #   the only bar for weights. Ship the license verbatim
                       #   and lead the card with a non-commercial banner
                       #   (SegFormer precedent); users are responsible for
                       #   complying with the weight license.
```

**Already-shipped reference cases**:

- D-FINE / DEIM / RF-DETR / EC — Apache-2.0 → clean.
- DEIMv2 — Apache-2.0 family code, **but** s/m/l/x sizes vendor DINOv3 (Meta's custom non-OSI license). Not "Apache-2.0 clean" wholesale.
- YOLOv9 — MIT, via `MultimediaTechLab/YOLO` (Kin-Yiu Wong & Hao-Tang Tsui). **Not** `WongKinYiu/yolov9` (GPL-3.0). When upstream has multiple forks, pick the permissive one.
- YOLO-NAS — Apache-2.0 code + Deci CDN weights (non-redistributable) → weights linked, not rehosted.
- YOLO-World — GPL-3.0 → flagged plugin-only in #108, never merged.
- L2CS-Net — portable code but redistribution-restricted weights → inference-only family, weights **not** rehosted, and the family is skiplisted in runtime auto-conversion (`_SKIP_FAMILIES` in `models/autoconvert.py`).

If a license problem surfaces mid-port, never "fix" it by rewriting or
renaming the offending code to obscure where it came from. Surface it as a
licensing decision — plugin, link-out, or drop.

### Vendored sub-components with separate licenses

If your family vendors a separately-licensed architectural sub-component
(DEIMv2 vendors **DINOv3** for s/m/l/x backbones; Meta's custom non-OSI
license), the family's own license isn't sufficient. Required:

- `LICENSE.md` next to the vendored code (e.g. `libreyolo/models/deimv2/engine/backbone/dinov3/LICENSE.md`).
- A NOTICE entry citing the sub-component, its upstream, and its license.
- A docstring annotation on the family's model class explaining which
  size variants depend on the differently-licensed sub-component.
- A clear rule for the user: do the constraints of the sub-component
  flow through to the produced weights? Document the answer.

Don't quietly bundle a non-OSI sub-component without these three artifacts.

## 2. Pick the implementation and evidence target

Pick one and document scope explicitly:

1. **Inference-only.** No trainer is wired, or `train()` rejects the task. The user can `model.predict(...)` but not `model.train(...)`. YOLO-NAS pose, L2CS, and CLIP are examples.
2. **Training with limited evidence.** Trainer exists and `train()` accepts the task. Record the exact completed checks and the missing convergence evidence; do not turn a user acknowledgement into the capability contract.
3. **Training with RF1 evidence.** `test_rf1_training` passes; row in `MODEL_CATALOG`; recipe gaps documented in the family docstring. YOLOX, YOLOv9, YOLO-NAS detect, D-FINE, DEIM, DEIMv2, RT-DETR, RF-DETR sit here, plus the classify families (MobileNetV4 / ConvNeXt / EfficientNetV2 / ResNet) on the shared classify path.

**Inference-only is a legitimate ship state.** Don't gate the port on a working trainer.

Which target to pick belongs to the PRD: `libreyolo-write-model-prd` section 4
checks the task/data shape and the available fine-tuning recipe. If the PRD is
silent on implementation scope or evidence, apply the same checks and record
the answer in the PR description.

## 3. Pick your scaffold

Use this decision tree to pick the family you'll clone as your starting point.

### 3.1 By architecture

| Upstream looks like | Closest scaffold | What to keep | What to swap |
|---|---|---|---|
| YOLO-grid + SimOTA / TaskAlignedAssigner | **YOLOX** (`models/yolox/`, 6 files, ~288 LoC `model.py`) | mosaic+mixup pipeline, label assignment idiom, BGR preprocess if YOLOX-style | head architecture, backbone, optionally loss |
| YOLO-grid + GFL/DFL + ESNet/light backbone | **PicoDet** (`models/picodet/`, 6 files, ~272 LoC `model.py`) | shared GFL head pattern, RGB+ImageNet norm, training implementation and evidence notes | backbone parser, neck, possibly DFL reg_max |
| YOLO-grid + ELAN/RepNCSPELAN, complex conversion | **YOLOv9** (`models/yolo9/`, 7 files) | aux-head-skip pattern, heavy structural converter | head modules |
| NMS-free YOLO-grid (one-to-one head, top-K) | **YOLOv9-E2E** (`models/yolo9_e2e/`, ~271 LoC `model.py`) | NMS-free postprocess pattern, parent-child sibling pattern | inherit from your detect parent rather than `BaseModel` |
| DETR — light, mostly metadata-wrap conversion | **D-FINE** (`models/dfine/`, 16 files) | encoder/decoder/MS-deform-attn modules, FlatCosineScheduler, deploy() wrapper | matcher, loss weights |
| DETR — sibling of an existing port (different loss/matcher) | **DEIM** (`models/deim/`) | inherit from D-FINE, override only what differs | loss, matcher |
| DETR — vendors a separately-licensed backbone | **DEIMv2** (`models/deimv2/`) | safetensors handling, DINOv3 vendoring + LICENSE.md | backbone-specific code |
| DETR — multi-task (detect + pose + segment) | **EC** (`models/ec/`, ~431 LoC `model.py`) | task-dispatch in `_init_model`/`_postprocess`, `is_pose_state_dict`/`is_seg_state_dict` discriminators | architecture |
| DETR — multi-task native port, backbone from an optional dep | **RF-DETR** (`models/rfdetr/`, 18 files) | DINOv2-via-transformers backbone + explicit state-dict transfer (`backbone.py`), lazy registration, family-local config | task heads, matcher |
| YOLO-grid pretrained on ImageNet w/ heavy backbone | **YOLO-NAS** (`models/yolonas/`) | in-process EMA unwrap (no separate converter), CDN-not-HF weight URL | head + backbone |
| Classify-only backbone (CNN / hybrid) | **MobileNetV4 / ConvNeXt / EfficientNetV2 / ResNet** (`models/<family>/`) | shared `BaseTrainer` classify path, `{"probs": ...}` postprocess, timm-parity test pattern | the backbone itself |
| Semantic segmentation | **PIDNet** (CNN) / **EoMT** (ViT) | `SemanticValidator` wiring, semantic mask contract | architecture |
| Dense prediction (depth / restore) | **Depth Anything V2** (`models/depth_anything/`) / **NAFNet** (`models/nafnet/`) | native-resolution inference path, dedicated validator | architecture |
| Zero-shot / text-conditioned | **CLIP** (`models/clip/`) for classify; the `models/openvocab/` tier for detection (§4.1) | text-tower handling, `set_classes` API | towers |

### 3.2 By task scope

| You're shipping | Multi-task pattern to follow |
|---|---|
| detect only | any single-task family above; declare `SUPPORTED_TASKS = ("detect",)` |
| detect + pose | YOLO-NAS (asymmetric sizes — pose has `n`, detect doesn't) |
| detect + segment | RF-DETR (native multi-task port; multi-task training via `RFDETR_SEG_TRAINERS`) |
| detect + pose + segment | EC (3-way dispatch in `_init_model`, three converters via `--task` flag) |
| classify only | MobileNetV4 / ConvNeXt / EfficientNetV2 / ResNet — shared `BaseTrainer` classify path |
| semantic / depth / restore / point / gaze only | PIDNet + EoMT / Depth Anything V2 / NAFNet / FOMO / L2CS — single-task `BaseModel` families, each with a dedicated validator |
| open-vocab detect, promptable seg, VLM, GUI grounding | not `BaseModel` families — sibling factories `LibreOpenVocab` / `LibreSAM` / `LibreVLM` / `LibreGround` (§4.1) |

### 3.3 By non-PyTorch upstream

- **Paddle / TensorFlow / safetensors**: PicoDet ports a community PyTorch reimplementation (Bo's). For Paddle direct, you'd write a heavier conversion script that handles framework-specific buffer cleanup. For safetensors, see DEIMv2's `weights/convert_deimv2_weights.py:19-43` — dispatch on `Path(input).suffix == ".safetensors"`, build a fresh native model, `safetensors.torch.load_model(model, path, strict=True)`.

## 4. Per-family ledger

Dense reference. Find the family closest to your port and clone its directory
as your starting scaffold. Each row tells you what's already solved.

| Family | Pattern | Sizes | Tasks | Training state (per task) | Files | Weight conversion | Notable |
|---|---|---|---|---|---|---|---|
| **YOLOX** | YOLO-grid (NMS) | n/t/s/m/l/x | detect | trainable; RF1 covered | 6 | none (in-process unwrap) | BGR 0–255 inference; mosaic+mixup; closest to upstream of any family |
| **YOLOv9** | YOLO-grid (NMS) | t/s/m/c | detect | trainable; RF1 covered | 7 | heavy structural | RGB 0–1; aux head dropped; `xyxy` normalized targets in loss; from-scratch recipe gap (3 param groups at same LR, no backbone-LR split) |
| **YOLOv9-E2E** | NMS-free YOLO-grid | t/s/m/c | detect | trainable; RF1 covered | 5 | reuses YOLOv9 converter (different `model_family` at wrap) | Inherits from `LibreYOLO9`. Postprocess does top-K only — `del iou_thres`. In the `_is_nms_free_family()` allowlist (was a miss for a while — since fixed) |
| **YOLO-NAS** | YOLO-grid (NMS) | n*/s/m/l (n* pose-only) | detect, pose | detect trainable; pose has no trainer | 7 | none (in-process EMA unwrap) | Weights from Deci CDN (license, not LibreYOLO HF). Only family with asymmetric-per-task `INPUT_SIZES` |
| **PicoDet** | YOLO-grid (NMS) | s/m/l | detect | trainable; RF1 gap documented | 6 | light structural (mmcv key remap + EMA drop) | GFL+DFL loss; ESNet backbone; per-size `INPUT_SIZES` (320/416/640) |
| **D-FINE** | DETR | n/s/m/l/x | detect | trainable; RF1 covered | 16 | metadata-wrap (~50 LoC) | Per-group LR via `lr_mult` in `_setup_optimizer` + `_train_epoch` override; `FlatCosineScheduler` (added by this family); `min_lr_ratio=0.05`; backbone-LR multiplier 0.5× |
| **DEIM** | DETR (D-FINE sibling) | n/s/m/l/x | detect | trainable; RF1 covered | small | metadata-wrap | Architecturally identical to D-FINE; `min_lr_ratio=0.5`; tie-break in factory (`models/__init__.py`, search "Ambiguous D-FINE/DEIM") |
| **DEIMv2** | DETR | atto/femto/pico/n/s/m/l/x | detect | trainable; RF1 covered | small | metadata-wrap + safetensors handling | s/m/l/x **vendor DINOv3** (separate license). Per-size `min_lr_ratio` overrides (`n` = 1.0, others 0.5). Documents `warmup_iters` epoch-override scaling |
| **EC** | DETR | s/m/l/x | detect, pose, segment | trainable; evidence recorded per task | many | metadata-wrap multi-task (`--task` flag) | 3-way `_init_model` dispatch; `is_pose_state_dict` / `is_seg_state_dict` discriminators; pose forces `nc=1, names={0:"person"}`; mask head DETR-style (transformer cross-attention, not YOLO proto) |
| **RT-DETR** | DETR | r18/r34/r50/r50m/r101/l/x | detect | trainable; RF1 covered | medium | light structural | Multi-char size codes (`r50m` vs `r50`) — overrides `detect_size_from_filename` with length-descending sort. **Per-group LR via `lr_ratio` + `_scale_lr` override** — better template than D-FINE for new DETR ports. Pretrained-backbone-download fix prototyped in `bf16a2b` but not in current code |
| **RF-DETR** | DETR (native port) | n/s/m/l | detect, segment, pose, obb | trainers implemented per task; evidence recorded separately | 18 | bespoke autoconvert recognizer (needs the full checkpoint for size detection + COCO class remap) | **No longer a PyPI-package wrapper — full native port.** DINOv2 backbone loaded via transformers `AutoBackbone` with an explicit state-dict transfer (`models/rfdetr/backbone.py`) because `from_pretrained` silently no-ops on the windowed subclass (landmine #37). Lazy-registered behind the `transformers` dep (`_ensure_rfdetr`). Family-local config (`models/rfdetr/config.py`). Multi-task training via compile-time `RFDETR_TRAINERS` vs `RFDETR_SEG_TRAINERS` selection |

**File-count signal**: 6–7 files = single-task YOLO-grid scaffold.
16+ = a DETR family with non-trivial loss + matcher + transforms.

### Families added since the table above was first written

Compact rows — clone these directly for the newer archetypes:

| Family | Pattern | Tasks | Training | Notable |
|---|---|---|---|---|
| **RTMDet** (`models/rtmdet/`) | YOLO-grid (NMS) | detect | trainable; evidence gaps documented | CSPNeXt; mm-series upstream; runtime auto-convert remap in `models/rtmdet/convert.py` |
| **RT-DETRv2** | DETR (child of RT-DETR) | detect | inherits RT-DETR | registered **after** v1 so metadata-less checkpoints default to v1 |
| **RT-DETRv4** | DETR (child of D-FINE) | detect | trainable | must register **before** D-FINE (more-specific `can_load`); own `models/rtdetrv4/convert.py` |
| **FOMO** | boxless point head | point | trainable | `point` task; `PointValidator`; per-size `imgsz` from family `CONFIGS` |
| **L2CS** | CNN gaze | gaze | inference-only | redistribution-restricted weights → autoconvert skiplist, no HF rehost |
| **Depth Anything V2** | ViT dense | depth | inference-only | `DepthValidator`; check per-size weight licenses (upstream b/l are CC-BY-NC) |
| **NAFNet** | encoder-decoder restore | restore | wired (see docstring) | native-resolution path (no letterbox); `RestoreValidator` |
| **EoMT** | ViT semantic | semantic | see docstring | `SemanticValidator`; unique query/mask keys make `can_load` trivial |
| **PIDNet** | CNN semantic | semantic | see docstring | 1024-px default input; fusion-key `can_load` |
| **MobileNetV4 / ConvNeXt / EfficientNetV2 / ResNet** | classify backbone | classify | trainable | shared `BaseTrainer` classify path; parity vs timm is bit-identical (`max_abs_diff == 0`) |
| **CLIP** | dual-tower zero-shot | classify | inference-only | pure-torch towers (no open_clip at runtime); `set_classes` API; `clip_validator.py` |
| **DINOv2** | ViT | semantic, classify | via `RFDETRConfig` | lazy-registered together with RF-DETR (transformers dep) |
| **YOLO9-P2** (`models/yolo9_p2/`) | YOLO-grid (child of YOLO9) | detect | trainable | stride-4 P2 head for small objects; sizes t/s; the `WEIGHT_VARIANTS = ("visdrone",)` precedent for dataset-variant weight suffixes |
| **Darknet lineage: YOLO2/3/4** (`models/darknet/` + thin `models/yolo{2,3,4}/`) | anchor-grid CNN | detect | inference-only | one shared `DarknetFamily` (cfg parser + blocks + anchor decode) serves all three; public-domain upstream; converter `weights/convert_darknet_weights.py`, parity via `weights/parity_darknet.py` |
| **Darknet lineage: YOLO1** (`models/darknet/` + thin `models/yolo1/`) | dense FC-head CNN | detect | inference-only | shares the `DarknetFamily` engine but the FC head (`[connected]`/`[local]`/`[detection]`) does NOT fit the anchor decode: v1-specific `decode_detection` (7x7x30, VOC-20, fixed 448, square-stretch preprocess). OpenCV can't oracle it, so faithfulness = byte-exact reader + dog/bicycle/car golden. `b` weights on HF; tiny `t` weights lost upstream (code-ready, BYO `.weights`) |
| **YOLO7** (`models/yolo7/`) | anchor-grid CNN | detect | training evidence recorded in the RF1 skip map + infer | MIT upstream (same repo as the YOLO9 source); own `v7.yaml` + net; converter `weights/convert_yolo7_weights.py`, parity via `weights/parity_yolo7.py`; training via SimOTA loss (`loss.py`) adapted from Apache-2.0 YOLOX |
| **TinyFormer** (`models/tinyformer/`) | DETR (DEIMv2 derivative) | detect | trainable (DEIMv2Trainer subclass) | Apache-2.0 upstream (mmpmmpmmpjosh/TinyFormer); reuses the vendored DEIMv2 engine + towers; family-local SSA backbone + PBM 4-scale encoder; routing vs DEIMv2 enforced by can_load in both directions (backbone.sda. markers), NOT registry order — importing it pulls in models.deimv2 first; `xl` size code needs multi-char filename precedence; `-visdrone`/`-obj2coco` WEIGHT_VARIANTS |
| **BiRefNet** (`models/birefnet/`) | Swin v1 + bilateral-reference decoder | matte | inference-only (v1) | MIT upstream; `matte` task (ADR 0010); `MatteValidator` (MAE + S-measure); **family-local Swin v1** (original lineage, NOT the timm `models/swin/` tower, see NOTICE); ASPP deformable conv exports to ONNX `DeformConv` (opset 19) via a registered symbolic; converter `weights/convert_birefnet_weights.py`, parity via `weights/parity_birefnet.py` (max_abs_diff == 0) |
| **Faster R-CNN** (`models/faster_rcnn/`) | two-stage RPN + RoIAlign + class-wise NMS | detect | inference-only | native port from torchvision v0.26.0 (BSD-3-Clause), sizes n/s/m/l; official state keys load strictly and all four variants have exact eager parity. COCO-91 sparse ids map to contiguous COCO-80. ONNX is batch-1/fixed-square and emits final already-NMSed boxes/scores/labels. Weight mirrors carry BSD-3-Clause on a disclosed implied basis plus torchvision's pretrained-model caveat; `weights/upload_faster_rcnn_hf.py` enforces the five-file contract |
| **FCN** (`models/fcn/`) | dilated ResNet + primary/auxiliary FCN heads | semantic | inference-only | native port from torchvision v0.26.0 (BSD-3-Clause), sizes r50/r101 at 520; this is not the original VGG FCN-8s graph. Both heads have exact eager parity. Semantic predict/val and ONNX/TorchScript/OpenVINO/TensorRT are validated. Weight mirrors carry BSD-3-Clause on a disclosed implied basis plus torchvision's pretrained-model caveat; `weights/upload_fcn_hf.py` enforces the five-file contract |

### 4.1 Sibling factories (not `BaseModel` families)

Prompt-driven models do not join the checkpoint-driven `BaseModel` registry.
They live in sibling factories with their own contracts:

- **`LibreOpenVocab`** (`models/openvocab/`) — text-conditioned open-vocabulary
  detectors returning standard detection `Results`: Grounding DINO
  (`models/grounding_dino/`), OWLv2 (`models/owlv2/`), and OMDet-Turbo, which
  runs through `transformers` with no vendored model source.
  Shared towers live in `models/bert/` (text) and `models/swin/` (vision).
- **`LibreSAM`** — promptable segmentation: SAM-1 and SAM-2 (`models/sam/`),
  MobileSAM (`models/mobilesam/`).
- **`LibreVLM`** (`models/vlm/`) — vision-language models.
- **`LibreGround`** (`models/ground/`) — instruction → click: screenshot +
  referring expression returns `Results.points` on the original canvas.
  Clone an existing snapshot adapter (`florence.py`, `showui.py`, or
  `qwen3vl.py`). Hosted weights are VLM-style snapshot
  mirrors under `LibreYOLO/`, not detect-family `.pt` files.

If your port is prompt-driven, clone one of these factories instead of a
`BaseModel` family. The license gate (§1), parity discipline (§12), and HF-upload rules
(commit 10) still apply in full; the `BaseModel` ABC contract (§8) does not.

## 5. Walking the port — commit sequence

Each numbered commit is a self-contained PR-able unit. Don't combine. Don't
write the trainer before the inference parity test passes (see §12).

### Commit 1 — Skeleton + factory recognition

Create `libreyolo/models/<family>/{__init__.py, model.py, nn.py, utils.py}`
using template §6.1. Implement:

- `Libre<FAMILY>` class with `FAMILY`, `FILENAME_PREFIX`, `INPUT_SIZES`, `SUPPORTED_TASKS`, `DEFAULT_TASK`, `TRAIN_CONFIG = None` (we'll wire later).
- `can_load(state_dict)` — pick a key **unique to your architecture**. Never `"backbone"` or `"weight"`.
- `detect_size(state_dict)` — infer size from a shape signature.
- `detect_nb_classes(state_dict)` — read nc from the head.
- `_init_model`, `_get_available_layers`, `_forward`, `_postprocess` (stub OK), `_preprocess` (use shared letterbox).

Add `from .<family>.model import Libre<FAMILY>` to `libreyolo/models/__init__.py` in the **right registry order** (most distinctive markers first). Add `Libre<FAMILY>` to `libreyolo/__init__.py` exports.

**Enroll the family in the model registry**: add one `"<family>": "<group>"`
line to `MODEL_GROUPS` in `libreyolo/models/registry.py`. Group semantics are
in `docs/nomenclature.md` ("Model groups"). Take the coverage group from the
PRD; it mirrors the implemented surface and never decides capability.
`tests/unit/test_model_registry.py` fails until the family is enrolled.

**Verify**: `python -c "from libreyolo import Libre<FAMILY>; m = Libre<FAMILY>(size='s'); print(m.task, m.family)"` runs.

### Commit 2 — `nn.py` + forward smoke

Port the model architecture into `libreyolo/models/<family>/nn.py`. Mirror
upstream attribute names where possible — it makes conversion a metadata-wrap
(see Commit 3) and keeps the parity diff readable.

**Verify**: model builds at all sizes, `model(torch.zeros(1, 3, 640, 640))` returns the expected tensor shape (or dict for DETR).

### Commit 3 — Conversion script + atomic write

Create `weights/convert_<family>_weights.py` using template §6.4 (single-task)
or §6.5 (multi-task). Use `wrap_libreyolo_checkpoint(...)` with `task`,
`supported_tasks`, `default_task` populated even for single-task ports —
free disambiguation later.

Write atomically: `tmp = output.with_suffix(".tmp"); save_checkpoint(wrapped, tmp); tmp.rename(output)`. Print a missing/unexpected-key diff after loading the wrapped dict into a fresh model.

**Verify**: `python weights/convert_<family>_weights.py upstream/x.pth weights/Libre<FAMILY>s.pt --size s` runs and produces a file. `LibreYOLO("weights/Libre<FAMILY>s.pt")` loads without errors.

### Commit 3b — Runtime auto-conversion recognizer

`libreyolo/models/autoconvert.py` makes `LibreYOLO("<upstream_file>.pth")`
work directly: the factory unwraps the common upstream layouts (`ema.module` /
`ema` / `net` / `model` / `state_dict` / plain), asks every registered family
via `BaseModel.convert_upstream_state_dict` whether it recognizes the tensors,
wraps the winner in v1.0 metadata (size / task / nc read from the tensors), and
writes `<source>-<Prefix><size>[-task].pt` beside the source.

- The `BaseModel` default (`models/base/model.py::convert_upstream_state_dict`)
  claims any layout your `can_load` accepts — if upstream keys already match
  your native port, auto-conversion is free and this commit is just tests.
- If upstream naming differs, add `models/<family>/convert.py` with
  `is_upstream_state_dict()` + `convert_upstream()` (shared by the runtime
  auto-converter **and** your offline `weights/convert_<family>_weights.py`),
  and override the hook on the model class (template §6.10). Precedents:
  `models/{yolo9,rtmdet,rtdetr,rtdetrv4,picodet}/convert.py`.
- Special cases: non-redistributable weights are skiplisted
  (`_SKIP_FAMILIES` — L2CS); RF-DETR registers a bespoke recognizer in
  `autoconvert.py` because it needs the full checkpoint (size detection +
  COCO class remap), not just the tensor dict.
- Arbitration when several families claim one file: subclass beats base, then
  registry order (= import order in `libreyolo/models/__init__.py`). The
  filename is consulted only for the D-FINE/DEIM tensor tie.

**Verify**: `LibreYOLO("<downloaded upstream>.pth")` loads your family end to
end, AND your recognizer returns `None` for every sibling family's upstream
checkpoints — a greedy recognizer steals other families' files (landmine #36).

### Commit 4 — Inference parity proof

This is the gate. Before any postprocess work, prove the model produces
identical outputs to upstream on identical inputs.

Use template §6.8 (cross-load script). Save it as
`tests/unit/test_<family>_parity.py` or as a one-off under
`weights/`. The test:

1. Imports the upstream model class and yours side by side.
2. Loads upstream weights into both.
3. Feeds the same `torch.zeros(...)` (or a fixed seed) through both.
4. Asserts `max_abs_diff == 0` in `eval()` mode on layers present in both.

**Verify**: parity script passes for every size of your family. Don't proceed until it does.

### Commit 5 — `_postprocess` returning the canonical dict

Implement `_postprocess` in `models/<family>/utils.py`. Return:

```python
{"boxes": (N, 4), "scores": (N,), "classes": (N,)}                # detect
# + "masks": (N, H, W) for segment
# + "keypoints": (N, K, 3) for pose (xy + visibility)
```

If your model emits keypoints as `(N, K, 2)`, append a column of ones for
visibility — `Keypoints.has_visible` requires column 3
(`libreyolo/postprocess/ec.py::postprocess_pose` for the precedent).

**Verify**: `tests/unit/test_<family>_postprocess.py` smoke-tests shape contracts on synthetic input.

### Commit 6 — End-to-end inference

`LibreYOLO("Libre<FAMILY>s.pt").predict("test.jpg")` returns a `Results`
object with the right slots populated, and `Results._select(idx)` slices
boxes ↔ masks ↔ keypoints in lockstep. Drawing dispatches on slot presence
(`if result.keypoints is not None: draw_keypoints(...)`), no task field
needed.

Add `<Family>ValPreprocessor` to `libreyolo/validation/preprocessors.py` (or
inherit existing). Set `uses_letterbox`, `custom_normalization`,
`wants_unresized_image` properties to match your training transform.

**Verify**: `model.predict("test.jpg")` runs end-to-end. `model.val(data="coco128.yaml")` runs.

**Verify in `libreyolo ui`** (required): the UI dropdown is built from
`get_all_cli_names()`, so a registered family appears automatically, but the
result card only works if `save=True` writes an annotated image: the server
calls `model(img, conf=..., save=True)` and shows the saved file plus a
task-aware summary from `_summarize_result` in `libreyolo/ui/server.py`.
Launch `libreyolo ui`, drop a test image, run your smallest size, and confirm:
(1) inference does not require extra kwargs the UI cannot supply (prompts,
auxiliary detectors); (2) an annotated image renders, not the unchanged
source; (3) the summary is task-appropriate, not a fallback "0 objects".
If the port introduces a new `Results` slot, extend `_summarize_result` (and
`Results.plot` if the slot has no drawing path) in the same PR.

### Commit 7 — ONNX export

For YOLO-grid: 1 output `"output"`, opset 13. For DETR: 2 outputs
`["pred_logits", "pred_boxes"]`, opset ≥ 16 (deformable attention needs
`grid_sample`).

DETR families: add yourself to `BaseBackend._is_nms_free_family()` at
`libreyolo/backends/base.py:211`. NMS-free YOLO-grid (yolo9_e2e-style)
**must** also be added or exported backends will wrongly apply NMS. NCNN
doesn't work for DETR — block early with `NotImplementedError`.

**Verify**: `model.export(format="onnx")` produces a working graph.
`OnnxBackend("Libre<FAMILY>s.onnx").predict(...)` matches PyTorch outputs.

### Commit 8 — Trainer (skip if shipping inference-only)

Decide implementation scope and evidence (§2). Inference-only: `TRAIN_CONFIG = None`, override
`train()` to raise `NotImplementedError`, and **stop here**. Otherwise:

Create `models/<family>/trainer.py` using template §6.6 (YOLO-grid) or §6.7
(DETR with per-group LR via `lr_ratio`). Append `<Family>Config(TrainConfig)`
to `libreyolo/training/config.py` (or use a family-local config — RF-DETR /
RT-DETR / YOLOv9-E2E do this).

For limited-evidence training, expose the same `train()` interface and document
the completed checks and known limits separately from the callable API.

For non-detect tasks: **override `best_metric_key`** explicitly. Default is
`"metrics/mAP50-95"` (bbox). Set `"metrics/mAP50-95(M)"` for segment,
`"metrics/keypoints_mAP50-95"` for pose.

**Verify**: `model.train(data="coco128.yaml", epochs=3)` runs and produces a
new `.pt`.

### Commit 9 — e2e catalog row

Append your family's sizes to `MODEL_CATALOG` in `tests/e2e/conftest.py:417`. Run:

```bash
pytest tests/e2e/test_val_coco128.py -k <family>
pytest tests/e2e/test_rf1_training.py -k <family>
```

DETR families: skip the `last_loss < first_loss` assertion in
`test_rf1_training` (DETR loss is too noisy on small datasets — RF-DETR and
D-FINE both exempt themselves).

**Verify**: both e2e tests pass for every size.

### Commit 10 — HuggingFace upload

**Only for redistributable weights** — re-check §1. Permissive-licensed
weights are rehosted under `LibreYOLO/`; anything else follows the YOLO-NAS
link-out or L2CS no-rehost path and this commit becomes documentation of
where the user gets the weights instead.

Follow the `libreyolo-upload-hf-model` skill. Cross-check your filename
against the whitelist there before uploading. The 5-file contract:
`.gitattributes`, `README.md`, `LICENSE`, `NOTICE`, `Libre<FAMILY><size>[-<task>].pt`.

Multi-task families: one HF repo per task variant
(`LibreYOLO/Libre<FAMILY>s` for detect, `LibreYOLO/Libre<FAMILY>s-pose` for
pose, `LibreYOLO/Libre<FAMILY>s-seg` for segment).

**Verify**: on a fresh machine / cleared cache (no `weights/Libre<FAMILY>s.pt`
staged), `LibreYOLO("Libre<FAMILY>s.pt")` auto-downloads from the new HF repo
and loads. There is no `from_pretrained` API; the bare canonical filename *is*
the download trigger (`BaseModel.get_download_url()` builds the
`huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt` URL).

## 6. Paste-ready templates

Copy these, fill in the `# TODO` markers. Mirror upstream attribute names
in `nn.py` whenever possible — it makes conversion a metadata-wrap.

### 6.1 Family directory layout

```
libreyolo/models/<family>/
├── __init__.py        # exports Libre<FAMILY>
├── model.py           # BaseModel subclass (template 6.2 or 6.3)
├── nn.py              # the actual nn.Module — port-specific, mirror upstream names
├── utils.py           # postprocess, preprocess_numpy
├── loss.py            # only if you ship a trainer
└── trainer.py         # BaseTrainer subclass (template 6.6 or 6.7) — optional
```

`__init__.py`:

```python
"""Libre<FAMILY> family: <one-line architecture summary>."""
from .model import Libre<FAMILY>

__all__ = ["Libre<FAMILY>"]
```

### 6.2 `model.py` — single-task detect-only

```python
"""Libre<FAMILY>: BaseModel subclass wiring <FAMILY> into the LibreYOLO factory."""

from __future__ import annotations
from typing import Any, Optional

import torch
import torch.nn as nn

from ...training.config import <FAMILY>Config  # delete if TRAIN_CONFIG=None
from ...validation.preprocessors import <FAMILY>ValPreprocessor
from ..base import BaseModel
from .nn import Libre<FAMILY>Model
from .utils import postprocess as _postprocess
from .utils import preprocess_numpy as _preprocess_numpy


class Libre<FAMILY>(BaseModel):
    """<one-line architecture summary>."""

    FAMILY = "<family>"                              # TODO: short lower-case ID
    FILENAME_PREFIX = "Libre<FAMILY>"                # TODO: PascalCase, no -det suffix
    INPUT_SIZES = {"s": 640, "m": 640, "l": 640}     # TODO: confirm with upstream
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = <FAMILY>Config                    # or None for inference-only
    val_preprocessor_class = <FAMILY>ValPreprocessor

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # TODO: pick a key UNIQUE to your architecture.
        # Never "backbone", "weight", or anything generic.
        # Cross-check against every existing family's state dict in tests.
        return any("<unique_token>" in k for k in weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        # TODO: read a shape signature that disambiguates sizes.
        # Common pattern: a head conv's out_channels.
        key = "<head.cls_pred.weight>"
        if key not in weights_dict:
            return None
        out_ch = int(weights_dict[key].shape[0])
        return {<channels>: "s", ...}.get(out_ch)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # TODO: read nc from a head weight shape, accounting for reg channels.
        key = "<head.cls_pred.weight>"
        if key not in weights_dict:
            return None
        return int(weights_dict[key].shape[0])  # adjust if reg channels are mixed in

    def _init_model(self) -> nn.Module:
        return Libre<FAMILY>Model(size=self.size, nc=self.nb_classes)

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "neck": self.model.neck,
            "head": self.model.head,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return _preprocess_numpy

    def _preprocess(self, image, *, color_format=None, **kwargs):
        # TODO: pick the shared letterbox helper or implement a family-local one.
        # See models/yolox/model.py or models/picodet/model.py for precedents.
        ...

    def _forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def _postprocess(self, raw, conf_thres: float, iou_thres: float, **kwargs):
        return _postprocess(raw, conf_thres, iou_thres, **kwargs)
```

### 6.3 `model.py` — multi-task (detect + pose [+ segment])

Add per-task class vars and dispatch. EC is the 3-task reference (`models/ec/model.py`).

```python
class Libre<FAMILY>(BaseModel):
    FAMILY = "<family>"
    FILENAME_PREFIX = "Libre<FAMILY>"
    SUPPORTED_TASKS = ("detect", "pose", "segment")  # subset as needed
    DEFAULT_TASK = "detect"

    INPUT_SIZES      = {"s": 640, "m": 640, "l": 640}
    POSE_INPUT_SIZES = {"s": 640, "m": 640, "l": 640}  # may add asymmetric sizes (e.g. "n")
    SEG_INPUT_SIZES  = {"s": 640, "m": 640, "l": 640}
    TASK_INPUT_SIZES = {
        "detect":  INPUT_SIZES,
        "pose":    POSE_INPUT_SIZES,
        "segment": SEG_INPUT_SIZES,
    }

    # State-dict discriminators — cross-test against sibling families!
    _POSE_HEAD_KEY = "<unique pose key>"
    _SEG_HEAD_KEY  = "<unique seg key>"

    @classmethod
    def is_pose_state_dict(cls, sd) -> bool:
        return cls._POSE_HEAD_KEY in sd

    @classmethod
    def is_seg_state_dict(cls, sd) -> bool:
        return any(k.startswith(cls._SEG_HEAD_KEY) for k in sd)

    @classmethod
    def detect_task_from_state_dict(cls, sd) -> Optional[str]:
        if cls.is_pose_state_dict(sd): return "pose"
        if cls.is_seg_state_dict(sd):  return "segment"
        return None  # falls back to detect via DEFAULT_TASK

    def _init_model(self) -> nn.Module:
        if self.task == "pose":    return Libre<FAMILY>PoseModel(size=self.size, nc=1)
        if self.task == "segment": return Libre<FAMILY>SegModel(size=self.size, nc=self.nb_classes)
        return Libre<FAMILY>Model(size=self.size, nc=self.nb_classes)

    def _postprocess(self, raw, conf_thres, iou_thres, **kwargs):
        if self.task == "pose":    return _postprocess_pose(raw, conf_thres, iou_thres, **kwargs)
        if self.task == "segment": return _postprocess_seg(raw, conf_thres, iou_thres, **kwargs)
        return _postprocess(raw, conf_thres, iou_thres, **kwargs)
```

### 6.4 Conversion script — single-task metadata-wrap

```python
"""Convert upstream <FAMILY> weights to LibreYOLO format.

Usage:
    python weights/convert_<family>_weights.py upstream/<file>.pth weights/Libre<FAMILY>s.pt --size s
"""
from __future__ import annotations
import argparse
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


def convert(input_path: str, output_path: str, size: str, nc: int = 80) -> None:
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw, prefer_ema=True)
    print(f"Extracted {len(state_dict)} parameter entries from {input_path}")

    # OPTIONAL — strip an upstream prefix if upstream wraps:
    # state_dict = strip_state_dict_prefix(state_dict, "model.")

    # OPTIONAL — print missing/unexpected diff after dry-load into your model:
    # add_repo_root_to_path(); from libreyolo import Libre<FAMILY>
    # m = Libre<FAMILY>(size=size); res = m.model.load_state_dict(state_dict, strict=False)
    # print("missing:", res.missing_keys); print("unexpected:", res.unexpected_keys)

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="<family>",
        size=size,
        nc=nc,
        task="detect",
        supported_tasks=("detect",),
        default_task="detect",
    )

    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.rename(out)  # atomic
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--size", required=True, choices=["s", "m", "l"])
    p.add_argument("--nc", type=int, default=80)
    args = p.parse_args()
    convert(args.input, args.output, args.size, args.nc)
```

### 6.5 Conversion script — multi-task with `--task` flag

EC pattern. One CLI invocation per task variant.

```python
"""Convert upstream <FAMILY> weights to LibreYOLO format (multi-task)."""
from __future__ import annotations
import argparse
from pathlib import Path

from _conversion_utils import (
    extract_state_dict, load_checkpoint, save_checkpoint, wrap_libreyolo_checkpoint,
)

_SUPPORTED = ("detect", "pose", "segment")
_DEFAULT = "detect"


def convert(input_path, output_path, size, task, nc):
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw, prefer_ema=True)

    # Per-task overrides — pose typically forces nc=1, names={0: "person"}.
    names = None
    if task == "pose":
        nc = 1
        names = {0: "person"}

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="<family>",
        size=size,
        nc=nc,
        names=names,
        task=task,
        supported_tasks=_SUPPORTED,
        default_task=_DEFAULT,
    )

    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.rename(out)
    print(f"Wrote {out} (task={task}, size={size})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--size", required=True)
    p.add_argument("--task", required=True, choices=list(_SUPPORTED), default=_DEFAULT)
    p.add_argument("--nc", type=int, default=80)
    args = p.parse_args()
    convert(args.input, args.output, args.size, args.task, args.nc)
```

### 6.6 Trainer subclass — YOLO-grid

```python
"""<FAMILY> trainer: BaseTrainer subclass."""

from __future__ import annotations
import torch

from ...training.trainer import BaseTrainer
from ...training.config import <FAMILY>Config
from ...training.scheduler import WarmupCosineScheduler
from .loss import <FAMILY>Loss


class <FAMILY>Trainer(BaseTrainer):
    @staticmethod
    def _config_class():
        return <FAMILY>Config

    @staticmethod
    def get_model_family() -> str:
        return "<family>"

    def get_model_tag(self) -> str:
        return f"<FAMILY>-{self.config.size}"

    def create_transforms(self):
        # Return (val_preprocessor, train_dataset_class). Most YOLO-grid
        # families use the shared MosaicMixupDataset wrapper.
        from ...training.augment import MosaicMixupDataset
        from ...validation.preprocessors import <FAMILY>ValPreprocessor
        return <FAMILY>ValPreprocessor(...), MosaicMixupDataset

    def create_scheduler(self, iters_per_epoch: int):
        return WarmupCosineScheduler(
            base_lr=self.config.lr0,
            warmup_iters=self.config.warmup_epochs * iters_per_epoch,
            total_iters=self.config.epochs * iters_per_epoch,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def get_loss_components(self, outputs) -> dict:
        return {k: float(v) for k, v in outputs.items() if k.startswith("loss_")}

    def on_forward(self, imgs, targets, polygons=None):
        # Detect-only families ignore polygons.
        outputs = self.wrapper_model.model(imgs)
        loss_dict = <FAMILY>Loss(...)(outputs, targets)
        return loss_dict
```

### 6.7 Trainer subclass — DETR with per-group LR

Use the **RT-DETR pattern** (`lr_ratio` + `_scale_lr`) — it's smaller than
D-FINE's `_train_epoch` fork and survives base-trainer changes automatically.

```python
class <FAMILY>Trainer(BaseTrainer):
    # ... (other overrides as in 6.6) ...

    def _scale_lr(self, base_lr: float, param_group: dict) -> float:
        return base_lr * param_group.get("lr_ratio", 1.0)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        # Group params by regex; store lr_ratio so _scale_lr applies it each step.
        backbone_params, head_params = [], []
        for name, param in self.wrapper_model.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        base_lr = self.config.lr0
        bb_mult = getattr(self.config, "backbone_lr_mult", 0.1)
        groups = [
            {"params": backbone_params, "lr": base_lr * bb_mult, "lr_ratio": bb_mult},
            {"params": head_params,     "lr": base_lr,           "lr_ratio": 1.0},
        ]
        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=self.config.weight_decay)

    def on_forward(self, imgs, targets, polygons=None):
        # DETR criteria want list[dict] targets, not padded (B, max_labels, 5).
        # Translate here. See models/dfine/trainer.py for the precedent.
        target_dicts = self._padded_to_dict_list(targets)
        outputs = self.wrapper_model.model(imgs, target_dicts)  # DETR forward needs targets in train()
        loss_dict = self.criterion(outputs, target_dicts)
        # DETR aux losses across decoder layers — keep them all in the dict.
        return loss_dict
```

### 6.8 Inference parity test

Save under `weights/parity_<family>.py` or `tests/unit/test_<family>_parity.py`. Skip CI (gate on env var).

```python
"""Cross-load upstream weights into our port and assert max_abs_diff == 0."""
import os
import torch

from libreyolo import Libre<FAMILY>

UPSTREAM_DIR = os.environ.get("<FAMILY>_OFFICIAL_CKPT_DIR")
if not UPSTREAM_DIR:
    raise SystemExit("Set <FAMILY>_OFFICIAL_CKPT_DIR to upstream checkpoint directory")


def main():
    from upstream_pkg import build_model as build_upstream  # TODO: import upstream

    sizes = ["s", "m", "l"]
    for size in sizes:
        upstream = build_upstream(size=size)
        upstream.load_state_dict(torch.load(f"{UPSTREAM_DIR}/<family>_{size}.pth"))
        upstream.eval()

        ours = Libre<FAMILY>(size=size).model
        # If your strict-loading needs adjustment, do it here:
        ours.load_state_dict(upstream.state_dict(), strict=True)
        ours.eval()

        x = torch.randn(1, 3, 640, 640)
        torch.manual_seed(0)
        with torch.no_grad():
            up_out = upstream(x)
            our_out = ours(x)

        # Detect: tensor or per-scale list. DETR: dict.
        # Compare every leaf tensor.
        for k in (up_out if isinstance(up_out, dict) else range(len(up_out))):
            diff = (up_out[k] - our_out[k]).abs().max().item()
            assert diff == 0.0, f"size={size} key={k} max_abs_diff={diff}"
        print(f"size={size}: OK")


if __name__ == "__main__":
    main()
```

### 6.9 Cross-load rejection test (siblings)

```python
# tests/unit/test_<family>_can_load.py
"""Bidirectional can_load rejection. Critical for sibling families."""
import torch

from libreyolo.models.<family>.model import Libre<FAMILY>
from libreyolo.models.<sibling>.model import Libre<SIBLING>


def test_<family>_rejects_<sibling>_state_dict():
    sibling_sd = torch.load("tests/fixtures/Libre<SIBLING>s.pt")["model"]
    assert Libre<FAMILY>.can_load(sibling_sd) is False


def test_<sibling>_rejects_<family>_state_dict():
    our_sd = torch.load("tests/fixtures/Libre<FAMILY>s.pt")["model"]
    assert Libre<SIBLING>.can_load(our_sd) is False
```

### 6.10 Runtime auto-convert recognizer (only if upstream keys differ)

Skip this when upstream keys already match your native port — the `BaseModel`
default hook claims `can_load`-matching layouts for free. RTMDet is the
smallest precedent (`models/rtmdet/convert.py`).

```python
# models/<family>/convert.py
"""Upstream <FAMILY> checkpoint key remapping.

Shared by the runtime auto-converter and ``weights/convert_<family>_weights.py``.
"""
from __future__ import annotations
from typing import Dict
import torch

DROP_PREFIXES = ("data_preprocessor.",)  # TODO: buffers your port doesn't carry


def is_upstream_state_dict(state_dict: dict) -> bool:
    """True ONLY for your upstream's naming — be strict, not greedy."""
    # TODO: require a conjunction of tokens unique to the upstream layout.
    return any(k.startswith("bbox_head.") and "rtm_cls" in k for k in state_dict)


def convert_upstream(state_dict: dict) -> Dict[str, torch.Tensor]:
    """Remap upstream keys to the native port's naming."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in DROP_PREFIXES):
            continue
        # TODO: syntactic renames only — numerics must stay identical.
        out[k.replace("bbox_head.", "head.", 1)] = v
    return out
```

```python
# in models/<family>/model.py
    @classmethod
    def convert_upstream_state_dict(cls, weights_dict: dict) -> Optional[dict]:
        """Remap upstream naming to Libre<FAMILY>'s, or None if unrecognized."""
        from .convert import convert_upstream, is_upstream_state_dict

        if not is_upstream_state_dict(weights_dict):
            return None
        return convert_upstream(weights_dict)
```

Pair it with rejection tests mirroring §6.9: feed every sibling family's
upstream checkpoints through `is_upstream_state_dict` and assert `False`.

---

# Reference

The rest of this document is reference material. You don't need to read it
end-to-end. Look up what you need.

## 7. Architectural patterns

### 7.1 YOLO-grid (NMS-using)

- **Output**: per-scale tensor list. Shape varies. YOLOX is `(B, 5+nc, H, W)` per scale (4 reg + 1 obj + nc cls). YOLOv9 / YOLO-NAS / PicoDet drop objectness, emit `(B, 4+nc, N)`. Confirm against upstream head.
- **Targets**: `(B, max_labels, 5)` padded with `[class, ...4 box coords]`. **Box convention is family-specific** — YOLOX uses `cx, cy, w, h` pixel; YOLOv9 uses `xyxy` normalized to [0, 1] (`models/yolo9/loss.py:89, 305`). Cross-check the loss's first computation against the dataset's emitted target.
- **Loss**: per-anchor + assignment (SimOTA / TaskAlignedAssigner / DFL).
- **Augmentation**: numpy/cv2, mosaic + mixup. Lives in `libreyolo/training/augment.py`.
- **ONNX**: 1 output `"output"`, opset 13 default works.
- **`self.export: bool`** flag flipped by exporter at `libreyolo/export/exporter.py:_model_context`.
- **NCNN / OpenVINO / TensorRT / TorchScript** all work out of the box.

### 7.2 NMS-free YOLO-grid hybrid (yolo9_e2e)

- One-to-one head + top-K inference (no NMS).
- Backbone, neck, training infra inherited from YOLOv9.
- Postprocess accepts `iou_thres` for API compat but `del`s it (`libreyolo/postprocess/yolo9_e2e.py::postprocess`).
- **Must add the family to `BaseBackend._is_nms_free_family()`** (`libreyolo/backends/base.py:211`) or exported backends will wrongly apply NMS. Current set: `{dfine, deim, deimv2, ec, rfdetr, rtdetr, rtdetrv2, rtdetrv4, yolo9_e2e}`.

### 7.3 DETR

- **Output**: dict `{"pred_logits": (B, Q, nc), "pred_boxes": (B, Q, 4)}` cxcywh in [0, 1]. Multi-task variants add `"pred_masks"` / `"pred_keypoints"`.
- **Targets**: `list[dict{labels, boxes_cxcywh_normalized}]` — no padding.
- **Loss**: Hungarian matching + auxiliary outputs across decoder layers.
- **Augmentation**: torchvision v2 with `tv_tensors.Image` + `tv_tensors.BoundingBoxes`. Family-local `transforms.py`.
- **Multi-scale**: per-batch random resize via custom collate.
- **Backbone LR multiplier** (0.1× / 0.5×) is standard. Per-group LR in `_setup_optimizer` + `_scale_lr` override (RT-DETR pattern, preferred) or `_train_epoch` override (D-FINE pattern, older).
- **Gradient clipping** (`max_norm=0.1`).
- **ONNX**: 2 outputs `["pred_logits", "pred_boxes"]`, opset ≥ 16.
- **Export wrapper**: small `nn.Module` calling `model.deploy()` (recursive `convert_to_deploy`) and flattening dict→tuple. D-FINE has the precedent at `models/dfine/nn.py:203-209`.
- **NCNN doesn't work** — `topk` not in NCNN's op registry. Block early.

### 7.4 Sibling / parent-child families

Three flavors, different disambiguation rules:

**Parent-child** (one extends another). YOLOv9-E2E extends YOLOv9. Rules:
- Re-declare `FAMILY` and `FILENAME_PREFIX` always.
- Inherit ClassVars that don't change.
- Override `can_load` to fingerprint a key the child has and the parent doesn't.
- **Bidirectional rejection test** (template §6.9) is mandatory.
- Trainer minimal-override: subclass parent trainer, override only `_config_class`.
- Reuse parent's converter by passing a different `model_family` at wrap.

**Same architecture, different loss/matcher** (D-FINE / DEIM). `can_load` collisions are inevitable. Disambiguate via:
1. Embed `model_family` in the converted checkpoint (the converter's job).
2. Use `FILENAME_PREFIX` in `detect_size_from_filename` as fallback.
3. Order registry imports in `libreyolo/models/__init__.py` so families with most distinctive markers load first; generic fallbacks last. Current order at `libreyolo/models/__init__.py:47-75`: `EC → YOLOX → YOLOv9-E2E → YOLOv9 → YOLO-NAS → DEIMv2 → RT-DETRv4 → D-FINE → DEIM → PicoDet → RT-DETR → RT-DETRv2 → RTMDet → L2CS → FOMO → DepthAnythingV2 → NAFNet → EoMT → PIDNet → MobileNetV4 → ConvNeXt → EfficientNetV2 → ResNet → CLIP` (RF-DETR and DINOv2 lazy-register behind the `transformers` dep). The same order arbitrates runtime auto-conversion claims (§ commit 3b).
4. Raise an explicit "ambiguous between {A, B}" error on a true tie. The D-FINE/DEIM-tie precedent lives in `libreyolo/models/__init__.py` (search "Ambiguous D-FINE/DEIM").

## 8. `BaseModel` contract

### 8.1 ClassVars (declare these explicitly, even when single-task)

| ClassVar | Default | Breaks if missing |
|---|---|---|
| `FAMILY: str` | `""` | family resolution fails |
| `FILENAME_PREFIX: str` | `""` | `_filename_regex` returns `None`, filename detection disabled |
| `WEIGHT_EXT: str` | `".pt"` | download URL construction breaks |
| `INPUT_SIZES: dict[str, int]` | `{}` | `__init__` raises `ValueError` |
| `SUPPORTED_TASKS: tuple[str, ...]` | `("detect",)` | `resolve_task` rejects non-default tasks |
| `DEFAULT_TASK: str` | `"detect"` | `_resolve_task` falls back wrong |
| `TASK_INPUT_SIZES: dict[str, dict[str, int]]` | `{}` (falls back to `INPUT_SIZES`) | per-task input size validation degrades |
| `TRAIN_CONFIG: type[TrainConfig] \| None` | `None` | inference-only port (legitimate) |
| `val_preprocessor_class` | `StandardValPreprocessor` | val pipeline uses generic letterbox |

`__init_subclass__` auto-registers on import (`models/base/model.py:62-69`). Import order in `libreyolo/models/__init__.py` = `can_load` priority.

### 8.2 The 4 ABCs

| ABC | File | Required overrides |
|---|---|---|
| `BaseModel` | `models/base/model.py` | `can_load`, `detect_size`, `detect_nb_classes`, `_init_model`, `_get_available_layers`, `_preprocess`, `_forward`, `_postprocess`, `_get_preprocess_numpy` |
| `BaseTrainer` | `training/trainer.py` | `_config_class`, `get_model_family`, `get_model_tag`, `create_transforms`, `create_scheduler`, `get_loss_components`, `on_forward` |
| `TrainConfig` | `training/config.py` | dataclass subclass with `kw_only=True`, override only fields that differ |
| `BaseValPreprocessor` | `validation/preprocessors.py` | `__call__`, `normalize`; optional property hooks `uses_letterbox`, `custom_normalization`, `wants_unresized_image` |

Family configs are split: most append to `libreyolo/training/config.py` (YOLOX, YOLO9, D-FINE, DEIM, DEIMv2, EC, YOLO-NAS, PicoDet); RF-DETR / RT-DETR / YOLOv9-E2E use family-local `models/<family>/config.py`. Either is fine.

### 8.3 Task resolution flow

When `LibreYOLO("LibreECs-pose.pt", task=None)` is called:

1. **Family resolution** (`libreyolo/models/__init__.py:288-331`):
   - `state_dict["model_family"]` → match registered family.
   - Else iterate `_registry`; first class where `detect_size_from_filename(name) is not None` AND `can_load(weights_dict)` is `True` wins.
   - Else fall back to `can_load` only.
2. **Task resolution** (`libreyolo/models/__init__.py:373-399` → `libreyolo/tasks.py:109-127`):
   - Precedence: **explicit `task=` → `checkpoint["task"]` → state-dict sniff → filename suffix → `DEFAULT_TASK`**.
   - State-dict sniff: family-specific `is_pose_state_dict` / `is_seg_state_dict`.
   - `resolve_task()` validates the resolved task is in `SUPPORTED_TASKS`.

`_filename_regex` (`models/base/model.py`) compiles
`{FILENAME_PREFIX}(?P<size>...)(?P<task>-<suffix>)?{WEIGHT_EXT}`.
Task suffix mapping at `libreyolo/tasks.py:70-80`:

```python
TASK_TO_SUFFIX = {
    "segment": "seg", "semantic": "sem", "pose": "pose", "classify": "cls",
    "gaze": "gaze", "obb": "obb", "point": "point", "depth": "depth",
    "restore": "restore",
}
```

Detect has **no suffix**.

If nothing recognizes the file as a LibreYOLO checkpoint, the factory
attempts runtime auto-conversion (`autoconvert_upstream_checkpoint`, called
from `libreyolo/models/__init__.py`) before failing — see commit 3b.

## 9. Multi-task internals

`Results` slots default `None`:

```python
Results(boxes=Boxes(...), masks=None, keypoints=None, probs=None, obb=None)
```

`Results._select(idx)` slices every non-None slot in lockstep. Drawing
dispatches on slot presence (`if result.keypoints is not None: draw_keypoints(...)`),
not on `self.task`. `draw_keypoints` uses `COCO_KEYPOINT_EDGES` skeleton at
`libreyolo/utils/drawing.py:222-286`.

`InferenceRunner._apply_classes_filter` takes both `masks_t` and
`keypoints_t` (`models/base/inference.py:241-258`). When you implement
`_postprocess`, return only the present slots; `_wrap_results` threads
them into `Results`.

**Postprocess return contract**:

| Task | Required keys | Shape |
|---|---|---|
| detect | `boxes, scores, classes` | `(N, 4)`, `(N,)`, `(N,)` |
| pose | + `keypoints` | `(N, K, 3)` xy+visibility |
| segment | + `masks` | `(N, H, W)` boolean |

If your model emits keypoints as `(N, K, 2)`, append a column of ones for visibility (`libreyolo/postprocess/ec.py::postprocess_pose`).

## 10. Validation

`BaseModel.val()` dispatches by `self.task` (see the validator-selection block
inside `models/base/model.py::val`; a family can also pin `validator_class`
directly):

```text
detect   -> DetectionValidator     (COCOeval iouType="bbox")
segment  -> SegmentationValidator  (iouType="bbox" + "segm", returns (B) and (M))
pose     -> PoseValidator          (iouType="keypoints")
classify -> ClassifyValidator      obb   -> OBBValidator
semantic -> SemanticValidator      point -> PointValidator
depth    -> DepthValidator         restore -> RestoreValidator
```

`ValidationConfig` (`libreyolo/validation/config.py`) exposes one of three
input modes — exactly one must be set:

- `data` (YAML) — detect/segment with COCO128-style YAML.
- `data_dir` — detect/segment with a flat dataset.
- `keypoints_json` + `images_dir` — pose only.

This skill's templates are detect/pose/segment-shaped. For the other tasks,
clone the merged exemplar family instead of adapting a template:

- **classify** → `models/{mobilenetv4,convnext,efficientnetv2,resnet}/` — reuse
  the shared `BaseTrainer` classify path (`_setup_classify_data` /
  `_run_classify_validation`), return `{"probs": ...}` from `_postprocess`,
  set `best_metric_key = "metrics/accuracy_top1"`. Build the predict transform
  with the shared `build_classify_transforms` (`data/augment/classify.py`), not a
  private copy. Declare the eval pipeline once on the model (`crop_pct`, `interpolation`, and `norm_mean` / `norm_std` /
  `resize_mode` when it is not ImageNet center-crop). `val()`, INT8
  calibration and export metadata read it through `_get_eval_transform`;
  override that method only when those settings cannot express the pipeline.
  Do not re-declare preprocessing in a validator subclass. Add the family to
  `tests/unit/test_classify_eval_parity.py`, which fails until you do.
- **zero-shot classify** → `models/clip/` (`set_classes`; `clip_validator.py`
  only for the open-vocabulary label indexing, not for preprocessing).
- **semantic** → `models/pidnet/` (CNN) or `models/eomt/` (ViT).
- **depth** → `models/depth_anything/`.
- **restore** → `models/nafnet/` (native-resolution, no letterbox).
- **point** → `models/fomo/`.
- **gaze** → `models/l2cs/` (inference-only precedent).

**Per-task `best_metric_key` override** is mandatory for non-detect
trainers. `BaseTrainer.best_metric_key = "metrics/mAP50-95"` (`training/trainer.py:35`)
selects on bbox. Pose / seg trainers tracking the wrong metric is the
silent issue this skill calls out as landmine #23.

## 11. Training, per task

### 11.1 Detection training

Families with detection training subclass `BaseTrainer` (or `DFINETrainer` for
DETR). Recorded E2E coverage lives in `tests/e2e/test_val_coco128.py` and
`tests/e2e/test_rf1_training.py`.

### 11.2 Segmentation — plumbing exists, native uptake limited

- `BaseTrainer.on_forward(imgs, targets, polygons=None)` accepts polygons (`training/trainer.py:130-149`).
- `BaseTrainer` batch-unpacks 5-tuples when segments are loaded (`trainer.py:451-455`).
- `YOLODataset(..., load_segments=True)` and `COCODataset(..., load_segments=True)` add a polygons stream.
- Polygons contract: `batch -> image -> instance -> ring`, each ring `Nx2` float32 in **pixel-space original-image coords** (`data/dataset.py:28-49`).
- `SegmentationValidator` runs two pycocotools evaluators (bbox + segm), reports `metrics/mAP50-95(B)` and `metrics/mAP50-95(M)`.
- **RF-DETR** is the only family with a multi-task **trainer** today, via compile-time `RFDETR_TRAINERS` vs `RFDETR_SEG_TRAINERS` selection (`models/rfdetr/trainer.py:27-106`).

To wire native segmentation training: proto/mask head + mask coefficients + polygon→mask rasterization + mask loss in family-local `nn.py` / `loss.py`. Override `best_metric_key = "metrics/mAP50-95(M)"`. Disable mosaic/mixup or use a family-local dataset wrapper (D-FINE pattern).

### 11.3 Pose training

- `PoseValidator` exists (`validation/pose_validator.py:35-270`), uses `COCOeval(iouType="keypoints")`.
- `Results.keypoints` plumbing is complete; `_select` preserves alignment.
- `YOLOPoseDataset` and the family-local EC/RF-DETR transforms carry keypoints.
- Augmentation support is family-specific; do not assume detection mosaic or
  mixup transforms keypoints correctly.

When shipping another pose trainer, reuse the closest keypoint-aware family and
document its dataset, augmentation, loss, and validation evidence explicitly.

## 12. Inference parity proof

Before writing a trainer, prove the model loads upstream weights and
produces identical outputs.

1. Import the upstream model class side-by-side with yours.
2. Build both with the same config / size; cross-load the upstream `state_dict` into yours and inspect the missing/unexpected key diff. Use `strict=True` only if your port loads the *full* upstream state dict. For ports that intentionally drop layers, use `strict=False` and assert the missing/unexpected set matches a documented expected set.
3. Run identical inputs through both at FP32 and assert `max_abs_diff == 0` on output tensors that come from layers present in both, in `eval()` mode.
4. Save the script as a one-off (template §6.8).

`_strict_loading=False` is the right default whenever upstream checkpoints
carry buffers the port doesn't materialize identically (regenerated `anchors`,
`valid_mask`). EC and D-FINE both return `False` unconditionally for this
reason — don't conflate it with task-aware logic.

For sibling-tier integrations that load upstream implementations from an
installed dependency (LibreSAM via transformers): substitute "import the
upstream package and verify it produces the documented outputs." Run the
parity check against the dependency versions users will actually install —
see landmine #37 for what silent dep drift does.

For multi-task families: do the parity check **for each task variant**.

## 13. Files-touched matrix

Always edited:

| File | Why |
|---|---|
| `libreyolo/models/<family>/{__init__.py, model.py, nn.py, utils.py}` | family-local code (preprocess + checkpoint helpers) |
| `libreyolo/postprocess/<family>.py` | postprocessing — one module per family (ADR 0005) |
| `libreyolo/models/__init__.py` | one-line family import (drives auto-registration order) |
| `libreyolo/models/registry.py` | one-line `MODEL_GROUPS` enrollment — `tests/unit/test_model_registry.py` fails without it |
| `libreyolo/__init__.py` | `Libre<Family>` export + `__all__` |
| `libreyolo/training/config.py` | append `<Family>Config(TrainConfig)` if shared route. Family-local `models/<family>/config.py` is also fine — RF-DETR, RT-DETR, YOLOv9-E2E |
| `libreyolo/validation/preprocessors.py` | append `<Family>ValPreprocessor` |
| `tests/unit/test_<family>_*.py` | parity / shape / loss / smoke / sibling-rejection |
| `tests/e2e/conftest.py` | task-appropriate registration: for detect, append rows to `MODEL_CATALOG` and `GENERAL_NIGHTLY_INFERENCE_MODELS` (and record the family in `_RF1_NOT_APPLICABLE` when training is out of scope, or `_RF1_VALIDATION_GAPS` when a callable trainer has not cleared RF1). `MODEL_CATALOG` is detect-only — its mAP gate fails classify / semantic / depth rows by construction; mirror the closest merged non-detect family instead |
| `CHANGELOG.md` | `Unreleased / Added` entry for the new family |
| `README.md` | one row in the family support table — nothing else (README policy in `AGENTS.md`; tick only export columns you actually ran) |
| `reports/export_inventory.json` | regenerate — `tests/unit` checks the committed snapshot matches the runtime inventory (including the family's `group`) |
| `docs/testing.md` | bump the general-nightly test count when nightly models were added |
| `pyproject.toml` | add the `<family>` pytest marker |
| `docs/nomenclature.md` | family / filename rows |

Conditional:

| File | When |
|---|---|
| `libreyolo/models/<family>/loss.py` | non-trivial loss (everyone except RF-DETR) |
| `libreyolo/models/<family>/transforms.py` | augmentation diverges from shared `training/augment.py` |
| `libreyolo/training/scheduler.py` | new LR shape needed (D-FINE added `FlatCosineScheduler`) |
| `libreyolo/training/ema.py` | EMA decay needs runtime change (`set_decay`) |
| `libreyolo/backends/base.py`, `tensorrt.py` | output shape diverges from YOLO-grid (DETR) |
| `libreyolo/backends/base.py:_is_nms_free_family` | family is NMS-free (DETR or yolo9_e2e-style hybrid) |
| `libreyolo/__init__.py` lazy import via `__getattr__` | family has heavy/optional runtime dep (RF-DETR pattern) |
| `libreyolo/export/exporter.py` | needs `_model_context` branch (D-FINE has one) |
| `libreyolo/export/onnx.py` | output count differs from 1 or 3 |
| `weights/convert_<family>_weights.py` | strongly recommended; existing exceptions are YOLOX, YOLO-NAS, YOLOv9-E2E (in-process unwrap), RF-DETR (bespoke autoconvert recognizer) |
| `libreyolo/models/<family>/convert.py` | upstream key naming differs from the native port — runtime auto-conversion remap shared with the offline script (commit 3b) |
| `pyproject.toml` | family needs an optional heavy dep (RF-DETR's `[rfdetr]` extra pulls transformers) |

## 14. Integration-proof tests

| Test | What it proves |
|---|---|
| `tests/e2e/test_val_coco128.py` | Inference loads + runs; preprocessing + class mapping + postprocessing are correct. Asserts mAP50-95 ≥ 0.18 |
| `tests/e2e/test_rf1_training.py` | Training improves the model on real data (marbles). 10 epochs, asserts post-mAP > pre-mAP and post-mAP ≥ 0.05 |
| `tests/e2e/test_rf5_training.py` | Broader fine-tune coverage: config-driven training on 5 representative Roboflow100 datasets (opt-in `rf5` marker) |

DETR families: skip the `last_loss < first_loss` assertion (loss too noisy).

### Per-family unit smoke tests

The floor below the e2e gate. For each family:
- filename detection
- can_load discriminator (bidirectional for siblings — template §6.9)
- forward shape
- loss parity
- export smoke
- trainer smoke
- multi-task: one file per task

### Optional faithfulness gate

`test_val_coco128`'s mAP50-95 ≥ 0.18 floor is a sanity check, not a faithfulness check. For published-number matching: `tests/nightly/test_<family>_official_ckpt_map.py`, gated on `<FAMILY>_OFFICIAL_CKPT_DIR` env var, opt-in.

## 15. Silent-corruption landmines

In priority order. Each line: *[which family hit it]* — what to do.

1. **Color space mismatch between training transform and val preprocessor** *(YOLOX BGR vs YOLOv9 RGB)* — pin the convention in both docstrings.
2. **Target format mismatch** *(D-FINE)* — DETR criteria want `list[dict]` but pipeline yields padded `(B, max_labels, 5)`; translate in `on_forward`.
3. **`can_load()` too greedy** *(RF-DETR almost stole D-FINE checkpoints)* — match on tokens unique to your architecture; never `"backbone"` or `"weight"`.
4. **Backbone LR multiplier missing** *(DETR families)* — silent ~0.5 mAP loss in fine-tuning. Implies per-group LR + `_scale_lr` override (RT-DETR pattern).
5. **Multi-scale collate epoch propagation** *(D-FINE)* — collate needs `set_epoch()` from trainer at each epoch start.
6. **Stop-epoch augmentation policy** *(D-FINE)* — disable `RandomZoomOut`/`RandomIoUCrop` at epoch N.
7. **`labels_getter=lambda` is unpicklable** under Python 3.14 `forkserver` *(D-FINE on macOS)* — module-level function for `SanitizeBoundingBoxes`.
8. **`RandomIoUCrop` has no `p` parameter** in tv2 — wrap with `RandomApply`.
9. **MPS-specific torch bugs in DETR backward** *(D-FINE)* — `_setup_device` override falls back to CPU; CUDA stays unchanged.
10. **Post-train device drift** *(D-FINE)* — end `train()` with `self.model.to(self.device)`.
11. **ONNX opset 13 default** — DETR with deformable attention needs ≥ 16. Set per-family default in `BaseExporter.__call__`.
12. **NCNN can't handle DETR ops** *(D-FINE)* — block the export early.
13. **`head.export` flag missing** *(YOLO-grid)* — without it, ONNX bakes static shapes.
14. **`strict=True` state-dict loading** — override `_strict_loading() = False` if upstream carries EMA buffers, profiling state, or aux heads.
15. **Cross-family rejection on cross-family transfer** — pop `model_family` from the donor checkpoint dict.
16. **EMA decay too low** — early-epoch evals show flat mAP because EMA hasn't settled.
17. **Letterbox vs. plain resize** — `uses_letterbox` property must match the training transform.
18. **`_train_epoch` override drift** *(DETR)* — leave a "kept in sync as of <commit>" comment if you fork the loop. Better: use `_scale_lr` (RT-DETR pattern) and don't fork.
19. **Per-size defaults copy-pasted from one size to all.** Build a side-by-side table of every override per size before assuming s applies to n.
20. **`min_lr_ratio` cross-family disagreement.** D-FINE: 0.05, DEIM: 0.5 (default), DEIMv2: 0.5 default but `n` overrides to 1.0 (`training/config.py:304`), EC: 0.5, YOLOv9: 0.01. Cross-check upstream YAML.
21. **Wasted ImageNet backbone download in `_init_model`.** When the user constructs from a LibreYOLO checkpoint, your weights overwrite anything `_init_model` initialised. Pattern (prototyped in `bf16a2b` but **not currently in `models/rtdetr/model.py`**): `BaseModel.__init__` peeks at the checkpoint via `cls.detect_size`; inside `_init_model`, check `self._loading_pretrained_checkpoint` and pass `backbone_pretrained=False` when set. New backbone-heavy ports should implement this from day one.
22. **`is_pose_state_dict` / `is_seg_state_dict` collision with sibling families.** Cross-test against every sibling family's known state dict in unit tests.
23. **`best_metric_key` left at default for non-detect trainers.** `BaseTrainer.best_metric_key = "metrics/mAP50-95"` selects on bbox. Pose / seg trainers must override.
24. **Mosaic/mixup applied to polygons/keypoints they don't transform.** Disable, or family-local dataset wrapper (D-FINE).
25. **`TASK_INPUT_SIZES` keys disagree with `INPUT_SIZES` keys.** Per-task entries needed for asymmetric task sizes (YOLO-NAS pose-`n`).
26. **Postprocess returning `keypoints` without a visibility column.** Append a column of ones (`libreyolo/postprocess/ec.py::postprocess_pose`).
27. **Multi-character size codes need length-descending regex sort.** RT-DETR's `r50` vs `r50m` — override `detect_size_from_filename` and sort sizes by length descending. Precedent: `models/rtdetr/model.py:253-270`.
28. **Pose head with single-class user-facing override.** DETR pose may pair multi-class internal head with single user-facing class. Converter must override `nc=1, names={0:"person"}` for the pose variant. EC: `weights/convert_ec_weights.py:53-55`.
29. **DETR pose may skip the box head entirely.** Some pose decoders derive boxes from keypoint extents rather than a dedicated regression head (EC: `libreyolo/postprocess/ec.py::postprocess_pose`). Document if yours does.
30. **NMS-free family not added to backend allowlist.** `BaseBackend._is_nms_free_family()` (`backends/base.py:211`) hardcodes `{"dfine","deim","deimv2","ec","rfdetr","rtdetr","rtdetrv2","rtdetrv4","yolo9_e2e"}`. Backends apply NMS post-export to anything not in this set. YOLOv9-E2E shipped with this miss and exported graphs were silently wrong until it was fixed — don't be the next entry.
31. **Sibling-family `can_load` collisions are bidirectional.** Both directions need tests (template §6.9).
32. **Sniff-key prefix-match collisions across multi-task families.** RF-DETR's `"segmentation_head"` is a shorter prefix than EC's `"decoder.decoder.segmentation_head"` — if a new family puts `segmentation_head.*` at a path not nested under `decoder.decoder.`, RF-DETR steals it.
33. **`warmup_iters` from upstream is recipe-tuned to a full epoch budget.** When user overrides `epochs`, fall back to `warmup_epochs`. DEIMv2: `models/deimv2/trainer.py:79-90` (commit `22f90518`).
34. **Asymmetric per-task sizes need explicit error messages.** YOLO-NAS pose ships `n` but YOLO-NAS detect doesn't. The factory's `TASK_INPUT_SIZES[task]` validation needs a per-task error path.
35. **Single-task families currently inherit `SUPPORTED_TASKS` defaults silently.** §8.1's recommendation is **aspirational** today — D-FINE, DEIM, DEIMv2, PicoDet, RT-DETR, YOLOX, YOLOv9, YOLOv9-E2E all rely on `BaseModel`'s `("detect",)` default. New ports should declare explicitly.
36. **Greedy `convert_upstream_state_dict` steals other families' upstream checkpoints.** Runtime auto-conversion asks *every* registered family; subclass-beats-base then registry order arbitrates. Recognize on a strict conjunction of upstream-unique tokens, return `None` for everything else, and add rejection tests against sibling families' upstream checkpoints (mirror §6.9).
37. **Dependency loads that silently no-op leave you training a random backbone.** *(RF-DETR)* `WindowedDinov2WithRegistersBackbone.from_pretrained(...)` silently skipped the weight transfer under recent Transformers releases — classify/seg/depth variants trained on a randomly-initialized ViT and nobody crashed. Fix: explicit reference-model state-dict transfer (`models/rfdetr/backbone.py`, see the NOTE around line 534). Your commit-4 parity gate is the detector — run it against the dependency versions users actually install, not only your lockfile.
38. **Third-party API drift breaks training while inference stays green.** *(all four timm-derived classify families, #496)* a timm kwarg rename crashed `model.train()` for every shipped classify family while `predict`/`val` stayed perfect, so it shipped broken. Inference parity proves nothing about the training path — run a real `model.train()` smoke per family against current deps before release.
39. **Augmentation knobs that silently don't fire.** *(YOLO9, #432/#484)* `degrees`/`translate`/`shear`/`scale` were dead no-ops in the mosaic path, and mixup dropped the second image's labels — silently, for a long time. When you wire a training pipeline, unit-test that each documented aug knob measurably changes pixels *and* labels.

## 16. Cross-references

- **Filename whitelist + HF 5-file contract**: `skills/libreyolo-upload-hf-model/SKILL.md`. Authoritative list of all valid weight names.
- **Nomenclature**: `docs/nomenclature.md`. Family table, casing, size codes, task suffixes.
- **Runtime auto-conversion**: `libreyolo/models/autoconvert.py` (its module
  docstring is the spec); hook default `models/base/model.py::convert_upstream_state_dict`;
  per-family remaps `models/{yolo9,rtmdet,rtdetr,rtdetrv4,picodet}/convert.py`;
  skiplist `_SKIP_FAMILIES`; RF-DETR's bespoke recognizer lives in `autoconvert.py` itself.
- **Sibling tiers (prompt-driven)**: `models/openvocab/` (Grounding DINO, OWLv2,
  OMDet-Turbo; towers in `models/bert/` + `models/swin/`), `models/sam/` +
  `models/mobilesam/` (LibreSAM), `models/vlm/` (LibreVLM),
  `models/ground/` (LibreGround).
- **Conversion tier examples**:
  - metadata-wrap (single-task): `weights/convert_dfine_weights.py`, `weights/convert_deim_weights.py`
  - metadata-wrap (multi-task `--task` flag): `weights/convert_ec_weights.py`
  - light structural (key remap + EMA drop): `weights/convert_rtdetr_hgnetv2_weights.py`, `weights/convert_picodet_weights.py`
  - heavy structural (numbered upstream → semantic): `weights/convert_yolo9_weights.py`
  - safetensors handling: `weights/convert_deimv2_weights.py:19-43`
- **Pattern references**:
  - NMS-free YOLO-grid hybrid: `libreyolo/models/yolo9_e2e/`
  - Parent-child sibling: YOLOv9-E2E inheriting from YOLOv9
  - Same-architecture siblings (D-FINE / DEIM): tie-break in `libreyolo/models/__init__.py` (search "Ambiguous D-FINE/DEIM")
  - Lazy import for optional dep: `_ensure_rfdetr` in `libreyolo/models/__init__.py` (registers RF-DETR + DINOv2 behind the `transformers` dep)
  - Training evidence: RF1 and family-specific trainer tests
  - Vendored sub-component license: `libreyolo/models/deimv2/engine/backbone/dinov3/`
  - Per-group LR via `lr_ratio` + `_scale_lr` (preferred): `libreyolo/models/rtdetr/trainer.py:185-260`
  - Per-group LR via `_train_epoch` fork (older): `libreyolo/models/dfine/trainer.py:148-212`
