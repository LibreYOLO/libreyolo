# Changelog

All notable changes to LibreYOLO are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases
before 1.4.0 are documented in the
[GitHub Releases](https://github.com/LibreYOLO/libreyolo/releases) only.

## [Unreleased]

### Added

- **Class-agnostic NMS. (#928)**
  `predict(agnostic_nms=True)` and `val(agnostic_nms=True)` (CLI `--agnostic-nms`) keep only the highest-scoring box among boxes that overlap above `iou`, whatever their classes. It is one step applied to a family's finished detections, after the `classes` filter, so it works for every family that returns boxes (NMS-free families and exported backends included) and for masks, keypoints and oriented boxes (rotated IoU). Predict covers detect, segment, pose and OBB; validation covers detect and segment. Other tasks and predict paths raise instead of ignoring it.

- **Confusion matrix on validation results. (#928)**
  `val()` returns `results.confusion_matrix` for detect, segment and classify: `matrix[predicted, true]` (detection adds a `background` row and column), with `summary()`, `to_df()` (Polars, optional), `to_csv()`, `to_json()`, `tp_fp()`, `nonzero()`, `plot()` and `class_accuracy()`, the share of found objects of each class that got that class. It is computed on every validation, not only with `plots=True`.

- **Per-class precision, recall and F1 across confidence thresholds. (#928)**
  Detect and segment results expose `box.p_curve`, `box.r_curve` and `box.f1_curve` over `box.px` (1000 confidences, IoU 0.5 matching), one row per class in `box.ap_class_index`, plus `box.p`, `box.r` and `box.f1` at the confidence that maximizes mean F1. `plots=True` also writes `f1_conf_box.png`.

- **Classification validation results object. (#928)**
  Classification `val()` results gain `top1`, `top5` and `confusion_matrix`, and `plots=True` saves the confusion matrix. The returned object is still the same flat metrics dict.

### Fixed

- **TensorRT batched predict with `classes` and a small `max_det`. (#928)**
  The batched path cut to `max_det` before the class filter, so it could return fewer boxes than `batch=1`. It now uses the same candidate budget as single-image predict.

### Changed

- **Confusion matrix plots. (#928)**
  `plots=True` now writes raw counts to `confusion_matrix.png` and per-true-class shares to `confusion_matrix_normalized.png` (previously one normalized image named `confusion_matrix.png`). Detections are counted at confidence 0.25, or `conf` if higher, instead of a fixed 0.15, matching `visualize` and `box.image_metrics`; with `classes=`, ground truth of the dropped classes is no longer counted as missed. The exported `libreyolo.validation.ConfusionMatrix` now indexes `matrix[predicted, true]` (it was `[actual, predicted]`).

- **`agnostic_nms` is no longer a silent no-op in `predict()`. (#928)**
  It used to warn and do nothing. `agnostic_nms=False` is still accepted everywhere.

## [1.6.0] - 2026-09-27

Three new tasks (`detect3d`, `albedo` and `act`, taking the canonical task list from 17 to 20), 34 new top-level `Libre*` exports (31 model classes, the `LibreLLM` client and the `LibreGround` and `LibreVLA` factories), validation error analysis, and a release QA pass over 1.5.0 behavior. 592 commits, 105 merged pull requests. Several changes move outputs or numbers you may be comparing against v1.5.0 (NumPy channel order, batched inputs, resume, RF-DETR preprocessing, EdgeCrafter pose, COCO evaluation ids): read **Changed** and **Breaking changes** before upgrading.

### Highlights

- **Three new tasks. (#845, #846, #847, #848, #850, #857, #858)**
  3D boxes, albedo and robot action chunks join the existing result contracts.

- **More detection, pose, semantic and classification models. (#752, #759, #750, #745, #751, #836, #868, #894)**
  PP-YOLOE, TinyFormer, DEKR, YOLO-NAS oriented boxes, PP-LiteSeg, U-Net, ConvNeXt V2 and six-task GTR expand the model choices.

- **Video representations. (#746, #747, #748, #832)**
  Perception Encoder, V-JEPA 2 and LeVJEPA add finite-clip embeddings with explicit export and training limits.

- **Image restoration and matting. (#799, #800, #803)**
  BEN2, QuickSRNet, DDColor, HVI-CIDNet, LaMa and ViTMatte add specialized image-to-image workflows.

- **Language, grounding and VLM training. (#802, #804, #791, #764, #788, #861, #767)**
  LibreGround and LibreLLM add sibling APIs; LibreVLM gains model choices and Qwen3-VL detection fine-tuning.

- **Validation explains more of its results. (#890, #902, #883, #778)**
  Error images, per-image box counts, classification macro metrics and detection confidence estimates expose distinct evaluation views.

- **Training control. (#812, #875, #900, #782, #819, #905)**
  Class filtering, custom fitness, opt-in training helpers, opt-in `torch.compile` and guardless local DDP extend existing training paths.

- **Data and deployment workflows. (#789, #780, #865, #893, #897)**
  Hub and FiftyOne integration, event histograms, TFLite INT8 I/O and expanded result plotting are available.

- **Release QA fixes. (#907)**
  Release testing with GPU runs and API, CLI and docs audits fixed 23 bugs present in 1.5.0 (the Fixed entries credited only to #907); resume, NumPy input channel order, batched inputs, validation scheduling and CLI exit codes change behavior (see Changed and Breaking changes).

### Added

- **Histogram-based calibration algorithms. (#774)**
  Quantization accepts `algorithm="mse"` and `algorithm="entropy"` in Python and `libreyolo quantize`. They choose activation clipping ranges by quantization-error or KL-divergence sweeps; `algorithm="auto"` continues to use min/max calibration.

- **Three new task contracts. (#845, #850, #857)**
  `detect3d`, `albedo` and `act` extend the canonical task list from 17 to 20. New exported payloads `Boxes3D`, `AlbedoMap` and `Actions` integrate with `Results`.

- **PP-YOLOE detection. (#752)**
  `LibrePPYOLOE` adds `s`, `m`, `l` and `x` at 640, detection-dataset training, two-stage ATSS/task-aligned assignment and class-head rebuilding. The ATSS-to-task-aligned switch scales with `epochs` unless `static_assigner_epochs` is set, and augmentation follows the source recipe (random affine scale and translation, MixUp, HSV, horizontal flips, 90-degree rotation and RGB-to-BGR swaps, no mosaic). ONNX, TorchScript, TensorRT and OpenVINO export paths are available. Pretrained files are downloaded from the source CDN and SHA-256 checked; they are not mirrored.

- **TinyFormer detection. (#759, #903)**
  `LibreTinyFormer` adds trainable `s`, `m`, `l`, `x` and `xl` sizes, with parsing and conversion for `-visdrone` and `-obj2coco` variants. ONNX and OpenVINO are the recorded export paths; rectangular inference is rejected. Hosted weights retain Apache-2.0 plus DINOv3 terms.

- **DEKR bottom-up pose. (#750)**
  `LibreDEKRw32-pose.pt` adds inference-only COCO-17 multi-person pose without a separate person detector, using the released `no_dc` graph at 640. `Results.boxes` are fitted to confident decoded joints. ONNX, TorchScript, TensorRT and OpenVINO export paths emit heatmaps and offsets for decoding. Weights remain linked to the source CDN.

- **YOLO-NAS gains oriented-box training and inference. (#745, #907)**
  `LibreYOLONAS{s,m,l}-obb.pt` uses a 1024 canvas and 18 DOTA2 classes, returning the existing `Results.obb` payload. Training uses rotated assignment/losses and `metrics/mAP50-95(OBB)` for checkpoint selection; `load_detect_weights_for_obb()` initializes from detection weights. ONNX, TorchScript, TensorRT and OpenVINO paths are available. Training defaults follow the upstream DOTA recipe (AdamW at `lr0=5e-5`, weight decay 3.5e-6, cosine decay to 0.1, EMA 0.9997, FP32); augmentation is flips and HSV. Pretrained weights remain non-commercial and non-redistributable, downloaded from the source CDN and SHA-256 checked. `libreyolo train` uses the OBB recipe unless options are set (#907, d6b5d27b).

- **GTR detection, segmentation, pose, oriented boxes, depth and semantic segmentation. (#894, 3e136716; #907)**
  `LibreGTR` adds 22 hosted weights: `LibreGTR{s,m,l,x}.pt` for COCO detection, `-seg`, `-pose` (17 COCO keypoints), `-depth` and `-sem` (19 Cityscapes classes) in the same four sizes, and `LibreGTR{s,x}-obb.pt` for the 15 DOTA v1.0 classes. Inputs are square, a multiple of 32 and at least 160 pixels, default 640; OBB uses a fixed 1024 canvas, and semantic segmentation slides 1024-pixel windows over a 1024×2048 canvas. Postprocessing is NMS-free. Depth returns relative inverse depth, so `1 / depth_map` gives upstream's metres. CPU and MPS run a portable PyTorch recurrence; CUDA uses `flash-linear-attention` when it is installed separately. All six tasks train, in FP32 by default. Detection defaults to 30 epochs, batch 4, AdamW and the flat-cosine schedule (other optimizers and schedulers raise), with upstream Mosaic and MixUp in the first 6 epochs. `lora=True` covers detection, segmentation, pose and OBB. Pose training requires a single-class 17-keypoint dataset at 640. ONNX and TorchScript export fixed-shape FP32 graphs; other formats are rejected. Resume continues the checkpoint's run with the call's device, `exist_ok=False` opens a new run, and `pretrained=` accepts only `True` or `False` (#907, a30759d4, 1777942b, 30decab6, 59019f1b). Code is MIT with retained Apache-2.0 notices; weights are MIT, as the upstream weight repository declares.

- **PP-LiteSeg semantic segmentation. (#751, #907)**
  `LibrePPLiteSeg{t50,b50,t75,b75}-sem.pt` adds trainable Cityscapes segmentation; `t`/`b` select the STDC1/STDC2 backbone and `50`/`75` the source evaluation scale. Evaluation uses 512×1024 for `t50`/`b50` and 768×1536 for `t75`/`b75`; default training crops are 512×1024 and 768×768 respectively. The recipe includes auxiliary heads, Dice/cross-entropy/edge loss, SGD with non-backbone parameters at 10x `lr0`, 10 warmup epochs, polynomial decay, EMA 0.9999 and FP32, and selects checkpoints by `metrics/mIoU`. ONNX, TorchScript, TensorRT and OpenVINO support the native rectangles. `resume=True` resumes the loaded checkpoint and keeps writing into its run, a path resumes that file, and `libreyolo train` keeps the size-aware recipe unless options are set (#907, ad52e7f5, 878464cd).

- **U-Net semantic segmentation. (#836, #908)**
  `LibreUNets-sem.pt` (`LibreUNet`, size `s`) downloads from `LibreYOLO/LibreUNets-sem` and loads the trainable same-padded UNet-S5-D16 graph with FCN head: 19 Cityscapes classes, 1024×2048 evaluation and 512×1024 training crops from a 0.5 to 2.0 rescale of the source frame, with cross-entropy plus 0.4× auxiliary cross-entropy, in FP32 by default. The weights are mmsegmentation's official Cityscapes checkpoint under the Apache-2.0 license it declares (#908, 781ffea5). Export is unsupported.

- **ConvNeXt V2 classification. (#868)**
  Eight sizes, `atto`, `femto`, `pico`, `n`, `t`, `b`, `l` and `h`, add 224-pixel classification, raw-checkpoint conversion, supervised fine-tuning/resume and ONNX/TorchScript export. The constructor defaults to `size="atto"`. Code is MIT; official ImageNet-1K weights retain CC-BY-NC-4.0.

- **Perception Encoder image, text and video embeddings. (#746, #892)**
  `LibrePE` adds `t16`, `s16`, `b16`, `l14` and `g14`: zero-shot classification through `set_classes()`, image/text embeddings and one embedding for a finite video. `clip_frames=8` controls uniform frame sampling; frame embeddings are averaged and L2-normalized. Image embedding families such as CLIP, SigLIP2 and DINOv2 keep one result per video frame; for PE, live cameras, network streams and screen captures raise `ValueError`. Canonical weights use `-cls`; select `task="embed"` for embeddings. Training is unsupported. ONNX/TorchScript include image and fixed-frame video graphs; exported video graphs require direct runtime input. Weights are Apache-2.0.

- **V-JEPA 2 video embeddings and attentive probes. (#747, #748, #892)**
  Four encoders, `l256`, `h256`, `g256` and `g384`, and four released classification probes support finite-video clips and still images. `embed_tokens()` returns `(B, T', H', W', D)`; the public embedding is the normalized mean of final tokens. Frozen-encoder probe training validates video manifests, takes the class count and names from the video dataset, and selects checkpoints by top-1 accuracy; validation rejects a manifest whose ordered class names differ from the model's. ONNX/TorchScript embedding graphs take direct 5D clip input; generic exported video preprocessing is unsupported. Full encoder fine-tuning and pretraining are unsupported. The two giant encoder artifacts are Apache-2.0; the other six artifacts are MIT.

- **LeVJEPA clip and patch embeddings. (#832)**
  `LibreLeVJEPAl-embed.pt` adds inference-only 16-frame ViT-L/16, normalized 1024-dimensional CLS embeddings and patch tokens through `embed_tokens()`. Finite videos use a centered window sampled at approximately 7.5 FPS. TorchScript export requires direct input at batch 1, 16 frames and 224×224. Weights retain CC-BY-NC-4.0.

- **BEN2 background removal. (#799)**
  `LibreBEN2b-matte.pt` adds fixed-1024 background removal, batched native prediction, matte validation and transparent cutout saving. ONNX/TorchScript exports use fixed resolution and batch 1. Weights are MIT.

- **QuickSRNet 2× super-resolution. (#800)**
  `LibreQuickSRNetm2-restore.pt` adds native-resolution Medium 2× upscaling, paired PSNR/SSIM validation, dynamic-spatial ONNX and fixed-canvas TorchScript export. Weights are BSD-3-Clause.

- **Colorization, low-light enhancement, inpainting and guided matting. (#803)**
  `LibreDDColor{t,l}-restore.pt` colorizes images; `LibreHVICIDNett-restore.pt` enhances low light with `gamma=1.0`, `saturation=1.0` and `intensity=1.0`; `LibreLaMab-restore.pt` requires `mask=` for inpainting; `LibreViTMattes-matte.pt` requires `trimap=` for matting. Python and predict CLI accept the single-image guides. All four support paired validation and reject training/export. LaMa runs an embedded ONNX graph and needs `libreyolo[onnx]`.

- **Four 3D detection APIs. (#845, #846, #847, #848, #851)**
  `LibreWildDet3D`, `Libre3DMOOD`, `LibreFCOS3D` and `LibreDetAny3D` return `Results.boxes3d`, with dedicated `wilddet3d`, `3dmood`, `fcos3d` and `detany3d` CLI commands. FCOS3D is native; the other three require separately installed upstream runtimes. WildDet3D, 3D-MOOD and FCOS3D require camera `intrinsics`; DetAny3D estimates them. The payload stores center, dimensions and a `wxyz` quaternion in camera coordinates, plus confidence/class fields and intrinsics. Plotting projects cuboids. Training, validation, tracking and export are unsupported for all four.

- **Marigold V2 depth, normals and albedo. (#850)**
  Nine adapters use a separately downloaded pinned base model, default `quantization="4bit"` and `seed=2025`. Four-bit inference requires CUDA; `quantization="none"` enables a CPU code path, whose full execution was not verified. `AlbedoMap` stores linear RGB and converts to sRGB for display; albedo validation reports PSNR/SSIM in linear RGB and uses PSNR as fitness. Training and export are unsupported. Adapters and base model are Apache-2.0.

- **LibreGround instruction-to-point API. (#802, #804)**
  `LibreGround()` defaults to ShowUI-2B, with Florence-2 base/large and Qwen3-VL 2B/4B/8B adapters. `prompt=` or `query=` sets a per-call instruction; `set_query()` persists it. Single-image multi-query calls merge at most one click per query into `Results.points`. This is inference-only. Florence-2 and ShowUI weights are MIT; Qwen3-VL weights are Apache-2.0.

- **Additional LibreVLM model choices. (#764, #788, #861)**
  Gemma 4 E2B/E4B adds detection, with `gemma-4` selecting E4B; Moondream 2 adds detection, points and native chat, one class per query (Moondream 3 does not load yet, see Known limitations); North Micro Vision runs one detection query per vocabulary class, and an absent but plausible class can produce a hallucinated box; `lfm2-vl-3b` adds LFM2.5's 3B size and 0-1000 box parsing. Molmo2 4B/8B/O-7B returns points, and custom pointing templates must include `{label}`. These additions are inference-only. LFM retains its Open License; the other listed snapshots are Apache-2.0.

- **Qwen3-VL detection fine-tuning. (#767, #834, #907)**
  `LibreVLM("qwen3-vl-2b").train(data=...)` adds LoRA by default, optional full language-model fine-tuning with the vision tower frozen, checkpoint-directory reload, callbacks/loggers and validation-loss checkpoint selection. Defaults include `epochs=10`, `batch=1`, `accumulate=8`, `workers=0`, `hflip=0.5` and gradient checkpointing. `lr0=None` resolves to `1e-4` for LoRA or `2e-5` for full language-model tuning. Runs default to `runs/vlm/train` with `exist_ok=False`. Optimizer-state resume, resuming full language-model fine-tuning, COCO-JSON training and detection-mAP validation are unsupported. `resume=True` on a model loaded from `<run>/weights/<best|last>` continues and writes into that run, and a resumed LoRA adapter is applied to the base weights once (#907, ce888888).

- **LibreVLA robot-policy API. (#857, #858, #860)**
  `LibreVLA()` defaults to SmolVLA; ACT and Diffusion Policy provide training from scratch without a pretrained policy base. Prediction accepts camera frames and `state=`, returning an action chunk `(T, D)` with names, control rate and instruction. SmolVLA needs an instruction; ACT and Diffusion Policy predict without one. LeRobot-data training defaults to `epochs=5`, `batch=8`, `val_split=0.1` and `workers=0`. Offline validation reports aggregate L1/MSE, first-action L1 and per-dimension L1. Checkpoints are directories with `libreyolo_vla.json`. Diffusion keeps observation history; use `reset()` between episodes. No CLI, tracking or export is provided. ACT may download its pretrained image backbone.

- **LibreLLM compatible-endpoint client. (#791)**
  `LibreLLM` supports `api="responses"` by default or `api="chat.completions"`, image requests via `image=`, streaming, `async_call()`, custom `base_url` and `openai/`/`openrouter/` provider prefixes. It returns native SDK responses. The constructor's default model is `gpt-5.6-luna`; no local weights are loaded.

- **RF-DETR UI detector weights. (#897)**
  `LibreRFDETRm-ui.pt` adds a class-agnostic screenshot detector with the single class `object`, routed to `LibreYOLO/LibreRFDETRm-ui`. The weights are racineai's UI-DETR-1, an RF-DETR-M fine-tune. Only the medium detection variant is accepted. The weight declaration is MIT.

- **Dome-DETR checkpoints auto-download. (#797)**
  The six official AI-TOD-V2 and VisDrone checkpoints download from LibreYOLO mirrors; 1.5.0 refused and printed a manual download command. Each download prints a notice that the weights are for academic research only and are not covered by LibreYOLO's MIT license.

- **Validation error images. (#890, #902)**
  `val(visualize=True)` draws class-aware box TP/FP/FN for detection/segmentation and label versus top-1 for ImageFolder classification. Matching uses IoU `0.5` and confidence `max(0.25, conf)`. True positives are drawn green, false positives red and false negatives orange, with a TP/FP/FN count per image; a wrong class counts as one false positive and one false negative. The metrics are unchanged. Images go to `visualize/errors/` or `visualize/correct/` under the run directory. Defaults are `visualize=False`, `show_labels=True`, `show_conf=True`; all are exposed on the val CLI. Unsupported tasks and the separate V-JEPA 2 clip validator reject visualization.

- **Per-image validation box metrics. (#902)**
  Detection and segmentation `val()` results remain dictionary-compatible and expose `.box.image_metrics`, keyed by image filename, with `precision`, `recall`, `f1`, `tp`, `fp` and `fn`. The first occurrence keeps its basename; subsequent duplicates use full paths. Counts use the visualization matching rule even when `visualize=False`; segmentation counts boxes. Zero denominators return `0.0`. These records are not gathered across distributed ranks.

- **Image classification macro metrics. (#883)**
  ImageFolder validation adds `metrics/precision`, `metrics/recall` and `metrics/f1`, averaging over classes present in validation targets, with zero precision for unpredicted classes. Top-1/top-5 keys remain, and default fitness remains top-1. CLI text and JSON include the new values, and the per-epoch `metrics.jsonl` rows and loggers gain three columns (`results.csv` too for DINOv2 and RF-DETR). Targets outside the classifier head's range raise a mismatch error.

- **F1-optimal detection confidence thresholds. (#778, #907)**
  Detection validation adds `metrics/best_conf` and `metrics/best_conf_f1`, computed at IoU `0.50` with micro F1 over classes; per-class thresholds, keyed by class name, are on `results.box.best_conf_per_class`. Equal-score detections are grouped, and F1 ties choose the higher threshold. When no threshold reaches F1 > 0 the threshold and F1 are `0.0`, so the returned metrics stay a flat dict of finite numbers (#907, 9831fa94). Segmentation's separate metrics implementation does not expose these keys.

- **Configurable validation sample plots. (#880)**
  `plot_samples=8` sets the retained sample-image budget on validation, training configuration and both CLI commands. `0` disables sample images and `-1` retains all. Detection, segmentation and pose validation honor it, and an invalid value raises `ValueError`. It does not limit evaluated images or the separate `visualize=True` output.

- **`val()` without `data=` reuses the training dataset. (#907, 625cfbe0, a725680d, 4bee7b7a)**
  Native `model.val()` with no `data`, `data_dir` or `keypoints_json` validates on the dataset path saved in the loaded checkpoint's training configuration, or on the dataset of a `train()` that finished in the same session. Released weights carry none and raise `ValueError` asking for `data=`.

- **`val(project=, name=, exist_ok=)`. (#907, 6595f5e8)**
  Native and exported-backend `val()` write to `project/name` (defaults `runs/val` and `exp`), incremented unless `exist_ok=True`; combining them with `save_dir` raises. Without either, output keeps the timestamped `runs/val/` directory.

- **Two-channel event-histogram detection. (#865)**
  YOLO9 and RF-DETR detection accept prepared `(H, W, 2)` `.npy` arrays containing finite, nonnegative positive/negative counts. Dataset `input_profile` declares `format="event_histogram"`, `layout="HWC"`, `polarity="positive_negative"`, `encoding="counts"`, positive `scale` and positive integer `window_us`. Preprocessing clips to `scale`, divides by it and resizes without RGB normalization. RGB-weight initialization sets both input channels to `1.5 * mean(R,G,B)`; checkpoints preserve profile and initialization metadata. Native train/val/predict and FP32 ONNX without embedded NMS are supported. This path requires fixed-batch, single-device training and excludes live event decoding, video, TTA, tiling, LoRA, distillation and quantization.

- **Custom tracker instances. (#869, #907)**
  `track(tracker=instance)` accepts the exported `libreyolo.tracking.Tracker` protocol: `reset()` and `update(results, image=None)`. Each run resets the instance once, unless `persist=True` continues it from the previous `persist=True` call on the same model and stream (#907, 48542ae9), and passes the original RGB PIL frame. Returned `track_id` must be a one-dimensional integer array/tensor aligned with boxes and on the matching backend/device. Custom instances use `track_conf=0.25` by default; configure them directly instead of passing `tracker_config` or tracker kwargs.

- **Tracking across image sequences. (#825, #907)**
  `track()` accepts individual images, filename-sorted folders, lists, tuples and lazy image iterators as consecutive frames. New `fps=30.0` and `color_format="auto"` control timing and the channel order of NumPy frames: `"auto"` reads them as BGR, like `cv2` frames, and `"rgb"` as RGB (#907, d80e1627). `vid_stride` reduces the output rate to `fps / vid_stride`; ByteTrack and BoT-SORT use that retained rate unless explicitly configured.

- **`track(..., persist=True)` keeps the tracker across calls. (#907, 48542ae9)**
  A call with `persist=True` continues the tracker and track IDs of the previous `persist=True` call on the same model, for per-frame loops such as `model.track(frame, persist=True)`. A different tracker or configuration, or another video file or directory, starts a fresh tracker; the default `persist=False` always does.

- **Hub checkpoint loading and publishing. (#789, #907)**
  `LibreYOLO("owner/repo")` or `LibreYOLO("hf://owner/repo@revision/path/model.pt")` loads schema-tagged checkpoints through the Hub cache; existing local paths take precedence. `model.push_to_hub("owner/repo", private=False)` publishes `model.pt` and a model card. `HuggingFaceHubLogger` or `loggers="hf:owner/repo"` uploads `weights/best.pt`, falling back to `last.pt`, at training end; the logger defaults to `private=True`. Ambiguous repositories require an explicit filename; a sole `.pt` takes precedence over `.safetensors` candidates. A bare `owner/name` whose last segment has a file extension (`weights/model.pt`, `ckpts/best.pth.tar`) or starts with `librefacerec-` is treated as a local path (#907, 5c8a9acb).

- **FiftyOne model application and dataset exchange. (#780)**
  `libreyolo.integrations.fiftyone` exports `apply_model`, `to_fiftyone_model`, `to_fiftyone` and `from_fiftyone`. Model application supports detect, segment, pose, OBB and classify. Dataset exchange covers detection/segmentation YOLO layouts and native COCO annotations, with curated views exportable to training YAML. `apply_model` defaults to `label_field="predictions"`, `conf=0.25`, `iou=0.45`, `max_det=300`, `mask_format="mask"` and `skip_failures=True`.

- **TFLite INT8 inputs and outputs for YOLOX and YOLO9 detection. (#893)**
  `export(format="tflite", int8=True, data=..., fraction=1.0, batch=1)` creates separate normalized-box and score outputs with independent quantization scales. The backend uses sidecar `output_layout` metadata to reconstruct predictions. Missing `data` falls back to `coco8.yaml` with a warning. `libreyolo export --model <weights> --format tflite --int8 --data <yaml>` does the same from the CLI. Requires `onnx2tf[tensorflow]`; `half=True`, `dynamic=True`, other INT8 families/tasks and batch sizes other than 1 are rejected. Some internal operators may remain floating point. On 200 held-out COCO val2017 images with 100 calibration images, mAP50-95 goes from 0.453 (FP32) to 0.416 for YOLOX-s and from 0.442 to 0.367 for YOLO9-t.

- **Expanded `Results.plot()` rendering. (#897, #907)**
  Detection, masks, pose, OBB, classification top-5, points, OCR, semantic/panoptic results and other image-overlay payloads use a shared renderer. Overlay output is contiguous `HxWx3` uint8 BGR; `pil=True` requests PIL. Depth, normal, edge and albedo maps, 3D cuboids and action chunks return a PIL image by default; `pil=False` returns the array. New controls include `img`, `conf=True`, `labels=True`, `boxes=True`, `masks=True`, `probs=True`, `line_width=None`, `pil=None`, `show=False`, `save=False` and `filename=None`. `orig_img` is an `HxWx3` uint8 BGR array: in-memory and URL sources keep their decoded pixels, and local image or video files are read back from `path` on first access (#907, f4da8e89); videos and GIFs collected without `stream=True` keep no frames. In 1.5.0 `plot()` raised `NotImplementedError` for every result except edge and normal maps.

- **Class-subset and single-class detection training. (#812, #875, #838, #907)**
  `train(classes=[...])` filters supervision while preserving original class IDs and dataset `nc`/`names`; `single_cls=True` collapses retained labels to class 0, named `object`. Defaults are `classes=None`, `single_cls=False`. Both options support G0/G1 detection, including YOLO9 and RF-DETR; resuming or validating a subset-trained checkpoint inherits its saved settings, while a new run validates with its own `classes`/`single_cls` (#907, 3ce36e24). CLI train/val expose class subsets, and `libreyolo val --single-cls` evaluates G0/G1 detectors with classes merged (#907, 9d77b56b). The public OBB dataset loader can collapse classes, but model OBB training rejects `single_cls`.

- **Minimum epoch sample count. (#777)**
  `min_samples=0` (CLI `--min-samples`) adds an opt-in floor for the shared detection loader, including RF-DETR detection, segmentation and OBB. A shorter dataset is sampled with replacement; DDP rounds draws to equal rank lengths and reshuffles by epoch. The D-FINE line (D-FINE, DEIM, DEIMv2, RT-DETRv4, TinyFormer, Dome-DETR, EC and GTR detection), FOMO, V-JEPA 2, the RF-DETR, EC, GTR and YOLO-NAS pose trainers and the classification, semantic, depth and restoration loaders ignore it without a warning.

- **Class-balanced detection sampling. (#782)**
  `class_balanced=False` enables repeat-factor sampling when set to true on shared detection loaders, using per-image class occurrence, exponent `0.5` and the median positive class frequency as threshold. It combines with `min_samples` and DDP. Empty images retain weight 1; single-class collapse makes balancing uniform. Trainers bypassing the shared sampler reject enabled balancing.

- **Averaging selected training checkpoints. (#782, #900)**
  `average_best=0` defaults off; `average_best=N` writes `weights/average.pt` from up to N best-scoring snapshots. Floating tensors are averaged uniformly, and integer/boolean buffers come from the best snapshot. The pool persists for supported ordinary resume, and final validation is attempted separately.

- **An export check before epoch one. (#782)**
  `export_check=False` defaults off; enabling it writes `export_check.onnx` and fails the run if export fails. Native/ONNX raw outputs are compared at `rtol=1e-3`, `atol=1e-4` only when tensor layouts and dtypes match; otherwise the skipped comparison is logged. ONNX Runtime is required.

- **Final BatchNorm recalibration. (#782)**
  `precise_bn=0` defaults off; `precise_bn=N` consumes up to N training-loader images before final validation, preserves frozen BatchNorm and refreshes a historical best checkpoint when needed. Under DDP the budget is per participating rank before moment reduction. Models without BatchNorm are unchanged. D-FINE, DEIM, DEIMv2, RT-DETRv4, EC, Dome-DETR and GTR detection skip the final-epoch refresh and run it only on early stop or when refreshing an earlier `best.pt`.

- **Custom training fitness callbacks. (#900)**
  One callback may implement the exported `TrainFitnessCallback` protocol's `fitness(metrics)` and return a finite, higher-is-better scalar for `best.pt`, patience and checkpoint averaging. Raw metrics remain; scores use `fitness/custom` and checkpoints record `fitness_source="callback"`. Custom-fitness runs cannot resume, and an ordinary resumed run cannot add a scorer: load weights and start `resume=False`. The hook is Python-only and rejected by VLM/VLA trainers.

- **Classification loss weighting. (#841, #842, #868)**
  `cls_pw=0.0` accepts `[0,1]`, computing `n_c ** (-cls_pw)` normalized to arithmetic mean 1. Alternatively, `class_weights=True` uses `N / (C*n_c)`; its default is false, and it cannot combine with `cls_pw>0`. Supported families are ResNet, ConvNeXt, ConvNeXt V2, MobileNetV4, EfficientNetV2 and DINOv2. Weighting settings must match on resume. EfficientNetV2 and MobileNetV4 skip the shared training-option checks, so they accept `cls_pw>0` together with `class_weights=True`.

- **Classification crop and augmentation controls. (#877, #879, #881)**
  `scale=(0.5,1.0)` accepts a lower-bound float or explicit crop-area pair; `crop_pct=None` preserves the family's evaluation ratio. Python and CLI accept the new crop controls; CLI adds `auto_augment`, `erasing`, `cutmix`, `fliplr` and `flipud`. Classification CLI `mixup` now routes to batch mixing, default `0.0`, rather than detection `mixup_prob`. `flip_prob=0.5` and `flipud=0.0` control flips. A `crop_pct` override affects train/val evaluation; export retains native family preprocessing. The defaults keep the 1.5.0 training crop (`scale=(0.5, 1.0)`) and each family's evaluation ratio, and the CLI warns when `scale` or `crop_pct` is set for a detection family, which ignores them.

- **Guardless automatic local DDP. (#819, #907)**
  `model.train(device=[0,1])` and `device="0,1"` use coordinator-managed ranks without replaying an ordinary unguarded training script's top-level code. Guarded scripts and explicit `torchrun` remain supported. Callback/logger objects requiring the documented standard-pickle fallback still need a main guard. This includes the default DataLoader `workers>0`: rank worker processes do not re-run the script (#907, b034110d).

- **Opt-in `torch.compile` training. (#905, 0ce0d51c, bfad4596, 7b39ffe5; #907, 5e63bbe5)**
  `train(compile=...)` accepts `False` (default), `True` or `"default"`, `"reduce-overhead"`, `"max-autotune"` and `"max-autotune-no-cudagraphs"`; other values raise `ValueError`, and `libreyolo train compile=` exits `config_type_error` for them. The network forward and backward are compiled at the boundary CUDA-graph capture uses, including YOLO9's default PGI branch and RF-DETR detection; the loss, optimizer, EMA, validation, checkpoints and export stay eager, and checkpoints load without compilation. Only single-GPU CUDA runs compile. CPU, MPS, distributed and distillation runs, tasks without a capture boundary (including RF-DETR segmentation, pose, OBB and classification) and compiler failures train eager after a warning. `"reduce-overhead"`, `"max-autotune"` or `cuda_graph=True` replay CUDA graphs, except under gradient accumulation, which RF-DETR's defaults (`batch=4`, `nbs=16`) use. With RF-DETR's default multi-scale training, dynamic-shape compilation failed in release GPU testing and the run continued eager; the warning suggests `multi_scale=False`, which compiled (#907, 5e63bbe5). On an RTX 4090, RF-DETR nano trained 1.44x faster at batch 4 and 1.08x faster at batch 16, and YOLO9-t at batch 16 did not speed up; accuracy was not measured. Compilation takes several minutes.

- **RF-DETR multi-class pose. (#888)**
  Dataset `kpt_names`, keyed by class index or name, retains the first `len(names)` keypoint rows per class; `[]` marks a box-only class. Head resizing and checkpoints retain `num_keypoints_per_class`, and predictions pad keypoints to `kpt_shape`, zeroing box-only classes. Multiclass datasets require `names`, and at least one class must have keypoints. Default keypoint-mAP fitness does not score box-only classes.

- **Rectangular semantic training datasets. (#751)**
  `SemanticDataset` accepts `(height, width)`, adds `rescale_crop` sampling with ignore padding and supports a family-provided `photometric` transform. The shared `PolyLRScheduler` supports the semantic recipes.

- **Model-diagram authoring skill. (#837)**
  `skills/libreyolo-make-diagram/` provides architecture verification, editable SVG authoring, route checks and an interactive viewer workflow with matching SVG/PNG outputs.

### Changed

- **Output change: NumPy image inputs are read as BGR by default. (#907, d80e1627)**
  `predict()`, `track()`, exported backends, `LibreEnsemble` and `ImageLoader.load()` treat 3-channel NumPy arrays as BGR when `color_format` is `"auto"` (the default) or `"bgr"`, matching `cv2.imread()` and OpenCV frames; `color_format="rgb"` reads RGB arrays. In 1.5.0 `"auto"` meant RGB, so `model(cv2.imread(path))` saw swapped channels. PIL images, tensors, files, bytes and video files are unchanged.

- **Output change: batched arrays and tensors return one `Results` per image. (#907, 1df8cec2)**
  A 4-D NumPy array (NHWC or NCHW) or 4-D tensor is split into its images and returns a list with one `Results` each, also for a batch of one. 1.5.0 used only the first image and returned a single `Results`. `ImageLoader.load()` raises `ValueError` for a batch larger than one.

- **Output change: RF-DETR fine-tunes keep pretrained class logits when dataset class names match the checkpoint. (#907, 020f8eaf)**
  When every dataset class name matches a checkpoint class, the classification head is rebuilt from the matching rows; the released 91-column COCO head maps class names to COCO category-id columns. 1.5.0 truncated the head to `nc + 1` columns, which shifted every label on COCO-named data (coco8 fine-tune mAP 0.053 before, 0.642 after). Other datasets keep the truncate/tile resize, with a warning that the rows do not correspond to the dataset classes.

- **Output change: ONNX INT8 export keeps each family's float layers. (#907, e9743d71)**
  Unless `nodes_to_exclude` is given, the layers `model.quantize()` keeps in float stay float in ONNX INT8 export: YOLO9 `head.` and `backbone.conv0.`, RF-DETR class, box, angle, keypoint and segmentation heads, embeddings and reference points, and the listed BiRefNet/FeyNobg modules. YOLO9 INT8 scores no longer saturate at 0.5: coco8 mAP rises from 0.322 to 0.471 (FP32 0.527). Re-export INT8 models to pick this up.

- **Output change: exported MiDaS and Depth Anything V2 resize inputs bicubically. (#907, 6a33162d)**
  Their exported backends resize the `[0, 1]` float image with bicubic interpolation, as native prediction does, instead of bilinear on uint8, so their depth maps change (MiDaS ONNX had differed about 9% from `.pt`). Other depth families are unchanged.

- **Output change: EdgeCrafter pose keypoints match upstream. (#894)**
  At inference the keypoint position embedding also reaches the attention value, residual, gate input and the previous decoder layer's refinement features, as in upstream EdgeCrafter; 1.5.0 added it to queries and keys only. Released checkpoints reproduce the upstream decoder to float rounding (logits had moved by up to 2.1), and coco8-pose keypoint mAP50-95 rises from 0.186 to 0.813 (S) and from 0.438 to 0.818 (M). EC pose predictions and exports made with 1.5.0 differ from 1.6.0. Training keeps the query/key-only form.

- **Output change: YOLO-format detection and segmentation validation no longer counts a duplicate detection as a true positive. (#906, d1948c44)**
  The first prediction of each evaluation had id 0, which COCO evaluation reads as "unmatched", so the ground truth it matched stayed free and a lower-scoring duplicate of that object also counted as a true positive. Prediction ids now start at 1, as in pycocotools. At most one ground truth per evaluation was affected, so box and mask mAP and AR can fall slightly, most on small validation sets. COCO-JSON datasets and `faster_coco_eval=True` were not affected.

- **The final training epoch always validates, and `val=False` turns validation off. (#907, b9faf668, 6f038fcf, 0cc2598d)**
  With validation on, training validates every `eval_interval` epochs and after the final epoch, so runs shorter than `eval_interval` (10 for YOLO9, YOLOX, YOLOv7, D-FINE, DEIM, DEIMv2, RT-DETRv4, TinyFormer, EC detect/segment, YOLO-NAS detect, PP-YOLOE, PicoDet, RTMDet and Dome-DETR; 5 for EC pose) report metrics and write `best.pt`, and `best.pt` can come from the final epoch. Python `train(val=False)` sets `eval_interval=0` and skips all validation, including final plots and precise-BN metrics, so the run writes `last.pt` but no `best.pt`; 1.5.0 warned about `val` as an unknown key and validated anyway. `libreyolo train val=false` is honored for RF-DETR too.

- **Resume continues the saved run: its arguments, directory and checkpoint. (#907, 606144f4, a5816bc4)**
  For YOLO9 (incl. E2E and P2), YOLOX, YOLOv7, D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4, EC, YOLO-NAS, PP-YOLOE, PicoDet, RTMDet, Dome-DETR, TinyFormer, FOMO, ConvNeXt, ConvNeXt V2, ResNet, MobileNetV4, EfficientNetV2, U-Net and NAFNet, `train(resume=True)` resumes the checkpoint the model was loaded from and `train(resume="path/to/last.pt")` resumes that file. Saved arguments such as `data`, `epochs`, `imgsz`, `batch`, `lr0` and `amp` are restored unless passed; `device` follows the call. A `<run>/weights/*.pt` checkpoint keeps writing into `<run>`, and `exist_ok=False` opens a new numbered run beside it. Released weights raise `ValueError` ("holds no training state") instead of `KeyError: 'epoch'`, and a finished run raises `ValueError` instead of training on. The CLI forwards only the options you set when resuming, so `libreyolo train model=<run>/weights/last.pt data=<yaml> resume=true` continues the run for YOLO9 (the CLI still requires `data`), and a missing checkpoint exits `checkpoint_not_found`. In 1.5.0 resume took the call's defaults (for YOLO9, a new 300-epoch run in a new directory) and resumed the loaded weights even when a path was given. An explicit `val=True` (CLI `val=true`) turns validation back on for a run saved with `val=False`, for these families, RF-DETR, DINOv2 and GTR (#909, 61592d7d, 65e72a16, 19b551d9).

- **RF-DETR resume restores the run's settings. (#907, 46bbff45, d19bb608, 59a347eb, 9f70b876)**
  `LibreRFDETR.train(resume=True|path)` takes the dataset, epochs, batch, lr0 and the other saved settings from the checkpoint unless passed, so `data=` is optional, and continues that run's directory; the CLI does the same, forwards only the options you set, and still requires `data`. 1.5.0 required `data=` and resumed with the signature defaults (100 epochs, batch 4, lr 1e-4), and the CLI opened a new run directory.

- **Validation during training writes to `<run>/val`. (#907, 654a17b3)**
  Every family's in-training validation writes `config.yaml`, plots and JSON inside the run folder. 1.5.0 created a `runs/val/<tag>_<time>/` directory in the working directory for each validation when plots were off.

- **Python `train()` honors `mosaic`, `fliplr` and detection `mixup`. (#907, baa54963, 073aeaab, 7a15208f)**
  `mosaic` sets `mosaic_prob`, `fliplr` sets `flip_prob`, and on non-classification models `mixup` sets `mixup_prob`, as the CLI does; on classification `mixup` stays batch MixUp. 1.5.0 ignored `mosaic` and `fliplr` with an unknown-key warning and routed detection `mixup` to the unused classification field. A call's value overrides the same setting from `cfg=`; an alias and its field with different values in one source raise `ValueError`.

- **`export(quantize=16|8|32)` selects precision. (#907, f8e12af1, 7c6d0119)**
  `16` exports FP16 (`half=True`), `8` INT8 (`int8=True`) and `32` FP32, in Python and with `libreyolo export quantize=`; other values or a conflicting `half`/`int8` raise. 1.5.0 accepted `quantize=16` silently and wrote FP32.

- **Unused export options are warned about. (#907, f8e12af1)**
  `export()` warns `Unknown <format> export arguments (ignored)` for keyword options the format's exporter does not take; 1.5.0 accepted them silently.

- **`track(conf=...)` sets the detection threshold. (#907, 220d9a72)**
  Detections below `conf` no longer reach the tracker, as with `predict(conf=)`. Without `conf` the detector keeps running at the tracker's low threshold. 1.5.0 warned about `conf` as an unknown tracker key and ignored it.

- **CLI YOLO-NAS training uses the task recipes. (#907, d6b5d27b)**
  `libreyolo train` for YOLO-NAS and PP-LiteSeg forwards only the options you set, so each YOLO-NAS task uses its Python recipe; pose gets AdamW, lr0 2e-3, weight decay 1e-6, 10 warmup epochs, 1000 epochs and AMP. 1.5.0 sent the generic CLI defaults (SGD, lr0 0.01, weight decay 5e-4, 5 warmup epochs, 300 epochs) to YOLO-NAS detection and pose training.

- **CLI exit codes follow the failure type. (#907, b5b9d406, d2c56c84, 7861400f)**
  Training failures from bad values or arguments exit 2 with `config_type_error`, `config_unknown_key` or `config_unsupported`; CUDA out-of-memory exits `cuda_oom` and missing device operators `device_not_available`; other failures stay `io_error` (exit 1). An unavailable CUDA/MPS device exits `device_not_available` instead of `model_load_failed`, an `imgsz` the model cannot run exits `invalid_imgsz`, a missing resume checkpoint exits `checkpoint_not_found` instead of `data_not_found`, and `libreyolo metadata` on an unreadable file exits `model_load_failed` instead of a traceback. 1.5.0 reported every training failure as `io_error`.

- **The CLI rejects out-of-range values. (#907, ebd26372)**
  `conf` and `iou` outside [0, 1], and `max_det`, `batch` or `epochs` below 1, exit `config_range_error` (exit 2) in predict, val, export and train; `batch=-1` still requests AutoBatch.

- **Unpublished weights fail fast, and `libreyolo models` lists only routable names. (#907, 9546891f, acef0fb1, 926d740a)**
  A weight URL answering 401, 403, 404 or 410 raises `WeightsNotPublishedError` ("No weights are published at ...") without retries. Names known to be unpublished, such as CLIP `l14`, Depth Anything V2 `g`, DINOv2 heads, EoMT `s`/`b` seg/sem and YOLOv1 `t`, explain why instead of downloading. `libreyolo models` drops names without a download route, lists known-unpublished names separately, and adds `cli_command`, `unpublished_names` and `python_class` to `--json`. It does not probe the network, so names whose repository is missing are still listed.

- **Unknown training keys suggest the close key. (#907, 6f2938e1)**
  `train(epoch=1)` warns `Unknown training config keys (ignored): ['epoch'] (did you mean 'epochs'?)` and still trains the default epochs; nothing is auto-corrected. Keys that are ecosystem arguments LibreYOLO lacks, such as `lrf`, get no suggestion.

- **Auto-converted upstream downloads use a new cache name. (#752)**
  An upstream checkpoint downloaded under its canonical name converts to `<canonical>-converted.pt` instead of `<canonical>-<canonical>.pt` (for example `LibreYOLONASs-LibreYOLONASs.pt`). Existing 1.5.0 caches convert once more on first load, and the old file stays until deleted. Other sources keep `<source stem>-<canonical>.pt`.

- **Expected optional-runtime fallbacks are quieter. (#811, #867)**
  Video output logs an H.264 fallback at INFO and caches failed probes by `(codec, width, height)` only after a later codec successfully opens the output. Missing xFormers in the depth DINOv2 layers is logged at DEBUG.

- **Package metadata and CLI text are updated. (#898, #773)**
  PyPI metadata gains keywords, classifiers and website, documentation, issue and changelog links. CLI help and status strings use plain punctuation. Export-support `since` metadata is restamped as release-line metadata; it is not a guarantee that a capability existed in v1.5.0.

- **Sponsorship links use Open Collective. (#855, #889)**
  The funding configuration, README badges and sponsorship policy point to the LibreYOLO collective.

- **Depth results carry an explicit encoding. (#850)**
  `DepthMap(..., encoding="inverse_depth")` keeps the existing default and adds `"depth"` and `"log_depth"`. Depth validation interprets the encoding before affine alignment; relative predictions do not acquire a metric-scale guarantee.

- **RF-DETR preprocessing changes detection results. (#897)**
  Detection, segmentation and OBB prediction/validation use floating-point OpenCV bilinear resize without antialiasing instead of PIL downscaling. Exported backends use the same preprocessing. Boxes, scores and validation metrics can change for the same checkpoint and input; on a 200-image COCO val2017 subset, RF-DETR-M mAP50-95 moved from 0.6195 to 0.6179. Pose retains its separate antialiased resize.

- **Classifier evaluation uses family-specific preprocessing. (#892)**
  Classification validation and INT8 calibration use the model's evaluation transform. Export metadata adds `norm_mean`, `norm_std` and `resize_mode`; exported ViT, CLIP and SigLIP2 validation uses their own normalization instead of ImageNet defaults. Legacy exports fall back to family values, so their validation scores can also change.

- **Saved classification predictions include the top five labels. (#897)**
  `predict(save=True)` and exported-backend saving use the shared result renderer, including tracking IDs. Matte saving still produces an RGBA cutout, while `plot()` provides a preview.

- **Classification augmentation can close for the final epochs. (#881)**
  `no_aug_epochs` disables `auto_augment`, `erasing`, `mixup` and `cutmix` in the final training tail while retaining crop and flips. Classification family configurations default that tail to `0`. Existing classifier helper imports remain re-exported from `data.classify_dataset` after the implementation moves to `data.augment.classify`.

- **Four detector recipes enable AMP by default. (#754, #907)**
  D-FINE, DEIM, RT-DETRv4 and YOLO-NAS detection now use `amp=True`; `amp_dtype="float16"` is unchanged. Dome-DETR keeps its FP32 default, and the new PP-YOLOE and YOLO-NAS OBB recipes also default to FP32. Resume restores a run's saved `amp` (#907, 606144f4), so a 1.5.0 FP32 run resumes in FP32.

- **YOLO9 training recipe and checkpoint geometry. (#796, #907)**
  New YOLO9 detection training runs (not YOLO9-E2E or YOLO9-P2) add a training-only PGI branch with `aux_weight=0.25`; `aux_weight=0` trains the main head only (Python, or `libreyolo train aux_weight=0` for YOLO9; #907, 3ce36e24). `max_labels` rises from 100 to 300 and SGD momentum warms from `0.8` to `0.937` over three epochs, for YOLO9, YOLO9-E2E and YOLO9-P2. Measured on YOLO9-s (coco128, batch 16, RTX 3090), peak training VRAM rises from 5.47 GB to 9.05 GB and training time by 37% compared with 1.5.0, so batch sizes that fit in 1.5.0 can run out of memory. Old single-head checkpoints resume as single-head. Unmarked checkpoints keep `letterbox_pad="topleft"`; new official conversions stamp `"center"`, and `letterbox_pad=None` inherits the checkpoint setting. Exports record `letterbox_pad`, and all exported backends, backend validation, INT8 calibration and DeepStream sidecars (`symmetric-padding=1` for center) use it; artifacts without the key stay top-left (#907, 0380a2d9, 9cc6e46b). Inference/export use the main head.

- **RF-DETR and DINOv2 use incremented training directories. (#834, #907)**
  Their Python `output_dir` default changes from `"runs/train"` to `None`, resolving to `runs/train/rfdetr_exp` and `runs/train/dinov2_exp`, with `exist_ok=False`. DINOv2's CLI default name is `dinov2_exp`. `resume=True` on a model loaded from `<run>/weights/*.pt`, or `resume="<run>/weights/last.pt"`, continues writing into that run in Python and the CLI, including multi-GPU runs and models built with `LibreRFDETR(path)`; `exist_ok=False` opens a new numbered run beside it (#907, 3ce36e24, d19bb608, 30decab6, 59a347eb, 3d814179; #908, fc06ed8c).

- **Mosaic and MixUp prefer annotated partners. (#776)**
  Mosaic partners are drawn with bounded retries in the shared YOLOX-style mosaic dataset (YOLOX, RTMDet, PicoDet, FOMO) and the YOLO9-style one (YOLO9 and its E2E/P2 variants, YOLOv7, RT-DETR); YOLO9-style MixUp does too. This applies where a family's recipe enables mosaic or MixUp. Up to 20 candidates are drawn, retaining the final draw if none have annotations. This changes the augmentation distribution for datasets containing empty images.

- **QAT disables incompatible training state. (#794, #782)**
  Quantized-model setup disables EMA and SyncBatchNorm and sets `average_best=0`, logging the changes. Ordinary floating-point training retains its requested settings.

- **Dataset-mutation hooks reject incompatible persistent workers. (#775, #881)**
  Training raises when active `close_mosaic`/`set_epoch` hooks cannot reach persistent multi-worker dataset copies. The default nonpersistent loader path is unaffected.

- **CUDA MSDA fallback offers an installation hint. (#793, #790)**
  Eager CUDA calls without an accepted accelerated provider emit one hint for `libreyolo[hub-kernels]`. `ms_deform_attn_available(value=None)` can inspect the actual tensor. `LIBREYOLO_HUB_KERNELS=0` disables Hub and its hint; Triton has a separate opt-out.

### Breaking changes

- **RGB NumPy arrays passed without `color_format` are read as BGR. (#907)**
  Code that passes `np.asarray(pil_image)` or other RGB arrays gets swapped channels.
  Migration: Pass `color_format="rgb"` for RGB arrays, or pass the PIL image; `cv2.imread()` output needs no change.

- **4-D array and tensor inputs return a list. (#907)**
  `model(batch)` returns one `Results` per image in a list, also for a batch of one.
  Migration: Index the list (`results[0]`) or pass a 3-D image.

- **Resume restores the run instead of starting from the call's defaults. (#907)**
  Resumed runs take their saved arguments and keep writing into the checkpoint's run directory; resuming released weights or a finished run raises.
  Migration: Pass the arguments you want to change (for example a larger `epochs=` to extend a finished run), pass `exist_ok=False` for a new run directory, and train without `resume` when starting from released weights.

- **Environments pinning ONNX Runtime below 1.18 no longer satisfy the ONNX extra. (#803)**
  The extra's lower bound changes from 1.16 to 1.18.
  Migration: Upgrade to `onnxruntime>=1.18.0` before installing `libreyolo[onnx]`.

- **SAM 3D Body accepts only the reviewed checkpoint/configuration asset set. (#807)**
  An explicit checkpoint path must identify `model.ckpt` beside its matching `model_config.yaml` and `LICENSE`, or the containing directory, with the pinned bytes. Arbitrary renamed or modified assets are rejected. Extra files outside the accepted snapshot inventories are rejected.
  Migration: Use automatic acquisition after installing `libreyolo[hf]` and obtaining access to the gated model, or provide the reviewed snapshot directory and pinned MHR asset on a trusted local filesystem.

- **Preprocessing corrections can invalidate saved validation baselines. (#897, #892)**
  RF-DETR non-pose resizing and exported classifier normalization change without a legacy-preprocessing compatibility flag.
  Migration: Re-run validation and recheck deployed confidence thresholds before comparing metrics or outputs with v1.5.0.

- **Default mixed precision changes four detector training recipes. (#754)**
  D-FINE, DEIM, RT-DETRv4 and YOLO-NAS detection use FP16 autocast by default.
  Migration: Pass `amp=False` to retain FP32 training.

- **New YOLO9 training runs use different defaults. (#796, #907)**
  PGI, the ground-truth cap and momentum warmup change new-run behavior.
  Migration: Use `aux_weight=0` (also `libreyolo train aux_weight=0`), `max_labels=100` and `warmup_momentum=0.937` for the former choices; the last two also apply to YOLO9-E2E and YOLO9-P2. Lower `batch` if a 1.5.0 batch size runs out of memory. Set `letterbox_pad="topleft"` when starting from a new center-stamped conversion if the old geometry is required.

- **Default RF-DETR and DINOv2 artifact paths change. (#834, #907)**
  Fresh Python runs use family-named, incremented directories; resumed runs keep writing into the checkpoint's own directory (#907).
  Migration: Update artifact-path consumers, or pass `output_dir="runs/train", exist_ok=True` to retain the old location and reuse behavior.

- **Invalid classification augmentation combinations now raise. (#881)**
  `mixup + cutmix > 1` is rejected at setup instead of silently reducing the effective CutMix probability.
  Migration: Keep classification mixing probabilities summing to at most 1.

- **QAT no longer keeps requested EMA or SyncBatchNorm active. (#794)**
  The quantized-model setup overrides both options.
  Migration: Use QAT best/last checkpoints without relying on EMA or SyncBatchNorm state; averaged fake-quant checkpoints are also disabled.

- **Custom persistent-worker loaders may now fail at mutation hooks. (#775, #881)**
  Loaders whose worker copies cannot observe scheduled dataset changes are rejected.
  Migration: Use `persistent_workers=False` for these loaders, or rebuild workers after dataset mutation.

- **CLI training errors use new codes. (#907)**
  Configuration errors exit 2 with `config_*` codes instead of `io_error` (exit 1); device, `imgsz` and missing-checkpoint failures have their own codes.
  Migration: Match the documented codes in scripts that parse `--json` errors or exit statuses.

### Fixed

- **YOLO text labels use the same geometry in training and validation. (#815, #827)**
  Finite boxes crossing an image border are clipped; boxes with no visible area, non-finite coordinates and malformed polygons are dropped. Training keeps segment rings aligned with accepted boxes and passes the configured class count to the parser, so out-of-range IDs are reported before target creation.

- **Training monitor completion counts and charts use the reported epoch. (#831)**
  `status.json` no longer adds one to the completed-epoch count, and the monitor no longer adds one to `current_epoch`, `best_epoch` or `metrics.jsonl` chart coordinates. ETA therefore includes the remaining final epoch.

- **RT-DETR weight names resolve to the right checkpoint. (#871, #850, #908, #909)**
  Size detection requires a delimiter around single-character `l` and `x` codes, and auto-download needs an RT-DETR name, so loading a missing `last.pt`, `lora.pt` or `model_xlnet.pt` no longer downloads `LibreRTDETRl.pt` or `LibreRTDETRx.pt` as in 1.5.0. `rtdetr-<size>.pt` and `LibreRTDETR-<size>.pt` download the canonical `LibreRTDETR<size>.pt`, and the r50 and r101 names now download `LibreRTDETRr50.pt` and `LibreRTDETRr101.pt` instead of the DETR checkpoints `LibreDETRr50.pt` and `LibreDETRr101.pt` (#908, 874f0c28). DINO-DETR and Deformable DETR names such as `dinodetr-r50.pt`, `dino-detr-r50.pt` and `deformable_detr-r50.pt` no longer download the DETR checkpoint either (#909, c0fd0ed9).

- **Custom RF-DETR pose checkpoints retain their names and device. (#884)**
  Checkpoint class names take precedence over legacy arguments instead of being replaced with `person`. Rebuilt GroupPose attention-mask buffers stay on the decoder device after a keypoint-schema change.

- **Skipped pose labels identify the file, line and parsing error. (#876)**
  Training and validation report malformed rows, quote at most the first five rejected lines and count the rest. Diagnostics distinguish non-numeric values from an incorrect field count and explain the expected `kpt_shape: [K, 2]` or `[K, 3]` layout.

- **Unavailable weight spellings fail with a useful alternative. (#773, #779)**
  Auto-download rejects task-suffixed FCN and Mask R-CNN names and names the hosted forms `LibreFCNr50.pt`, `LibreFCNr101.pt` and `LibreMaskRCNNr50.pt`. `lingbotvision-g` explains that no checkpoint is published and points to `s`, `b`, `l` or a local checkpoint.

- **Classifier preprocessors supply calibration arrays. (#892, #907)**
  AlexNet, VGG, ResNet, EfficientNetV2, ConvNeXt, MobileNetV4, DeiT, Swin and ViT satisfy the calibration `(CHW array, ratio)` contract, and their evaluation transforms accept the `(height, width)` size that export passes (#907, 12a2d321), so INT8 calibration no longer drops the images; non-square classification sizes raise `NotImplementedError`. DINOv2 classifier/embed calibration uses its classification pipeline instead of semantic preprocessing.

- **Rectangular fine-tunes restore their saved input canvas. (#901)**
  Loading restores `imgsz_h`/`imgsz_w`, and the live model adopts the training rectangle or clears it after a later square run. Default exports of square-native families that cannot use the rectangle (YOLOX, YOLOv7, RTMDet and PicoDet, and YOLO9 to ExecuTorch, Paddle or MNN) fall back to `max(height, width)` with a warning; explicitly requested unsupported rectangles still raise. Export CLI JSON reports the actual canvas, and tiling defaults to its long side.

- **Rectangular auxiliary inference paths use the requested size. (#901)**
  Fixes YOLOX validation ratios, TTA letterbox reversal, YOLO9-E2E default postprocessing, `InferenceProfiler`, validation preprocessors and `capture_graph(imgsz=(h,w))`. Finalized quantized `.pt` exports retain rectangular dimensions.

- **DETR-line rectangular prediction handles height and width separately. (#903)**
  RF-DETR, D-FINE, DEIM and EdgeCrafter accept `imgsz=(height, width)` in their resize paths; in 1.5.0 these calls failed with a `TypeError`. RF-DETR still checks the task-specific patch/window grid. DEIMv2 and TinyFormer give an explicit `ValueError` for rectangular predict/val requests. This does not add universal rectangular training or export.

- **Raw checkpoints retain the correct class metadata. (#897, #745)**
  RF-DETR's one-output class head converts as `nc=1` instead of receiving an 80-class record. YOLO-NAS detect/pose conversion reads `processing_params.class_names` when ordinary argument metadata is absent.

- **Converted RF-DETR checkpoints with empty `names` keep their class labels. (#906, 45d62c26, caef122d)**
  Auto-conversion of a raw RF-DETR checkpoint whose top-level `names` is empty takes labels from `args.class_names` or `hyper_parameters.class_names` when their count matches the resolved class count, and a checkpoint whose metadata identifies COCO gets the COCO-80 names. 1.5.0 labelled both `class_0`, `class_1` and so on.

- **DINOv2 resume restores training state. (#834, #907, #908)**
  `resume=` calls the trainer's resume path before continuing instead of only carrying the value in configuration, keeps the task's best score (`metrics/accuracy_top1` or `metrics/mIoU`) so the first resumed epoch no longer overwrites `best.pt`, takes the dataset, epochs, batch, lr0, imgsz and the other saved settings from the checkpoint unless passed (so `data=` is optional), and continues the checkpoint's run directory, in Python and the CLI (#907, a0222aaa, 59a347eb, 9f70b876; #908, 44829b22, fc06ed8c).

- **YOLO9 fine-tunes from COCO load at their own class count. (#907, 5c997862)**
  YOLO9-t, YOLO9-E2E-t, YOLO9-P2-t and YOLO9-P2-s models fine-tuned from COCO to another class count keep 80-wide class towers, which a fresh build at that count does not reproduce (64 wide for a 2-class YOLO9-t). Building such a checkpoint at its class count, as multi-GPU workers and `LibreYOLO(path, nb_classes=2)` do, now takes the tower width from the checkpoint, including the YOLO9-E2E one-to-one towers. In 1.5.0 these loads failed with a size mismatch, so a multi-GPU resume of such a run, or a new multi-GPU run started from its checkpoint, crashed in every worker.

- **OBB validation works with DataLoader workers on macOS and Windows. (#894, a5177119)**
  The OBB validation preprocessor recursed without end when unpickled, so spawned worker processes (the default on macOS and Windows) failed with `RecursionError` whenever `workers` was above 0, as it is by default (4). In 1.5.0 this affected RF-DETR and RT-DETRv2 OBB validation.

- **`cuda_graph=True` with gradient accumulation sums gradients correctly. (#905, 0ce0d51c)**
  Autograd adopted a replayed graph's gradient buffers as `.grad` and the next replay overwrote them, so an accumulation window stepped with twice the last micro-batch's gradient instead of the sum. Live gradients now get their own storage before each replay when accumulating. This affected every `cuda_graph=True` run with `nbs` above `batch`, including RF-DETR's defaults (`batch=4`, `nbs=16`); runs without accumulation are unchanged.

- **Loading models no longer changes `torch.compile` settings for the whole process. (#905, 0ce0d51c)**
  In 1.5.0, importing `libreyolo.models`, which every model load does, set `torch._dynamo.config.automatic_dynamic_shapes=False` and `accumulated_cache_size_limit=1024` through the vendored DINOv3 backbone, affecting any model compiled in the same process.

- **RF-DETR no longer prints a Transformers deprecation notice. (#905, 0ce0d51c)**
  The DINOv2 backbone reads `config.return_dict` instead of `config.use_return_dict`, which recent Transformers releases report as deprecated on the first prediction. Eager outputs and gradients are bitwise unchanged.

- **`track(tracker="bytetrack.yaml")` works. (#907, 462f2910)**
  The `.yaml`/`.yml` spellings of the built-in tracker names select that tracker instead of raising "Unknown tracker"; a local file with that name raises, because tracker yaml files are not read.

- **Resuming a non-SGD run without `optimizer=` works. (#907, 3ce36e24)**
  Resume uses the checkpoint's optimizer, so an AdamW run no longer rebuilds SGD and fails with `KeyError: 'momentum'`.

- **DINOv2 `train(batch=...)` is honored. (#907, 43da519b)**
  Every DINOv2 run trained at batch 4 regardless of the requested batch.

- **An ecosystem `cls` train key no longer crashes. (#907, de7807f5)**
  `train(cls=0.5)` and `cfg=` files with the `cls` loss gain are warned about as unknown keys instead of raising `TypeError: got multiple values for argument 'cls'`.

- **YOLO9 rounds `imgsz` up to a multiple of 32. (#907, eda9b46d)**
  Predict, val, export and train warn and round, for example 333 to 352, instead of failing with "Sizes of tensors must match".

- **`predict(classes=0)` and YOLO9 `imgsz=[h, w]` lists work. (#907, e72f3396, 5a40955a)**
  Native, exported-backend and ensemble predict accept a single int class id; YOLO9 predict and export accept list sizes.

- **`classes=` is applied before `max_det`. (#907, 6cef64d2)**
  In predict, TTA and exported backends, `classes=[16], max_det=1` returns the best class-16 box instead of nothing.

- **Tiling rejects `overlap_ratio` outside [0, 1). (#907, bd3be61a)**
  Values of 1 or more raise `ValueError` instead of never finishing.

- **FP16 ONNX and TorchScript exports load and predict. (#907, 321a9f93, 3be89e20)**
  YOLO9 and RF-DETR `half=True` ONNX files load in ONNX Runtime, backends cast inputs to the graph dtype, and the in-memory model is restored exactly after a half export.

- **ONNX INT8 export works with the default `dynamic=True`. (#907, 16e26385)**
  ONNX Runtime symbolic shape inference, which failed on dynamic-batch graphs, is skipped.

- **`val(device=...)` moves the model. (#907, 4a071ed1)**
  As `predict(device=...)` does, instead of failing with a mixed-device error; YOLO9 rebuilds its anchor cache after a device or dtype switch, including MPS to CPU.

- **TorchScript exports load on Apple Silicon. (#907, 32c0c4ff)**
  `device="auto"` picks CPU instead of MPS, which cannot load the traced float64 constants.

- **Quantization and QAT run on Apple Silicon. (#907, 7861400f)**
  Fake-quantized models move to CPU with a warning instead of failing on MPS.

- **Finalized fp16-remainder checkpoints return to float32 cleanly. (#907, 32a3e933)**
  Their half-width I/O hooks are removed, so export, QAT and dequantize work.

- **EfficientNetV2 and EfficientDet ONNX exports match PyTorch. (#907, 31a04e9b, 0124a09f)**
  SAME padding is baked as constants when tracing, so onnxsim no longer folds it into wrong pads; re-export existing ONNX files.

- **YOLO9 CoreML export works and letterboxes like `.pt`. (#907, 7d0869cf, 9cc6e46b)**
  Export no longer fails with `IndexError` (empty frozen anchor grid), and the CoreML backend letterboxes with the exported pad instead of stretching; a non-square coco8 image matches `.pt` with IoU 1.0.

- **Fixed-shape exports name the expected size. (#907, 01ac6b3d, ef43f7a0)**
  ONNX, TensorRT, OpenVINO and CoreML exports run at another `imgsz` raise `ValueError` with both sizes; the CLI exits `invalid_imgsz` and suggests the exported size.

- **`val(split=...)` with an empty dataset entry names the split. (#907, e5dc0225)**
  Detection, segmentation and OBB validation raise `FileNotFoundError` instead of `TypeError: PosixPath / NoneType`.

- **RF-DETR `batch=-1` fits in memory. (#907, 43ad8560)**
  Losses compute paired IoU/GIoU instead of an N x N matrix (same values), and AutoBatch probes the real loss at the largest multi-scale size, so it picks a smaller batch; on a 24 GB GPU it had estimated 10.6 GiB for a step that used more than 21 GiB.

- **Failed training setup leaves no `status.json` in the working directory. (#907, 544b44c4)**
  Failures after setup still record `state=failed` in the run folder.

- **CLI `--json` output is one JSON document. (#907, e4750317, 67b1f91e, e4afc028)**
  Third-party output (pycocotools tables, torch ONNX logs) goes to stderr, and predict failures and usage errors print JSON errors with documented codes instead of tracebacks.

- **CLI `--key value` values are not rewritten. (#907, 4e5fc889)**
  `--name show` names the run `show` instead of `--show`.

### Performance

- **Training reduces matcher transfers and optimizer launches. (#761, #762, #765, #771, #772)**
  Affected RF-DETR, D-FINE/DEIM and RT-DETR matchers separate cost building from host assignment with a depth-two pipeline. Broadcast L1 costs replace `cdist(p=1)` where applicable; YOLO9 batches loss logging transfers. Eligible CUDA Adam/AdamW construction is fused; SGD and non-CUDA parameters retain stock construction.

- **Denoising indices and EC tensors avoid repeated work. (#772)**
  D-FINE/DEIM derive denoising positive indices from known target counts and retain tensor normalizers. EC caches eligible anchor grids and projection vectors, bypassing caches for trainable parameters.

- **Hub MSDA accepts half-precision inputs. (#760, #762)**
  FP16/BF16 inputs are upcast to FP32 for the kernel and outputs are restored to the value dtype, with gradients retained through the casts.

- **In-tree Triton MSDA for eligible CUDA inference. (#784, #787, #790)**
  The provider supports FP32/FP16/BF16 when Triton is installed, rejects gradient-requiring inputs and falls back to portable attention. It does not need the `hub-kernels` extra; Hub remains preferred when installed, and `LIBREYOLO_TRITON_MSDA=0` disables Triton. On an RTX 5070 Ti the isolated kernel measured about 4x faster on RF-DETR-s shapes and about 12x on multi-level DETR shapes; RF-DETR-n end to end gained about 7%. EC pose adapts its split values to the shared slot. Provider fallback, device selection, capture handling and disable-after-failure behavior are included.

- **Training keeps PyTorch's CPU threads within the process's CPU allowance. (#905, 0ce0d51c, 888e810a, 7b39ffe5)**
  During `train()`, unless `OMP_NUM_THREADS` is set, the PyTorch thread count is lowered to the smaller of the cgroup CPU quota and the CPU affinity count, divided among local ranks, and restored afterwards; it is never raised. This stops CPU-limited containers from throttling every step: on an RTX 4090 host with a 30-core CPU quota, RF-DETR nano's default recipe went from 81 s to 9.7 s per epoch. It applies to all families on the shared trainer, not to LibreVLM or LibreVLA fine-tuning.

### Security/Licensing

- **Existing-family source and weight notices are completed. (#773, #779)**
  Notices add or clarify attribution and weight terms for LW-DETR, LingBot-Vision, face embedding/detection, SAM 3D Body, SenseNova-Vision and several existing classifier/detector families. FeyNobg mirror descriptions retain `fp16`/`fp8`; the withdrawn `nvfp4` claim is removed.

- **SAM 3D Body and MHR loading validate pinned assets. (#807)**
  SAM snapshots are keyed by immutable revisions and verify the checkpoint, configuration and license bytes. MHR acquisition checks the reviewed release archive and member identities. Integrity checks reject linked paths and compare files before and after construction; the path-based upstream constructor still requires a trusted, quiescent local filesystem.

- **Weight delivery preserves model-specific restrictions. (#753, #797, #862, #864, #851, #845, #846, #848)**
  MiDaS `s/l`, MoGe2 `s/l` and SAM1 base/large/huge route to LibreYOLO mirrors; MoGe2 `b` remains upstream. Six Dome-DETR mirrors are academic-research-only. Twelve EC `obj2coco` detect/segment/pose variants are opt-in and non-commercial; original COCO variants retain their recorded Apache-2.0 grant. DetAny3D weights are CC-BY-NC-4.0, WildDet3D retains the SAM license, and 3D-MOOD is Apache-2.0. These weight terms are separate from the library's MIT license. MiDaS is mirrored under the publisher's MIT grant.

### Dependencies & extras

- **Automatic DDP adds one core dependency. (#819)**
  `cloudpickle>=3.0.0` transports coordinator jobs. The package-wide Python requirement remains `>=3.10`.

- **The ONNX runtime floor rises. (#803)**
  `libreyolo[onnx]` now requires `onnxruntime>=1.18.0` instead of `>=1.16.0`, for LaMa's opset-21 graph.

- **New integration and language extras. (#789, #791, #780, #802, #767, #807, #908)**
  `hf` adds `huggingface_hub>=1.0.0`; `llm` adds `openai>=1.66.0`; `fiftyone` adds `fiftyone>=1.0.0` and caps `graphql-core<3.3`, whose 3.3.0 release breaks `import fiftyone` (#908, e122db89); `ground` installs `libreyolo[vlm]`; `vlm-train` adds that VLM stack and `peft>=0.17.0`. `all` gains `hf` and `llm`. FiftyOne remains separate because its headless OpenCV dependency overlaps the core `cv2` installation.

- **VLA, Marigold and Molmo2 have separate dependency stacks. (#857, #858, #860, #850, #861)**
  `vla` adds `lerobot[smolvla,diffusion,dataset]>=0.6.1` only on Python `>=3.12`. `marigold` pins `diffusers==0.38.0`, `peft==0.18.1`, `accelerate==1.13.0`, `transformers==5.4.0` and, on Linux/Windows, `bitsandbytes==0.49.2`. `molmo2` pins `transformers==4.57.1` with `einops>=0.8.0` and `accelerate>=0.26.0`. These extras are outside `all`; uv declares Molmo2 conflicts with the newer Transformers/Hub stacks, including `hf`, `vlm`, `ground`, `marigold` and `all`.

- **Some VLM families require newer Transformers than the common VLM extra floor. (#764, #788)**
  North Micro Vision checks for `transformers>=5.16.0`, and Gemma 4 for `>=5.10.0`, before loading; the shared `vlm` extra still declares `>=5.1.0`.

### Known limitations

- Detection and segmentation `metrics/precision` and `metrics/recall` (and their `(B)`/`(M)` names) hold mAP50-95 and AR100, not a precision/recall pair, and `metrics.box` has no `map`, `map50`, `map75` or `maps`, as in 1.5.0.
- Some call shapes differ from the ecosystem: `track()` returns a generator (`[0]` fails), single-image `predict()` returns a `Results` rather than a list, `show=True` is ignored for image, list and folder inputs (it works for video, streams and `track()`), display options such as `line_width` and `predict(half=True)` on `.pt` models only warn, `Results.save(filename)` works only for matte results and there is no `Results.show()` (use `plot(save=True, filename=...)` or `plot(show=True)`), and Python `export()` defaults to `dynamic=True` while the CLI defaults to `False`.
- YOLO9 and other families on the shared training configuration warn about and ignore these ecosystem training keys: `lrf`, `cos_lr`, `close_mosaic`, `deterministic`, `fraction`, `rect`, `plots`, `time`, `dropout`, `box`, `dfl`, `label_smoothing` and `overlap_mask`. `optimizer="auto"` raises `ValueError` (pass `sgd`, `adam` or `adamw`). Glob sources, `save_txt` and `save_crop` are not supported, and `Results.speed` is empty.
- Results from `stream=True`, `track()` and in-memory image inputs keep the decoded frame as `orig_img` (about 6 MB per 1080p frame; 1.5.0 kept none), so code that stores every result of a long stream or tracking run should keep only what it needs.
- CLI training of D-FINE, EC and SegFormer uses the generic `epochs=300` and `lr0=0.01` unless you set them, instead of the values their Python `train()` uses. `--dry-run` shows generic defaults for YOLO-NAS, PP-LiteSeg and GTR, although their training uses the family recipe.
- TorchScript exports are traced on CPU by default and fail when run on CUDA; export with `device=0` to trace on the GPU. Same as 1.5.0.
- Training checkpoints keep optimizer and EMA state (an RF-DETR `best.pt` is about 488 MB, and every `save_period` save adds another); `val(conf=)` is ignored by the DETR-family validators; validation on a split without labels reports mAP -1.0; RF-DETR ignores `seed=`, so its runs do not repeat.
- ONNX parity gaps, same as 1.5.0: Depth Anything V2 differs from PyTorch by about 2.5%, ZipDepth by 4% and Depth Anything 3 by 2-5%; HRNet ONNX treats the whole image as one person; BiRefNet and FeyNobg ONNX graphs contain DeformConv, which ONNX Runtime cannot run.
- Exported YOLO-NAS OBB models validate with the detection preprocessor, so their `val()` mAP is near 0; prediction is unaffected.
- GTR ONNX models failed the GPU end-to-end test for all four sizes in release testing; CPU ONNX export and prediction match PyTorch.
- `compile=True` needs a C compiler and Python headers on the training host; without them training continues uncompiled after a warning.
- Some advertised weight names have no published file, so loading them fails with "No weights are published at ...": `LibreNAFNet{s,l}-restore.pt` (`LibreNAFNetl-restore-sidd.pt` is published), `LibreDINOv2n.pt`, `LibreBiRefNett-matte.pt` (`LibreBiRefNetl-matte.pt` is published) and `LibreYOLO9P2{t,s}.pt` (`LibreYOLO9P2s-visdrone.pt` is published). FOMO weights have no download route, as in 1.5.0, and the L2CS upstream download returns 404.
- `from_fiftyone()` exports boxes only: instance masks on FiftyOne detections are not written, so an exported view trains detection, not segmentation.
- `LibreVLM("moondream-3")` does not load: the `LibreYOLO/LibreMoondream3` mirror's index names weight shards that the repository does not hold. Moondream 2 is unaffected.

### Contributors

- **François Joudelat** (fjoudelat), first-time contributor, 1 commit: training-monitor epoch numbers (#831).
- **Gordon Böer** (gboeer), first-time contributor, 29 commits: training and validating on a subset of a dataset's classes (#875), tracking over image sequences (#825), the class-count bounds check for non-OBB training and validation (#827), and RF-DETR and DINOv2 runs that no longer overwrite earlier runs (#834).
- **Javier Boix** (JBoixCampos), first-time contributor, 1 commit: DINOv2's missing-xFormers message moved to debug level (#867).
- **lwlovelmm**, first-time contributor, 1 commit: RT-DETR size detection for single-character size codes (#871).
- **massimiliano1991**, first-time contributor, 1 commit: OBB rows kept when `single_cls` collapses class ids (#838).
- **velu-97**, first-time contributor, 2 commits: classification macro precision, recall and F1 (#883).
- **Xuban Ceccon**: everything else, 550 commits.

Thanks to the people whose issues closed in this window: christuchez (#839, #870, #878, #886, #887, #899), gboeer (#808, #809, #823, #826, #828, #833), wuhaoyu010 (#872, #873, #874), thi23459112 (#756, #844), ajinkya-kulkarni (#863), ChocolatesX (#822), fjoudelat (#829), gokulkato2001 (#816), JBoixCampos (#866), MARIANOCEREDA (#830), ML-rookie (#810), tildekarthik (#805), velu-97 (#852).

### Stats

592 commits, 819 files changed, +127,443 / -3,421 lines, 105 merged pull requests (#745-#909), 51 issues closed. 136 new test modules (453 to 589). 28 new documentation pages.

```
pip install --upgrade libreyolo
```

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
