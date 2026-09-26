# GTR detection

`LibreGTR` supports detection, instance segmentation, pose, oriented boxes,
monocular depth and semantic segmentation with the upstream architectures.
The GTR source pin is `782e737efe2e6437ac537fbdcee089673d3376c1` (MIT).
The upstream weight repository explicitly declares MIT. Required inherited
Apache-2.0 notices are retained in `libreyolo/models/gtr/NOTICE`.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreGTRs.pt")
results = model.predict("image.jpg")
model.train(data="dataset.yaml", epochs=30, device="cuda")
model.export(format="onnx", dynamic=False)
```

Canonical names download converted checkpoints from `LibreYOLO/LibreGTR{s,m,l,x}`
at pinned revisions. Their learned tensors match the authors' published EMA
parameters exactly. Local
`gtr_s_coco.pth` files also auto-convert. An explicit converter is available as
`weights/convert_gtr_weights.py`. Learned tensor names and values are preserved.
Each hosted repository includes the upstream MIT license and checkpoint provenance.

## Execution and training

CPU and MPS use a differentiable PyTorch recurrence. CUDA uses the MIT
`flash-linear-attention` package when installed, otherwise the same portable
recurrence. Install `flash-linear-attention==0.5.0` separately on a compatible
CUDA system for that optional path. Neither Triton nor transformers is required
to import the model. The upstream custom CUDA/TensorRT plugin is not vendored.
The portable recurrence prioritizes compatibility; it does not reproduce the
published GPU latency.

Inputs are square, at least 160 pixels, and divisible by 32. Preprocessing
uses square resize, RGB 0-1 and ImageNet normalization. Default resolution is
640. Postprocessing uses DETR top-K selection without NMS.

Training uses upstream group-DETR matching and MAL/FGL/DDF losses through the
shared D-FINE trainer. FP32 is the default. Backbone LR multipliers and weight
decay follow the size-specific upstream recipe, with identical Python/CLI
defaults. Quadratic warmup, the six-epoch flat phase, cosine decay and the final
two-epoch plateau follow the upstream 30-epoch recipe. Shorter runs bound these
phases to fit their budget.

Training transforms receive the original-resolution image and apply square
resize, ImageNet normalization, horizontal flip, photometric distortion,
zoom-out and IoU cropping. The final two epochs disable the strong transforms.
The upstream Mosaic and batch MixUp are included: during the first
`mosaic_epochs` (default 6) epochs, each sample becomes a four-tile mosaic with
probability `mosaic_prob` (default 0.5), built from a 50-image per-worker cache
of half-resolution tiles and followed by a random affine (`degrees`,
`translate`, `mosaic_scale`). Mosaic samples skip zoom-out and IoU cropping.
Each batch is blended with its shifted copy with probability `mixup_prob`
(default 0.5) at a 0.45-0.55 ratio, keeping both label sets. Set
`mosaic_prob=0` or `mixup_prob=0` to turn either off. `mosaic_epochs` is a
Python training argument only. Multi-scale training defaults off.
Unsupported optimizer and scheduler choices are rejected instead of being
silently ignored.

`lora=True` fine-tunes with LoRA adapters on the backbone q/k/v projections
and the decoder layers while the backbone base stays frozen (see
[lora.md](lora.md)); it needs `pip install "libreyolo[lora]"` and works from
Python and the CLI. Adapter checkpoints reload directly and export merges the
adapters into dense weights. Detection LoRA does not apply to depth (see
[Depth](#depth)). The other GTR tasks are described below.

```python
model = LibreYOLO("LibreGTRs.pt")
model.train(data="dataset.yaml", epochs=30, lora=True)
```

ONNX and TorchScript export use a fixed square FP32 graph. Other export
formats and dynamic spatial shapes are not enabled. The portable recurrence
exports as an ONNX Loop rather than an unrolled graph.
GTR-S at 640px exports in about 2.2 seconds locally with 4,094 graph nodes.
This is export usability evidence, not a GPU deployment-speed benchmark.

## Instance segmentation

`task="segment"` loads the upstream GTRSeg model: the detector plus a per-query
mask head fed by the highest-resolution encoder level
(`configs/seg/coco_seg_finetune`). Canonical names are
`LibreGTR{s,m,l,x}-seg.pt`, converted from `seg/gtrseg_{s,m,l,x}_coco.pth` in
the same MIT weight repository with EMA tensors unchanged; the raw upstream
files also auto-convert. Masks come out as logits at a quarter of the input
resolution and are thresholded at zero after bilinear upsampling, as upstream.

```python
model = LibreYOLO("LibreGTRs-seg.pt")
results = model.predict("image.jpg")  # results.masks
model.train(data="dataset.yaml", epochs=30)  # YOLO polygon labels
```

Training reuses the EC segmentation data path (square resize, polygon
rasterization, ImageNet normalization) with the GTR recipe: grouped matching,
MAL/box/FGL/DDF losses plus point-sampled mask BCE and Dice (weights from
`gtrseg_base.yml`), the GTR schedule and backbone LR multipliers. Mosaic and
MixUp are not applied to segmentation. `lora=True` works as for detection and
keeps the mask head trainable. Upstream initializes segmentation from the whole
COCO detector; the same start is available as an explicit transfer,
`LibreGTR("LibreGTRs.pt", size="s", task="segment",
allow_detect_to_segment_transfer=True)` or
`libreyolo train model=LibreGTRs.pt task=segment ...`, and the mask head then
starts untrained. ONNX and TorchScript export the raw logits, boxes and mask
logits with the same fixed-shape FP32 constraints as detection.

Segmentation evidence:

- All four checkpoints convert with identical tensors and load strictly. At
  640px on CPU they match the pinned upstream GTRSeg graph exactly
  (logits, boxes and mask logits, maximum absolute difference 0) with the same
  portable operator substitutes as the detection check.
- GTR-S on a 200-image COCO val2017 subset (the coco1000 validation split,
  polygons from `instances_val2017.json`, one polygon per instance, crowd
  regions excluded): GTR-S box mAP50-95 0.545, mask mAP50-95 0.493, mask
  mAP50 0.698; GTR-X box 0.605, mask 0.551, mask mAP50 0.773. Upstream
  reports mask AP 45.0/49.8 and AP50 67.9/74.2 (S/X) on full val2017. This is
  a subset sanity check, not a reproduction.
- GTR-S ONNX and TorchScript exports reproduce PyTorch predictions on a COCO
  image (same classes, score difference below 3e-6, mask IoU 1.0).
- CPU segment training runs end to end, including LoRA and the CLI
  detect-to-segment transfer. Starting from GTR-S detect weights, 90 plain
  AdamW steps on one 13-instance COCO image raise the mean best-query mask IoU
  from 0.085 to 0.62, so the fresh mask head learns. Short runs with the
  default EMA (tau 2000) still validate with near-initial EMA weights, so an
  untrained mask head scores zero mask mAP for the first few thousand steps.

## Depth

`task="depth"` loads the upstream `gtrdepth_{s,m,l,x}` models: the GTR
backbone and encoder with a DPT depth head (upstream's adaptation of the
Depth-Anything-V2 metric head, Apache-2.0), pretrained on upstream's mixed
metric-depth corpus with a log-depth head (metres, `exp(clamp(logit, -4, 5))`).
There is no NYU fine-tune; upstream reports NYU Depth V2 after a per-image
log-affine fit with six test-time views.

```python
model = LibreYOLO("LibreGTRs-depth.pt")
result = model.predict("image.jpg")[0]
inverse_depth = result.depth_map  # higher is closer; 1 / value is metres
```

`Results.depth_map` follows the LibreYOLO depth contract (ADR 0006): relative
inverse depth on the original canvas. The graph emits the exact reciprocal of
upstream's metre output, so `1 / depth_map` recovers it; the metre scale holds
only for cameras and scenes like the training data. ImageNet normalization is
inside the graph, and native preprocessing stretch-resizes to a square
`imgsz` (default 640), as upstream validation does. Exported backends use the
shared cv2 stretch resize, so exported predictions on downscaled images are an
approximation of native ones; given the same input tensor, ONNX matches within
5e-6 and TorchScript exactly.

`val()` uses the shared depth validator (per-image scale and shift fitted in
inverse-depth space), not upstream's log-affine, six-view, two-tile protocol,
so its numbers are not comparable to the upstream table. Fine-tuning uses the
upstream SILog loss and `configs/depth/gtrdepth_base.yml` recipe (AdamW 2e-4,
backbone at 0.1x, warmup, flat then cosine to 0.1x, EMA 0.999, clipping 0.1,
FP32) on the shared depth dataset; supply depth in metres to keep the metric
scale. LoRA, TTA, tiled inference and tracking are rejected for depth.

Depth evidence:

- All four real checkpoints load strictly, and at 640px the converted models
  match the pinned upstream graph exactly (max abs difference 0.0 on
  `pred_depth`) with the same portable operators as the detection check.
- On 9 NYU Depth V2 validation images (metres, stretch 640, shared
  validator): delta1 0.706 / 0.762 / 0.808 / 0.843 and AbsRel 0.172 / 0.155 /
  0.141 / 0.127 for S / M / L / X. This is a pipeline check, not a benchmark.
- Converted checkpoints are `LibreYOLO/LibreGTR{s,m,l,x}-depth`, with download
  revisions pinned in `LibreGTR.HF_TASK_REVISIONS`.

## Pose

`LibreGTR{s,m,l,x}-pose.pt` are COCO person keypoint models (17 keypoints) with
the same backbone and encoder as detection and a DETRPose-style keypoint
decoder. They load from `LibreYOLO/LibreGTR{s,m,l,x}-pose`; local upstream
`gtrpose_{s,m,l,x}_coco.pth` files also convert, and
`weights/convert_gtr_weights.py` handles both tasks.

```python
model = LibreYOLO("LibreGTRs-pose.pt")
results = model.predict("image.jpg")  # results[0].keypoints: (N, 17, 3)
model.train(data="coco8-pose.yaml", epochs=10, allow_download_scripts=True)
```

The decoder reuses LibreYOLO's ECPose port, whose architecture and tensor
names match upstream GTR's pose decoder, with one GTR-specific layer: the
keypoint position embedding stays on the attention value, the residual path
and the gated cross-attention input, as in upstream. Postprocessing is
DETR-style top-K without NMS; boxes are the keypoint extents and keypoint
visibility is reported as 1.

Training uses the ECPose DETRPose recipe (Hungarian matching with VFL,
keypoint L1 and OKS losses, GO union matching and contrastive keypoint
denoising) with GTR's optimizer settings: AdamW at 5e-4, backbone LR multiplier
0.05 (S/M) or 0.005 (L/X), weight decay 1e-4 (S/M) or 1.25e-4 (L/X), a
500-iteration linear warmup then a constant learning rate, FP32, and 92 (S/M) or
74 (L/X) default epochs. Upstream's PoseMosaic, MixUpCopyPaste and zoom-out
augmentations are not reproduced; the ECPose flip, color and affine transforms
are used instead. Pose training requires the native 640px input and a
single-class 17-keypoint dataset. `lora=True` adapts the backbone q/k/v and the
pose decoder layers' Linears (see [lora.md](lora.md)); adapter checkpoints
reload and export merges them. A 3-epoch CPU LoRA fine-tune of GTR-S on
coco8-pose kept keypoint mAP50-95 at 0.73 to 0.75 (0.746 before training)
while validation loss fell from 102.5 to 90.7, and its merged ONNX and
TorchScript exports match PyTorch within 4.3e-4 px. ONNX and TorchScript
export produce `(pred_logits, pred_keypoints)` at a fixed 640px.

## Validation evidence

- All four real EMA checkpoints strictly load without missing learned tensors.
- At 640px, all four models match the pinned upstream graph exactly for logits
  and boxes on CPU. Both graphs use the same portable attention/normalization
  substitutes. This establishes graph and key-mapping parity, not parity with
  the upstream Triton or custom CUDA kernels.
- The portable recurrence and its gradients agree with an independent explicit
  causal-sum formulation in unit tests.
- All four sizes pass real-weight ONNX and TorchScript reload and public
  detection parity at the default 640px on CPU. The ONNX gate uses a stable
  log-sigmoid expression to
  avoid underflow in the legacy exporter's log(sigmoid(x)) lowering.
- CPU training completes and saves checkpoints. A 30-epoch, two-image synthetic
  square fixture is memorized; the reloaded best checkpoint reaches 0.9731 box
  IoU against its training target. This is a pipeline smoke check, not a
  real-data convergence result.
- Resume restores saved training settings before applying explicit overrides;
  image size, batch, learning rate, EMA and schedule settings are regression-tested.
- A clean-directory canonical download and the HTTP UI inference path render
  an annotated image with a detection summary.

Still required before performance or production-readiness claims: native FLA
and custom-kernel parity on CUDA, COCO AP reproduction, RF1/RF5 training,
real-data convergence and deployment-speed measurements. True multi-rank DDP
has not been validated. Native MPS prediction was smoke-tested
at 160px and 640px. GPU, COCO and RF1 checks are explicitly deferred to the
v1.6.0 release validation.

Pose evidence:

- All four upstream EMA pose checkpoints load strictly and convert with
  learned tensors unchanged.
- At 640px on CPU, all four match the pinned upstream pose graph with the same
  portable-operator substitution. Maximum absolute differences are at most
  1.9e-6 for logits and 8.9e-7 for keypoints. The shared ECPose `Integral`
  sums elementwise products instead of calling `F.linear` (an MPS backward
  workaround), so agreement is to float32 rounding rather than bit-exact.
- On `coco8-pose` validation, keypoint mAP50-95 is 0.746 (S), 0.814 (M),
  0.806 (L) and 0.795 (X). Eight images is a smoke check, not a COCO result.
- A 3-epoch CPU fine-tune of GTR-S pose on `coco8-pose` lowered validation loss
  from 100.6 to 88.7, kept keypoint mAP50-95 at 0.73-0.75, and the best
  checkpoint reloads as a pose model.
- ONNX and TorchScript exports of all four pose sizes reproduce the PyTorch
  scores, with keypoints within 2.5e-4 pixels.

Reproduce the opt-in real-weight checks without network access:

```sh
GTR_UPSTREAM=/path/to/GTR GTR_CHECKPOINTS=/path/to/detection/weights \
  pytest tests/unit/test_gtr_parity.py -m 'unit and external_data'
GTR_CHECKPOINTS=/path/to/detection/weights \
  pytest tests/unit/test_gtr_export.py -m 'unit and external_data'
GTR_UPSTREAM=/path/to/GTR GTR_POSE_CHECKPOINTS=/path/to/pose/weights \
  pytest tests/unit/test_gtr_pose_parity.py -m 'unit and external_data'
```

## Oriented boxes (DOTA)

`task="obb"` runs the upstream DOTA v1.0 oriented-box models. Upstream publishes
S and X weights only; its M and L OBB configs use a different, unreleased plain
ViTAdapter backbone, so LibreYOLO builds S and X.

```python
model = LibreYOLO("LibreGTRs-obb.pt")
result = model.predict("aerial.jpg")
result.obb.xywhr  # (cx, cy, w, h, theta) in pixels, theta in [0, pi)
```

- Canonical names `LibreGTR{s,x}-obb.pt` download from `LibreYOLO/LibreGTR{s,x}-obb`.
  Raw upstream `gtrobb_{s,x}_dota.pth` files auto-convert with EMA tensors unchanged.
- Input is a fixed 1024px square. Images are resized to fit without distortion,
  padded at the bottom and right with zeros (the DOTA split convention),
  converted to RGB 0-1 and ImageNet-normalized. Decoding is NMS-free top-K.
- Angles use the long-edge convention: `w >= h` and `theta` in `[0, pi)`, the same
  contract as RT-DETRv2 OBB.
- Class ids follow the upstream DOTA v1.0 order (`plane`, `baseball-diamond`,
  `bridge`, `ground-track-field`, `small-vehicle`, `large-vehicle`, `ship`,
  `tennis-court`, `basketball-court`, `storage-tank`, `soccer-ball-field`,
  `roundabout`, `harbor`, `swimming-pool`, `helicopter`). Datasets labeled in a
  different order, such as Ultralytics' DOTA8, need their label ids remapped
  before `val()`.
- ONNX and TorchScript export a fixed 1024px FP32 graph; exported models predict
  and validate through the same decoding. GTR-S OBB ONNX export takes about three
  minutes on CPU.
- `train()` fine-tunes on YOLO OBB datasets with the upstream DOTA recipe
  (`configs/obb/dota_finetune`): MAL, L1 and KLD losses with Chamfer/KLD
  Hungarian matching over three query groups and oriented denoising; AdamW at
  5e-4 with backbone LR 0.36x (S) or 0.032x (X), weight decay 1e-4 (S) or
  1.25e-4 (X); 2000-iteration quadratic warmup then a flat LR; 20 (S) or 30 (X)
  epochs; EMA, clipping 0.1, FP32. Augmentation is upstream's: one random
  horizontal, vertical or diagonal flip (`flip_prob`, 0.75) and a random
  rotation (probability 0.5, uniform within `degrees`, 180; a multiple of 90
  degrees when the image holds `storage-tank` or `roundabout`), on the square
  canvas. Upstream trains on pre-split 1024px tiles; LibreYOLO resizes and
  pads arbitrary images first, as at inference. A dataset with a different
  class count rebuilds the heads; `lora=True` adapts the backbone q/k/v and the
  decoder layers as for detection, and export merges the adapters.

```python
model = LibreYOLO("LibreGTRs-obb.pt")
model.train(data="dota.yaml", epochs=20)          # YOLO OBB labels
model.train(data="dota.yaml", epochs=20, lora=True)
```

Evidence, CPU:

- Both checkpoints strictly load, and their logits and boxes match the pinned
  upstream graph exactly (max abs diff 0.0) at 1024px with the same portable
  attention substitution as detection. Decoded rotated boxes and scores match
  upstream's `OBBPostProcessor` exactly (`tests/unit/test_gtr_obb_parity.py`).
- The publisher reports DOTA-v1.0 test AP50 of 80.0 (S) and 81.3 (X); LibreYOLO
  has not reproduced these.
- On the 4 DOTA8 val tiles (label ids remapped to the upstream order), S reaches
  mAP50 1.000 / mAP50-95 0.525 and X 1.000 / 0.657. This is a pipeline check on
  a tiny sample that may overlap upstream training data, not a DOTA benchmark.
- ONNX and TorchScript exports of S reproduce PyTorch predictions (max abs diff
  9e-4 px and 6e-8) and the same DOTA8 metrics.
- Training: on a fixed batch the ported criterion and matcher reproduce the
  pinned upstream `OBBGTRCriterion` exactly (all 33 loss terms, max abs diff
  0.0; `tests/unit/test_gtr_obb_train.py`). Three CPU epochs from GTR-S on the
  4 DOTA8 train tiles (batch 2) raise DOTA8 val mAP50-95(OBB) from 0.525 to
  0.619 at the best epoch, with mAP50 0.98 to 1.0; with `lora=True` it
  reaches 0.579, leaves the frozen backbone base unchanged, and the reloaded
  checkpoint and its TorchScript export give the same metrics. A dataset with
  a different class count rebuilds the heads, and `resume=True` continues from
  `last.pt`. These are pipeline checks, not DOTA fine-tuning results.

Reproduce the parity check without network access:

```sh
GTR_UPSTREAM=/path/to/GTR GTR_CHECKPOINTS=/path/to/weights \
  pytest tests/unit/test_gtr_obb_parity.py -m 'unit and external_data'
```

## Semantic segmentation

`LibreYOLO("LibreGTR{s,m,l,x}-sem.pt")` loads the upstream Cityscapes
checkpoints (`semseg/gtrsemseg_*_cityscapes.pth` at the pinned weight
revision, EMA tensors unchanged) into the same `LibreGTR` class with
`task="semantic"`. The network is the GTR backbone, a one-level stride-8
encoder and an FCN head (conv3x3, BN, ReLU, dropout, conv1x1) over the 19
Cityscapes train IDs; 255 is ignore.

```python
model = LibreYOLO("LibreGTRs-sem.pt")
mask = model.predict("street.jpg")[0].semantic_mask
model.val(data="cityscapes.yaml")
model.train(data="my_semantic.yaml", epochs=30)
```

Geometry follows upstream evaluation. The canvas is 1024x2048. Images are
letterboxed into it (top-left, grey pad) and the network averages the logits
of overlapping 1024px square windows at a 768px stride, then crops the padding
and resizes to the source image. A Cityscapes frame therefore runs exactly as
upstream's slide inference: three windows, no resizing. A square input whose
side is divisible by 32 runs in one pass; inputs shorter than 1024px are
rescaled for the windows and rescaled back. `predict`, `val` and exported
backends share this letterbox, and ImageNet normalization happens inside the
network (the input is RGB in [0, 1]).

ONNX and TorchScript export a fixed 1024x2048 FP32 graph that contains the
three windows. ONNX export skips onnxsim by default because it spends about
15 minutes folding the recurrence Loops on CPU without shrinking the graph;
pass `simplify=True` to run it anyway. GTR-S exports in about 8 seconds.

Training uses the shared semantic trainer with the upstream recipe: AdamW,
base LR 5e-4, backbone LR multiplier 0.3 (S), 0.36 (M) or 0.24 (L/X), weight
decay 1e-4 (S/M) or 1.25e-4 (L/X) and none on norm/BN/bias, 2000-iteration
quadratic warmup, 6 flat epochs, cosine decay to half the LR, EMA 0.9999,
gradient clipping 0.1, FP32, 30 epochs, batch 8 and 1024px square crops.
Augmentation is flip, torchvision photometric distortion and large-scale
jitter (long side fit to 1024 times 1 to 4, then a crop), through the shared
semantic dataset. It differs from upstream in the grey image padding and in
not re-sampling crops dominated by one class. The head and patch embedding use
BatchNorm; upstream trains with SyncBN over a global batch of 8, and much
smaller single-process batches give noisy batch statistics.

Semantic validation evidence:

- All four checkpoints strictly load. On CPU with the portable operators, all
  four match the pinned upstream graph exactly (max abs diff 0.0) for a
  single 1024px window and for upstream's own `slide_inference` over a
  1024x2048 input.
- GTR-S predicts sensible maps on the bundled photos (sky, building, person,
  vegetation, sidewalk).
- GTR-S ONNX and TorchScript exports reload and reproduce the PyTorch mask
  with 100% pixel agreement on a 3072x1194 photo.
- `val()` on two photos labeled with the model's own `predict()` output gives
  99.5% pixel accuracy; the residue is the canvas-resolution comparison.
- A two-epoch CPU fine-tune from GTR-S trains, validates, saves and reloads.
  It is a pipeline check, not a convergence result.

Not yet measured: Cityscapes mIoU (no Cityscapes copy was available), GPU
speed, and real-data fine-tuning convergence.

```sh
GTR_UPSTREAM=/path/to/GTR GTR_SEM_CHECKPOINTS=/path/to/semseg/weights \
  pytest tests/unit/test_gtr_sem_parity.py -m 'unit and external_data'
```
