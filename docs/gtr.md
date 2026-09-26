# GTR detection

`LibreGTR` supports detection and pose with the upstream S, M, L and X
architectures.
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
adapters into dense weights. Pose is described below. The other GTR tasks
(segmentation, depth, OBB, semantic segmentation) are not implemented.

```python
model = LibreYOLO("LibreGTRs.pt")
model.train(data="dataset.yaml", epochs=30, lora=True)
```

ONNX and TorchScript export use a fixed square FP32 graph. Other export
formats and dynamic spatial shapes are not enabled. The portable recurrence
exports as an ONNX Loop rather than an unrolled graph.
GTR-S at 640px exports in about 2.2 seconds locally with 4,094 graph nodes.
This is export usability evidence, not a GPU deployment-speed benchmark.

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
single-class 17-keypoint dataset. LoRA is not supported for pose yet. ONNX and
TorchScript export produce `(pred_logits, pred_keypoints)` at a fixed 640px.

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
