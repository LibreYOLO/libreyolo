# GTR detection

`LibreGTR` supports detection with the upstream S, M, L and X architectures.
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
Upstream mosaic/mixup is not implemented. Multi-scale training defaults off.
LoRA and the other GTR tasks are not implemented. Unsupported optimizer and
scheduler choices are rejected instead of being silently ignored.

ONNX and TorchScript export use a fixed square FP32 graph. Other export
formats and dynamic spatial shapes are not enabled. The portable recurrence
exports as an ONNX Loop rather than an unrolled graph.
GTR-S at 640px exports in about 2.2 seconds locally with 4,094 graph nodes.
This is export usability evidence, not a GPU deployment-speed benchmark.

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

Reproduce the opt-in real-weight checks without network access:

```sh
GTR_UPSTREAM=/path/to/GTR GTR_CHECKPOINTS=/path/to/detection/weights \
  pytest tests/unit/test_gtr_parity.py -m 'unit and external_data'
GTR_CHECKPOINTS=/path/to/detection/weights \
  pytest tests/unit/test_gtr_export.py -m 'unit and external_data'
```
