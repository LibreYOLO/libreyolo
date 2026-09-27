# Kernel registry

`libreyolo/kernels/` hosts the library-wide registry of pluggable accelerated
implementations. Every op has a portable default; accelerated variants
register on top and are selected per-op by predicate. A missing optional
dependency is never an error, only a fallback.

## Layout (purpose first, backend second)

- `kernels/quant/simulate/`: fake-quantization Triton kernels. Numerics-true
  simulation with STE backward, any device. They serve QAT/QAD **and**
  simulated PTQ/`val()` inference; the enforced boundary is
  `is_finalized`, not train-vs-deploy.
- `kernels/quant/execute/`: finalized-only real-precision paths. No backward,
  real hardware: the `fp8_gemm` tensor-core GEMM (`torch._scaled_mm`), its
  fused Triton prologue/epilogue, and the packed-weight unpack kernels.
- `kernels/attention/`: attention ops shared across model families. The
  `ms_deform_attn` slot (multi-scale deformable attention) consumed by the
  Deformable-DETR-lineage families, plus `sdpa.py`, which is policy rather
  than a kernel: it records when a family may hand its attention to torch's
  fused `scaled_dot_product_attention` and provides the opt-in switch.
- The reference implementations stay in `libreyolo/quant/fake_quant.py` and
  `libreyolo/quant/packing.py`: `quant/` defines what the numbers mean,
  `kernels/` makes them fast. `packing.py` never has variants because it is
  the checkpoint contract.

## Selection

Implementations are tried newest-first; `resolve()` returns the first whose
predicate passes. Slots that may return None for a given input
(`ms_deform_attn`) then walk the remaining eligible providers so a Hub
reject does not hide Triton. `libreyolo.kernels.active()` reports the
first eligible name.

- `LIBREYOLO_KERNELS=off|reference` forces the reference implementations;
  any other value selects only implementations registered under that name.
  `LIBREYOLO_QUANT_KERNELS` is honored as a legacy alias.
- GEMM and attention slots (`fp8_gemm`, `ms_deform_attn`, `nvfp4_gemm`, ...)
  have no reference implementation. Callers must check `resolve()` returns
  non-None and keep their portable path as the fallback; exported graphs
  (ONNX, TensorRT, torch.export) always use the portable path.

## Hub kernels

Compiled kernels published on the Hugging Face Hub load at runtime through
the optional `kernels` package. Installing the extra is the opt-in:
`pip install "libreyolo[hub-kernels]"` enables them. Without the package, an
eager CUDA DETR call that no in-tree provider accepts uses the portable path
and logs one install hint. Set `LIBREYOLO_HUB_KERNELS=0` to disable them
without uninstalling and silence the hint. Nothing is vendored; artifacts are
fetched and cached by the `kernels` package. The global
`LIBREYOLO_KERNELS=off|reference` portable-path override also suppresses the
hint. A
kernel that fails to load or run disables itself for the process and falls
back to the portable path with one warning. When the installed `kernels`
release cannot resolve the pinned commit (newer releases reject SHA
revisions and validate a metadata schema older kernel repos predate), the
provider fetches the pinned snapshot directly via `huggingface_hub` and
imports the matching build variant itself — same binary, same pin. Every hub kernel is pinned to an
audited commit revision in its provider module — a moved branch on the Hub
can never change the binary that runs in-process. Bumping a pin requires a
GPU parity run of the provider's `*_matches_portable_on_cuda` test.

Current `ms_deform_attn` providers, newest-first:

- `hub` <- [`kernels-community/deformable-detr`](https://huggingface.co/kernels-community/deformable-detr)
  (Apache-2.0): the compiled CUDA multi-scale deformable attention
  forward/backward from Deformable DETR. Eligible inputs are CUDA
  fp32/fp16/bf16 in eager mode (half inputs are upcast at the call
  boundary). Training is accelerated too (the compiled backward registers
  through an autograd bridge). Active whenever the `kernels` package is
  installed, unless `LIBREYOLO_HUB_KERNELS=0`.
- `triton` (in-tree): same equation, no extra package. CUDA fp32/fp16/bf16
  inference; inputs that require grad fall through so training keeps a
  backward. On by default when Triton imports and CUDA is visible;
  `LIBREYOLO_TRITON_MSDA=0` disables it. Hub stays preferred when both
  are eligible.

  Wired into every Deformable-DETR-lineage family: RF-DETR,
  LibreDeformableDETR, LibreDINO-DETR, LW-DETR, Grounding DINO, RT-DETR,
  RT-DETRv2, D-FINE (and RT-DETRv4), DEIM (and DEIMv2), EC, and OV-DEIM.
  Families whose core carries a different layout adapt to the slot's before
  calling it; a shape that cannot be expressed in the slot's layout falls
  through to the portable path instead. Two cases do that today: a
  `num_points_list` with a different point count per level, and the
  `method='discrete'` integer-index sampling, which is a different equation.
  The EC pose core (`MSDeformAttnPose`) adapts its pre-split
  `(bs*heads, c, hw)` levels onto the slot layout and falls through when
  that reshape is impossible.

Out-of-tree compiled kernels can also ship as a `libreyolo_kernels` package,
which self-registers on import (e.g. a future CUTLASS NVFP4 GEMM for the
documented `nvfp4_gemm` slot).

## Fused attention (SDPA)

Model families written as `q @ k.T -> softmax -> @ v` can hand their attention
to `torch.nn.functional.scaled_dot_product_attention`, which dispatches to the
flash / memory-efficient / cuDNN kernels instead of materialising the score
matrix. This needs no optional dependency: it is stock torch.

Two rules decide when a family uses it.

**Graph capture never does.** Every swapped call site keeps the primitive-op
equation behind `manual_attention_required()`, which covers ONNX export
(LibreYOLO defaults to opset 13, which has no symbolic for fused SDPA) *and*
`torch.jit.trace`, the capture used by the TorchScript, CoreML and NCNN
exporters. Exported graphs are unchanged.

The dynamo-based captures are deliberately excluded. `torch.compile` lowers
SDPA better than the manual equation, so forcing primitives there would
deoptimize compiled inference — and neither predicate can single out
`torch.export`: on torch 2.11 both `is_compiling()` and `is_exporting()` are
True under plain `torch.compile` too. `torch.export` does not need the gate
anyway, because both backends decompose SDPA to core ATen before their
converter runs (Core AI through `run_decompositions()`, ExecuTorch through
`to_edge_transform_and_lower`).

This is a *looser* rule than the `ms_deform_attn` slot's, which refuses under
both dynamo predicates as well. That slot dispatches to a runtime-fetched
compiled binary that must never be captured at all and has no compiled-
inference story to protect; SDPA is stock torch either way, so only the
capture formats that care about the op spelling are gated.

**A byte-exact parity bar keeps manual math by default.** Several ports are
pinned to `max_abs_diff == 0` against a reference that itself runs manual
attention (the Swin and OWLv2 harnesses explicitly switch the reference's fused
path *off* to get there). Fused kernels accumulate in a different order, so
those families keep manual attention and expose a `fused_attn` flag:

```python
from libreyolo.kernels.attention import set_fused_attention

set_fused_attention(model)  # returns how many attention modules switched
```

The count is the number of `fused_attn` flags that moved, and every module
carrying one honors it. `LibreViT` and `LibreDeiT` also carry the flag (they
default to SDPA, following timm) so the switch turns them off too.

That trades byte-exact agreement with the family's upstream reference for the
fused kernels. On an RTX 5070 Ti under fp16 autocast, Swin window attention
(512 windows x 49 tokens x 384) goes from 1.278 ms to 0.721 ms (1.77x) and
OWLv2 vision attention (3600 tokens x 1024) from 6.483 ms to 1.735 ms (3.74x).

| Route | Families |
| --- | --- |
| SDPA by default (bar is a tolerance) | SegFormer, Depth Anything (and MoGe-2), BERT, Grounding DINO, SwinIR, PP-OCR |
| Opt-in via `set_fused_attention` (bar is byte-exact) | Swin, LibreDINO-DETR's Swin backbone, BiRefNet (and FeyNoBG), OWLv2, LW-DETR, SigLIP 2, ZipDepth, MobileSAM |

## Hardware behavior

Accelerated implementations engage only where their predicate passes; every
op always has a portable path, so no platform needs configuration:

- CPU-only and Apple Silicon (MPS): all CUDA/Triton predicates fail;
  reference implementations run as plain torch ops.
- NVIDIA CUDA: Triton kernels and eligible hub/GEMM kernels engage.
- AMD ROCm: torch reports CUDA and ROCm wheels ship Triton's AMD backend,
  so Triton kernels can engage; parity is currently only exercised on
  NVIDIA in CI. Hub kernels resolve per-variant; a missing variant simply
  falls back.

## Adding an implementation

Register a callable with the slot's reference signature and a cheap
predicate:

```python
from libreyolo import kernels

kernels.register("fake_quant_fp8", my_impl, name="mybackend", predicate=my_check)
```

Parity is gated by `tests/unit/kernels/` against shapes harvested from real
models into `tools/shapes.json`; any accelerated implementation must match
the reference exactly (forward) and to 1e-6 (STE gradients).
