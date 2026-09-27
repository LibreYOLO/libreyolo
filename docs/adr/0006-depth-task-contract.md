# ADR 0006: Depth Task Contract

## Status

Accepted.

## Context

Monocular depth estimation predicts a dense per-pixel depth map from one image.
Single-image metric depth in meters does not transfer cleanly across cameras:
focal length, sensor size, and scene scale are entangled. A model that emits
meters for one camera can silently overclaim on another camera.

LibreYOLO supports dedicated monocular-depth families with different encoders
and decoder heads. The depth task needs a family-agnostic public contract so
model families can share dataset, result, validation, and export semantics
without tying them to a detector architecture.

## Decision

LibreYOLO defines a canonical `depth` task whose prediction primitive is
`Results.depth_map`: a dense `(H, W)` float map on the original image canvas.
Values are relative inverse depth, where higher values mean closer to the
camera. No metric unit is implied.

Amendment (Marigold V2, 2026-09): `inverse_depth` remains the default encoding.
Families may explicitly declare affine-relative `depth` or `log_depth` through
`DepthMap.encoding`. These preserve native numeric values and validate in their
declared space. See ADR 0025; existing inverse-depth families are unchanged.

Training targets are plain depth maps in any dataset-consistent unit. Pixels
with `0`, negative, NaN, or inf values are invalid. Training support and loss
design are family-specific and are not defined by this contract.

Validation aligns predictions to ground truth with a per-image positive scale
and shift in inverse-depth space. Non-positive fitted scales fall back to a
median shift so inverted predictions cannot validate as perfect. Reported
metrics are AbsRel, RMSE, and delta1/2/3; best-checkpoint fitness is delta1.

Depth checkpoints use `task: "depth"`, `nc: 1`, and
`names: {0: "depth"}` for checkpoint-schema compatibility. The class slot does
not represent a semantic class.

Export, tiled inference, tracking, TTA, augmented validation, and LoRA are
explicitly rejected until each has a depth-aware runtime contract.

Amendment (ZipDepth, 2026-07): export now has that contract, following the
fixed-resolution v1 pattern from restore/matte. The exported graph emits one
`depth` output of shape `(B, 1, H, W)` at the export canvas; exported backends
stretch-resize the input to the canvas (no padding, so padded pixels cannot
leak depth context through the receptive field) and bilinearly resize the depth
map back to the original canvas with `align_corners=True`, matching native
postprocess. Native predict keeps each family's own keep-aspect preprocessing,
so exported-backend outputs on non-square images are a documented approximation
of native predict. Families opt in individually; Depth Anything V2 keeps its
family-level export block. Tiled inference, tracking, TTA, augmented
validation, and LoRA remain rejected.

## Consequences

- Depth results never fabricate boxes; `Results.boxes` is `None`.
- Metric depth requires user-side calibration against known distances.
- Hosted depth weights must be redistributable. Redistribution rests on the
  licence the publisher applied to the released checkpoint, not on clearing
  every dataset in the training mixture: a permissive checkpoint licence is
  the grant LibreYOLO relies on, and any non-commercial terms inherited from
  the training data are stated on the model card and in the download notice so
  the user can judge their own use.

  This supersedes the original wording, which additionally required the
  training data itself to permit commercial use. That bar kept MiDaS
  unhostable indefinitely despite isl-org publishing the checkpoints under
  MIT, and it does not match how LibreYOLO treats non-commercial weights
  elsewhere (SegFormer, the NVIDIA Source Code License tier), which are hosted
  with the licence shipped verbatim.
- Future model families can implement `depth` without redefining dataset,
  results, validator, or checkpoint behavior.
