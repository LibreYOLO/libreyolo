# ADR 0032: Flow Task Contract

## Status

Accepted.

## Context

Optical flow predicts, for every pixel of one frame, its displacement to the
next frame. The output is dense and two-channel, the input is a pair of frames
rather than one image, and the quality metrics (end-point error, outlier
rates) compare displacement vectors, not boxes, classes or single-channel maps.
No existing task fits: `depth` is single-image and single-channel, `point`
localizes objects, tracking associates detections across frames and produces
no per-pixel motion.

Every optical flow family from FlowNet to the 2026 transformers produces the
same primitive, so one contract can serve all of them. The families differ in
how many frames of context they consume, in optional side outputs (occlusion,
covisibility, uncertainty) and in preprocessing geometry; none of that changes
the public result.

## Decision

LibreYOLO defines a canonical `flow` task whose prediction primitive is
`Results.flow`, a `FlowField`: a dense `(H, W, 2)` float32 field on the
original canvas of the first frame. `flow.data[y, x]` is `(dx, dy)`, the
displacement in pixels from frame `t` to frame `t + 1`, `+x` to the right and
`+y` downwards. Vectors are expressed on the original canvas: families that
resize or pad for inference must un-pad or resize the field back and scale the
vectors by the resize ratio before returning it.

`FlowField.confidence` is an optional `(H, W)` map in `[0, 1]`. A family may
fill it with whatever it has (an occlusion or covisibility estimate, a score
derived from a predicted variance) and documents what it means; families with
nothing leave it `None`.

`flow` uses `-flow` as its filename suffix. Public aliases are `optical-flow`,
`optical_flow` and `opticalflow`; only `flow` appears in filenames and
metadata.

Sequence semantics: a single image is an error. Any multi-image source (a
list or tuple of images, a directory sorted by name, a video, a stream) is a
sequence; `N` frames produce `N - 1` results, and result `i` carries the flow
from frame `i` to frame `i + 1` with frame `i` as its `orig_img`. On videos
`vid_stride` selects the frame gap. A family that needs more than two frames
declares `FLOW_CONTEXT = (before, after)`, the extra frames it wants on each
side of the pair; the runner provides the window and repeats the edge frames
at sequence boundaries.

Flow checkpoints use `task: "flow"`, `nc: 1` and `names: {0: "flow"}` for
checkpoint-schema compatibility; the class slot does not represent a semantic
class.

Validation aligns nothing: predicted and ground-truth vectors are compared
directly on valid pixels. Reported metrics are `metrics/epe` (mean end-point
error, the Euclidean distance between predicted and true vectors, the
headline), `metrics/px1` and `metrics/px3` (the fraction of valid pixels with
an error under 1 and 3 pixels), and `metrics/fl_all` (the KITTI outlier rate,
pixels whose error exceeds both 3 pixels and 5 percent of the true magnitude).
Best-checkpoint fitness is `metrics/px1` because it increases with quality.

Ground truth is a per-frame flow file next to the frames: Middlebury `.flo`,
KITTI 16-bit PNG, or `.npy`, as described in `docs/dataset_schema.md`.

Export emits one `flow` output of shape `(B, 2, H, W)` at the export canvas
from two named inputs, `image1` and `image2`, each `(B, 3, H, W)`. Iteration
counts are baked into the graph. Exported backends stretch-resize both frames
to the canvas, resize the field back to the original canvas and multiply the
vectors by the resize ratio, matching native postprocess.

Rendering has three views, all read-only over the stored vectors. The default
is the Middlebury colour wheel (hue is direction, saturation is magnitude,
zero motion is white). `plot(arrows=True)` draws the source frame with one
yellow displacement arrow every `arrow_step` pixels, stretched by
`arrow_scale`, skipping vectors under 1 px. `plot(warp=True)` is the
correctness check: frame `t` on the left and frame `t + 1` warped back onto
it with the flow on the right, so a correct field makes both panels match.
For that, predict attaches the next frame to the payload as
`FlowField.next_frame`, a uint8 `(H, W, 3)` RGB array on the canvas of frame
`t`. `predict(save=True)` writes the chosen view (`arrows=`, `warp=`), and
`predict(save_flow=True)` also writes the raw field of every pair as a
Middlebury `.flo` file next to it, named `<frame stem>_<frame index>.flo` so
pairs stay unique across folders; `FlowField.save(path)` writes `.flo` or
`.npy` directly.

Amendment (Video Depth Anything, 2026-11): the sequence runner is shared with
per-frame sequence families of other tasks. A family that sets
`SEQUENCE_INPUT = True` gets the same source handling (ordering, `vid_stride`,
lazy reading, `stream=True`, saving), but no pairing: the runner hands
`_forward_sequence` an iterator of `(preprocessed frame, original size)` in
source order, and the family yields exactly one raw output per frame, in
order, as soon as it is final. The runner pairs output `t` with frame `t`,
calls `_postprocess` and raises if the counts differ. How many frames a family
reads ahead (a fixed window, a memory bank) is the family's business; pulling
frames lazily keeps live sources streaming. A single image is a one-frame
sequence for these families, unlike flow. The task keeps its own result
contract; flow itself is unchanged.

## Consequences

- Flow results never fabricate boxes; `Results.boxes` is `None`.
- Tiled inference, tracking, test-time augmentation, augmented validation and
  LoRA reject flow results until each has a flow-aware runtime contract.
- The task ships inference-only. `train()` on a flow family raises
  `NotImplementedError` with a clear message until a family wires a trainer.
- Hosted flow weights follow the repository rule for weights: the code must be
  permissive, the checkpoint licence is shipped verbatim, and non-commercial
  weights lead their model card with that notice.
- Future families implement `flow` without redefining dataset, results,
  validator, export or checkpoint behaviour.

## Non-Goals

This ADR does not add a sparse keypoint-matching task, scene flow, flow-based
tracking, frame interpolation or a classical (non-learned) flow family.
