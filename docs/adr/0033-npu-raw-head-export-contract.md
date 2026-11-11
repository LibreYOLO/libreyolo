# ADR 0033: Raw-head export contract for integer NPUs

## Status

Accepted.

## Context

Integer accelerators (Axelera Metis first) run convolutions well but give
every tensor one INT8 scale. A decoded YOLO-NAS graph ends in pixel-range box
coordinates and post-sigmoid scores; quantizing those costs resolution, and
the decode operators (distribution softmax, reshapes, transposes) are not
accelerated anyway.

## Decision

Exports for integer NPUs stop after the detection head's final convolutions
(`_YOLONASRawHeadsWrapper`) and LibreYOLO decodes on the host
(`libreyolo/postprocess/yolonas_raw.py`), producing exactly the tensors the
decoded export emits, so the shared backend parsers are reused unchanged.

- The artifact metadata names the contract in `output_layout`
  (`yolonas_raw_heads_v1`, `yolonas_pose_raw_heads_v1`,
  `yolonas_obb_raw_heads_v1`, `yolo9_raw_heads_v1`) and carries the decode
  constants (`strides`, `reg_max`, `grid_cell_offset`, pose offsets). YOLO9's
  `reg_max` counts the distance bins (0..reg_max-1); YOLO-NAS's is the last
  bin index.
- `output_names` lists the outputs in the order the compiled model returns
  them; compilers may reorder graph outputs, so backends map by name.
- `output_windows` (optional, `{part: [low, high]}`) marks parts whose
  logits the graph maps from `[low, high]` onto `[0, 6]` and clamps there,
  with the map folded into the final convolution so the clamp lowers to a
  fused ReLU6. Each such tensor then gets an INT8 range of `[low, high]`
  instead of its full calibrated range. YOLO-NAS-pose needs this: its class
  and distribution logits reach about -1900 at strides 16 and 32, and one
  INT8 scale over that range rounds every confident score to 0.5. Backends
  map windowed parts back before decoding and reject windows for parts the
  layout does not have.
- A layout's meaning never changes. Any change to what an output holds gets a
  new layout name (`..._v2`); backends reject layouts they do not know.
- The host decode is tested against the in-graph decode at more than one input
  size (`tests/unit/test_yolonas_raw_heads.py`).

## Consequences

Future integer backends (Hailo raw HEFs, INT8 RKNN, other NPUs) can reuse the
wrapper and decoders. The decode costs host CPU time per image; the network
part runs on the accelerator.
