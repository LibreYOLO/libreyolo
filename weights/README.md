# Weight Conversion

Weight conversion is not one uniform operation across model families.

Some upstream checkpoints already use the same parameter names as LibreYOLO and
only need LibreYOLO metadata around the raw `state_dict`. Others need key
renaming, key dropping, or fixed tensor injection before they can load into the
local implementation.

This folder keeps family-specific conversion scripts, plus shared helpers in
[`_conversion_utils.py`](_conversion_utils.py) for the repeated plumbing:
- repo-root imports
- checkpoint loading
- common state-dict extraction
- metadata wrapping
- saving

## Conversions

### D-FINE

Script: [`convert_dfine_weights.py`](convert_dfine_weights.py)

Nature of the conversion:
- unwrap the upstream checkpoint layout
- keep parameter names unchanged
- add LibreYOLO metadata: `model_family`, `size`, `nc`, `names`

This is a metadata-wrap conversion. There is no model-specific key remapping.

### DEIMv2

Script: [`convert_deimv2_weights.py`](convert_deimv2_weights.py)

Nature of the conversion:
- unwrap the upstream checkpoint layout
- keep parameter names unchanged
- add LibreYOLO metadata: `model_family`, `size`, `nc`, `names`

This is a metadata-wrap conversion. The LibreYOLO native implementation vendors
the DEIMv2 component graph so upstream parameter names remain loadable.

### RT-DETR HGNetv2

Script: [`convert_rtdetr_hgnetv2_weights.py`](convert_rtdetr_hgnetv2_weights.py)

Nature of the conversion:
- unwrap the EMA checkpoint
- remap a small set of encoder and decoder keys
- drop tensors that exist in the upstream v2 checkpoint but not in LibreYOLO's
  RT-DETR implementation
- save a flat converted `state_dict`

This is a light structural adaptation, not just metadata wrapping.

### YOLOv9

Script: [`convert_yolo9_weights.py`](convert_yolo9_weights.py)

Nature of the conversion:
- load one of the supported upstream checkpoint layouts
- translate numbered YOLO layer indices into LibreYOLO semantic module names
- remap sublayer names for ELAN, RepNCSPELAN, AConv, ADown, SPP, and detection
  heads
- skip unsupported auxiliary-head weights
- inject fixed DFL weights
- save a flat converted `state_dict`

This is the heaviest conversion in this folder because the upstream naming
scheme and module structure differ substantially from LibreYOLO's.

### PicoDet

Script: [`convert_picodet_weights.py`](convert_picodet_weights.py)

Nature of the conversion:
- unwrap a Bo396543018/Picodet_Pytorch checkpoint
- remap top-level prefixes: `bbox_head.* -> head.*`,
  `neck.trans.trans.* -> neck.trans.*`
- flatten `backbone.<stage>_<i>.*` (Bo's per-stage `setattr` naming) into
  `backbone.blocks.<flat>.*` for our `nn.ModuleList` layout
- unwrap mmcv's `ConvModule` inside SE layers (`*.se.conv{1,2}.conv.X
  -> *.se.conv{1,2}.X`)
- add LibreYOLO metadata
- optionally emit the 5-file HuggingFace upload bundle via `--hf-bundle`

This is a light structural adaptation: every learned tensor maps 1-to-1,
no DFL injection or auxiliary-head dropping needed. Round-trip
bit-equivalence is verified by `tests/unit/test_picodet_parity.py`.
