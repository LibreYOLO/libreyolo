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

All converted LibreYOLO `.pt` files must satisfy
[`docs/checkpoint_schema.md`](../docs/checkpoint_schema.md). In v1.0 that means
every converted checkpoint carries `model`, `schema_version`,
`libreyolo_version`, `model_family`, `size`, `task`, `nc`, `names`, and `imgsz`
at the top level.

## Automatic conversion (flagship models)

The two flagship families convert automatically. When `LibreYOLO("...")` is
pointed at an upstream YOLO9 or RF-DETR checkpoint, it is converted to a v1.0
LibreYOLO checkpoint on the fly, written next to the source under the canonical
`Libre<FAMILY><size>[-task].pt` name, and then loaded:

```python
from libreyolo import LibreYOLO

LibreYOLO("v9-t.pt")           # -> writes LibreYOLO9t.pt, then loads it
LibreYOLO("rf-detr-nano.pth")  # -> writes LibreRFDETRn.pt, then loads it
```

Class count and names are read from the upstream head, so fine-tuned (non-COCO)
checkpoints convert correctly. The shared YOLO9 remapping lives in
[`libreyolo/models/yolo9/convert.py`](../libreyolo/models/yolo9/convert.py) and
the runtime orchestration in
[`libreyolo/models/autoconvert.py`](../libreyolo/models/autoconvert.py); the
YOLO9 script below is a thin CLI wrapper over the same logic. The non-flagship
families still require their explicit `convert_*.py` scripts.

## Conversions

### D-FINE

Script: [`convert_dfine_weights.py`](convert_dfine_weights.py)

Nature of the conversion:
- unwrap the upstream checkpoint layout
- keep parameter names unchanged
- add LibreYOLO metadata required by schema v1.0

This is a metadata-wrap conversion. There is no model-specific key remapping.

### DEIMv2

Script: [`convert_deimv2_weights.py`](convert_deimv2_weights.py)

Nature of the conversion:
- unwrap the upstream checkpoint layout
- keep parameter names unchanged
- add LibreYOLO metadata required by schema v1.0

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

### RF-DETR

No script — RF-DETR converts automatically (see *Automatic conversion* above).

Nature of the conversion:
- extract the model `state_dict` from the upstream checkpoint, stripping
  `module.`/`model.`/`_orig_mod.` prefixes and dropping non-tensor training
  state (optimizer, EMA, the embedded `argparse.Namespace`)
- keep parameter names unchanged — the LibreYOLO native port vendors the
  RF-DETR component graph, so upstream keys remain loadable
- map the COCO 91-class arch head to LibreYOLO's contiguous COCO-80 interface
  (fine-tuned heads keep their own class count)
- add LibreYOLO metadata required by schema v1.0

This is a metadata-wrap conversion. The only reason it cannot be loaded
directly by the safe loader is the embedded `argparse.Namespace`, which the
auto-conversion path strips.

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

### PICODET

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

This is a light structural adaptation: every learned tensor maps 1-to-1,
no DFL injection or auxiliary-head dropping needed. Round-trip
bit-equivalence is verified by `tests/unit/test_picodet_parity.py`.
