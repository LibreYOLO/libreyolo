# Axelera Metis export integration

- **LibreYOLO modules:** `libreyolo/export/axelera.py`, `AxeleraExporter`,
  `_YOLONASRawHeadsWrapper`, `_YOLO9RawHeadsWrapper` and `_YOLOXRawHeadsWrapper` in
  `libreyolo/export/exporter.py`,
  `libreyolo/backends/axelera.py`, `libreyolo/postprocess/yolonas_raw.py`,
  `libreyolo/models/mobilenetv4/quantize.py` (MobileNetV4 residual-stream
  equalization and bounds for INT8)
- **Origin:** original code written for this integration
- **Runtime/compiler:** `axelera-devkit` and `axelera-rt` (Voyager SDK 1.9.0),
  proprietary software under Axelera's EULA, called as user-installed optional
  dependencies; nothing from them is vendored, and LibreYOLO does not
  redistribute them, Axelera's documentation or compiler build files
- **YOLO-NAS decode:** the raw-head decoders reimplement LibreYOLO's existing
  YOLO-NAS decode (SuperGradients-derived, Apache-2.0, listed in
  `THIRD_PARTY_NOTICES.txt`), verified against it by
  `tests/unit/test_yolonas_raw_heads.py`
- **YOLO9 decode:** `decode_yolo9_raw` reimplements LibreYOLO's own YOLO9 head
  decode (`libreyolo/models/yolo9/nn.py`) with the same helpers, verified
  against it by `tests/unit/test_yolo9_raw_heads.py`; the class-tower
  activation bound and output windows are original code
- **YOLOX decode:** `decode_yolox_raw` reimplements LibreYOLO's existing
  YOLOX export decode (Megvii YOLOX-derived, Apache-2.0, listed in
  `THIRD_PARTY_NOTICES.txt`), verified against it by
  `tests/unit/test_yolox_raw_heads.py`
- **Darknet decode:** `decode_anchor_yolo_raw` reimplements LibreYOLO's
  existing Darknet export decode (public-domain Darknet cfgs and weights,
  see `libreyolo/models/darknet/`), verified against it by
  `tests/unit/test_darknet_raw_heads.py`
- **Evidence:** fake-SDK unit tests; real compilation and on-device validation
  on a Metis M.2 Max, recorded in `docs/axelera.md`
- **Evaluation data:** COCO val2017, DOTA-v2.0 and ImageNetV2 subsets used
  locally for measurement only; not in the repository
