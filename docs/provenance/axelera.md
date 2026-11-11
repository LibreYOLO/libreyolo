# Axelera Metis export integration

- **LibreYOLO modules:** `libreyolo/export/axelera.py`, `AxeleraExporter` and
  `_YOLONASRawHeadsWrapper` in `libreyolo/export/exporter.py`,
  `libreyolo/backends/axelera.py`, `libreyolo/postprocess/yolonas_raw.py`
- **Origin:** original code written for this integration
- **Runtime/compiler:** `axelera-devkit` and `axelera-rt` (Voyager SDK 1.9.0),
  proprietary software under Axelera's EULA, called as user-installed optional
  dependencies; nothing from them is vendored, and LibreYOLO does not
  redistribute them, Axelera's documentation or compiler build files
- **YOLO-NAS decode:** the raw-head decoders reimplement LibreYOLO's existing
  YOLO-NAS decode (SuperGradients-derived, Apache-2.0, listed in
  `THIRD_PARTY_NOTICES.txt`), verified against it by
  `tests/unit/test_yolonas_raw_heads.py`
- **Evidence:** fake-SDK unit tests; real compilation and on-device validation
  on a Metis M.2 Max, recorded in `docs/axelera.md`
- **Evaluation data:** COCO val2017 and DOTA-v2.0 subsets used locally for
  measurement only; not in the repository
