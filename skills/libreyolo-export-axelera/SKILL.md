---
name: libreyolo-export-axelera
description: >-
  Run LibreYOLO models on Axelera Metis AIPUs: export with format="axelera"
  (INT8, compiled to .axm), load the *_axelera_model directory, predict and
  val on the device, and diagnose install, device and accuracy
  problems. Use when a user mentions Axelera, Metis, AIPU, Voyager SDK, .axm,
  axdevice or format=axelera, or asks whether a model can run on Axelera.
  Covers which families and tasks are supported, prerequisites, the export /
  validate / deploy recipes, failure signatures and the evidence needed before
  claiming support. For other formats see libreyolo-export-formats; for
  general usage see use-libreyolo.
---

# Run LibreYOLO on Axelera Metis

`format="axelera"` turns a LibreYOLO checkpoint into an INT8 `.axm` compiled
for the Metis AIPU, and `LibreYOLO("<dir>_axelera_model")` runs it with the
normal API. The contract and validation record live in `docs/axelera.md`;
this skill is the working recipe.

## 1. Will it work? Check before compiling

Support is per (family, task), and only what the matrix says is runnable:

```bash
libreyolo formats --family yolonas     # look at the axelera column
```

- YOLO-NAS detect (s/m/l): validated (tier in `docs/export_support.md`).
- YOLO-NAS-pose: validated for m and l (within 1.0 keypoint mAP of FP32);
  n and s are refused with their measured loss (docs/axelera.md sec 7).
- YOLO-NAS-R (obb): validated s/m/l (300 DOTA-v2.0 tiles, within 1.0 mAP50-95 of FP32). Pass `data=` (no default calibration set).
- YOLO9: blocked for now (planned next, same approach).
- RF-DETR and every transformer family: blocked. They need attention,
  data-dependent TopK/Gather and grid sampling, which the compiler does not
  place on the AIPU, and the host can only run work before or after the
  accelerated graph.
- Anything else: blocked; export raises `NotImplementedError` with the reason
  before doing any work. Do not try to force it.

## 2. Prerequisites

```bash
libreyolo checks          # shows axelera-devkit / axelera-rt versions and Metis devices
axdevice                  # Axelera's device tool: firmware, board controller, clock
```

- Export needs Linux x86_64 (or WSL2), `libreyolo[onnx]` and
  `axelera-devkit`; no device needed.
- Inference needs a Metis device, Axelera's `metis-dkms` driver and
  `axelera-rt`.
- Both packages install from Axelera's index (exact commands in
  `docs/axelera.md` section 3). They are proprietary software under Axelera's
  EULA; tell the user before installing. `LIBREYOLO_AUTOINSTALL=1` lets
  LibreYOLO install a missing one with a log line.
- Recommend a dedicated virtual environment: the SDK pins click, typer, rich,
  onnx, opencv and pillow (table in `docs/axelera.md`). Never "fix" a conflict
  by raising LibreYOLO's own dependency floors.

## 3. Export

```python
from libreyolo import LibreYOLO

path = LibreYOLO("LibreYOLONASs.pt").export(format="axelera", verify=True)
# -> weights/LibreYOLONASs-converted_axelera_model/ (named after the checkpoint; Deci weights
#    are converted to LibreYOLONASs-converted.pt first): .axm, compiler_config_final.toml,
#    metadata.yaml, and with verify=True a .parity.json
```

```bash
libreyolo export model=LibreYOLONASs.pt format=axelera verify=true
```

- INT8 is automatic. `half=True`, `quantize=16/32` and `dynamic=True` raise.
- Calibration: default `coco128.yaml` for detect and pose. For a fine-tuned
  model pass `data=` with 100 to 400 images from its own domain;
  `fraction=` picks a share of the split.
- A compile takes minutes to tens of minutes and a few GB of RAM. Run it in
  the background and do not start a second compile on the same machine.
- `verify=True` writes `<stem>.parity.json`: quantized model vs FP32 per
  output, and device vs quantized model when a device is present.

## 4. Validate

```bash
libreyolo val model=weights/LibreYOLONASs-converted_axelera_model data=coco128.yaml
```

Compare against the PyTorch checkpoint on the same data
(`libreyolo val model=LibreYOLONASs.pt data=...`). The recorded deltas per
size are in `docs/axelera.md` section 7; a much larger drop usually means
unrepresentative calibration images.

## 5. Deploy

```python
npu = LibreYOLO("weights/LibreYOLONASs-converted_axelera_model")
results = npu.predict("images/", batch=8)
for r in npu.predict("video.mp4", stream=True): ...
```

- The directory is self-contained: copy it to any Linux machine with a Metis
  device and `axelera-rt` (no `axelera-devkit` needed there).
- The input size is fixed at export; re-export for another size.
- `device=` accepts `auto`, `axelera`, `metis`, `aipu`, `cpu`; CUDA is ignored.
- `track()` exists on PyTorch models only; exported models (any format) do not
  track yet.

## 6. Failure signatures

| Symptom | Fix |
| --- | --- |
| `requires Axelera's axelera-devkit package` / `axelera-rt package` | run the printed pip command, or `LIBREYOLO_AUTOINSTALL=1` |
| `Axelera export compiles with Axelera's toolchain, which runs on Linux` | export on Linux or WSL2, copy the directory |
| `export to axelera is blocked` | the family/task is not supported (section 1) |
| `Could not run the model on an Axelera Metis device` | run `axdevice`; check the driver against the SDK version; another process may hold the device |
| `exported with a fixed NxN input` | use the exported `imgsz` or re-export |
| `is not a LibreYOLO Axelera export` | the directory came from another tool; export with LibreYOLO |
| `Axelera compilation failed` | read the compiler message and the saved error archive it names; see `docs/axelera.md#troubleshooting` |
| accuracy much lower than PyTorch | calibrate with domain images (100-400), then compare `verify=True` cosines |

Work from LibreYOLO's messages, `axdevice` and Axelera's public
documentation at docs.axelera.ai. Do not read or analyze the SDK's
installed source or internals.

## 7. Support-claim discipline

Say "this model runs on Axelera" only when one of these exists:

1. its (family, task) row is `validated` in `docs/export_support.md`, or
2. a fresh on-device `val` of the exact export, compared with the PyTorch
   checkpoint on the same data, with SDK, driver and firmware versions noted.

Compiling successfully is not evidence of accuracy.

## 8. Licensing

- LibreYOLO is MIT; the Axelera SDK is proprietary (EULA) and is never
  bundled or redistributed, nor are Axelera's docs or compiler build files.
- YOLO-NAS pretrained weights are Deci's, non-commercial and
  non-redistributable; `.axm` files compiled from them carry the same terms.
  LibreYOLO hosts none; users compile their own.

## Related

- `docs/axelera.md`: contract, validation record, troubleshooting.
- `skills/libreyolo-export-formats/`: the exporter/backend architecture.
- `skills/use-libreyolo/`: general predict / val / export usage.
- `skills/libreyolo-export-hailo/`: the other NPU path (no `format=` target).
