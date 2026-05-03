# CoreML Validation Report

Date: 2026-04-29

Issue: [#90 Add CoreML support](https://github.com/LibreYOLO/libreyolo/issues/90)

## Summary

CoreML export is implemented as Apple `.mlpackage` output using the direct
PyTorch trace to `coremltools.convert` path.

Validated working:

- YOLOX CoreML export on macOS: fp32, fp16, and embedded CoreML NMS.
- YOLO9 CoreML export on macOS: fp32 and embedded CoreML NMS.
- YOLOX CoreML export on a physical iPhone over Wi-Fi: raw and embedded-NMS
  packages both compile and run on-device.
- The iPhone YOLOX scores match the macOS CoreML reference to about `3e-7`.

Known gaps:

- RT-DETR PyTorch inference works, but CoreML conversion currently fails in
  `coremltools` because the converted graph contains a rank-6 tensor in
  `sampling_offsets_1`; Core ML only supports tensors up to rank 5.
- RF-DETR was not validated in this checkout because `weights/rf-detr-nano.pth`
  is not present. Embedded CoreML NMS is intentionally blocked for RF-DETR with
  `NotImplementedError`.

## Test Environment

| Item | Value |
| --- | --- |
| macOS | 26.3.1 |
| Machine | arm64 |
| Python | 3.12.13 |
| PyTorch | 2.11.0 |
| coremltools | 9.0 |
| Xcode | 26.4.1, build 17E202 |
| iPhone | iPhone 13 mini, iPhone14,4 |
| iPhone OS | iOS 26.3.1, build 23D771330a |
| iPhone transport | Wi-Fi / CoreDevice local network tunnel |

`coremltools` warns that PyTorch 2.11.0 is newer than its latest tested PyTorch
version, which is 2.7.0.

## macOS Results

### Automated Tests

| Command | Result |
| --- | --- |
| `.venv-libreyolo/bin/python -m pytest tests/unit/test_export_coreml.py -q` | `16 passed` |
| `.venv-libreyolo/bin/python -m pytest tests/e2e/test_coreml_roundtrip.py -q -m coreml` | `5 passed, 1 skipped` |
| `.venv-libreyolo/bin/python -m pytest -q` | `149 passed, 416 deselected`; unit suite only because repo default is `-m unit` |

The e2e skip is the RF-DETR NMS test, because `weights/rf-detr-nano.pth` is not
present.

### Model-Family Sweep

These checks exported a temporary `.mlpackage`, loaded it back through
`LibreYOLO(...)` on macOS, and ran inference on the bundled sample image.

| Model | Export | Result | PyTorch detections / top score | CoreML detections / top score | Delta vs PyTorch |
| --- | --- | --- | --- | --- | --- |
| YOLOX nano | fp32 | Passed | `4 / 0.836616873741` | `3 / 0.836616218090` | `0.000000655651` |
| YOLOX nano | fp16 | Passed | `4 / 0.836616873741` | `3 / 0.831334590912` | `0.005282282829` |
| YOLOX nano | embedded NMS | Passed | `4 / 0.836616873741` | `3 / 0.836616218090` | `0.000000655651` |
| YOLO9 tiny | fp32 | Passed | `3 / 0.925528109074` | `3 / 0.925528109074` | `0.000000000000` |
| YOLO9 tiny | embedded NMS | Passed | `3 / 0.925528109074` | `3 / 0.925528109074` | `0.000000000000` |
| RT-DETR r18 | fp32 | Failed conversion | `8 / 0.967821240425` | N/A | N/A |
| RT-DETR r18 | embedded NMS | Failed conversion | `8 / 0.967821240425` | N/A | N/A |
| RF-DETR nano | embedded NMS | Skipped / unsupported | N/A | N/A | N/A |

RT-DETR failure:

```text
ValueError: Core ML only supports tensors with rank <= 5.
Layer "sampling_offsets_1", with type "reshape", outputs a rank 6 tensor.
```

## iPhone Results

The iPhone test used the hosted XCTest project in
`validation/ios_coreml_smoke`. It ran against the physical iPhone over Wi-Fi.
Xcode reported `device_isWireless = 1`.

Command:

```bash
xcodebuild test \
  -project validation/ios_coreml_smoke/CoreMLSmoke.xcodeproj \
  -scheme CoreMLSmoke \
  -destination 'platform=iOS,id=00008110-001640392278801E' \
  -destination-timeout 180 \
  -allowProvisioningUpdates \
  -allowProvisioningDeviceRegistration \
  CODE_SIGN_IDENTITY='Apple Development'
```

Result:

```text
Executed 2 tests, with 0 failures
** TEST SUCCEEDED **
```

Exact score comparison:

| iPhone test | iPhone top score | macOS CoreML reference | Delta |
| --- | --- | --- | --- |
| YOLOX embedded NMS | `0.836616516113` | `0.836616218090` | `0.000000298023` |
| YOLOX raw | `0.836616514551` | `0.836616218090` | `0.000000296461` |

This is not bit-for-bit identical, but it is numerically the same for practical
use. The difference is normal floating-point/runtime variance.

The temporary `CoreMLSmokeApp` was removed from the iPhone after validation.
`Fooding` was removed earlier to free a free-provisioning app slot.

## How CoreML Export Works

Plain version:

1. LibreYOLO loads the PyTorch model.
2. The exporter wraps the model so CoreML always receives a normal RGB image.
3. PyTorch traces the wrapped model at the model input size.
4. `coremltools.convert` converts the trace into an Apple ML Program.
5. The result is saved as a `.mlpackage`.
6. Optional: `nms=True` wraps the model with Apple's CoreML Non-Maximum
   Suppression stage, so the package returns filtered `confidence` and
   `coordinates` outputs.

The exported package targets iOS 15 / macOS 12 or newer.

## How To Export

Install CoreML support:

```bash
pip install 'libreyolo[coreml]'
```

From this checkout, use the local environment:

```bash
.venv-libreyolo/bin/python -m pip install -e '.[coreml]'
```

Basic export:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("weights/LibreYOLOXn.pt")
model.export(
    format="coreml",
    output_path="LibreYOLOXn.mlpackage",
)
```

fp16 export:

```python
model.export(
    format="coreml",
    output_path="LibreYOLOXn_fp16.mlpackage",
    half=True,
)
```

Export with embedded CoreML NMS:

```python
model.export(
    format="coreml",
    output_path="LibreYOLOXn_nms.mlpackage",
    nms=True,
)
```

Choose CoreML compute units:

```python
model.export(
    format="coreml",
    output_path="LibreYOLOXn_cpu.mlpackage",
    compute_units="cpu_only",
)
```

Accepted `compute_units` values:

- `all`
- `cpu_and_gpu`
- `cpu_and_ne`
- `cpu_only`

## How To Run A CoreML Package On macOS

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXn.mlpackage")
results = model("path/to/image.jpg")

print(results.boxes.conf)
print(results.boxes.xyxy)
```

## How To Re-run The iPhone Smoke Test

Prepare the resources:

```bash
.venv-libreyolo/bin/python validation/ios_coreml_smoke/prepare_resources.py
```

Unlock the iPhone, keep it awake, and run:

```bash
xcodebuild test \
  -project validation/ios_coreml_smoke/CoreMLSmoke.xcodeproj \
  -scheme CoreMLSmoke \
  -destination 'platform=iOS,id=00008110-001640392278801E' \
  -destination-timeout 180 \
  -allowProvisioningUpdates \
  -allowProvisioningDeviceRegistration \
  CODE_SIGN_IDENTITY='Apple Development'
```

Expected success output:

```text
testEmbeddedNMSPackageCompilesAndPredictsOnPhysicalIPhone passed
testRawCoreMLPackageCompilesAndPredictsOnPhysicalIPhone passed
Executed 2 tests, with 0 failures
** TEST SUCCEEDED **
```
