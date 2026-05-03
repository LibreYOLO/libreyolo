# iOS CoreML Smoke Test

This package verifies that LibreYOLO's CoreML export can compile and run on a
physical iPhone.

```bash
python validation/ios_coreml_smoke/prepare_resources.py
xcodebuild test \
  -project validation/ios_coreml_smoke/CoreMLSmoke.xcodeproj \
  -scheme CoreMLSmoke \
  -destination 'platform=iOS,id=<DEVICE_UDID>' \
  -allowProvisioningUpdates \
  -allowProvisioningDeviceRegistration \
  CODE_SIGN_IDENTITY='Apple Development'
```

The XCTest bundle copies the exported `.mlpackage` resources, compiles them on
the device with `MLModel.compileModel(at:)`, feeds a canonical YOLOX preprocessed
sample image, and checks the raw and embedded-NMS outputs against macOS CoreML
reference scores.
