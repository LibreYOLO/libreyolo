# Event histogram input contract

YOLO9 and RF-DETR detection accept prepared two-polarity event histograms.
This is an input representation of the existing `detect` task. Event decoding,
accumulation, camera drivers and intensity reconstruction are external.

## Dataset

Use NumPy `.npy` files and the normal YOLO detection text labels:

```text
images/train/frame.npy -> labels/train/frame.txt
images/val/frame.npy   -> labels/val/frame.txt
```

Each array has shape `(height, width, 2)`: positive activity first, negative
activity second. Values are finite nonnegative counts, stored as integer or
floating-point numbers. Object, complex, boolean, signed activity, CHW, RGB,
NaN and infinity inputs are rejected. Files load with `allow_pickle=False`.
Empty event windows are zero arrays with the full sensor dimensions.

```yaml
path: /absolute/path/to/histograms
train: images/train
val: images/val
names:
  0: person
input_profile:
  format: event_histogram
  layout: HWC
  polarity: positive_negative
  encoding: counts
  scale: 16.0
  window_us: 40000
```

Every `input_profile` field is required. `scale` is a positive finite count
saturation level. `window_us` is the positive integer accumulation duration in
microseconds. The producer owns window boundaries, polarity conversion and
annotation association; an array cannot prove these facts. Do not infer a
window boundary from the first or last event, or treat missing annotation
coverage as a negative sample. Keep recordings in their publisher-defined
splits. Save a conversion manifest separately from the library.

`channels: 2` is optional; any other value conflicts with this profile. Split
paths can be directories, text manifests or lists, as for RGB data. Histogram
YAMLs discover `.npy` files only. Native COCO JSON input is outside this profile.
Labels remain `class cx cy width height`, normalized to the original canvas.

## Numerical and geometry rules

Preprocessing computes `clip(counts / scale, 0, 1)` in float32 before bilinear
resize. It never applies RGB/BGR conversion, ImageNet normalization, division
by 255, per-image normalization or polarity exchange.

YOLO9 preserves aspect ratio and uses zero padding. Its existing saved
`letterbox_pad` selects top-left or center placement. The shared letterbox
geometry also undoes predictions. RF-DETR stretches directly to its requested
canvas; its patch/window divisibility requirements still apply. All paths use
the same numerical preprocessing. Results boxes stay on the original canvas.

Training supports horizontal and vertical flips. Color changes, mosaic,
mixup, affine/projective augmentation, random crops, multi-scale RF-DETR
training, LoRA, distillation and quantization are outside this initial profile.
Unsupported explicit options raise; the histogram defaults disable them.
Training uses one device and a fixed positive batch. CUDA graph training and
auto-batch are not supported. Prediction supports arrays, files, lists,
directories and `stream=True` over those finite sources; TTA, tiling, videos,
live cameras and raw event streams are not supported.

## Training and use

These commands use an already prepared dataset named `data.yaml`. Scratch
training uses the architecture named by `model` without downloading weights:

```bash
libreyolo train model=LibreYOLO9t.pt pretrained=False data=data.yaml epochs=100 batch=4 imgsz=128 workers=0
libreyolo train model=LibreRFDETRn.pt pretrained=False data=data.yaml epochs=100 batch=4 imgsz=128 workers=0
libreyolo val model=path/to/best.pt data=data.yaml split=val workers=0
libreyolo predict model=path/to/best.pt source=images/val/frame.npy save=True
libreyolo export model=path/to/best.pt format=onnx imgsz=128 dynamic=False simplify=False
libreyolo predict model=path/to/best.onnx source=images/val/frame.npy save=True
```

The corresponding Python verbs are unchanged:

```python
import numpy as np
from libreyolo import LibreYOLO, LibreYOLO9

model = LibreYOLO9(None, size="t", device="cpu")
run = model.train(data="data.yaml", pretrained=False, epochs=100,
                  batch=4, imgsz=128, workers=0)
model = LibreYOLO(run["best_checkpoint"], device="cpu")
result = model.predict(np.load("images/val/frame.npy", allow_pickle=False), save=True)
```

Loading an RGB checkpoint and training with a histogram dataset adapts the
input convolution to two channels: each channel receives `1.5 * mean(R,G,B)`
of each original kernel, with the original bias retained. All other compatible
parameters are preserved. Scratch training randomly initializes the input
convolution. Neither initialization is a pretrained event detector. The saved
`input_initialization` records `rgb_mean` or `random`.

`best.pt` and `last.pt`, `model.save()` and ONNX metadata preserve the full
profile. Reload requires no dataset configuration for prediction. Training and
validation reject a dataset whose profile differs from the model. ONNX accepts
FP32 without embedded NMS; its graph input is normalized NCHW float32 with two
channels. Other export formats and quantized export are not supported.
Prepared-data prediction requires no Prophesee SDK. In an ONNX-only
installation without PyTorch, use `libreyolo.backends.onnx.OnnxBackend` directly;
the `LibreYOLO` factory requires PyTorch.

## Visualization and producer integration

`predict(save=True)` draws detections on an RGB preview: positive red, negative
blue, overlap magenta, zero black. Validation sample plots use the same preview.
For label inspection, generate the preview separately:

```python
from PIL import Image
from libreyolo.utils.event_histogram import visualize_histogram

rgb = visualize_histogram("images/val/frame.npy", scale=16.0)
Image.fromarray(rgb).save("histogram-preview.png")
```

The preview is never fed back into the detector. Prophesee/OpenEB is an optional
producer; use its public Core preprocessing API, convert its output layout and
polarity order explicitly, and undo any producer normalization before saving
counts. See the [Core preprocessing guide](https://docs.prophesee.ai/stable/guides/events_preprocessing.html)
and [OpenEB installation documentation](https://github.com/prophesee-ai/openeb).
An SDK runtime, RAW decoding and a live camera have not been validated by this
change. Available local evidence and its limits are recorded in the PR.
