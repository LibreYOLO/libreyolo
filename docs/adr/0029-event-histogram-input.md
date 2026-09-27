# ADR 0029: Prepared event histogram detection inputs

Status: accepted for implementation, validation evidence reported separately.

## Decision

Keep the `detect` task, existing verbs, YOLO labels and Results contract. Add
an explicit numerical input profile for YOLO9 and RF-DETR only. The profile
is self-contained in checkpoints and ONNX metadata. No SDK is a dependency.

Use two HWC count planes, positive then negative, and a producer-declared
window duration. Normalize by one dataset-wide count saturation level before
resize. Preserve the existing per-family box geometry, with zero padding for
YOLO9 and direct stretch for RF-DETR. Use only flips for training augmentation.
Replace the actual input convolution; do not encode counts into RGB images.

The numerical helpers live in `utils/event_histogram.py` so ONNX-only clients
can import them without the PyTorch training stack. `data/event_histogram.py`
assembles the training loader. The shared image dataset retains its label and
cache machinery, with scoped NumPy discovery/decoding. See
[Input profiles](../input_profiles.md) for the public contract.

## Initialization and export

Scratch uses a random two-channel convolution. RGB transfer assigns each new
kernel channel 1.5 times the original channel mean, preserving the sum of
responses when all input planes are identical. This is an initialization
policy, not an accuracy claim. Keep all other compatible weights.

FP32 ONNX is the initial export. RF-DETR histogram export bakes positional
embeddings at the requested graph resolution and restores live model state
after export. A second interpolation from the default RGB resolution changes
learned positions and fails parity on trained weights.

## Provenance

The implementation is original code based on LibreYOLO's own loaders,
transforms, model wrappers, serialization and export contracts. No event
fork, incompatible implementation, or prior research-session code was used.

Producer API verification used OpenEB revision
`9003b5416676e78ba994d912087486cfa94fae73`, its `licensing/LICENSE_OPEN`, and the
Apache-2.0 headers on `histo_processor.h` and
`event_preprocessor_python.cpp`. These files were inspected to verify the
optional public API, not copied or adapted into the library. PEDRo data and
its CC-BY-4.0 dataset license are separate from implementation provenance;
no dataset or learned weights are bundled.
