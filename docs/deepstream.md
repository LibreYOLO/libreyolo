# DeepStream export

`deepstream=True` on the ONNX export produces artifacts ready for NVIDIA
DeepStream's `nvinfer` element (Jetson and x86 dGPU):

```python
from libreyolo import LibreYOLO9

model = LibreYOLO9("libreyolo9s.pt", size="s")
model.export(format="onnx", deepstream=True)
```

This writes an ONNX graph and an `nvinfer` config next to each other. Tasks
with class labels also get a labels file. For the detection example above:

- `libreyolo9s.onnx`: the detection graph with a single output tensor of
  shape `(batch, num_detections, 6)`, rows `[x1, y1, x2, y2, score,
  class_id]` in network-input pixel coordinates.
- `config_infer_primary_libreyolo9s.txt`: an `nvinfer` configuration with
  the family's preprocessing constants, class count, clustering (NMS)
  thresholds, and parser wiring filled in.
- `libreyolo9s_labels.txt`: one class name per line.

DeepStream builds the TensorRT engine from the ONNX on first run and caches
it next to the model. Detection configs use the parser builder's required
`model_b{batch}_gpu0_{precision}.engine` cache name; other tasks keep a
model-specific engine filename.

## The parser library

`nvinfer` needs a custom bounding-box parser for this output layout. The
generated config targets `NvDsInferParseYolo` from the MIT-licensed
[DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)
project. Build it once per device, **inside the DeepStream container**.
A host-side `CUDA_VER=12.8 make` is not enough: on the validated
`nvcr.io/nvidia/deepstream:8.0-samples-multiarch` image, `/usr/local/cuda-12`
is a stub (`fatal error: crt/host_defines.h: No such file or directory`)
and the image ships `libcublas.so.12` but not the unversioned
`libcublas.so` that `-lcublas` needs.

```bash
git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo.git

# Prefer a toolkit that actually carries the CUDA headers. On the 8.0
# image that is cuda-12.5, not the cuda-12 stub.
CUDA_DIR=$(readlink -f /usr/local/cuda)
[ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
  CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
             while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

# Give the linker the unversioned sonames it asks for.
mkdir -p /tmp/cudalibs
for lib in cublas cublasLt cudart; do
  real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
  ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
done
export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"

make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo \
  CUDA_VER="${CUDA_DIR##*/cuda-}"
```

Adjust `custom-lib-path` in the generated config to the built
`libnvdsinfer_custom_impl_Yolo.so`. No NMS is embedded in the ONNX graph;
the parser applies the confidence threshold and DeepStream's clustering
stage (`cluster-mode=2`) suppresses using `nms-iou-threshold`.

## Supported tasks and families

**Detection** (`network-type=0`, needs the parser library above):
yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox,
yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr,
rtdetrv2, rtdetrv4.

DETR heads and yolo9_e2e's one-to-one head emit at most one prediction per
object, so their configs set `cluster-mode=4`: DeepStream must not run NMS
over them or it merges distinct detections. Anchor and grid heads get
`cluster-mode=2` with `nms-iou-threshold`.

**Classification** (`network-type=1`, no parser library needed):
mobilenetv4, convnext, efficientnetv2, resnet, dinov2. The graph emits
softmax probabilities, which is what `classifier-threshold` expects. Set
`process-mode=2` and `operate-on-gie-id` in the generated config to run one
as a secondary classifier behind a detector.

**Instance segmentation** (`network-type=3`, needs the *seg* parser build):
rfdetr, dfine, ec. Rows are the detection row followed by that instance's
mask, flattened at `(netH / 4, netW / 4)` (the resolution the seg parser
hardcodes) as probabilities for `segmentation-threshold`. This parser is a
separate MIT project and a separate build, using the same CUDA_DIR and
LIBRARY_PATH setup as the detector parser:

```bash
git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git
make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
  CUDA_VER="${CUDA_DIR##*/cuda-}"
```

LibreYOLO's seg families export per-query masks directly, so the graph only
resizes and sigmoids them. No RoI pooling and no custom TensorRT plugin are
involved, unlike prototype-coefficient heads. RTMDet-Ins and YOLO9 are
excluded because their segmentation export is blocked in LibreYOLO itself.

**Semantic segmentation** (`network-type=2`, no parser library needed):
pidnet, eomt, dinov2, lingbotvision. The graph emits `(C, H, W)` per-class
probabilities; `nvinfer` applies `segmentation-threshold` and produces the
class map. `segformer` is excluded because it is not wired to the shared
semantic export contract and cannot export to ONNX in any format.

EoMT's semantic head already emits probabilities rather than logits, so its
graph skips the softmax the other families need. Only the `l` semantic
checkpoint is published; `s` and `b` semantic weights do not exist.
DINOv2 classification has no published checkpoint at all, so use it with
your own fine-tuned weights.

**Raw-tensor tasks** (`network-type=100` with `output-tensor-meta=1`, no
parser library): DeepStream has no post-processor for these, so the graph's
native outputs pass through untouched and the application decodes them from
the tensor metadata. Multi-output graphs are fine; every output layer
reaches the metadata with the same output names and dynamic axes as a
regular ONNX export. No labels file is written.

| Task | Families |
|---|---|
| Depth | depth_anything, zipdepth |
| Pose | yolo9, yolonas, rfdetr, ec |
| Restoration | nafnet, quicksrnet, realesrgan, swinir |
| Matting | birefnet |
| Gaze | l2cs |

Gaze is a head-only contract: each input is one face crop, so run it as a
secondary GIE (`process-mode=2` plus `operate-on-gie-id`) behind a face
detector.

Families whose native preprocessing cannot be expressed by `nvinfer`'s
scalar `net-scale-factor` (per-channel std: rfdetr, ec, DINO-backboned
deimv2 sizes, rtmdet, picodet, and every classification family) have the
normalization baked into the exported graph; the generated config feeds the
graph the matching raw input space, so no manual preprocessing
configuration is needed. The semantic families normalize inside their own
forward, so their graphs take plain `[0, 1]` RGB and add nothing.

## Preprocessing approximations

Known deviations from the native Python pipelines, documented here for
benchmark accounting:

- Letterbox families (yolo9, yolox, yolonas, rtmdet, yolo2/3/4/7) pad with
  gray natively; `nvinfer` pads black. YOLO9-family configs set
  `symmetric-padding=1` when the checkpoint's `letterbox_pad` is `center`.
- yolonas detection natively resizes the longest side to 636 inside its 640
  canvas; `nvinfer`'s `maintain-aspect-ratio` uses the full 640. Yolonas pose
  already uses the full 640, BGR input, and bottom/right padding; its generated
  config preserves those task-specific settings.
- Classification natively resizes the shortest side then centre-crops;
  `nvinfer` stretches the frame or object ROI to the network input. Expect
  small differences on tightly cropped subjects.
- EoMT natively runs sliding-window tiles for semantic segmentation; the
  exported graph is a single stretched canvas, which is faster and less
  accurate.
- pidnet emits a class map at 1/8 of the input resolution and
  lingbotvision at 1/16; DeepStream upsamples the class map for display.

The ONNX parity gate feeds already-preprocessed tensors, so it validates graph
outputs but cannot detect a wrong `nvinfer` color order or padding policy.
Config-level tests cover the declared settings; end-to-end DeepStream
validation is still required.

For exact-parity workloads, validate on your data before deploying; all
other math is parity-tested against each family's native postprocess.

## Options

- `conf` / `iou` (defaults 0.25 / 0.45) seed `pre-cluster-threshold` and
  `nms-iou-threshold` in the generated config.
- `dynamic=True` exports a dynamic batch axis; set `batch-size` in the
  config to the engine batch you want DeepStream to build.
- `half=True` marks the config `network-mode=2` (fp16 engine build).
- `deepstream=True` and `nms=True` are mutually exclusive.
