# Restoration and guided matting specialists

DDColor, HVI-CIDNet, and LaMa use LibreYOLO's existing `restore` task.
ViTMatte uses the existing `matte` task. Colorization, low-light enhancement,
and inpainting are capabilities of those families, not additional task keys;
their outputs therefore keep the standard `Results.restored` or
`Results.matte` contracts.

## Quick start

### DDColor automatic colorization

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDDColort-restore.pt")
result = model("black-and-white.jpg")
result.restored.save("colorized.png")
```

`t` is the recommended default. `l` uses the larger ConvNeXt-L encoder. Both
predict chroma at 512 square and reconstruct RGB on the source canvas with the
source image's original-resolution Lab luminance plane.

### HVI-CIDNet low-light enhancement

```python
model = LibreYOLO("LibreHVICIDNett-restore.pt")
result = model(
    "night.jpg",
    gamma=1.0,
    saturation=1.0,
    intensity=1.0,
)
result.restored.save("enhanced.png")
```

All three controls are per-call and must be positive. Their neutral value is
`1.0`, which reproduces the published generalization checkpoint's evaluation
configuration. Prediction preserves the source canvas and pads internally to
a multiple of eight.

### LaMa mask-guided inpainting

Install the ONNX Runtime extra once:

```bash
pip install "libreyolo[onnx]"
```

```python
model = LibreYOLO("LibreLaMab-restore.pt")
result = model("photo.jpg", mask="erase-mask.png")
result.restored.save("inpainted.png")
```

The image and mask must share one canvas. Every nonzero mask pixel means
"fill"; zero means preserve. LibreYOLO executes the exact embedded OpenCV Zoo
QDQ ONNX graph at 512 square and copies every unmasked source pixel back
exactly on the output canvas.

The CLI exposes the same input directly:

```bash
libreyolo predict model=lama-b source=photo.jpg mask=erase-mask.png save=true
```

### ViTMatte trimap-guided matting

> **Pretrained-weight restriction:** the Composition-1k checkpoint is
> non-commercial under Adobe's Deep Image Matting Dataset License Agreement.
> Any permitted checkpoint redistribution must include the exact Deep Image
> Matting CVPR 2017 attribution recorded in the family `NOTICE`.

```python
model = LibreYOLO("LibreViTMattes-matte.pt")
result = model("portrait.jpg", trimap="trimap.png")
alpha = result.matte.array
```

The trimap must use exactly `0/128/255` or normalized `0/0.5/1`: known
background, unknown, and known foreground. The output is a soft float32 alpha
matte on the source canvas. Known background and foreground are forced to
exactly zero and one. Inference stays at native resolution with bottom/right
padding to a multiple of 32; a guide on a different canvas is resized with
nearest-neighbor sampling. Use `trimap=...` from the CLI as well.

## Validation data

DDColor and HVI-CIDNet use the normal paired restore schema described in
`docs/dataset_schema.md`. LaMa uses the same input/target pairs plus a
same-stem binary mask directory selected by `mask_dir` in the dataset YAML.
For example, `LibreYOLO("LibreLaMab-restore.pt").val(data="inpaint.yaml")`
reads the required YAML key.
ViTMatte accepts `model.val(data="matte.yaml", trimap_dir="trimaps")` with one
same-stem guide per image. When omitted, its validator thresholds the
ground-truth alpha at 0.5 and derives a deterministic guide with fixed
15-pixel erosion/dilation (`trimap_radius=15`).

## License boundaries

Code, checkpoint, and training-data terms are separate:

| Family | Runtime code | Published checkpoint | Training-data caveat |
|---|---|---|---|
| DDColor | Apache-2.0 port | Publisher labels the selected files Apache-2.0 | ImageNet training and ImageNet-22K initialization lineage; ImageNet access terms are non-commercial research/education; the Artistic checkpoint also uses undisclosed private data and is excluded |
| HVI-CIDNet | MIT port | Publisher labels the Generalization file MIT | The named LOLv2-Synthetic source publishes no explicit dataset license |
| LaMa | MIT adapter around an unmodified Apache-2.0 OpenCV Zoo artifact | OpenCV Zoo publishes the ONNX file under Apache-2.0 | Places365-Challenge image terms are non-commercial research/education; no training images are redistributed |
| ViTMatte | Apache-2.0 Transformers port with MIT architecture lineage | **Non-commercial** under the Adobe Deep Image Matting Dataset License Agreement | Composition-1k terms restrict trained-model use and distribution to non-commercial purposes and require attribution |

The family `NOTICE` files pin every source commit, checkpoint revision, size,
SHA-256 digest, and required attribution. Converting or mirroring a checkpoint
does not make it MIT and does not erase its training-data restrictions.
