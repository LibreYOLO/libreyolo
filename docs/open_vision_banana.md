# Open Vision Banana

Open Vision Banana (OVB, FunAILab, [paper](https://arxiv.org/abs/2611.15844))
is FLUX.2 [klein] Base fine-tuned to answer perception tasks with an image. The
prompt names the task. The output image encodes the answer: segmentation as
flat colours, depth as grey, surface normals as RGB. One set of weights also
keeps image editing and text-to-image.

The architecture is unchanged, only the transformer weights are new. LibreYOLO
loads it with diffusers' `Flux2KleinPipeline` and adds no network code. What
you write is the prompt and a few lines of NumPy to read the colours back.
This page has both.

| Size | Alias | Mirror | Peak VRAM | One image, 28 steps |
| --- | --- | --- | --- | --- |
| 4b | `open-vision-banana` | `LibreYOLO/LibreOpenVisionBanana4b` | 31 GB | 49 s |

Measured on one RTX 6000 Ada (48 GB), 1 megapixel, bf16. The mirror holds
the paper's non-zero-shot 4B checkpoint (trained with the benchmarks' training
splits). Weights are under the FLUX Non-Commercial License: no commercial or
production use.

## Install

```bash
pip install "libreyolo[gen]"
```

## First call

```python
from libreyolo import LibreGen, SAMPLE_IMAGE

model = LibreGen("open-vision-banana")      # 4b; downloads the mirror once
DEPTH = (
    "Estimate a dense relative inverse-depth map for the input image. Produce exactly one output value "
    "for every input pixel at the same pixel location, with closer points brighter and farther points "
    "darker. Encode the result as neutral grayscale (R=G=B) using one consistent mapping over the "
    "complete image. Preserve object silhouettes, thin structures, occlusion boundaries, smooth surfaces, "
    "and small depth changes. Do not alter the scene or add texture, shading, color, text, labels, "
    "legends, borders, or objects; return only the registered depth map."
)
result = model(DEPTH, source=SAMPLE_IMAGE, seed=0)
result.generated.save("depth.png")
rgb = result.generated.array                # (H, W, 3) uint8
```

`source` takes a path, URL, `PIL.Image` or BGR NumPy array. The output keeps
the photo's aspect ratio, capped at one megapixel (the sample photo comes back
at 1152x896). Knobs: `steps` (28) and
`guidance` (3.5), the paper's settings. Without `source` it is a normal
text-to-image model:

```python
model("an empty tram at a cobblestone stop on a rainy night", imgsz=1024, seed=0).generated.save("tram.png")
```

Use `offload=True` to keep idle parts off the GPU, and
`LibreGen("open-vision-banana", checkpoint_dir="path/to/folder")` for a local
copy or another OVB checkpoint (the 9B, the zero-shot ones) saved as a
diffusers pipeline.

## Prompts and readouts

The model was trained on many phrasings of each task. These are FunAILab's
quickstart prompts; keep their structure and change the classes, colours or
description.

Every readout starts from the output resized back to the photo:

```python
import numpy as np
from PIL import Image

def to_photo_size(result, photo, mask=False):
    """Output RGB at the photo's size: NEAREST keeps mask colours exact."""
    image = Image.fromarray(result.generated.array)
    size = Image.open(photo).size if isinstance(photo, str) else photo.size
    return np.asarray(image.resize(size, Image.NEAREST if mask else Image.BILINEAR))
```

### Semantic segmentation

List the classes with one saturated RGB colour each. Unlisted pixels come out
black.

```python
import json

legend = {"building": [255, 8, 232], "sky": [2, 204, 254], "vegetation": [10, 255, 2],
          "water": [8, 4, 249], "person": [255, 0, 13], "road": [255, 249, 23]}

def semantic_prompt(legend):
    return (
        "[TASK: SEMANTIC_SEGMENTATION] Generate a semantic segmentation visualization of the input image, "
        f"using this color mapping: {json.dumps(legend, separators=(', ', ': '))} "
        "Color each pixel of a listed class with its color and leave everything else black. "
        "Output only the segmentation visualization image."
    )

def read_classes(rgb, legend):
    """Class index per pixel: 0 is background, i + 1 is the i-th legend class."""
    palette = np.array([[0, 0, 0], *legend.values()], dtype=np.float32)
    distance = ((rgb[:, :, None, :].astype(np.float32) - palette) ** 2).sum(-1)
    return distance.argmin(-1)

rgb = to_photo_size(model(semantic_prompt(legend), source=SAMPLE_IMAGE, seed=0), SAMPLE_IMAGE, mask=True)
classes = read_classes(rgb, legend)          # (H, W) int, names = ["background", *legend]
```

**Legend ensembling** is how the paper gets its best segmentation scores: run
the same classes with different colour assignments and take the per-pixel
majority. On Cityscapes it adds about 8 mIoU (65.6 to 73.9 for 9B zero-shot,
20 passes). It costs one full generation per pass.

```python
def segment(model, photo, names, passes=5, seed=0):
    rng = np.random.default_rng(seed)
    colours = [list(map(int, c)) for c in legend.values()]   # any distinct saturated colours
    votes = None
    for p in range(passes):
        order = rng.permutation(len(names))
        pass_legend = {names[i]: colours[j] for j, i in enumerate(order)}
        rgb = to_photo_size(model(semantic_prompt(pass_legend), source=photo, seed=seed + p), photo, mask=True)
        index = read_classes(rgb, pass_legend)              # in pass_legend order
        label = np.where(index == 0, 0, np.concatenate([[0], order + 1])[index])
        one_hot = np.eye(len(names) + 1, dtype=np.int32)[label]
        votes = one_hot if votes is None else votes + one_hot
    return votes.argmax(-1)                                 # 0 background, i + 1 is names[i]

classes = segment(model, SAMPLE_IMAGE, list(legend), passes=5)
```

### Referring segmentation

```python
def referring_prompt(text):
    return (
        "[TASK: REFERRING_SEGMENTATION] Segment the region that best matches this "
        f'description: "{text}". Paint the matching region red [220, 30, 30] and '
        "leave everything else black [0, 0, 0]. Output only the segmentation mask image."
    )

rgb = to_photo_size(model(referring_prompt("the person in mid-air"), source=SAMPLE_IMAGE, seed=0), SAMPLE_IMAGE, mask=True)
mask = read_classes(rgb, {"target": [220, 30, 30]}) == 1     # (H, W) bool
```

### Instance segmentation

One class per call. Each instance gets its own colour, ordered left to right.

```python
COLOURS = ("blue lime red sky_blue dark_green pink apricot mint wine magenta navy green yellow brown white olive "
           "lavender teal orange hot_pink plum gray purple beige steel_blue cyan royal_blue sea_green rust mustard "
           "mauve maroon").split()

def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

def instance_prompt(noun):
    order = ", ".join(f"the {ordinal(i + 1)} {noun} is {c.replace('_', ' ')}" for i, c in enumerate(COLOURS))
    return (
        f"[TASK: INSTANCE_SEGMENTATION] This image shows segmentation masks for {noun} from the input image. "
        "The background is set to black color [0, 0, 0]. Each " + noun + " instance is represented by a solid mask, "
        "and a different color is used for each mask. Instances are numbered by where their mask first "
        "appears in a left-to-right scan of the image (ties broken top to bottom). "
        + order[0].upper() + order[1:] + "."
    )

def read_instances(rgb, radius=60, min_fraction=0.001):
    """Instance id per pixel (0 is background); groups each mask's soft-edged shades."""
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    ids = np.zeros(len(pixels), np.int32)
    todo = pixels.sum(1) > 90                      # not background
    while todo.sum() >= min_fraction * len(pixels):
        colours, counts = np.unique(pixels[todo].round(-1), axis=0, return_counts=True)
        centre = colours[counts.argmax()]
        member = todo & (((pixels - centre) ** 2).sum(1) < radius**2)
        if member.sum() < min_fraction * len(pixels):
            break
        ids[member] = ids.max() + 1
        todo &= ~member
    return ids.reshape(rgb.shape[:2])

rgb = to_photo_size(model(instance_prompt("palm tree"), source=SAMPLE_IMAGE, seed=0), SAMPLE_IMAGE, mask=True)
instances = read_instances(rgb)              # the sample photo: 2 palm trees
```

The model does not always follow the colour order. That is fine: each
instance only needs a distinct colour. `read_instances` takes the most common
colour, gathers every pixel within `radius` of it as one instance, and
repeats; `min_fraction` drops edge halos.

### Depth

The `DEPTH` prompt above. Brighter is closer. The output is relative inverse
depth (disparity) up to an unknown scale and shift:

```python
disparity = to_photo_size(model(DEPTH, source=SAMPLE_IMAGE, seed=0), SAMPLE_IMAGE).mean(-1) / 255.0
```

To compare with metric depth, fit scale and shift on known points first, as
the paper does.

### Surface normals

```python
NORMALS = (
    "Create a dense camera-space unit surface-normal map registered to the input. Use +x image-right, "
    "+y image-up, +z toward the camera and encode (x,y,z) as R=(1-x)/2, G=(1+y)/2, B=(1+z)/2. "
    "Preserve planes, curved surfaces, thin structures, and sharp orientation boundaries; "
    "return only the RGB normal map."
)
rgb = to_photo_size(model(NORMALS, source=SAMPLE_IMAGE, seed=0), SAMPLE_IMAGE) / 255.0
normals = np.stack([1 - 2 * rgb[..., 0], 2 * rgb[..., 1] - 1, 2 * rgb[..., 2] - 1], -1)
normals /= np.linalg.norm(normals, axis=-1, keepdims=True).clip(1e-6)
```

### Image editing

Keep the tag and write the edit as a plain instruction:

```python
model("[TASK: IMAGE_EDITING] Make it a snowy winter evening with warm street lights.",
      source=SAMPLE_IMAGE, seed=0).generated.save("winter.png")
```

## What to expect

Paper numbers, non-zero-shot checkpoints (the 4B is mirrored here):

| | 4B | 9B |
| --- | --- | --- |
| Cityscapes mIoU, 20-pass legend vote (one pass) | 76.4 (69.2) | 75.7 (69.7) |
| RefCOCOg cIoU | 65.6 | 69.2 |
| NYUv2 depth delta1 | 98.7 | 99.0 |
| NYUv2 normals mean error (deg) | 13.21 | 13.51 |

- It is slow: every answer is a 28-step diffusion run, and the best
  segmentation needs several. Use it when one model for many tasks matters
  more than speed.
- Masks have soft, slightly shifted edges after resizing; depth and normals
  are smooth but not metric.
- Not supported: training, export, `val`. The CLI `libreyolo generate`
  covers text-to-image only (`model=open-vision-banana prompt=...`).

## Where the weights come from

The mirror is a complete diffusers pipeline. The transformer is FunAILab's
non-zero-shot checkpoint from `FunAILab/Open-Vision-Banana` cast from float32
to bfloat16, the precision it runs in; the rest is BFL's FLUX.2 [klein] Base
pipeline, byte-identical. Outputs match FunAILab's own quickstart (their fp32
transformer loaded in bf16) bit for bit on the same seed. The 9B, the zero-shot
checkpoints and the LoRA adapters stay on FunAILab's repo; save one as a full
pipeline and pass it with `checkpoint_dir=`.

Weights: FLUX Non-Commercial License. Section 2(e) asks for content filtering
or review of generated images before they are shared. Code: OVB's training
code is Apache-2.0; LibreYOLO ports none of it.
