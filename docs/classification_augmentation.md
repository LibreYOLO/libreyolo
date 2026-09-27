# Classification augmentation

Image-classification training uses the ImageFolder pipeline in
`libreyolo/data/augment/classify.py`. It is the classification counterpart of
the per-family detection recipes in the same package. This page is the
contract for its knobs: names, defaults, where each one applies, and how the
`no_aug_epochs` tail behaves. Which model family honours which knob is
declared in `libreyolo/data/augment/spec.py` and pinned by unit tests; the CLI
warns when an explicitly set knob is ignored by the selected family.

## Knobs

All knobs are `TrainConfig` fields, so they work identically from Python
(`model.train(data=..., knob=value)`) and both CLI grammars
(`libreyolo train knob=value` or `--knob value`). Defaults reproduce the
historical pipeline exactly; nothing changes unless a knob is set.

| Knob | Default | Applies to | Effect |
|---|---|---|---|
| `scale` | `0.5` (that is `(0.5, 1.0)`) | train | `RandomResizedCrop` area range. A float is the lower bound; a `(min, max)` pair is explicit. |
| `flip_prob` (CLI alias `fliplr`) | `0.5` | train | Horizontal-flip probability. `0` removes the op. |
| `flipud` | `0.0` | train | Vertical-flip probability. `0` removes the op. |
| `auto_augment` | `None` | train, strong | One of `randaugment`, `autoaugment`, `augmix`. Runs in PIL space before `ToTensor`. |
| `erasing` | `0.0` | train, strong | `RandomErasing` probability in `[0, 1)`, applied after normalization. |
| `mixup` | `0.0` | train, strong | Batch MixUp probability (soft labels). On a classification model the CLI `mixup` is this knob; on detection models it is `mixup_prob`. |
| `cutmix` | `0.0` | train, strong | Batch CutMix probability (soft labels). One op runs per batch, so `mixup + cutmix` must be at most 1; larger sums are rejected. |
| `crop_pct` | family value (`0.875` when the family has none) | eval | Shorter-side resize ratio before `CenterCrop`; `resize = floor(imgsz / crop_pct)`. Also a `val()` argument. |

Train pipeline, in order: `RandomResizedCrop(imgsz, scale)`,
`RandomHorizontalFlip(flip_prob)`, `RandomVerticalFlip(flipud)`, the
`auto_augment` policy, `ToTensor`, `Normalize`, `RandomErasing(erasing)`.
Ops whose knob is off are omitted, not inserted with probability zero.

Eval pipeline: `Resize(floor(imgsz / crop_pct))`, `CenterCrop(imgsz)`,
`ToTensor`, `Normalize`. Families with a native square resize (SigLIP2, PE)
skip the center crop unless `crop_pct` is set explicitly. `val()`, INT8
calibration and exported backends take this pipeline from the model
(`_get_eval_transform`), so they preprocess exactly like `predict()`. `crop_pct` changes
train-time and `val()` evaluation only; export runtime metadata keeps the
family's native value.

`hsv_prob`, `degrees`, `translate`, `shear`, `perspective`, `mosaic` and
`mosaic_scale` are detection knobs. Classification ignores them and the CLI
says so when they are set explicitly.

## The `no_aug_epochs` tail

`no_aug_epochs` means the same thing for every task: the final N epochs train
without strong augmentation. Detection closes mosaic and MixUp. Classification
switches off `auto_augment`, `erasing`, `mixup` and `cutmix` while keeping the
crop and flip geometry. The classification family configs default
`no_aug_epochs` to `0`, so the tail is opt-in.

The switch is a main-process mutation of the dataset transform and the
collate function. Dataloader workers are respawned each epoch and pick it up;
a loader built with `persistent_workers=True` would not, so the trainer
raises instead of silently continuing with augmentation on.

## Python

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreResNet18-cls.pt")
model.train(
    data="/path/to/imagefolder",
    scale=(0.35, 1.0),
    flipud=0.2,
    auto_augment="randaugment",
    erasing=0.25,
    mixup=0.2,
    cutmix=0.2,
    crop_pct=1.0,
    epochs=60,
    no_aug_epochs=5,
)
```

The recipe is also usable directly, for example to build the exact train
transform for inspection:

```python
from libreyolo.data.augment.classify import ClassifyAugKnobs, build_classify_transforms

knobs = ClassifyAugKnobs.from_config(model_train_config)
transform = build_classify_transforms(224, augment=True, **knobs.transform_kwargs())
```

## CLI

```bash
libreyolo train model=LibreResNet18-cls.pt data=/path/to/imagefolder \
    scale=0.35 fliplr=0.5 flipud=0.2 auto_augment=randaugment erasing=0.25 \
    mixup=0.2 cutmix=0.2 crop_pct=1.0 no_aug_epochs=5
libreyolo val model=runs/classify/exp/best.pt data=/path/to/imagefolder crop_pct=1.0
```
