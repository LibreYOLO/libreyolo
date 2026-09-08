# Classification class weighting

Image-classification training uses unweighted cross-entropy by default.
Set `cls_pw` between `0` and `1` to control inverse-frequency weighting:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreResNet18-cls.pt")
model.train(data="/path/to/imagefolder", cls_pw=0.5)
```

Both CLI grammars expose the same numeric option:

```bash
libreyolo train model=LibreResNet18-cls.pt data=/path/to/imagefolder cls_pw=0.5
libreyolo train --model LibreResNet18-cls.pt --data /path/to/imagefolder --cls-pw 0.5
```

`cls_pw` accepts a finite number in `[0, 1]`, not a boolean or a custom vector.
Its default `0.0` disables weighting, `1.0` gives full inverse-frequency
weighting, and intermediate values give partial weighting. For class `c`
with `n_c` training images, first compute `u_c = n_c ** (-cls_pw)`, then
`w_c = u_c / mean(u)`. This makes the arithmetic mean of the class weights
`1.0`. Counts come from the complete training split's label mapping before
distributed sharding or batch mixing. Every class must contain training images.

The parameter name, range, default, power, and mean-one class normalization
follow the public contract documented in the
[Ultralytics training arguments](https://docs.ultralytics.com/modes/train/)
and [configuration guide](https://docs.ultralytics.com/usage/cfg/), checked
2026-09-08. LibreYOLO applies that interface to its supported image classifiers.
This is not a claim of identical standalone-classification behavior, loss
reduction, or whole-library numerical parity; those have not been verified.

Supported image-classification trainers are ResNet, ConvNeXt, MobileNetV4,
EfficientNetV2, and DINOv2. DINOv2 uses the shared RF-DETR trainer's
classification path; RF-DETR itself no longer exposes classification as a
public task. Other tasks and trainers reject enabled weighting. YOLO9
detection losses are unchanged.

## Existing boolean option

`class_weights=True` remains supported with exactly the behavior introduced
in PR #841: `w_c = N / (C * n_c)`, where `N` is the number of training images
and `C` the number of classes. This normalization makes the average weight
across training images one. It differs from `cls_pw`'s mean across classes.

For a two-class 9:1 training split:

| Setting | Majority weight | Minority weight |
| --- | ---: | ---: |
| `cls_pw=0` (default, unweighted) | 1 | 1 |
| `cls_pw=0.5` | 0.5 | 1.5 |
| `cls_pw=1` | 0.2 | 1.8 |
| `class_weights=True` | 5/9 | 5 |

Enabling `class_weights=True` together with `cls_pw>0` raises an error;
neither silently overrides the other. `class_weights=True, cls_pw=0` retains
legacy behavior. `class_weights=False, cls_pw>0` uses the power option.
The boolean default remains `False`, and its CLI forms remain
`class_weights=true` or `--class-weights`.

## Loss, validation, and resume

LibreYOLO's loss multiplies each class's negative log-probability by its
weight and target probability, sums across classes, then averages across
images. Hard labels and MixUp/CutMix soft labels use this same normalization.
It differs from dividing each batch by the sum of its observed class weights;
homogeneous batches still retain the intended weight. Equal-sized DDP ranks
average these local means without extra world-size scaling. The weights
follow the loss logits' device; half-precision logits use float32 loss
arithmetic to avoid overflowing large rare-class weights.

This changes the training objective, not the sampling frequency. The existing
`class_balanced` option is a separate detection sampler and is not supported by
these classification loaders. Class weighting is not a guarantee of improved
accuracy. Validation accuracy remains unweighted. For CNN trainers, optional
`val_loss=True` uses the same training-set weights and loss normalization.
Standalone validation without a training loss adapter remains unweighted.

Both settings are saved in the training checkpoint's existing `config` field.
When resuming, pass the same `cls_pw` and `class_weights` values and use the
same dataset; weights are recomputed from its training split. A different
setting raises an error rather than silently changing the objective. Missing
fields in older checkpoints imply `cls_pw=0.0` and `class_weights=False`.
Checkpoints from PR #841 that saved `class_weights=True` therefore still resume
with that boolean enabled and `cls_pw=0`. Start a new fine-tuning run to change
the setting. No inference checkpoint tensors or metadata schema change.

The four CNN trainers use the weighted loss in both eager and CUDA-graph
assembly paths. DINOv2's RF-DETR classification trainer retains its existing
eager fallback when CUDA-graph training is requested. Hardware and convergence
validation evidence is recorded with the change, separately from this API.
