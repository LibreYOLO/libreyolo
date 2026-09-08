# Classification class weighting

Image-classification training uses unweighted cross-entropy by default.
Set `class_weights=True` to weight rare classes more heavily:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreResNet18-cls.pt")
model.train(data="/path/to/imagefolder", class_weights=True)
```

Both CLI grammars expose the same boolean option:

```bash
libreyolo train model=LibreResNet18-cls.pt data=/path/to/imagefolder class_weights=true
libreyolo train --model LibreResNet18-cls.pt --data /path/to/imagefolder --class-weights
```

`class_weights` is an additive LibreYOLO option. It accepts only `True` or
`False`, not custom vectors. Supported image-classification trainers are
ResNet, ConvNeXt, MobileNetV4, EfficientNetV2, and DINOv2. DINOv2 uses
the shared RF-DETR trainer's classification path; RF-DETR itself no longer
exposes classification as a public task. Other tasks and trainers reject enabled weighting.
YOLO9 detection losses are unchanged.

For `C` classes, `N` training images, and `n_c` images in class `c`, the
weight is `N / (C * n_c)`. Counts come from the complete training split's
label mapping before distributed sharding or batch mixing. Every class must
contain training images. For a 9:1 split the weights are `5/9` and `5`.

The loss multiplies each class's negative log-probability by its weight and
target probability, sums across classes, then averages across images. Hard
labels and MixUp/CutMix soft labels use this same normalization. It differs
from dividing each batch by the sum of its observed class weights; homogeneous
batches still retain the intended weight. Equal-sized DDP ranks average these
local means without extra world-size scaling. The weights follow the loss
logits' device; half-precision logits use float32 loss arithmetic to avoid
overflowing large rare-class weights.

This changes the training objective, not the sampling frequency. The existing
`class_balanced` option is a separate detection sampler and is not supported by
these classification loaders. Class weighting is not a guarantee of improved
accuracy. Validation accuracy remains unweighted. For CNN trainers, optional
`val_loss=True` uses the same training-set weights and loss normalization.
Standalone validation without a training loss adapter remains unweighted.

The setting is saved in the training checkpoint's existing `config` field.
When resuming, pass the same `class_weights` setting and use the same dataset;
weights are recomputed from its training split. A different setting raises an
error rather than silently changing the objective. Older checkpoints imply
`class_weights=False`. Start a new fine-tuning run to change the setting.
No inference checkpoint tensors or metadata schema change.

The four CNN trainers use the weighted loss in both eager and CUDA-graph
assembly paths. DINOv2/RF-DETR classification retains its existing eager
fallback when CUDA-graph training is requested. Hardware and convergence
validation evidence is recorded with the change, separately from this API.
