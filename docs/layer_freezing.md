# Layer Freezing

Version: 1.0

Layer freezing is a training API for transfer learning. It prevents selected
model weights from being updated while the rest of the model trains.

## Public API

`freeze` is optional. The default is no freezing.

Accepted values:

| Value | Meaning |
| --- | --- |
| `None`, `false`, or empty string | Train all parameters. |
| Integer `N` | Freeze the first `N` family-defined freeze groups. |
| List of integers | Freeze those zero-based family-defined groups. |
| String | Freeze matching family-defined group, module, or parameter prefixes. |
| List of strings/integers | Freeze each listed selector. |

Examples:

```bash
libreyolo train --model LibreYOLO9t.pt --data data.yaml --freeze 10
libreyolo train --model LibreYOLO9t.pt --data data.yaml --freeze backbone
libreyolo train --model LibreRFDETRn.pt --data data.yaml --freeze backbone
```

Python training APIs accept the same `freeze` values.

## Decision

LibreYOLO exposes familiar YOLO training options, but freeze selectors are
defined by LibreYOLO model-family contracts rather than by raw YAML layer
positions.

Reason: LibreYOLO model families are not all represented as one shared
YAML-indexed sequential graph. Raw positional layer numbers would be fragile and
misleading across YOLO9, RF-DETR, and future families.

Instead, each trainable family owns an ordered list of stable freeze groups.
Integer freezing addresses that list. Name freezing addresses stable family
selectors or normal model prefixes.

## Flagship Contract

YOLO9:

- `freeze=10` freezes the complete backbone.
- `backbone`, `neck`, and `head` are stable high-level selectors.
- Integer groups start at the input side of the backbone, then proceed through
  neck and head.

RF-DETR:

- `backbone`, `decoder`, `queries`, `transformer.encoder_output`, and `head`
  are stable selectors.
- `backbone.encoder` and `backbone.projector` are stable narrower selectors.
- With `lora=True`, adapter parameters remain trainable even when their parent
  backbone group is frozen.
- Name selectors are preferred for RF-DETR because transformer components do
  not map naturally to YOLO-style layer numbers.

## Internal Contract

Families that support freezing must expose a stable group order for integer
selectors. The order is part of the family training contract and should not be
changed casually, because `freeze=N` depends on it.

The generic fallback may freeze direct parameter-owning children, but flagship
families must provide semantic groups.

Freezing is applied after model setup and any data-driven head rebuilds, and
before optimizer creation.

Optimizers must only receive trainable parameters.

Frozen BatchNorm-style statistics must not continue updating during training.

Invalid selectors must fail loudly when they match no parameters.

Configurations that freeze every trainable parameter must fail loudly.

The API is for static freezing at training startup. Scheduled unfreezing or
progressive freezing is a separate feature and is not part of this contract.
