"""Convert the official PP-DocLayoutV3 safetensors release to LibreYOLO format.

Source: ``PaddlePaddle/PP-DocLayoutV3_safetensors`` (``model.safetensors``,
Apache-2.0). The release's keys already match ``libreyolo.models.ppdoclayoutv3``
under a ``model.`` prefix, so conversion strips that prefix and wraps the
tensors in v1.0 metadata. Learned parameters are unchanged. The class names come
from the release's ``inference.yml`` ``label_list`` (the ``config.json``
``id2label`` repeats some names).

Usage::

    python weights/convert_ppdoclayoutv3_weights.py \
        --src path/to/PP-DocLayoutV3_safetensors/model.safetensors \
        --dst weights/LibrePPDocLayoutV3l.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


def convert(src: str, dst: str) -> Path:
    add_repo_root_to_path()
    from safetensors.torch import load_file

    from libreyolo.models.ppdoclayoutv3.convert import (
        convert_upstream,
        is_upstream_state_dict,
    )
    from libreyolo.models.ppdoclayoutv3.nn import PPDocLayoutV3Model
    from libreyolo.models.ppdoclayoutv3.utils import DOC_LAYOUT_CLASSES

    raw = load_file(src, device="cpu")
    if not is_upstream_state_dict(raw):
        raise ValueError(f"{src} is not the PP-DocLayoutV3 safetensors layout")
    state_dict = convert_upstream(raw)
    nc = int(state_dict["enc_score_head.weight"].shape[0])
    if nc != len(DOC_LAYOUT_CLASSES):
        raise ValueError(f"expected {len(DOC_LAYOUT_CLASSES)} classes, found {nc}")

    target = PPDocLayoutV3Model(nc=nc)
    result = target.load_state_dict(state_dict, strict=True)
    print(f"Loaded {len(state_dict)} tensors strictly ({result})")

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="ppdoclayoutv3",
        size="l",
        nc=nc,
        names=dict(enumerate(DOC_LAYOUT_CLASSES)),
        task="detect",
        imgsz=800,
    )
    out = Path(dst)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(out)
    print(f"Wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Path to model.safetensors")
    parser.add_argument("--dst", default="weights/LibrePPDocLayoutV3l.pt")
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
