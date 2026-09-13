"""Pinned official 224px ImageNet-1K EMA checkpoint identities."""

import hashlib
from pathlib import Path

SOURCE_COMMIT = "2553895753323c6fe0b2bf390683f5ea358a42b9"
SOURCE_REPO = "https://github.com/facebookresearch/ConvNeXt-V2"
SOURCE_SHA256 = {
    "atto": "6ddb7c8a4ccfe4c9e1210d931efa139627d75465cb3a46ebecb79f128bccfaec",
    "b": "1bfd2dc2b27908062d185ef81e6bb5e36a0a10850b70d1bf010a945158e15971",
    "femto": "6bbfccae4ba55122e0fdf63498e0ae643323fabc704874c055f4051ab3c6c482",
    "h": "8b93444dacc21613c58c5efbfa594a46e019e735686b9c22c05208ea739bfeb9",
    "l": "2d8cda2e2385ba7218a9c1cd84b07ce62f806582cc58313f97fe6683a84ba69d",
    "n": "ac0c7985ee9e34f0b9e06f52abb6e72fa6127c42979191664901b15fcd185674",
    "pico": "99612a03567360eb99a803e01017d8177c237d64f63d36c8fc7d553d3b5da27a",
    "t": "b23c89e2e601d2459bc35f8fef1bf6a3e9c90ab44b2f5b2d1117b984ac7f89a4",
}


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
