"""PicoDet implementation for LibreYOLO.

Port of Bo396543018/Picodet_Pytorch (Apache-2.0), which itself ports
PaddlePaddle's PicoDet to PyTorch via mmdet/mmcv. This port strips the
mmcv/mmdet dependency: ``ConvModule`` is replaced with plain
``nn.Conv2d + nn.BatchNorm2d + activation``, ``BaseModule`` with
``nn.Module``, registry decorators removed, and the head/loss path
rewritten against LibreYOLO's ``BaseModel`` / ``BaseTrainer`` ABCs.

Activations (HardSwish in backbone/neck/head, HardSigmoid in SE) are
preserved exactly so upstream PaddlePaddle/Bo checkpoints can be
loaded with bit-equivalent forwards.
"""
