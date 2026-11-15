"""Minimal inference-only copy of WAFT (princeton-vl/WAFT, BSD-3-Clause).

Upstream: https://github.com/princeton-vl/WAFT at commit
b152ff1cad1af8c185ee7b141997c48ff3334c87. See ``LICENSE`` in this folder.
Training code, datasets, losses and the xformers and timm dependencies are
dropped; attention runs on ``torch.nn.functional.scaled_dot_product_attention``.
"""
