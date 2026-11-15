"""Vendored UniMatch model code (MIT, see LICENSE in this folder).

Source: https://github.com/autonomousvision/unimatch at commit
943a99006950ab6c84ea72a72beb147fdd03c19a, ``unimatch/`` package only, copied
verbatim. Training, dataloaders, the DepthSplat model and its DPT, ViT-FPN and
latent-diffusion UNet modules are not vendored. The flow, stereo and depth
code paths inside ``unimatch.py`` are kept unchanged so the stereo and depth
tasks can be wired later without re-vendoring.
"""
