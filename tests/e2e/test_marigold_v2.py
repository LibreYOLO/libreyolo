"""Manual same-worker parity against the pinned Apache-2.0 upstream graph.

Set LIBREYOLO_MARIGOLD_V2_UPSTREAM to the installed upstream checkout. This
downloads the public base/adapters as needed and needs a CUDA GPU. References
are computed in this process because cached BF16 outputs from another worker
are not a substitute for a controlled implementation comparison.
"""

import gc
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.e2e.conftest import require_test_weights

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.marigold_v2,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.slow,
    pytest.mark.timeout(1800),
]


def test_all_released_variants_match_upstream(tmp_path):
    source = os.environ.get("LIBREYOLO_MARIGOLD_V2_UPSTREAM")
    if not source:
        pytest.skip(
            "Set LIBREYOLO_MARIGOLD_V2_UPSTREAM to the pinned installed upstream checkout."
        )
    if not torch.cuda.is_available():
        pytest.skip("Marigold V2 NF4 parity requires CUDA.")
    import cv2
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf

    from libreyolo import LibreYOLO
    from libreyolo.models.marigold_v2.config import (
        MIRRORS,
        PROMPTS,
        SOURCE_REVISION,
        UPSTREAM_REPO,
        UPSTREAM_REVISION,
        VARIANTS,
        canonical_filename,
    )
    from libreyolo.models.marigold_v2.nn import BASE_REPO, BASE_REVISION
    from libreyolo.utils.hf_hub import HubRef, resolve_hub_checkpoint

    root = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == SOURCE_REVISION
    sys.path.insert(0, str(root))
    try:
        from marigoldv2.core.registry import REGISTRY
        from marigoldv2.dataset.dataloading.transform import (
            ReadRGBImage,
            Reshape,
            ReshapeToMultiple,
        )
        from marigoldv2.script.train.util import make_load_trainables_hook
        from marigoldv2.validation.folder_steps import (
            FolderDepthPrediction,
            FolderNormalizeSurfaceNormals,
            FolderRGBAlbedoPrediction,
        )
        from marigoldv2.validation.validate_steps import RunInference

        loader = importlib.import_module(
            "marigoldv2.experiments.20260316_qwen_depth.component_loader"
        )
        graph = importlib.import_module(
            "marigoldv2.experiments.20260316_qwen_depth.network_graph"
        )
        base = snapshot_download(
            BASE_REPO, revision=BASE_REVISION, allow_patterns=["transformer/*", "vae/*"]
        )
        adapters = snapshot_download(
            UPSTREAM_REPO,
            revision=UPSTREAM_REVISION,
            allow_patterns=["*/trainables.safetensors", "qwen_text_embeddings/*512*"],
        )
        cases = [("church.jpg", (517, 339), 0), ("squirrel.jpg", (321, 483), 512)]
        images = []
        for name, shape, imgsz in cases:
            path = tmp_path / (Path(name).stem + ".png")
            with Image.open(root / "assets/examples" / name) as image:
                image.convert("RGB").resize(shape).save(path)
            images.append((path, shape, imgsz))
        cfg = OmegaConf.load(root / "evaluation/config/inference_depth.yaml")
        cfg.paths = {
            "ckpt_qwen_image_edit": base,
            "embed_dir": str(Path(adapters) / "qwen_text_embeddings"),
        }
        REGISTRY["cfg"] = cfg
        loader.LoadQwenImageEditVAE()()
        loader.LoadQwenImageEditTransformerFlexible()()
        vae, dit = (
            REGISTRY["network_components"]["VAE"],
            REGISTRY["network_components"]["Diffuser"],
        )
        base_decoder = {
            key: value.cpu().clone()
            for key, value in vae.state_dict().items()
            if key.startswith(("decoder.", "post_quant_conv."))
        }
        expected = {}
        for variant, info in VARIANTS.items():
            vae.load_state_dict(base_decoder, strict=False)
            make_load_trainables_hook(REGISTRY, None)(
                [], str(Path(adapters) / info.subfolder)
            )
            vae.requires_grad_(False).eval()
            dit.requires_grad_(False).eval()
            output_type = {
                "depth": FolderDepthPrediction,
                "normal": FolderNormalizeSurfaceNormals,
                "albedo": FolderRGBAlbedoPrediction,
            }[info.task]
            nodes = (
                graph.QwenImageEncode(),
                graph.QwenImageEdit2509Step({"prefix": PROMPTS[info.task]}),
                graph.QwenImageDecode(),
                output_type(
                    {
                        "input_key": "out/pixel_pred",
                        "output_key": "out/pred",
                        "output_color_space": "linear",
                    }
                ),
            )

            def reference_graph(batch, nodes=nodes):
                for node in nodes:
                    node(batch)

            REGISTRY["built_network_graphs"] = {"train": reference_graph}
            for image, size, imgsz in images:
                batch = {"annotation": {"rgb_path": str(image)}}
                ReadRGBImage(key="rgb_path", name="rgb", use_rel_dir=False)(batch)
                transform = (
                    Reshape(imgsz, imgsz, use_lanczos=True)
                    if imgsz
                    else ReshapeToMultiple(16, use_lanczos=True)
                )
                transform(batch)
                batch = {
                    "rgb_norm": batch["rgb_norm"][None].cuda(),
                    "dataset_disp_name": "parity",
                }
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(2025)
                    RunInference()(batch)
                array = batch["out"]["pred"][0].float().cpu().numpy()
                array = np.stack(
                    [
                        cv2.resize(channel, size, interpolation=cv2.INTER_LINEAR)
                        for channel in array
                    ]
                )
                if info.task == "normal":
                    array *= np.array([1, -1, -1], np.float32)[:, None, None]
                    array /= np.linalg.norm(array, axis=0, keepdims=True)
                    array = array.transpose(1, 2, 0)
                elif info.task == "albedo":
                    array = array.transpose(1, 2, 0).clip(0, 1)
                else:
                    array = array[0]
                expected[variant, image.name] = array
                del batch
        REGISTRY["network_components"].pop("VAE")
        REGISTRY["network_components"].pop("Diffuser")
        REGISTRY["built_network_graphs"].clear()
        del nodes, vae, dit, base_decoder
        gc.collect()
        torch.cuda.empty_cache()
        for variant, info in VARIANTS.items():
            filename = canonical_filename(variant)
            staged = resolve_hub_checkpoint(
                HubRef(
                    repo_id=f"LibreYOLO/{filename[:-3]}",
                    filename=filename,
                    revision=MIRRORS[variant][0],
                )
            )
            checkpoint = require_test_weights(staged)
            model = LibreYOLO(checkpoint, device="cuda")
            for image, _, imgsz in images:
                result = model(image, imgsz=imgsz)
                payload = {
                    "depth": result.depth_map,
                    "normal": result.normal_map,
                    "albedo": result.albedo,
                }[info.task]
                np.testing.assert_array_equal(
                    payload.numpy().data, expected[variant, image.name]
                )
            del model
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        sys.path.remove(str(root))
