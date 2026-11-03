"""Unit tests for the optical flow dataset loader."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo.data.flow_dataset import (
    FlowDataset,
    build_flow_pairs,
    flow_collate_fn,
    img2flow_path,
    normalize_flow_format,
    read_flo,
    read_flow_file,
    read_kitti_flow,
    resolve_flow_data,
    write_flo,
    write_kitti_flow,
)

pytestmark = pytest.mark.unit

H, W = 12, 16


def _write_frame(path, shade):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (W, H), color=(shade, shade, shade)).save(path)


def _flow_for(index):
    """A distinct constant field per pair: (index + 1, -index) pixels."""
    flow = np.zeros((H, W, 2), dtype=np.float32)
    flow[..., 0] = index + 1
    flow[..., 1] = -index
    return flow


def _make_dataset(tmp_path, flow_format="flo", sequences=("seq_a", "seq_b"), frames=3, extra_yaml=None):
    pair_index = 0
    for sequence in sequences:
        for i in range(frames):
            _write_frame(tmp_path / "images" / "val" / sequence / f"frame_{i:03d}.png", 40 + 10 * i)
        for i in range(frames - 1):
            flow = _flow_for(pair_index)
            flow_dir = tmp_path / "flows" / "val" / sequence
            if flow_format == "flo":
                write_flo(flow_dir / f"frame_{i:03d}.flo", flow)
            elif flow_format == "kitti":
                valid = np.ones((H, W), dtype=bool)
                valid[0, :] = False
                write_kitti_flow(flow_dir / f"frame_{i:03d}.png", flow, valid)
            else:
                flow_dir.mkdir(parents=True, exist_ok=True)
                flow[0, 0] = np.nan
                np.save(flow_dir / f"frame_{i:03d}.npy", flow)
            pair_index += 1
    config = {"path": str(tmp_path), "val": "images/val", "flow_format": flow_format}
    if extra_yaml:
        config.update(extra_yaml)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(config))
    return yaml_path


def test_flo_round_trip(tmp_path):
    flow = np.random.RandomState(0).randn(H, W, 2).astype(np.float32)
    write_flo(tmp_path / "x.flo", flow)
    assert np.array_equal(read_flo(tmp_path / "x.flo"), flow)


def test_flo_rejects_wrong_magic(tmp_path):
    (tmp_path / "bad.flo").write_bytes(b"\x00" * 32)
    with pytest.raises(ValueError, match="magic"):
        read_flo(tmp_path / "bad.flo")


def test_kitti_round_trip_quantizes_to_1_over_64(tmp_path):
    flow = np.random.RandomState(1).randn(H, W, 2).astype(np.float32) * 10
    valid = np.random.RandomState(2).rand(H, W) > 0.3
    write_kitti_flow(tmp_path / "x.png", flow, valid)
    decoded, decoded_valid = read_kitti_flow(tmp_path / "x.png")
    assert np.array_equal(decoded_valid, valid)
    assert np.abs(decoded - flow).max() <= 1.0 / 64.0 + 1e-6


def test_npy_non_finite_pixels_are_invalid(tmp_path):
    flow = np.ones((H, W, 2), dtype=np.float32)
    flow[2, 3, 0] = np.inf
    np.save(tmp_path / "x.npy", flow)
    decoded, valid = read_flow_file(tmp_path / "x.npy", "npy")
    assert not valid[2, 3]
    assert valid.sum() == H * W - 1
    assert decoded[2, 3, 0] == 0.0


def test_normalize_flow_format():
    assert normalize_flow_format(None) == "flo"
    assert normalize_flow_format(".KITTI") == "kitti"
    with pytest.raises(ValueError, match="flow_format"):
        normalize_flow_format("exr")


def test_pairs_stay_inside_a_sequence_directory(tmp_path):
    frames = [
        tmp_path / "images/val/a/f2.png",
        tmp_path / "images/val/a/f1.png",
        tmp_path / "images/val/b/f1.png",
        tmp_path / "images/val/b/f2.png",
        tmp_path / "images/val/c/only.png",
    ]
    pairs = build_flow_pairs(frames)
    assert pairs == [
        (tmp_path / "images/val/a/f1.png", tmp_path / "images/val/a/f2.png"),
        (tmp_path / "images/val/b/f1.png", tmp_path / "images/val/b/f2.png"),
    ]


def test_img2flow_path_mirrors_tree_and_extension(tmp_path):
    frame = tmp_path / "images" / "val" / "seq" / "frame_000.png"
    assert img2flow_path(frame) == tmp_path / "flows" / "val" / "seq" / "frame_000.flo"
    assert img2flow_path(frame, "gt", "kitti") == tmp_path / "gt" / "val" / "seq" / "frame_000.png"
    assert img2flow_path(frame, flow_format="npy").suffix == ".npy"


@pytest.mark.parametrize("flow_format", ["flo", "kitti", "npy"])
def test_dataset_yields_n_minus_one_pairs_per_sequence(tmp_path, flow_format):
    yaml_path = _make_dataset(tmp_path, flow_format=flow_format)
    dataset = FlowDataset(resolve_flow_data(yaml_path), split="val")
    assert len(dataset) == 4  # two sequences of three frames

    img_a, img_b, flow, valid, info, index = dataset[0]
    assert img_a.shape == (3, H, W) and img_b.shape == (3, H, W)
    assert img_a.dtype == torch.float32 and float(img_a.max()) <= 1.0
    assert flow.shape == (2, H, W) and flow.dtype == torch.float32
    assert valid.shape == (H, W) and valid.dtype == torch.bool
    assert info["sequence"] == "seq_a"
    assert info["img_path"].endswith("frame_000.png")
    assert info["img_path_next"].endswith("frame_001.png")
    assert info["orig_shape"] == (H, W)
    assert index == 0

    expected = _flow_for(0)
    sample = flow.permute(1, 2, 0).numpy()
    if flow_format == "kitti":
        assert not valid[0].any() and valid[1:].all()
        assert np.abs(sample[valid.numpy()] - expected[valid.numpy()]).max() <= 1 / 64 + 1e-6
    elif flow_format == "npy":
        assert not valid[0, 0] and int(valid.sum()) == H * W - 1
    else:
        assert valid.all()
        assert np.array_equal(sample, expected)

    # The second sequence starts at pair 2 and reads its own files.
    _, _, flow2, _, info2, _ = dataset[2]
    assert info2["sequence"] == "seq_b"
    assert float(flow2[0].mean()) == pytest.approx(3.0, abs=1 / 32)


def test_missing_flow_file_is_an_error(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    (tmp_path / "flows" / "val" / "seq_a" / "frame_001.flo").unlink()
    with pytest.raises(FileNotFoundError, match="flow file"):
        FlowDataset(resolve_flow_data(yaml_path), split="val")


def test_single_frame_sequences_give_no_pair(tmp_path):
    yaml_path = _make_dataset(tmp_path, frames=1)
    with pytest.raises(ValueError, match="at least two frames"):
        FlowDataset(resolve_flow_data(yaml_path), split="val")


def test_valid_mask_png_is_applied(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    mask = np.ones((H, W), dtype=np.uint8) * 255
    mask[:, :4] = 0
    Image.fromarray(mask).save(tmp_path / "flows" / "val" / "seq_a" / "frame_000_valid.png")
    dataset = FlowDataset(resolve_flow_data(yaml_path), split="val")
    _, _, _, valid, _, _ = dataset[0]
    assert not valid[:, :4].any() and valid[:, 4:].all()
    _, _, _, valid1, _, _ = dataset[1]
    assert valid1.all()


def test_flows_dir_and_valid_suffix_keys(tmp_path):
    yaml_path = _make_dataset(tmp_path, extra_yaml={"flows_dir": "gt", "flow_valid_suffix": "_ok"})
    (tmp_path / "flows").rename(tmp_path / "gt")
    mask = np.zeros((H, W), dtype=np.uint8)
    Image.fromarray(mask).save(tmp_path / "gt" / "val" / "seq_a" / "frame_000_ok.png")
    dataset = FlowDataset(resolve_flow_data(yaml_path), split="val")
    assert dataset.flows_dir == "gt"
    _, _, _, valid, _, _ = dataset[0]
    assert not valid.any()


def test_collate_stacks_uniform_batches(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    dataset = FlowDataset(resolve_flow_data(yaml_path), split="val")
    imgs_a, imgs_b, flows, valids, infos, ids = flow_collate_fn([dataset[0], dataset[1]])
    assert imgs_a.shape == (2, 3, H, W) and imgs_b.shape == (2, 3, H, W)
    assert flows.shape == (2, 2, H, W) and valids.shape == (2, H, W)
    assert len(infos) == 2 and ids == [0, 1]


def test_collate_keeps_lists_for_mixed_sizes():
    small = (torch.zeros(3, 4, 4), torch.zeros(3, 4, 4), torch.zeros(2, 4, 4), torch.ones(4, 4, dtype=torch.bool), {"a": 1}, 0)
    big = (torch.zeros(3, 8, 8), torch.zeros(3, 8, 8), torch.zeros(2, 8, 8), torch.ones(8, 8, dtype=torch.bool), {"a": 2}, 1)
    imgs_a, _, flows, valids, _, ids = flow_collate_fn([small, big])
    assert isinstance(imgs_a, list) and isinstance(flows, list) and isinstance(valids, list)
    assert ids == [0, 1]


def test_mismatched_flow_shape_is_an_error(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    write_flo(tmp_path / "flows" / "val" / "seq_a" / "frame_000.flo", np.zeros((H + 1, W, 2), np.float32))
    dataset = FlowDataset(resolve_flow_data(yaml_path), split="val")
    with pytest.raises(ValueError, match="does not match frame shape"):
        dataset[0]
