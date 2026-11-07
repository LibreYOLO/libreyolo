"""Unit tests for the InstantHMR MHR decode helpers."""

import math

import pytest
import torch
from torch import nn

from libreyolo.models.instanthmr.decode import (
    PRED_DIM,
    angles_from_continuous,
    cross6d_to_matrix,
    decode_mhr,
    euler_xyz_to_matrix,
    matrix_to_euler_xyz,
    rot6d_to_matrix,
)

pytestmark = pytest.mark.unit


def _assert_rotation(R, atol=1e-5):
    eye = torch.eye(3, dtype=R.dtype).expand_as(R)
    torch.testing.assert_close(R.transpose(-1, -2) @ R, eye, atol=atol, rtol=0)
    torch.testing.assert_close(
        torch.linalg.det(R), torch.ones(R.shape[:-2], dtype=R.dtype), atol=atol, rtol=0
    )


@pytest.mark.parametrize("fn", [rot6d_to_matrix, cross6d_to_matrix])
def test_6d_gives_proper_rotations(fn):
    torch.manual_seed(0)
    R = fn(torch.randn(4, 7, 6, dtype=torch.float64))
    assert R.shape == (4, 7, 3, 3)
    _assert_rotation(R)


def test_rot6d_first_column_is_normalized_a():
    x = torch.randn(10, 6, dtype=torch.float64)
    R = rot6d_to_matrix(x)
    torch.testing.assert_close(
        R[..., 0], x[..., :3] / x[..., :3].norm(dim=-1, keepdim=True)
    )


def test_euler_round_trip():
    torch.manual_seed(1)
    n = 1000
    a = (torch.rand(n, dtype=torch.float64) * 2 - 1) * (math.pi - 1e-3)
    b = (torch.rand(n, dtype=torch.float64) * 2 - 1) * (math.pi / 2 - 0.1)
    c = (torch.rand(n, dtype=torch.float64) * 2 - 1) * (math.pi - 1e-3)
    e = torch.stack([a, b, c], -1)
    R = euler_xyz_to_matrix(e)
    _assert_rotation(R)
    torch.testing.assert_close(matrix_to_euler_xyz(R), e, atol=1e-9, rtol=0)


def test_euler_matches_axis_products():
    e = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64)
    zero = torch.zeros((), dtype=torch.float64)
    rx = euler_xyz_to_matrix(torch.stack([e[0], zero, zero]))
    ry = euler_xyz_to_matrix(torch.stack([zero, e[1], zero]))
    rz = euler_xyz_to_matrix(torch.stack([zero, zero, e[2]]))
    torch.testing.assert_close(euler_xyz_to_matrix(e), rz @ ry @ rx)
    assert rx[0, 0] == 1 and ry[1, 1] == 1 and rz[2, 2] == 1


@pytest.mark.parametrize("b", [math.pi / 2, -math.pi / 2])
def test_euler_gimbal_lock(b):
    e = torch.tensor([[0.4, b, 0.9], [-1.2, b, -0.3]], dtype=torch.float64)
    R = euler_xyz_to_matrix(e)
    out = matrix_to_euler_xyz(R)
    assert torch.isfinite(out).all()
    assert (out[:, 2] == 0).all()
    torch.testing.assert_close(euler_xyz_to_matrix(out), R, atol=1e-9, rtol=0)


def test_angles_from_continuous_shapes_and_hinge():
    torch.manual_seed(2)
    n3, n1, n_raw = 3, 5, 4
    t = torch.linspace(-3.0, 3.0, n1, dtype=torch.float64)
    pairs = torch.stack([torch.sin(t), torch.cos(t)], -1).reshape(-1) * 2.5
    raw = torch.randn(n_raw, dtype=torch.float64)
    x = torch.cat([torch.randn(6 * n3, dtype=torch.float64), pairs, raw])
    x = x.expand(2, 4, -1)
    out = angles_from_continuous(x, n3, n1)
    assert out.shape == (2, 4, 3 * n3 + n1 + n_raw)
    torch.testing.assert_close(out[0, 0, 3 * n3 : 3 * n3 + n1], t)
    torch.testing.assert_close(out[1, 3, 3 * n3 + n1 :], raw)
    # The first group decodes the same as the standalone path.
    g0 = matrix_to_euler_xyz(cross6d_to_matrix(x[0, 0, :6]))
    torch.testing.assert_close(out[0, 0, :3], g0)


def _random_tables(seed=0, flex=True):
    g = torch.Generator().manual_seed(seed)
    tables = dict(
        body_scatter=torch.randperm(133, generator=g),
        hand_cont_gather=torch.randperm(54, generator=g),
        hand_param_scatter=torch.randperm(27, generator=g),
        full_gather=torch.randint(0, 190, (136,), generator=g),
        hand_mean=torch.randn(54, generator=g),
        hand_comps=torch.randn(54, 54, generator=g) * 0.1,
        scale_mean=torch.randn(68, generator=g),
        scale_comps=torch.randn(28, 68, generator=g) * 0.1,
        coeff_lo=-torch.rand(28, generator=g) - 0.5,
        coeff_hi=torch.rand(28, generator=g) + 0.5,
    )
    if flex:
        tables["flex_lo"] = -torch.rand(6, generator=g)
        tables["flex_hi"] = torch.rand(6, generator=g)
    return tables


@pytest.mark.parametrize("flex", [True, False])
def test_decode_mhr_shapes(flex):
    tables = _random_tables(flex=flex)
    pred = torch.randn(3, PRED_DIM)
    params, shape, root_R = decode_mhr(pred, **tables)
    assert PRED_DIM == 447
    assert params.shape == (3, 204)
    assert shape.shape == (3, 45)
    assert root_R.shape == (3, 3, 3)
    assert torch.isfinite(params).all()
    torch.testing.assert_close(shape, pred[:, 266:311])
    _assert_rotation(root_R.double())


class _Decoder(nn.Module):
    def __init__(self, tables):
        super().__init__()
        for k, v in tables.items():
            self.register_buffer(k, v)

    def forward(self, pred):
        names = [n for n, _ in self.named_buffers()]
        return decode_mhr(pred, **{n: getattr(self, n) for n in names})


def test_decode_mhr_onnx_export(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    model = _Decoder(_random_tables(seed=3)).eval()
    pred = torch.randn(2, PRED_DIM)
    path = tmp_path / "decode.onnx"
    torch.onnx.export(
        model,
        (pred,),
        str(path),
        opset_version=17,
        dynamo=False,
        input_names=["pred"],
        output_names=["mhr_params", "shape", "root_R"],
        dynamic_axes={"pred": {0: "batch"}},
    )
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(5, PRED_DIM)
    with torch.no_grad():
        ref = model(x)
    got = sess.run(None, {"pred": x.numpy()})
    for r, o in zip(ref, got):
        torch.testing.assert_close(torch.from_numpy(o), r, atol=1e-5, rtol=1e-5)
