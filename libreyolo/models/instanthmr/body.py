"""MHR body for InstantHMR: model parameters -> 70 keypoints and the posed mesh.

Ported from InstantHMR ``training/mhr_body.py``
(https://github.com/Naturalpad-studio/InstantHMR, Apache-2.0).

The body is rebuilt from about 12 MB of constants read once from Meta's MHR
TorchScript rig (Apache-2.0): forward kinematics of the 127 joints, identity
blend shapes and linear blend skinning of the 18,439-vertex mesh, and the 70
keypoints as a fixed linear readout of joints and mesh vertices. MHR's pose
correctives are left out, as in upstream; the network was trained without them.

Differences from upstream:
- The constants are persistent buffers, so a LibreYOLO checkpoint carries the
  body and inference needs neither the 700 MB rig nor any MHR package.
- The mesh faces are stored as a buffer too.
- The camera-axes conversion builds its scale on the input's device.

Every op is static, so the module exports to ONNX together with the network.
Outputs are body-centred, in metres, camera axes (x right, y down, z forward).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

N_MODEL_PARAMS = 204
N_SHAPE_PARAMS = 45

# Buffers that make up a body, in the order a checkpoint stores them.
BODY_BUFFERS = (
    "parameter_transform",
    "joint_offsets",
    "joint_prerotations",
    "fk_partner",
    "fk_active",
    "inverse_bind_pose",
    "base_shape",
    "shape_vectors",
    "skin_joint",
    "skin_weight",
    "landmark_vertices",
    "W_joint",
    "W_vert",
    "faces",
)


def _fk_levels(parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prefix-multiplication schedule: at level l, every active joint j becomes partner(j) o j."""
    chains = []
    for j in range(len(parents)):
        chain = [j]
        while parents[chain[-1]] >= 0:
            chain.append(int(parents[chain[-1]]))
        chains.append(chain[::-1])
    partners, actives = [], []
    level = 0
    while True:
        partner, active = np.arange(len(parents)), np.zeros(len(parents), bool)
        for chain in chains:
            i = len(chain) - 1
            if (i >> level) & 1:
                partner[chain[i]] = chain[((i >> level) << level) - 1]
                active[chain[i]] = True
        if not active.any():
            return np.stack(partners), np.stack(actives)
        partners.append(partner)
        actives.append(active)
        level += 1


def _normalize(q):
    return q / torch.sqrt((q * q).sum(-1, keepdim=True)).clamp_min(1e-12)


def _qmul(a, b):
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        -1,
    )


def _cross(a, b):
    ax, ay, az = a.unbind(-1)
    bx, by, bz = b.unbind(-1)
    return torch.stack((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), -1)


def _rotate(q, v):
    axis, w = q[..., :3], q[..., 3:]
    av = _cross(axis, v)
    return v + 2 * (av * w + _cross(axis, av))


def _euler_xyz_to_quat(e):
    cr, sr = torch.cos(e[..., 0] / 2), torch.sin(e[..., 0] / 2)
    cp, sp = torch.cos(e[..., 1] / 2), torch.sin(e[..., 1] / 2)
    cy, sy = torch.cos(e[..., 2] / 2), torch.sin(e[..., 2] / 2)
    return torch.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        -1,
    )


def _compose(a, b):
    """a o b for skeleton states [t(3), q(4) xyzw, s(1)]."""
    ta, qa, sa = a[..., :3], _normalize(a[..., 3:7]), a[..., 7:]
    tb, qb, sb = b[..., :3], _normalize(b[..., 3:7]), b[..., 7:]
    return torch.cat((ta + sa * _rotate(qa, tb), _qmul(qa, qb), sa * sb), -1)


def _to_camera_axes(xyz):
    """MHR (cm, y up, z back) -> metres, y down, z forward."""
    return xyz * torch.tensor([0.01, -0.01, -0.01], dtype=xyz.dtype, device=xyz.device)


class MHRBody(torch.nn.Module):
    """``(B, 204)`` mhr_params + ``(B, 45)`` shape -> ``(B, 70, 3)`` keypoints, ``(B, 18439, 3)`` vertices."""

    def __init__(self, constants: dict | None = None):
        super().__init__()
        for name in BODY_BUFFERS:
            value = (
                torch.empty(0)
                if constants is None
                else torch.as_tensor(constants[name])
            )
            self.register_buffer(name, value)

    @property
    def is_built(self) -> bool:
        return self.parameter_transform.numel() > 0

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # The constants' shapes come from the rig, not from a config, so an
        # empty body adopts whatever the checkpoint carries.
        for name in BODY_BUFFERS:
            key = prefix + name
            if key in state_dict:
                setattr(self, name, state_dict[key].detach().clone())
            elif strict:
                missing_keys.append(key)

    @classmethod
    def from_rig(
        cls,
        rig_path: str | Path,
        landmark_vertices: np.ndarray,
        W_joint: np.ndarray,
        W_vert: np.ndarray,
    ) -> MHRBody:
        """Build the body constants from the MHR TorchScript rig and a 70-keypoint readout."""
        rig = torch.jit.load(str(rig_path), map_location="cpu").eval()
        ct = rig.character_torch
        sk, lbs, bs = ct.skeleton, ct.linear_blend_skinning, ct.blend_shape
        transform = ct.parameter_transform.parameter_transform
        if transform[:, N_MODEL_PARAMS:].any():
            raise ValueError(
                f"{rig_path}: the parameter transform uses the identity columns"
            )

        n_verts = bs.base_shape.shape[0]
        vert = lbs.vert_indices_flattened.long()
        counts = torch.bincount(vert, minlength=n_verts)
        order = torch.argsort(vert, stable=True)
        vert = vert[order]
        slot = torch.arange(len(vert)) - (torch.cumsum(counts, 0) - counts)[vert]
        skin_joint = torch.zeros(n_verts, int(counts.max()), dtype=torch.long)
        skin_weight = torch.zeros(n_verts, int(counts.max()))
        skin_joint[vert, slot] = lbs.skin_indices_flattened.long()[order]
        skin_weight[vert, slot] = lbs.skin_weights_flattened.float()[order]

        partner, active = _fk_levels(sk.joint_parents.numpy())
        return cls(
            dict(
                parameter_transform=transform[:, :N_MODEL_PARAMS].float(),
                joint_offsets=sk.joint_translation_offsets.float(),
                joint_prerotations=sk.joint_prerotations.float(),
                fk_partner=torch.from_numpy(partner).long(),
                fk_active=torch.from_numpy(active),
                inverse_bind_pose=lbs.inverse_bind_pose.float(),
                base_shape=bs.base_shape.float(),
                shape_vectors=bs.shape_vectors.float(),
                skin_joint=skin_joint,
                skin_weight=skin_weight,
                landmark_vertices=torch.as_tensor(landmark_vertices, dtype=torch.long),
                W_joint=torch.as_tensor(W_joint, dtype=torch.float32),
                W_vert=torch.as_tensor(W_vert, dtype=torch.float32),
                faces=ct.mesh.faces.long(),
            )
        )

    def skeleton(self, mhr_params):
        """``(B, 127, 8)`` global joint states, MHR units."""
        jp = (mhr_params @ self.parameter_transform.T).unflatten(-1, (-1, 7))
        q = _qmul(self.joint_prerotations, _euler_xyz_to_quat(jp[..., 3:6]))
        state = torch.cat(
            (
                jp[..., :3] + self.joint_offsets,
                q,
                torch.exp(math.log(2.0) * jp[..., 6:]),
            ),
            -1,
        )
        for level in range(self.fk_partner.shape[0]):
            composed = _compose(state.index_select(-2, self.fk_partner[level]), state)
            state = torch.where(self.fk_active[level, :, None], composed, state)
        return state

    def skin(self, state, shape_params):
        """``(B, 18439, 3)`` posed vertices, MHR units."""
        n_verts, n_inf = self.skin_joint.shape
        rest = self.base_shape + (
            shape_params @ self.shape_vectors.flatten(1)
        ).unflatten(-1, (n_verts, 3))
        bone = _compose(state, self.inverse_bind_pose)
        bone = bone.index_select(-2, self.skin_joint.flatten()).unflatten(
            -2, (n_verts, n_inf)
        )
        t, q, s = bone[..., :3], _normalize(bone[..., 3:7]), bone[..., 7:]
        return (
            (t + _rotate(q, s * rest.unsqueeze(-2))) * self.skin_weight[..., None]
        ).sum(-2)

    def forward(self, mhr_params, shape_params):
        state = self.skeleton(mhr_params)
        vertices = _to_camera_axes(self.skin(state, shape_params))
        joints = _to_camera_axes(state[..., :3])
        keypoints = self.W_joint @ joints + self.W_vert @ vertices.index_select(
            -2, self.landmark_vertices
        )
        return keypoints, vertices
