"""Upstream RAFT vs LibreRAFT parity on consecutive frame pairs (run on the box)."""
import argparse, glob, json, os, sys

import numpy as np
import torch
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--upstream", default="/workspace/raft_up")
p.add_argument("--frames", default="/workspace/clip/frames")
p.add_argument("--ckpt", required=True)
p.add_argument("--libre", required=True)
p.add_argument("--small", action="store_true")
p.add_argument("--iters", type=int, default=20)
a = p.parse_args()

torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, os.path.join(a.upstream, "core"))
from raft import RAFT  # noqa: E402
from utils.utils import InputPadder  # noqa: E402

ns = argparse.Namespace(small=a.small, mixed_precision=False, alternate_corr=False)
up = torch.nn.DataParallel(RAFT(ns))
up.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
up = up.module.cuda().eval()

sys.path = [s for s in sys.path if "raft_up" not in s]
for m in [k for k in sys.modules if k in ("raft", "update", "extractor", "corr", "utils", "utils.utils")]:
    del sys.modules[m]
from libreyolo import LibreYOLO  # noqa: E402

lib = LibreYOLO(a.libre, device="cuda")

frames = sorted(glob.glob(os.path.join(a.frames, "*.png")))
results = lib(frames)
assert len(results) == len(frames) - 1, len(results)

def load(f):
    img = np.array(Image.open(f).convert("RGB")).astype(np.uint8)
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].cuda()

maxd, means, epes = 0.0, [], []
with torch.no_grad():
    for i, (f1, f2) in enumerate(zip(frames[:-1], frames[1:])):
        i1, i2 = load(f1), load(f2)
        padder = InputPadder(i1.shape)
        i1, i2 = padder.pad(i1, i2)
        _, fu = up(i1, i2, iters=a.iters, test_mode=True)
        ref = padder.unpad(fu)[0].permute(1, 2, 0).cpu().numpy()
        got = np.asarray(results[i].flow.data, dtype=np.float32)
        assert got.shape == ref.shape, (got.shape, ref.shape)
        d = np.abs(got - ref)
        maxd = max(maxd, float(d.max()))
        means.append(float(d.mean()))
        epes.append(float(np.linalg.norm(got - ref, axis=-1).mean()))
out = {
    "ckpt": os.path.basename(a.ckpt), "libre": os.path.basename(a.libre), "pairs": len(epes),
    "shape": list(ref.shape), "iters": a.iters, "max_abs_diff_px": maxd,
    "mean_abs_diff_px": float(np.mean(means)), "mean_epe_between_px": float(np.mean(epes)),
    "mean_flow_magnitude_px": float(np.linalg.norm(ref, axis=-1).mean()),
    "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
}
print(json.dumps(out))
