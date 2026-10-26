"""External baseline: Depth Anything V2 Small (25M) per-frame, same protocol
as Table 1 (delta1 + scale-invariant TAE/tau5 + latency)."""
import json
import os
import sys
import time

import numpy as np

ROOT = "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "evals"))

import torch  # noqa: E402

import sintel_io  # noqa: E402
from eval_sintel import delta1_aligned, tae_relative, align_to_gt  # noqa: E402
from tau5 import tau5_sequence  # noqa: E402

sys.path.insert(0, "/home/cq/depthart_test/Depth-Anything-V2-main")
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

cfg = {"encoder": "vits", "features": 64,
       "out_channels": [48, 96, 192, 384],
       "use_bn": False, "use_clstoken": False}
m = DepthAnythingV2(**cfg)
m.load_state_dict(torch.load(
    "/home/cq/depthart_test/dav2_vits.pth", map_location="cpu"), strict=False)
m = m.cuda().eval().half()

import cv2  # noqa: E402


def infer(frame_bgr, long_side=518):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    s = long_side / max(h, w)
    nw, nh = int(round(w * s / 14)) * 14, int(round(h * s / 14)) * 14
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).cuda().half()[None]
    with torch.no_grad():
        d = m(x)
    d = d.squeeze().float().cpu().numpy()
    if d.shape != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    return d


seqs = sintel_io.list_depth_seqs("/home/cq/depthart_test")
acc = {"d1": [], "tae": [], "tau5": [], "ms": []}
for scene, seq, _ in seqs:
    imgs, gts, flows = sintel_io.load_seq("/home/cq/depthart_test", scene,
                                          seq, 50)
    if len(imgs) < 4:
        continue
    t0 = time.perf_counter()
    outs = [infer(im) for im in imgs]
    ms = (time.perf_counter() - t0) * 1000 / len(imgs)
    souts = [align_to_gt(o, g) for o, g in zip(outs, gts)]
    d1s = [d1 for d1 in (delta1_aligned(o, g) for o, g in zip(outs, gts))
           if d1 is not None]
    taes = [te for t in range(len(outs) - 1) if flows[t] is not None
            for te in [tae_relative(souts[t], souts[t + 1], flows[t])]
            if te is not None]
    t5 = tau5_sequence(souts, flows, gts)
    acc["d1"].append(np.mean(d1s))
    acc["tae"].append(np.mean(taes))
    if t5 is not None:
        acc["tau5"].append(t5)
    acc["ms"].append(ms)
    print(f"{scene}/{seq} d1={np.mean(d1s):.4f} tae={np.mean(taes):.4f} "
          f"ms={ms:.1f}", flush=True)

table = {k: round(float(np.mean(v)), 4) for k, v in acc.items() if v}
os.makedirs(os.path.join(ROOT, "benchmarks", "rerun"), exist_ok=True)
with open(os.path.join(ROOT, "benchmarks", "rerun",
                       "eval_baseline_dav2s.json"), "w") as f:
    json.dump(table, f, indent=1)
print(json.dumps(table, indent=1))
