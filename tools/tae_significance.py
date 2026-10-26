"""Per-sequence paired significance of the si-TAE gain (dense vs kf3)."""
import math
import os
import sys

import numpy as np

ROOT = "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "evals"))

from depthstream import DepthStream  # noqa: E402
from depthstream.backends.trt_gpu import TrtDepth  # noqa: E402
from depthstream.utils import depth_fullres, align_to_gt  # noqa: E402
import sintel_io  # noqa: E402
from eval_sintel import tae_relative  # noqa: E402

model = TrtDepth("/home/cq/depthart_test/depthart_fp16.engine")
seqs = sintel_io.list_depth_seqs("/home/cq/depthart_test")
per_seq = {}
for scene, seq, _ in seqs:
    imgs, gts, flows = sintel_io.load_seq("/home/cq/depthart_test", scene,
                                          seq, 50)
    if len(imgs) < 4:
        continue
    row = {}
    for name in ["dense", "kf3"]:
        if name == "dense":
            outs = [depth_fullres(model(im), im.shape[:2]) for im in imgs]
        else:
            st = DepthStream(model, keyframe=3, flow="dis", flow_width=256,
                             ema=1.0)
            outs = [st.process(im) for im in imgs]
        souts = [align_to_gt(o, g) for o, g in zip(outs, gts)]
        taes = [te for t in range(len(outs) - 1) if flows[t] is not None
                for te in [tae_relative(souts[t], souts[t + 1], flows[t])]
                if te is not None]
        row[name] = float(np.mean(taes))
    per_seq[seq] = row
    print(f"{scene}/{seq}: dense={row['dense']:.4f} kf3={row['kf3']:.4f}",
          flush=True)

d = [per_seq[s]["dense"] - per_seq[s]["kf3"] for s in per_seq]
d = np.array(d)
n = len(d)
m, sd = d.mean(), d.std(ddof=1)
t = m / (sd / math.sqrt(n))
ci = 2.074 * sd / math.sqrt(n)
a = np.abs(d)
ranks = np.empty(n)
ranks[np.argsort(a)] = np.arange(1, n + 1)
for u in np.unique(a):
    msk = a == u
    if msk.sum() > 1:
        ranks[msk] = ranks[msk].mean()
w = ranks[d > 0].sum()
mu = n * (n + 1) / 4
z = (w - mu) / math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
print(f"\npaired: mean={m:.4f} sd={sd:.4f} t={t:.2f} CI95=+-{ci:.4f} "
      f"wilcoxon z={z:.2f} p={p:.5f}")
