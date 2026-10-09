"""Diagnose the dav2l224-vs-518 delta1 non-monotonicity: per-seq + two alignments."""
import sys

import numpy as np

sys.path.insert(0, "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper")
import sintel_io  # noqa: E402
from eval_sintel_v2 import DAv2Baseline, delta1_aligned  # noqa: E402


def delta1_median(pred, gt, cap=500):
    """Median-scaling alignment (MiDaS convention)."""
    p, g = pred.astype(np.float64), gt.astype(np.float64)
    m = (g > 0) & (g < cap) & np.isfinite(p) & np.isfinite(g)
    if m.sum() < 500:
        return None
    s = np.median(g[m]) / np.median(p[m])
    p2 = np.maximum(p[m] * s, 1e-6)
    return float((np.maximum(g[m] / p2, p2 / g[m]) < 1.25).mean())


seqs = sintel_io.list_depth_seqs()[:4]
dav2 = DAv2Baseline()
print(f"{'seq':<12}", *[f"{ls:>10}px" for ls in (518, 336, 224)],
      "   | median-scaling:", *[f"{ls:>10}px" for ls in (518, 336, 224)])
for scene, seq, _ in seqs:
    imgs, gts, flows = sintel_io.load_seq(sintel_io.SINTEL_ROOT, scene, seq, 30)
    row_l, row_m = [], []
    for ls in (518, 336, 224):
        d1s_l, d1s_m = [], []
        for t in range(0, len(imgs), 2):   # every 2nd frame for speed
            d = dav2(imgs[t], long_side=ls)
            a = delta1_aligned(d, gts[t])
            b = delta1_median(d, gts[t])
            if a is not None:
                d1s_l.append(a)
            if b is not None:
                d1s_m.append(b)
        row_l.append(np.mean(d1s_l))
        row_m.append(np.mean(d1s_m))
    print(f"{seq:<12}", *[f"{v:10.4f}" for v in row_l],
          "   |               ", *[f"{v:10.4f}" for v in row_m])
