"""DAv2-L per-frame baseline on all Bonn sequences (for the paper table)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from depthstream.backends.trt_gpu import TrtDepth  # noqa: E402  (unused here)
import bonn_io  # noqa: E402
from eval_sintel import DAv2Baseline, delta1_aligned  # noqa: E402

REPO = os.environ["DEPTHSTREAM_DAV2_REPO"]
WEIGHTS = os.environ["DEPTHSTREAM_DAV2_WEIGHTS"]
BONN = os.environ.get("DEPTHSTREAM_BONN", "/home/cq/depthart_test/bonn")

dav2 = DAv2Baseline(REPO, WEIGHTS)
d1s, ms_all = [], []
for seq, _ in bonn_io.list_seqs(BONN):
    imgs, gts = bonn_io.load_seq(seq, 60, root=BONN)
    if len(imgs) < 10:
        continue
    t0 = time.perf_counter()
    outs = [dav2(im) for im in imgs]
    ms = (time.perf_counter() - t0) * 1000 / len(imgs)
    d1 = [delta1_aligned(o, g) for o, g in zip(outs, gts)]
    d1 = [d for d in d1 if d is not None]
    d1s.append(np.mean(d1))
    ms_all.append(ms)
    print(seq, round(float(np.mean(d1)), 4), round(ms, 1), flush=True)

print(json.dumps({
    "dav2l518_bonn9": {
        "delta1": round(float(np.mean(d1s)), 4),
        "ms_per_frame": round(float(np.mean(ms_all)), 1),
        "n_seqs": len(d1s),
    }
}, indent=1))
