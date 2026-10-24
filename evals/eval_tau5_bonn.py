"""tau5 on Bonn (estimated backward flow convention, like TAE_est)."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depthstream import DepthStream  # noqa: E402
from depthstream.backends.ort_cpu import OnnxDepth  # noqa: E402
from depthstream.flow import make_flow_backend, upsample_flow  # noqa: E402
from depthstream.utils import depth_fullres, align_to_gt  # noqa: E402
import bonn_io  # noqa: E402
from tau5 import tau5_pair  # noqa: E402

BONN = os.environ.get("DEPTHSTREAM_BONN", "/home/cq/depthart_test/bonn")
ONNX = os.environ.get("DEPTHSTREAM_ONNX", "/home/cq/depthart_test/mnv4m_slim_spf_224.onnx")

model = OnnxDepth(ONNX, threads=4)
calc = make_flow_backend("dis")
names = ["dense", "kf3", "kf10-adp"]
acc = {n: [] for n in names}

for seq, _ in bonn_io.list_seqs(BONN):
    imgs, gts = bonn_io.load_seq(seq, 60, root=BONN)
    if len(imgs) < 10:
        continue
    H, W = imgs[0].shape[:2]
    fw = 256
    fh = int(round(H / W * fw / 8) * 8)

    for name in names:
        if name == "dense":
            outs = [depth_fullres(model(im), im.shape[:2])
                    for im in imgs]          # pure per-frame baseline
        else:
            kf = int("".join(c for c in name if c.isdigit()) or 1)
            st = DepthStream(model, keyframe=kf, flow="dis", flow_width=fw,
                             ema=1.0, adaptive="adp" in name)
            outs = [st.process(im) for im in imgs]
        outs = [align_to_gt(o, g) for o, g in zip(outs, gts)]
        vals = []
        for t in range(len(outs) - 1):
            g0 = cv2.resize(cv2.cvtColor(imgs[t], cv2.COLOR_BGR2GRAY), (fw, fh))
            g1 = cv2.resize(cv2.cvtColor(imgs[t + 1], cv2.COLOR_BGR2GRAY), (fw, fh))
            fl = upsample_flow(calc(g1, g0), W / fw, (W, H))
            v = tau5_pair(outs[t], outs[t + 1], fl,
                          gt_t=np.where(np.isfinite(gts[t]), gts[t], -1.0))
            if v is not None:
                vals.append(v)
        if vals:
            acc[name].append(np.mean(vals))
    print(seq, "done", flush=True)

table = {k: round(float(np.mean(v)), 4) for k, v in acc.items() if v}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "benchmarks", "rerun", "eval_tau5_bonn.json")
with open(out, "w") as f:
    json.dump(table, f, indent=1)
print(json.dumps(table, indent=1))
