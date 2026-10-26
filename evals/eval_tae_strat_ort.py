"""C4: TAE stratified by pair type on the deterministic ORT backend,
same protocol/run style as the tau5 stratification."""
import json
import os
import sys

import numpy as np

ROOT = "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "evals"))

from depthstream import DepthStream  # noqa: E402
from depthstream.backends.ort_cpu import OnnxDepth  # noqa: E402
from depthstream.utils import depth_fullres, align_to_gt  # noqa: E402
import sintel_io  # noqa: E402
from eval_sintel import tae_relative  # noqa: E402

model = OnnxDepth("/home/cq/depthart_test/mnv4m_slim_spf_224.onnx",
                  threads=4)
seqs = sintel_io.list_depth_seqs("/home/cq/depthart_test")
res = {}
for name in ["dense", "kf3", "kf10-adp"]:
    acc = {"chain": [], "rekey": [], "all": []}
    for scene, seq, _ in seqs:
        imgs, gts, flows = sintel_io.load_seq("/home/cq/depthart_test",
                                              scene, seq, 50)
        if len(imgs) < 4:
            continue
        if name == "dense":
            outs = [depth_fullres(model(im), im.shape[:2]) for im in imgs]
            keys = set(range(len(imgs)))
        else:
            kf = int("".join(c for c in name if c.isdigit()) or 1)
            st = DepthStream(model, keyframe=kf, flow="dis", flow_width=256,
                             ema=1.0, adaptive=name.endswith("adp"))
            outs, keys = [], set()
            for t, im in enumerate(imgs):
                outs.append(st.process(im))
                if st.last_key == t:
                    keys.add(t)
        souts = [align_to_gt(o, g) for o, g in zip(outs, gts)]
        vals = {"chain": [], "rekey": [], "all": []}
        for t in range(len(outs) - 1):
            if flows[t] is None:
                continue
            typ = "rekey" if (t + 1) in keys else "chain"
            te = tae_relative(souts[t], souts[t + 1], flows[t])
            if te is not None:
                vals[typ].append(te)
                vals["all"].append(te)
        for k in vals:
            if vals[k]:
                acc[k].append(np.mean(vals[k]))
    res[name] = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    print(name, res[name], flush=True)

with open(os.path.join(ROOT, "benchmarks", "rerun",
                       "tae_strat_ort.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
