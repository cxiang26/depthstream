"""Count adaptive-trigger statistics on Sintel (kf10-adp, TRT backend)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depthstream import DepthStream  # noqa: E402
import sintel_io  # noqa: E402
from depthstream.backends.trt_gpu import TrtDepth  # noqa: E402

model = TrtDepth("/home/cq/depthart_test/depthart_fp16.engine")
warm = np.zeros((480, 640, 3), np.uint8) if False else None
import numpy as np  # noqa: E402
for _ in range(3):
    model(np.zeros((480, 640, 3), np.uint8))

seqs = sintel_io.list_depth_seqs("/home/cq/depthart_test")[:23]
tot_frames = tot_adaptive = tot_infer = 0
per_seq = {}
for scene, seq, _ in seqs:
    imgs, gts, flows = sintel_io.load_seq("/home/cq/depthart_test", scene,
                                          seq, 50)
    if len(imgs) < 4:
        continue
    st = DepthStream(model, keyframe=10, flow="dis", flow_width=256,
                     ema=1.0, adaptive=True)
    for im in imgs:
        st.process(im)
    tot_frames += st.i
    tot_adaptive += st.n_adaptive
    tot_infer += st.n_infer
    per_seq[seq] = st.n_adaptive

print(json.dumps({
    "frames": tot_frames,
    "n_infer": tot_infer,
    "n_adaptive": tot_adaptive,
    "adaptive_rate": round(tot_adaptive / max(tot_frames, 1), 4),
    "per_seq_adaptive": per_seq,
}, indent=1))
