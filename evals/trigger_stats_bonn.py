"""Count adaptive-trigger statistics on Sintel (kf10-adp, TRT backend)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depthstream import DepthStream  # noqa: E402
import bonn_io  # noqa: E402
from depthstream.backends.ort_cpu import OnnxDepth  # noqa: E402

model = OnnxDepth("/home/cq/depthart_test/mnv4m_slim_spf_224.onnx", threads=4)
import numpy as np  # noqa: E402
for _ in range(3):
    model(np.zeros((480, 640, 3), np.uint8))

seqs = bonn_io.list_seqs("/home/cq/depthart_test/bonn")
tot_frames = tot_adaptive = tot_infer = 0
per_seq = {}
for seq, _ in seqs:
    imgs, gts = bonn_io.load_seq(seq, 60, root="/home/cq/depthart_test/bonn")
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
