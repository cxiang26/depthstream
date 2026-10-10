"""Compare desktop ORT-fp32 stream vs Nano TRT-fp32 stream outputs."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depthstream import DepthStream  # noqa: E402
from depthstream.backends.ort_cpu import OnnxDepth  # noqa: E402

ONNX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.expanduser("~"), ".cache", "depthstream",
    "depthart_mnv4m_slim_spf_224.onnx")
VIDEO = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_video.mp4")
NANO_DIR = sys.argv[3] if len(sys.argv) > 3 else "results/nano_dump"
N = 30

model = OnnxDepth(ONNX, threads=4)
stream = DepthStream(model, keyframe=10, flow="dis", flow_width=256, ema=1.0)

cap = cv2.VideoCapture(VIDEO)
rs = []
for t in range(N):
    ok, frame = cap.read()
    if not ok:
        break
    desk = stream.process(frame)
    nano = np.load(os.path.join(NANO_DIR, "d_%05d.npy" % t))
    a = desk.astype(np.float64).ravel()
    b = nano.astype(np.float64).ravel()
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        continue
    rs.append(float(np.corrcoef(a, b)[0, 1]))
cap.release()

print(json.dumps({
    "frames_compared": len(rs),
    "pearson_r_mean": round(float(np.mean(rs)), 6),
    "pearson_r_min": round(float(np.min(rs)), 6),
}))
