"""ORT-alone test: load smoke ONNX, run 3000 inferences, log incrementally."""
import os
import sys
import time

import numpy as np
import onnxruntime as ort

LOG = os.path.expanduser("~/ort_test.log")
out = open(LOG, "w")

def log(m):
    out.write(m + "\n")
    out.flush()

log(f"start {time.strftime('%H:%M:%S')}")
path = "logs/rsl_rl/microduck_smoke/2026-09-08_19-37-13_smoke1/2026-09-08_19-37-13_smoke1.onnx"
log(f"creating session: {path}")
s = ort.InferenceSession(path)
log(f"providers: {s.get_providers()}")
inp = s.get_inputs()[0]
log(f"input: {inp.name} {inp.shape}")
obs = np.zeros((1, 61), dtype=np.float32)
t0 = time.time()
for i in range(3000):
    s.run(None, {inp.name: obs})
    if i % 1000 == 999:
        log(f"{i+1} inferences ok ({time.time()-t0:.1f}s)")
log("ORT ALONE OK")
out.close()
