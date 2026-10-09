"""Unit-test sintel_io.read_dmm with a synthetic DMM file (format hypothesis A)."""
import struct
import sys

import numpy as np

sys.path.insert(0, "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper")
from sintel_io import read_dmm  # noqa: E402

W, H, NF = 64, 48, 2

# corners every 16 px: value = (1 + x/W) * 300 cm (1m..1.75m depth ramp)
d16 = np.zeros((H, W), np.uint16)
for cy in range(0, H, 16):
    for cx in range(0, W, 16):
        d16[cy, cx] = int((1 + cx / W) * 300)

blob = b"DMM_" + struct.pack("<ii", W, H)
for f in range(NF):
    blob += struct.pack("<ii", f + 1, 0) + d16.tobytes()

with open("/tmp/test.dmm", "wb") as fh:
    fh.write(blob)

frames = read_dmm("/tmp/test.dmm")
assert len(frames) == NF, f"frame count {len(frames)} != {NF}"
d = frames[0]
print("decoded shape:", d.shape, "range:", d.min(), "-", d.max(), "m")

# verify: corner (0,0) = 3.0m; beyond last corner (x>48) border-replicates 5.25m
assert abs(d[0, 0] - 3.0) < 0.02, d[0, 0]
assert abs(d[0, W - 1] - 5.25) < 0.05, d[0, W - 1]
mid = (1 + 0.5) * 3.0  # bilinear at x=W/2 (= corner cx=32 exactly)
assert abs(d[8, 32] - mid) < 0.05, (d[8, 32], mid)
print("DMM reader unit test: PASS")
