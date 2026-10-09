"""Download Bonn RGB-D dynamic sequences (zips)."""
import os
import time
import urllib.request

BASE = "/home/cq/depthart_test/bonn"
URL = "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_{}.zip"
SEQS = ["balloon2", "kidnapping_box2", "person2"]

os.makedirs(BASE, exist_ok=True)
for s in SEQS:
    dst = os.path.join(BASE, f"rgbd_bonn_{s}.zip")
    if os.path.exists(dst) and os.path.getsize(dst) > 10**6:
        print("skip", s)
        continue
    t0 = time.time()
    try:
        urllib.request.urlretrieve(URL.format(s), dst)
        mb = os.path.getsize(dst) / 2**20
        print(f"ok {s}: {mb:.0f} MB in {time.time()-t0:.0f}s")
    except Exception as e:
        print("FAIL", s, repr(e)[:100])
