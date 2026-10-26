"""Multi-connection range downloader for Bonn zips."""
import os
import threading
import time
import urllib.request

BASE = "/home/cq/depthart_test/bonn"
URL = "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_{}.zip"
SEQS = ["balloon", "balloon_tracking", "balloon_tracking2",
        "crowd", "crowd3", "kidnapping_box"]
NCONN = 8


def get_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url, start, end, part, done):
    for attempt in range(6):
        try:
            got = os.path.getsize(part) if os.path.exists(part) else 0
            if got >= end - start + 1:
                done.append(part)
                return
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start + got}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r, \
                    open(part, "ab") as f:
                while True:
                    chunk = r.read(1 << 18)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(part) >= end - start + 1:
                done.append(part)
                return
        except Exception as e:
            print(f"  part{part[-5:]} retry{attempt}: {repr(e)[:60]}")
            time.sleep(2)


def fetch_multi(url, dst):
    size = get_size(url)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        print("skip", dst, size)
        return
    print(f"{os.path.basename(dst)}: {size/2**20:.0f} MiB, {NCONN} connections")
    bounds = [(i * size // NCONN, (i + 1) * size // NCONN - 1) for i in range(NCONN)]
    bounds[-1] = (bounds[-1][0], size - 1)
    parts = [f"{dst}.part{i}" for i in range(NCONN)]
    threads, done = [], []
    for (s, e), part in zip(bounds, parts):
        th = threading.Thread(target=fetch_range, args=(url, s, e, part, done))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    assert len(done) == NCONN, "some parts failed"
    with open(dst, "wb") as out:
        for part in parts:
            with open(part, "rb") as f:
                while True:
                    chunk = f.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
            os.remove(part)
    assert os.path.getsize(dst) == size
    print("done:", dst, f"{size/2**20:.0f} MiB")


for s in SEQS:
    t0 = time.time()
    fetch_multi(URL.format(s), os.path.join(BASE, f"rgbd_bonn_{s}.zip"))
    print(f"  took {time.time()-t0:.0f}s")
