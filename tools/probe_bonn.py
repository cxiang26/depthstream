"""Probe correct Bonn sequence names, download available extras."""
import urllib.request
import urllib.error

URL = "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_{}.zip"

for s in ["person2", "person", "skeleton", "crowd2", "plunderbox"]:
    req = urllib.request.Request(URL.format(s), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print(s, r.status, f"{int(r.headers['Content-Length'])/2**20:.0f} MiB")
    except urllib.error.HTTPError as e:
        print(s, e.code)
    except Exception as e:
        print(s, "ERR", repr(e)[:60])
