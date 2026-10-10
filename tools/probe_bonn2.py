"""Probe available Bonn dynamic sequences."""
import urllib.request
import urllib.error

URL = "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_{}.zip"
CANDIDATES = [
    "balloon", "balloon_tracking", "balloon_tracking2",
    "crowd", "crowd3", "kidnapping_box", "kidnapping",
    "person", "skeleton", "skeleton2", "plunderbox",
    "cooking", "cleaning", "moving", "moving_nonobstructing",
    "opening", "placing", "removing", "searching",
    "showing", "static", "constructing",
]

for s in CANDIDATES:
    req = urllib.request.Request(URL.format(s), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"OK   {s:<20} {int(r.headers['Content-Length'])/2**20:5.0f} MiB")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"{e.code} {s}")
    except Exception:
        pass
