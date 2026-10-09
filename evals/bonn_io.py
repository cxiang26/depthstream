"""Bonn RGB-D Dynamic readers: <root>/rgbd_bonn_<seq>/{rgb,depth}/*.png."""
import os

import cv2
import numpy as np

MAX_DEPTH = 10.0   # Kinect practical range (m)


def list_seqs(root):
    out = []
    for name in sorted(os.listdir(root)):
        if not name.startswith("rgbd_bonn_") or not os.path.isdir(
                os.path.join(root, name)):
            continue
        seq = name[len("rgbd_bonn_"):]
        n = len(os.listdir(os.path.join(root, name, "depth")))
        if n >= 20:
            out.append((seq, n))
    return out


def load_seq(seq, n_frames=None, root=None):
    """(rgb BGR list, gt depth m list; NaN outside [0, MAX_DEPTH])."""
    root = root or os.environ.get("DEPTHSTREAM_BONN", "./datasets/bonn")
    rgb_dir = os.path.join(root, f"rgbd_bonn_{seq}", "rgb")
    dep_dir = os.path.join(root, f"rgbd_bonn_{seq}", "depth")
    rgb_f = sorted(os.listdir(rgb_dir))
    dep_f = sorted(os.listdir(dep_dir))
    rgb_ts = np.array([float(f[:-4]) for f in rgb_f])
    dep_ts = [float(f[:-4]) for f in dep_f]
    n = len(dep_f) if n_frames is None else min(n_frames, len(dep_f))
    imgs, gts = [], []
    for t in range(n):
        i = int(np.argmin(np.abs(rgb_ts - dep_ts[t])))   # nearest-timestamp pair
        img = cv2.imread(os.path.join(rgb_dir, rgb_f[i]))
        if img is None:
            break
        d = cv2.imread(os.path.join(dep_dir, dep_f[t]),
                       cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        d[(d <= 0) | (d > MAX_DEPTH)] = np.nan
        imgs.append(img)
        gts.append(d)
    return imgs, gts
