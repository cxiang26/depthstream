"""Sintel data IO: .dpt depth (PIEH float32 meters), .flo flow, sequence utils."""
import os
import struct

import cv2
import numpy as np

SINTEL_ROOT = "/home/cq/depthart_test"


def read_dpt(path):
    """Sintel .dpt: b'PIEH' + width(i32) + height(i32) + HxW float32 meters."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"PIEH":
        raise ValueError(f"bad PIEH header in {path}")
    w, h = struct.unpack("<ii", data[4:12])
    d = np.frombuffer(data, dtype="<f4", count=w * h, offset=12).reshape(h, w)
    return d.astype(np.float32)


def read_flo(path):
    """Standard .flo reader -> (H,W,2) float32."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"PIEH":
            raise ValueError(f"bad flo header in {path}")
        w, h = struct.unpack("<ii", f.read(8))
        data = np.fromfile(f, dtype="<f4", count=w * h * 2)
    return data.reshape(h, w, 2)


def list_depth_seqs(root=SINTEL_ROOT):
    """[(scene, seq, n_frames)] — depth is flat: training/depth/<seq>/*.dpt."""
    out = []
    droot = os.path.join(root, "training", "depth")
    for seq in sorted(os.listdir(droot)):
        q = os.path.join(droot, seq)
        if not os.path.isdir(q):
            continue
        n = len([f for f in os.listdir(q) if f.endswith(".dpt")])
        if n:
            out.append((seq.rsplit("_", 1)[0], seq, n))
    return out


def load_seq(root, scene, seq, n_frames=None):
    """Returns (images BGR list, gt depths list, gt forward flows list).
    Layout is flat: training/{clean,depth,flow}/<seq>/frame_%04d.*"""
    base = os.path.join(root, "training")
    img_dir = os.path.join(base, "clean", seq)
    dpt_dir = os.path.join(base, "depth", seq)
    dpts = sorted(f for f in os.listdir(dpt_dir) if f.endswith(".dpt"))
    n = len(dpts) if n_frames is None else min(n_frames, len(dpts))
    imgs, gts, flows = [], [], []
    for t in range(n):
        img = cv2.imread(os.path.join(img_dir, f"frame_{t+1:04d}.png"))
        if img is None:
            break
        imgs.append(img)
        gts.append(read_dpt(os.path.join(dpt_dir, f"frame_{t+1:04d}.dpt")))
        fp = os.path.join(base, "flow", seq, f"frame_{t+1:04d}.flo")
        flows.append(read_flo(fp) if os.path.exists(fp) else None)
    return imgs, gts, flows
