"""Paper experiment: DepthStream on Bonn RGB-D (real dynamic indoor scenes).

Metrics: delta1 (aligned, GT capped at 10 m Kinect range), TAE with estimated
DIS flow (identical across configs; labeled as estimated-flow warping error),
ms/frame. Configs: dense, kf3, kf10-adp, dav2l518 reference.

Usage: python eval_bonn.py [--frames 60] [--skip-dav2]
"""
import argparse
import json
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper")
from depthestream import DepthStream, TrtRelativeDepth  # noqa: E402
import bonn_io  # noqa: E402
from eval_sintel_v2 import DAv2Baseline, delta1_aligned  # noqa: E402

OUT = "/home/cq/depthart_test"


def tae_est(pred_t, pred_t1, flow_bwd):
    """Warp pred_t1 back to t with backward flow; scale-align; rel abs err."""
    m = np.isfinite(pred_t) & np.isfinite(pred_t1)
    w = pred_t1.copy()
    w[~np.isfinite(pred_t1)] = np.nan
    warped = cv2.remap(pred_t1, flow_bwd[..., 0].astype(np.float32),
                       flow_bwd[..., 1].astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = m & np.isfinite(warped)
    a, b = warped[m], pred_t[m]
    if a.size < 500:
        return None
    A = np.stack([a, np.ones_like(a)], 1)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(np.abs(A @ coef - b).mean()) / max(float(np.abs(b).mean()), 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--skip-dav2", action="store_true")
    args = ap.parse_args()

    seqs = bonn_io.list_seqs()
    print("bonn seqs:", seqs)
    model = TrtRelativeDepth()
    dav2 = None if args.skip_dav2 else DAv2Baseline()

    names = ["dense", "kf3", "kf10-adp"]

    def make(name):
        kf = int("".join(c for c in name if c.isdigit()) or 1)
        st = dict(keyframe=max(1, kf), flow="dis", flow_width=256, ema=1.0)
        if "adp" in name:
            st["adaptive"] = True
        return DepthStream(model, **st)

    acc = {n: {"d1": [], "tae": [], "ms": []} for n in names + ["dav2l518"]}
    calc = __import__("depthestream").make_flow_backend("dis")

    for seq, _ in seqs:
        imgs, gts = bonn_io.load_seq(seq, args.frames)
        if len(imgs) < 10:
            continue
        print(f"[{seq}] {len(imgs)} frames", flush=True)
        H, W = imgs[0].shape[:2]
        fw = 256
        fh = int(round(H / W * fw / 8) * 8)

        outs_all = {}
        for name in names:
            st = make(name)
            outs = []
            t0 = time.perf_counter()
            for im in imgs:
                outs.append(st.process(im))
            ms = (time.perf_counter() - t0) * 1000 / len(imgs)
            outs_all[name] = outs

            d1s, taes = [], []
            for t in range(len(outs)):
                d1 = delta1_aligned(outs[t], gts[t])
                if d1 is not None:
                    d1s.append(d1)
                if t + 1 < len(outs):
                    g0 = cv2.resize(cv2.cvtColor(imgs[t], cv2.COLOR_BGR2GRAY), (fw, fh))
                    g1 = cv2.resize(cv2.cvtColor(imgs[t + 1], cv2.COLOR_BGR2GRAY), (fw, fh))
                    fl_s = calc(g1, g0)   # backward: t+1 -> t
                    fl = __import__("depthestream").upsample_flow(
                        fl_s, W / fw, (W, H))
                    te = tae_est(outs[t], outs[t + 1], fl)
                    if te is not None:
                        taes.append(te)
            acc[name]["d1"].append(np.mean(d1s))
            acc[name]["tae"].append(np.mean(taes))
            acc[name]["ms"].append(ms)

        if dav2 is not None:
            t0 = time.perf_counter()
            outs = [dav2(im) for im in imgs]
            ms = (time.perf_counter() - t0) * 1000 / len(imgs)
            d1s, taes = [], []
            for t in range(len(outs)):
                d1 = delta1_aligned(outs[t], gts[t])
                if d1 is not None:
                    d1s.append(d1)
                if t + 1 < len(outs):
                    g0 = cv2.resize(cv2.cvtColor(imgs[t], cv2.COLOR_BGR2GRAY), (fw, fh))
                    g1 = cv2.resize(cv2.cvtColor(imgs[t + 1], cv2.COLOR_BGR2GRAY), (fw, fh))
                    fl_s = calc(g1, g0)
                    fl = __import__("depthestream").upsample_flow(
                        fl_s, W / fw, (W, H))
                    te = tae_est(outs[t], outs[t + 1], fl)
                    if te is not None:
                        taes.append(te)
            acc["dav2l518"]["d1"].append(np.mean(d1s))
            acc["dav2l518"]["tae"].append(np.mean(taes))
            acc["dav2l518"]["ms"].append(ms)
            outs_all["dav2l518"] = outs

    table = {}
    for name, v in acc.items():
        if not v["d1"]:
            continue
        table[name] = {
            "delta1": round(float(np.mean(v["d1"])), 4),
            "tae_est": round(float(np.mean(v["tae"])), 4),
            "ms_per_frame": round(float(np.mean(v["ms"])), 2),
            "n_seqs": len(v["d1"]),
        }
    with open(f"{OUT}/eval_bonn_result.json", "w") as f:
        json.dump(table, f, indent=1)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
