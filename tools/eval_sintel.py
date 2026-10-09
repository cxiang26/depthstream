"""Paper experiment 1: DepthStream vs baselines on Sintel (GT depth + GT flow).

Metrics:
  delta1  - per-frame affine-invariant delta1 (aligned to GT, higher better)
  tae_rel - temporal warping error with GT flow, relative (lower better)
Also reports wall-clock ms/frame per config.

Usage:
  python eval_sintel.py [--max-seqs 6] [--frames 50] [--skip-dav2]
"""
import argparse
import json
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper")
from depthestream import DepthStream, TrtRelativeDepth, backward_warp  # noqa: E402
import sintel_io  # noqa: E402


def delta1_aligned(pred, gt):
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    m = (g > 0) & np.isfinite(p) & np.isfinite(g)
    p, g = p[m], g[m]
    if p.size < 500:
        return None
    A = np.stack([p, np.ones_like(p)], 1)
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    p2 = np.maximum(A @ coef, 1e-6)
    ratio = np.maximum(g / p2, p2 / g)
    return float((ratio < 1.25).mean())


def tae_relative(pred_t, pred_t1, flow_t):
    """Warp pred_t forward into t+1 with GT flow (coord-shift approximation),
    scale-align, then relative abs error."""
    h, w = pred_t.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    warped = cv2.remap(pred_t, gx + flow_t[..., 0], gy + flow_t[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = np.isfinite(warped) & np.isfinite(pred_t1)
    a, b = warped[m], pred_t1[m]
    if a.size < 500:
        return None
    A = np.stack([a, np.ones_like(a)], 1)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    diff = float(np.abs(A @ coef - b).mean())
    return diff / max(float(np.abs(b).mean()), 1e-6)


class DAv2Baseline:
    def __init__(self, encoder="vitl"):
        import torch
        sys.path.insert(0, "/home/cq/depthart_test/Depth-Anything-V2-main")
        from depth_anything_v2.dpt import DepthAnythingV2
        cfg = {"encoder": encoder, "features": 256,
               "out_channels": [256, 512, 1024, 1024],
               "use_bn": False, "use_clstoken": False}
        m = DepthAnythingV2(**cfg)
        m.load_state_dict(torch.load(
            "/home/cq/depthart_test/depth_anything_v2_vitl.pth", map_location="cpu"))
        self.torch = torch
        self.model = m.cuda().eval().half()

    def __call__(self, frame_bgr, long_side=518):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]
        s = long_side / max(h, w)
        nw, nh = int(round(w * s / 14)) * 14, int(round(h * s / 14)) * 14
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        x = self.torch.from_numpy(rgb.transpose(2, 0, 1)).cuda().half()[None]
        with self.torch.no_grad():
            d = self.model(x)
        d = d.squeeze().float().cpu().numpy()
        if d.shape != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        return d


def run_stream(imgs, keyframe, flow, model, ema=0.6):
    st = DepthStream(model, keyframe=keyframe, flow=flow, flow_width=256, ema=ema)
    outs, ages = [], []
    t0 = time.perf_counter()
    for im in imgs:
        outs.append(st.process(im))
        ages.append(st.i % keyframe if keyframe > 1 else 0)
    ms = (time.perf_counter() - t0) * 1000 / len(imgs)
    return outs, ms, st.stats(), ages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seqs", type=int, default=6)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--skip-dav2", action="store_true")
    ap.add_argument("--flow", default="dis", choices=["dis", "farneback"])
    ap.add_argument("--ema", type=float, default=0.6)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    seqs = sintel_io.list_depth_seqs()[:args.max_seqs]
    print("sequences:", seqs)

    model = TrtRelativeDepth()
    results = {}

    configs = [("kf1", 1), ("kf3", 3), ("kf10", 10)]
    if args.flow == "farneback":
        configs = [(f"{n}-fb", k) for n, k in configs]

    acc = {n: {"d1": [], "tae": [], "ms": []} for n, _ in configs}
    # delta1 bucketed by distance-from-keyframe (for the accumulation figure)
    accum = {n: {} for n, _ in configs}
    dav2_sizes = []
    if not args.skip_dav2:
        dav2_sizes = [518, 336, 224]
        for ls in dav2_sizes:
            acc[f"dav2l{ls}"] = {"d1": [], "tae": [], "ms": []}
        dav2 = DAv2Baseline()

    for si, (scene, seq, _) in enumerate(seqs):
        imgs, gts, flows = sintel_io.load_seq(sintel_io.SINTEL_ROOT, scene, seq,
                                              args.frames)
        if len(imgs) < 4:
            continue
        print(f"[{si+1}/{len(seqs)}] {scene}/{seq}  {len(imgs)} frames")

        for name, kf in configs:
            outs, ms, stats, ages = run_stream(imgs, kf, args.flow, model,
                                               ema=args.ema)
            d1s, taes = [], []
            for t in range(len(outs)):
                d1 = delta1_aligned(outs[t], gts[t])
                if d1 is not None:
                    d1s.append(d1)
                    accum[name].setdefault(ages[t], []).append(d1)
                if t + 1 < len(outs) and flows[t] is not None:
                    te = tae_relative(outs[t], outs[t + 1], flows[t])
                    if te is not None:
                        taes.append(te)
            acc[name]["d1"].append(np.mean(d1s))
            acc[name]["tae"].append(np.mean(taes))
            acc[name]["ms"].append(ms)

        for ls in dav2_sizes:
            d1s, taes = [], []
            t0 = time.perf_counter()
            outs = [dav2(im, long_side=ls) for im in imgs]
            ms = (time.perf_counter() - t0) * 1000 / len(imgs)
            for t in range(len(outs)):
                d1 = delta1_aligned(outs[t], gts[t])
                if d1 is not None:
                    d1s.append(d1)
                if t + 1 < len(outs) and flows[t] is not None:
                    te = tae_relative(outs[t], outs[t + 1], flows[t])
                    if te is not None:
                        taes.append(te)
            key = f"dav2l{ls}"
            acc[key]["d1"].append(np.mean(d1s))
            acc[key]["tae"].append(np.mean(taes))
            acc[key]["ms"].append(ms)

    table = {}
    for name, v in acc.items():
        table[name] = {
            "delta1": round(float(np.mean(v["d1"])), 4),
            "tae_rel": round(float(np.mean(v["tae"])), 4),
            "ms_per_frame": round(float(np.mean(v["ms"])), 2),
            "n_seqs": len(v["d1"]),
        }
    accum_out = {n: {str(k): round(float(np.mean(v)), 4)
                     for k, v in sorted(b.items())}
                 for n, b in accum.items() if b}
    table["_accum_curve"] = accum_out
    suffix = f"_{args.tag}" if args.tag else (f"_ema{args.ema}" if args.ema != 0.6 else "")
    with open(f"/home/cq/depthart_test/eval_sintel_result{suffix}.json", "w") as f:
        json.dump(table, f, indent=1)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
