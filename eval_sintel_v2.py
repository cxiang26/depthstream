"""Paper experiments v2: DepthStream on Sintel — richer configs + visuals.

Configs:
  dense        per-frame student (ema=1, kf=1)
  kf3 / kf10   fixed-interval streaming
  kf10-adp     adaptive re-anchor (max gap 10, photometric trigger)
  kf10-oracle  propagation with GT flow (ablation: flow headroom)
  kf3-hiflow   flow_width 384 (ablation: flow resolution)
  dav2l518     per-frame DAv2-L (reference)

Outputs: eval_v2_result.json (+ per-seq breakdown), qual strips in qual/.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper")
from depthestream import (DepthStream, TrtRelativeDepth, backward_warp,  # noqa
                          colorize, depth_fullres)
import sintel_io  # noqa: E402

OUT = "/home/cq/depthart_test"


def delta1_aligned(pred, gt):
    p, g = pred.astype(np.float64), gt.astype(np.float64)
    m = (g > 0) & (g < 500) & np.isfinite(p) & np.isfinite(g)  # cap outliers
    p, g = p[m], g[m]
    if p.size < 500:
        return None
    A = np.stack([p, np.ones_like(p)], 1)
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    p2 = np.maximum(A @ coef, 1e-6)
    return float((np.maximum(g / p2, p2 / g) < 1.25).mean())


def tae_relative(pred_t, pred_t1, flow_t):
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
    return float(np.abs(A @ coef - b).mean()) / max(float(np.abs(b).mean()), 1e-6)


class DAv2Baseline:
    def __init__(self):
        import torch
        sys.path.insert(0, "/home/cq/depthart_test/Depth-Anything-V2-main")
        from depth_anything_v2.dpt import DepthAnythingV2
        cfg = {"encoder": "vitl", "features": 256,
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


def err_heat(pred, gt):
    p, g = pred.astype(np.float64), gt.astype(np.float64)
    m = (g > 0) & np.isfinite(p)
    A = np.stack([p[m], np.ones(m.sum())], 1)
    coef, *_ = np.linalg.lstsq(A, g[m], rcond=None)
    p2 = np.zeros_like(pred)
    p2[m] = np.maximum(A @ coef, 1e-6)
    err = np.zeros_like(pred)
    err[m] = np.abs(p2[m] - g[m]) / np.maximum(g[m], 1e-6)
    vis = np.zeros((*pred.shape, 3), np.uint8)
    vis[..., 1] = np.clip(err / 0.25 * 255, 0, 255)          # green = good
    vis[..., 2] = np.clip((err - 0.25) / 0.5 * 255, 0, 255)  # red = bad
    vis[~m] = 40
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seqs", type=int, default=6)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--skip-dav2", action="store_true")
    ap.add_argument("--backend", default="trt", choices=["trt", "cpu"])
    ap.add_argument("--qual-seq", type=int, default=0)
    args = ap.parse_args()

    seqs = sintel_io.list_depth_seqs()[:args.max_seqs]
    if args.backend == "cpu":
        from bench_cpu import OnnxCpuDepth
        model = OnnxCpuDepth("/home/cq/depthart_test/mnv4m_slim_spf_224.onnx",
                             threads=4)
    else:
        model = TrtRelativeDepth()

    def make(name):
        kf = 1
        if name.startswith("kf"):
            digits = "".join(ch for ch in name[2:] if ch.isdigit())
            kf = int(digits) if digits else 1
        st = dict(keyframe=max(1, kf), flow="dis", flow_width=256, ema=1.0)
        if "adp" in name:
            st["adaptive"] = True
        if "oracle" in name:
            st["oracle_flow"] = True
        if "hiflow" in name:
            st["flow_width"] = 384
        return DepthStream(model, **st)

    names = ["dense", "kf3", "kf10", "kf10-adp", "kf10-oracle", "kf3-hiflow"]
    acc = {n: {"d1": [], "tae": [], "ms": []} for n in names}
    if not args.skip_dav2:
        acc["dav2l518"] = {"d1": [], "tae": [], "ms": []}
    per_seq = {n: {} for n in names + (["dav2l518"] if not args.skip_dav2 else [])}
    accum = {n: {} for n in names}
    dav2 = None if args.skip_dav2 else DAv2Baseline()

    qual_saved = False
    for si, (scene, seq, _) in enumerate(seqs):
        imgs, gts, flows = sintel_io.load_seq(sintel_io.SINTEL_ROOT, scene, seq,
                                              args.frames)
        if len(imgs) < 4:
            continue
        print(f"[{si+1}/{len(seqs)}] {scene}/{seq}  {len(imgs)} frames", flush=True)

        outs_all = {}
        for name in names:
            st = make(name)
            outs, ages = [], []
            t0 = time.perf_counter()
            for t, im in enumerate(imgs):
                gt_flow = flows[t - 1] if (t > 0 and flows[t - 1] is not None) else None
                outs.append(st.process(im, gt_fwd_flow=gt_flow))
                ages.append(st.i - 1 - st.last_key)
            ms = (time.perf_counter() - t0) * 1000 / len(imgs)
            outs_all[name] = outs

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
            per_seq[name][seq] = round(float(np.mean(d1s)), 4)

        if not args.skip_dav2:
            d1s, taes = [], []
            t0 = time.perf_counter()
            outs = [dav2(im) for im in imgs]
            ms = (time.perf_counter() - t0) * 1000 / len(imgs)
            for t in range(len(outs)):
                d1 = delta1_aligned(outs[t], gts[t])
                if d1 is not None:
                    d1s.append(d1)
                if t + 1 < len(outs) and flows[t] is not None:
                    te = tae_relative(outs[t], outs[t + 1], flows[t])
                    if te is not None:
                        taes.append(te)
            acc["dav2l518"]["d1"].append(np.mean(d1s))
            acc["dav2l518"]["tae"].append(np.mean(taes))
            acc["dav2l518"]["ms"].append(ms)
            per_seq["dav2l518"][seq] = round(float(np.mean(d1s)), 4)
            outs_all["dav2l518"] = outs

        # qualitative strips for one sequence: frame 10 of dense/kf10-oracle/DAv2
        if si == args.qual_seq and not qual_saved:
            qdir = f"{OUT}/qual"
            os.makedirs(qdir, exist_ok=True)
            t = min(10, len(imgs) - 1)
            cv2.imwrite(f"{qdir}/rgb.png",
                        cv2.resize(imgs[t], (512, 218)))
            for name in ["dense", "kf10", "kf10-oracle", "dav2l518"]:
                if name not in outs_all:
                    continue
                cv2.imwrite(f"{qdir}/{name}_depth.png",
                            cv2.resize(colorize(outs_all[name][t]), (512, 218)))
                cv2.imwrite(f"{qdir}/{name}_err.png",
                            cv2.resize(err_heat(outs_all[name][t], gts[t]),
                                       (512, 218)))
            qual_saved = True

    table = {}
    for name, v in acc.items():
        table[name] = {
            "delta1": round(float(np.mean(v["d1"])), 4),
            "tae_rel": round(float(np.mean(v["tae"])), 4),
            "ms_per_frame": round(float(np.mean(v["ms"])), 2),
            "n_seqs": len(v["d1"]),
        }
    table["_per_seq"] = per_seq
    table["_accum"] = {n: {str(k): round(float(np.mean(v)), 4)
                           for k, v in sorted(b.items())}
                       for n, b in accum.items() if b}
    suffix = "_cpu" if args.backend == "cpu" else ""
    with open(f"{OUT}/eval_v2_result{suffix}.json", "w") as f:
        json.dump(table, f, indent=1)
    print(json.dumps(table["_per_seq"] and table, indent=1))


if __name__ == "__main__":
    main()
