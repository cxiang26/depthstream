"""DepthStream evaluation on Sintel (GT depth + GT flow).

Usage:
  python evals/eval_sintel.py --sintel-root /path/to/Sintel \
      --student-onnx /path/to/student.onnx [--backend ort|trt] \
      [--max-seqs 23] [--frames 50] [--dav2-repo D --dav2-weights W]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depthstream import DepthStream  # noqa: E402
from depthstream.backends.ort_cpu import OnnxDepth  # noqa: E402
import sintel_io  # noqa: E402


def delta1_aligned(pred, gt):
    p, g = pred.astype(np.float64), gt.astype(np.float64)
    m = (g > 0) & (g < 500) & np.isfinite(p) & np.isfinite(g)
    p, g = p[m], g[m]
    if p.size < 500:
        return None
    A = np.stack([p, np.ones_like(p)], 1)
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    p2 = np.maximum(A @ coef, 1e-6)
    return float((np.maximum(g / p2, p2 / g) < 1.25).mean())


def tae_relative(pred_t, pred_t1, flow_t):
    import cv2
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel-root", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--student-onnx", default="")
    ap.add_argument("--student-engine", default="",
                    help="TensorRT engine; overrides --student-onnx")
    ap.add_argument("--backend", default="ort", choices=["ort", "trt"])
    ap.add_argument("--max-seqs", type=int, default=23)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--skip-dav2", action="store_true")
    ap.add_argument("--dav2-repo", default="")
    ap.add_argument("--dav2-weights", default="")
    args = ap.parse_args()

    if args.backend == "trt":
        from depthstream.backends.trt_gpu import TrtDepth
        model = TrtDepth(args.student_engine)
    else:
        model = OnnxDepth(args.student_onnx)

    seqs = sintel_io.list_depth_seqs(args.sintel_root)[:args.max_seqs]
    print("sequences:", seqs, flush=True)

    names = ["dense", "kf3", "kf10", "kf10-adp"]
    acc = {n: {"d1": [], "tae": [], "ms": []} for n in names}
    if not args.skip_dav2:
        from dav2 import DAv2Baseline
        dav2 = DAv2Baseline(args.dav2_repo, args.dav2_weights)
        acc["dav2l518"] = {"d1": [], "tae": [], "ms": []}

    for si, (scene, seq, _) in enumerate(seqs):
        imgs, gts, flows = sintel_io.load_seq(args.sintel_root, scene, seq,
                                              args.frames)
        if len(imgs) < 4:
            continue
        print(f"[{si+1}/{len(seqs)}] {scene}/{seq}  {len(imgs)} frames",
              flush=True)

        def stream(name, kf):
            st = DepthStream(model, keyframe=kf, flow="dis", flow_width=256,
                             ema=1.0,
                             adaptive=name.endswith("adp"))
            outs, t0 = [], time.perf_counter()
            for im in imgs:
                outs.append(st.process(im))
            return outs, (time.perf_counter() - t0) * 1000 / len(imgs)

        outs_by = {}
        for name in names:
            kf = int("".join(c for c in name if c.isdigit()) or 1)
            outs, ms = stream(name, kf)
            outs_by[name] = outs
            d1s, taes = [], []
            for t in range(len(outs)):
                d1 = delta1_aligned(outs[t], gts[t])
                if d1 is not None:
                    d1s.append(d1)
                if t + 1 < len(outs) and flows[t] is not None:
                    te = tae_relative(outs[t], outs[t + 1], flows[t])
                    if te is not None:
                        taes.append(te)
            acc[name]["d1"].append(np.mean(d1s))
            acc[name]["tae"].append(np.mean(taes))
            acc[name]["ms"].append(ms)

        if not args.skip_dav2:
            t0 = time.perf_counter()
            outs = [dav2(im) for im in imgs]
            ms = (time.perf_counter() - t0) * 1000 / len(imgs)
            d1s, taes = [], []
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

    table = {}
    for name, v in acc.items():
        if not v["d1"]:
            continue
        table[name] = {
            "delta1": round(float(np.mean(v["d1"])), 4),
            "tae_rel": round(float(np.mean(v["tae"])), 4),
            "ms_per_frame": round(float(np.mean(v["ms"])), 2),
            "n_seqs": len(v["d1"]),
        }
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "eval_sintel_result.json")
    with open(out_path, "w") as f:
        json.dump(table, f, indent=1)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
