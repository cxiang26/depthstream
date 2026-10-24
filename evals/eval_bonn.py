"""DepthStream evaluation on Bonn RGB-D Dynamic (real data, Kinect GT).

Usage:
  python evals/eval_bonn.py --bonn-root /path/to/bonn \
      --student-onnx /path/to/student.onnx [--frames 60]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depthstream import DepthStream  # noqa: E402
from depthstream.backends.ort_cpu import OnnxDepth  # noqa: E402
from depthstream.flow import make_flow_backend, upsample_flow  # noqa: E402
from depthstream.utils import depth_fullres  # noqa: E402
import bonn_io  # noqa: E402


def delta1_aligned(pred, gt, cap=10.0):
    p, g = pred.astype(np.float64), gt.astype(np.float64)
    m = (g > 0) & (g <= cap) & np.isfinite(p) & np.isfinite(g)
    p, g = p[m], g[m]
    if p.size < 500:
        return None
    A = np.stack([p, np.ones_like(p)], 1)
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    p2 = np.maximum(A @ coef, 1e-6)
    return float((np.maximum(g / p2, p2 / g) < 1.25).mean())


def tae_est(pred_t, pred_t1, flow_bwd):
    warped = cv2.remap(pred_t1, flow_bwd[..., 0].astype(np.float32),
                       flow_bwd[..., 1].astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = np.isfinite(warped) & np.isfinite(pred_t)
    a, b = warped[m], pred_t[m]
    if a.size < 500:
        return None
    A = np.stack([a, np.ones_like(a)], 1)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(np.abs(A @ coef - b).mean()) / max(float(np.abs(b).mean()), 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bonn-root", default=os.environ.get(
        "DEPTHSTREAM_BONN", "./datasets/bonn"))
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--student-onnx", default="")
    ap.add_argument("--backend", default="ort", choices=["ort", "trt"])
    ap.add_argument("--student-engine", default="")
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    if args.backend == "trt":
        from depthstream.backends.trt_gpu import TrtDepth
        model = TrtDepth(args.student_engine)
    else:
        model = OnnxDepth(args.student_onnx)

    calc = make_flow_backend("dis")
    names = ["dense", "kf3", "kf10-adp"]
    acc = {n: {"d1": [], "tae": [], "ms": []} for n in names}
    per_seq = {n: {} for n in names}

    for seq, _ in bonn_io.list_seqs(args.bonn_root):
        imgs, gts = bonn_io.load_seq(seq, args.frames, root=args.bonn_root)
        if len(imgs) < 10:
            continue
        print(f"[{seq}] {len(imgs)} frames", flush=True)
        H, W = imgs[0].shape[:2]
        fw = 256
        fh = int(round(H / W * fw / 8) * 8)

        for name in names:
            if name == "dense":
                # Pure per-frame baseline: no scheduling, no re-anchoring.
                outs = []
                t0 = time.perf_counter()
                for im in imgs:
                    outs.append(depth_fullres(model(im), im.shape[:2]))
                ms = (time.perf_counter() - t0) * 1000 / len(imgs)
            else:
                kf = int("".join(c for c in name if c.isdigit()) or 1)
                st = DepthStream(model, keyframe=kf, flow="dis",
                                 flow_width=fw, ema=1.0,
                                 adaptive="adp" in name)
                outs = []
                t0 = time.perf_counter()
                for im in imgs:
                    outs.append(st.process(im))
                ms = (time.perf_counter() - t0) * 1000 / len(imgs)

            d1s, taes = [], []
            for t in range(len(outs)):
                d1 = delta1_aligned(outs[t], gts[t])
                if d1 is not None:
                    d1s.append(d1)
                if t + 1 < len(outs):
                    g0 = cv2.resize(cv2.cvtColor(imgs[t], cv2.COLOR_BGR2GRAY),
                                    (fw, fh))
                    g1 = cv2.resize(cv2.cvtColor(imgs[t + 1],
                                                 cv2.COLOR_BGR2GRAY), (fw, fh))
                    fl = upsample_flow(calc(g1, g0), W / fw, (W, H))
                    te = tae_est(outs[t], outs[t + 1], fl)
                    if te is not None:
                        taes.append(te)
            acc[name]["d1"].append(np.mean(d1s))
            acc[name]["tae"].append(np.mean(taes))
            acc[name]["ms"].append(ms)
            per_seq.setdefault(name, {})[seq] = round(float(np.mean(d1s)), 4)

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
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "eval_bonn_result.json"), "w") as f:
        json.dump(table, f, indent=1)
    with open(os.path.join(args.out_dir, "eval_bonn_perseq.json"), "w") as f:
        json.dump(per_seq, f, indent=1)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
