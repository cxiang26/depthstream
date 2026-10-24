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

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depthstream import DepthStream  # noqa: E402
from depthstream.backends.ort_cpu import OnnxDepth  # noqa: E402
from depthstream.flow import backward_warp  # noqa: E402
from depthstream.utils import depth_fullres  # noqa: E402
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
    def __init__(self, repo_dir, weights, encoder="vitl"):
        import torch
        sys.path.insert(0, repo_dir)
        from depth_anything_v2.dpt import DepthAnythingV2
        cfg = {"encoder": encoder, "features": 256,
               "out_channels": [256, 512, 1024, 1024],
               "use_bn": False, "use_clstoken": False}
        m = DepthAnythingV2(**cfg)
        m.load_state_dict(torch.load(weights, map_location="cpu"))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel-root", default=os.environ.get(
        "DEPTHSTREAM_SINTEL", "./datasets/sintel"))
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--student-onnx", default="")
    ap.add_argument("--student-engine", default="")
    ap.add_argument("--backend", default="ort", choices=["ort", "trt"])
    ap.add_argument("--max-seqs", type=int, default=23)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--skip-dav2", action="store_true")
    ap.add_argument("--ema", type=float, default=1.0)
    ap.add_argument("--tag", default="")
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
    per_seq = {n: {} for n in names}
    if not args.skip_dav2:
        acc["dav2l518"] = {"d1": [], "tae": [], "ms": []}
        dav2 = DAv2Baseline(args.dav2_repo, args.dav2_weights)

    for si, (scene, seq, _) in enumerate(seqs):
        imgs, gts, flows = sintel_io.load_seq(args.sintel_root, scene, seq,
                                              args.frames)
        if len(imgs) < 4:
            continue
        print(f"[{si+1}/{len(seqs)}] {scene}/{seq}  {len(imgs)} frames",
              flush=True)

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
                                 flow_width=256, ema=args.ema,
                                 adaptive=name.endswith("adp"))
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
                if t + 1 < len(outs) and flows[t] is not None:
                    te = tae_relative(outs[t], outs[t + 1], flows[t])
                    if te is not None:
                        taes.append(te)
            acc[name]["d1"].append(np.mean(d1s))
            acc[name]["tae"].append(np.mean(taes))
            acc[name]["ms"].append(ms)
            per_seq.setdefault(name, {})[seq] = round(float(np.mean(d1s)), 4)

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
    suffix = ("_" + args.tag) if args.tag else ("_ema" + str(args.ema) if args.ema != 1.0 else "")
    with open(os.path.join(args.out_dir, f"eval_sintel_result{suffix}.json"), "w") as f:
        json.dump(table, f, indent=1)
    with open(os.path.join(args.out_dir, f"eval_sintel_perseq{suffix}.json"), "w") as f:
        json.dump(per_seq, f, indent=1)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
