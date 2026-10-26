"""Keyframe-cadence sweep on Sintel: K in {1,2,3,5,8,10,15,20}.

Measures delta1 / TAE(GT flow) / ms-per-frame per K (TRT backend by
default). Produces the canonical amortization curve (Fig: ksweep).

Usage:
  python evals/eval_ksweep.py --sintel-root R [--engine E | --onnx P] \
      [--backend trt|ort] [--max-seqs 23] [--frames 50] [--out results/ksweep.json]
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
import sintel_io  # noqa: E402
from eval_sintel import delta1_aligned, tae_relative  # noqa: E402
from depthstream.utils import align_to_gt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel-root", default=os.environ.get(
        "DEPTHSTREAM_SINTEL", "./datasets/sintel"))
    ap.add_argument("--student-onnx", default="")
    ap.add_argument("--student-engine", default="")
    ap.add_argument("--backend", default="trt", choices=["ort", "trt"])
    ap.add_argument("--max-seqs", type=int, default=23)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--out", default="benchmarks/eval_ksweep.json")
    args = ap.parse_args()

    if args.backend == "trt":
        from depthstream.backends.trt_gpu import TrtDepth
        model = TrtDepth(args.student_engine)
    else:
        from depthstream.backends.ort_cpu import OnnxDepth
        model = OnnxDepth(args.student_onnx)

    # warmup (CUDA context / workspace init)
    warm = np.zeros((480, 640, 3), np.uint8)
    for _ in range(3):
        model(warm)

    seqs = sintel_io.list_depth_seqs(args.sintel_root)[:args.max_seqs]
    Ks = [1, 2, 3, 5, 8, 10, 15, 20]
    acc = {K: {"d1": [], "tae": [], "ms": []} for K in Ks}

    for si, (scene, seq, _) in enumerate(seqs):
        imgs, gts, flows = sintel_io.load_seq(args.sintel_root, scene, seq,
                                              args.frames)
        if len(imgs) < 4:
            continue
        print(f"[{si+1}/{len(seqs)}] {scene}/{seq}", flush=True)
        for K in Ks:
            st = DepthStream(model, keyframe=K, flow="dis", flow_width=256,
                             ema=1.0)
            outs = []
            t0 = time.perf_counter()
            for im in imgs:
                outs.append(st.process(im))
            ms = (time.perf_counter() - t0) * 1000 / len(imgs)
            d1s, taes = [], []
            souts = [align_to_gt(o, g) for o, g in zip(outs, gts)]
            for t in range(len(outs)):
                d1 = delta1_aligned(outs[t], gts[t])
                if d1 is not None:
                    d1s.append(d1)
                if t + 1 < len(outs) and flows[t] is not None:
                    te = tae_relative(souts[t], souts[t + 1], flows[t])
                    if te is not None:
                        taes.append(te)
            acc[K]["d1"].append(np.mean(d1s))
            acc[K]["tae"].append(np.mean(taes))
            acc[K]["ms"].append(ms)

    table = {}
    for K, v in acc.items():
        if not v["d1"]:
            continue
        table[str(K)] = {
            "delta1": round(float(np.mean(v["d1"])), 4),
            "tae_rel": round(float(np.mean(v["tae"])), 4),
            "ms_per_frame": round(float(np.mean(v["ms"])), 2),
        }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=1)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
