"""Re-measure the propagation ablation variants (Table 3) with the
paper-true runtime: kf10 with GT forward flow, kf3 with 384-px flow.

Usage: python evals/eval_sintel_variants.py --sintel-root R --backend trt \
           --student-engine E --out benchmarks/rerun/eval_sintel_variants.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel-root", default="/home/cq/depthart_test")
    ap.add_argument("--student-engine",
                    default="/home/cq/depthart_test/depthart_fp16.engine")
    ap.add_argument("--backend", default="trt", choices=["ort", "trt"])
    ap.add_argument("--max-seqs", type=int, default=23)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--out", default="benchmarks/rerun/eval_sintel_variants.json")
    args = ap.parse_args()

    if args.backend == "trt":
        from depthstream.backends.trt_gpu import TrtDepth
        model = TrtDepth(args.student_engine)
    else:
        from depthstream.backends.ort_cpu import OnnxDepth
        model = OnnxDepth(args.student_onnx)

    variants = [
        ("kf10-oracle", dict(keyframe=10, oracle_flow=True)),
        ("kf3-hiflow", dict(keyframe=3, flow_width=384)),
    ]
    seqs = sintel_io.list_depth_seqs(args.sintel_root)[:args.max_seqs]
    table = {}
    for name, kw in variants:
        acc = {"d1": [], "tae": []}
        for scene, seq, _ in seqs:
            imgs, gts, flows = sintel_io.load_seq(args.sintel_root, scene,
                                                  seq, args.frames)
            if len(imgs) < 4:
                continue
            st = DepthStream(model, flow="dis", ema=1.0, **kw)
            outs = []
            for t, im in enumerate(imgs):
                fwd = flows[t] if t < len(flows) else None
                outs.append(st.process(im, gt_fwd_flow=fwd))
            d1s, taes = [], []
            for t in range(len(outs)):
                d1 = delta1_aligned(outs[t], gts[t])
                if d1 is not None:
                    d1s.append(d1)
                if t + 1 < len(outs) and flows[t] is not None:
                    te = tae_relative(outs[t], outs[t + 1], flows[t])
                    if te is not None:
                        taes.append(te)
            acc["d1"].append(np.mean(d1s))
            acc["tae"].append(np.mean(taes))
            print(f"[{name}] {scene}/{seq}", flush=True)
        table[name] = {"delta1": round(float(np.mean(acc["d1"])), 4),
                       "tae_rel": round(float(np.mean(acc["tae"])), 4)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=1)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
