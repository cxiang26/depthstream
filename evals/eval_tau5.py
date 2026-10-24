"""tau5 (VeloDepth-style temporal consistency) for DepthStream on Sintel.

Runs the student stream (ORT CPU or TRT) over Sintel sequences and reports
tau5 per config: dense / kf3 / kf10-adp / (optional teacher). GT depth used
for the relative-error denominator (full VeloDepth recipe sans egomotion;
poses are not consumed by our runtime). Higher is better.

Usage:
  python evals/eval_tau5.py --sintel-root R --student-onnx P [--skip-dav2]
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
from depthstream.utils import depth_fullres  # noqa: E402
import sintel_io  # noqa: E402
from tau5 import tau5_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel-root", default=os.environ.get(
        "DEPTHSTREAM_SINTEL", "./datasets/sintel"))
    ap.add_argument("--student-onnx", default="")
    ap.add_argument("--engine", default="")
    ap.add_argument("--backend", default="ort", choices=["ort", "trt"])
    ap.add_argument("--max-seqs", type=int, default=23)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--skip-dav2", action="store_true")
    ap.add_argument("--dav2-repo", default="")
    ap.add_argument("--dav2-weights", default="")
    ap.add_argument("--out", default="results/eval_tau5.json")
    args = ap.parse_args()

    if args.backend == "trt":
        from depthstream.backends.trt_gpu import TrtDepth
        model = TrtDepth(args.engine)
    else:
        model = OnnxDepth(args.student_onnx)

    seqs = sintel_io.list_depth_seqs(args.sintel_root)[:args.max_seqs]
    names = ["dense", "kf3", "kf10-adp"]
    acc = {n: [] for n in names}
    if not args.skip_dav2:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from eval_sintel import DAv2Baseline  # noqa: E402
        dav2 = DAv2Baseline(args.dav2_repo, args.dav2_weights)
        acc["dav2l518"] = []

    for si, (scene, seq, _) in enumerate(seqs):
        imgs, gts, flows = sintel_io.load_seq(args.sintel_root, scene, seq,
                                              args.frames)
        if len(imgs) < 4:
            continue
        print(f"[{si+1}/{len(seqs)}] {scene}/{seq}", flush=True)

        for name in names:
            if name == "dense":
                outs = [depth_fullres(model(im), im.shape[:2])
                        for im in imgs]      # pure per-frame baseline
            else:
                kf = int("".join(c for c in name if c.isdigit()) or 1)
                st = DepthStream(model, keyframe=kf, flow="dis",
                                 flow_width=256, ema=1.0,
                                 adaptive=name.endswith("adp"))
                outs = [st.process(im) for im in imgs]
            v = tau5_sequence(outs, flows, gts)
            if v is not None:
                acc[name].append(v)

        if not args.skip_dav2:
            outs = [dav2(im) for im in imgs]
            v = tau5_sequence(outs, flows, gts)
            if v is not None:
                acc["dav2l518"].append(v)

    table = {k: round(float(np.mean(v)), 4) for k, v in acc.items() if v}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=1)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
