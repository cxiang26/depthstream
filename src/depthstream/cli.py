"""Command-line entry point: depthstream run VIDEO [options]."""
import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="depthstream",
        description="Training-free amortized video depth streaming "
                    "with a tiny monocular depth model.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="stream depth for a video file")
    run.add_argument("video")
    run.add_argument("-o", "--out", default="depthstream_out.mp4")
    run.add_argument("-K", "--keyframe", type=int, default=10,
                     help="keyframe cadence in frames (default 10)")
    run.add_argument("--backend", default="ort", choices=["ort", "trt"])
    run.add_argument("--flow", default="dis", choices=["dis", "farneback"])
    run.add_argument("--flow-width", type=int, default=256)
    run.add_argument("--adaptive", action="store_true",
                     help="photometric-residual early re-anchoring")
    run.add_argument("--onnx", default="", help="student ONNX path "
                     "(default: auto-download to ~/.cache/depthstream)")
    run.add_argument("--engine", default="", help="TensorRT engine path "
                     "(backend=trt)")
    run.add_argument("--threads", type=int, default=4,
                     help="CPU threads for the ort backend")
    run.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args(argv)

    import cv2
    import numpy as np

    from depthstream import DepthStream
    from depthstream.utils import colorize

    if args.backend == "trt":
        from depthstream.backends.trt_gpu import TrtDepth
        model = TrtDepth(args.engine or args.onnx)
    else:
        from depthstream.backends.ort_cpu import OnnxDepth
        path = args.onnx
        if not path:
            from depthstream.model_zoo import ensure_student_onnx
            path = ensure_student_onnx()
        model = OnnxDepth(path, threads=args.threads)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stream = DepthStream(model, keyframe=args.keyframe, flow=args.flow,
                         flow_width=args.flow_width, adaptive=args.adaptive)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                         (W * 2, H))
    it = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and it >= args.max_frames):
            break
        out = stream.process(frame)
        panel = np.hstack([frame, colorize(out)])
        vw.write(panel)
        it += 1
    cap.release()
    vw.release()
    print(json.dumps(stream.stats(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
