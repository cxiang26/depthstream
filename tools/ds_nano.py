#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DepthStream on Jetson Nano (self-contained, Python 3.6 compatible).

Bundles: TensorRT 8.2 backend + DepthStream runtime (keyframe scheduling,
backward-flow propagation, affine re-anchoring, adaptive scheduling).

Usage:
  python3 ds_nano.py VIDEO [-K 10] [--max-frames N] [--flow dis|farneback]
"""
import argparse
import json
import sys
import time

import cv2
import numpy as np

# --------------------------------------------------------------------------
# TensorRT 8.2 backend (execute_v2 API, torch for CUDA memory)
# --------------------------------------------------------------------------
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class Trt82Backend(object):
    def __init__(self, engine_path, size=224):
        import torch
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = engine.create_execution_context()
        self.bindings = [None, None]
        for i in range(engine.num_bindings):
            ptr = torch.empty(
                tuple(engine.get_binding_shape(i)),
                dtype=torch.float32, device="cuda").data_ptr()
            self.bindings[i] = ptr
            if engine.binding_is_input(i):
                self.in_idx = i
        self.torch = torch
        self.size = size
        self.inp = torch.zeros(1, 3, size, size, device="cuda")

    def __call__(self, frame_bgr):
        torch = self.torch
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = ((img - MEAN) / STD).transpose(2, 0, 1)[None]
        self.inp.copy_(torch.from_numpy(img).cuda())
        out = torch.empty(1, 1, self.size, self.size, device="cuda")
        self.bindings[self.in_idx] = self.inp.data_ptr()
        self.bindings[1 - self.in_idx] = out.data_ptr()
        self.ctx.execute_v2(self.bindings)
        torch.cuda.synchronize()
        return out[0, 0].cpu().numpy()


# --------------------------------------------------------------------------
# DepthStream runtime (mirrors depthestream.DepthStream, ddof-free align)
# --------------------------------------------------------------------------
def depth_fullres(d_small, shape_hw):
    return cv2.resize(d_small, (shape_hw[1], shape_hw[0]),
                      interpolation=cv2.INTER_LINEAR)


def backward_warp(img, flow_bwd):
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow_bwd[..., 0], gy + flow_bwd[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def make_flow_backend(name):
    if name == "dis":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

        def calc(a, b):
            return dis.calc(a, b, None)
        return calc
    return lambda a, b: cv2.calcOpticalFlowFarneback(a, b, None,
                                                     0.5, 3, 15, 3, 5, 1.2, 0)


def upsample_flow(fl, up, size):
    fx = cv2.resize(fl[..., 0] * up, size, interpolation=cv2.INTER_LINEAR)
    fy = cv2.resize(fl[..., 1] * up, size, interpolation=cv2.INTER_LINEAR)
    return np.stack([fx, fy], -1)


def align_affine(a_depth, b_depth, valid):
    a = np.asarray(a_depth)[valid].astype(np.float64)
    b = np.asarray(b_depth)[valid].astype(np.float64)
    if a.size < 100:
        return 1.0, 0.0
    da = a - a.mean()
    db = b - b.mean()
    denom = float(np.sum(da * da))
    if denom < 1e-9:
        return 1.0, 0.0
    scale = float(np.clip(np.sum(da * db) / denom, 0.25, 4.0))
    shift = float(b.mean() - scale * a.mean())
    return scale, shift


class DepthStream(object):
    def __init__(self, model, keyframe=10, flow="dis", flow_width=256,
                 ema=1.0, adaptive=False, min_key_gap=2,
                 res_tau=0.15, res_frac=0.30):
        self.model = model
        self.keyframe = max(1, keyframe)
        self.calc = make_flow_backend(flow)
        self.flow_width = flow_width
        self.ema = ema
        self.adaptive = adaptive
        self.min_key_gap = min_key_gap
        self.res_tau = res_tau
        self.res_frac = res_frac
        self.i = 0
        self.last_key = 0
        self.prev_out = None
        self.prev_gray = None
        self.t_infer = self.t_flow = self.t_warp = 0.0
        self.n_infer = self.n_flow = 0
        self.n_adaptive = 0

    def process(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        fw = self.flow_width
        fh = int(round(H / W * fw / 8) * 8)
        gray_s = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (fw, fh))

        due = (self.i - self.last_key) >= self.keyframe
        is_key = due or self.prev_out is None
        d_prop = None

        if not is_key:
            t0 = time.time()
            fl = upsample_flow(self.calc(gray_s, self.prev_gray), W / fw, (W, H))
            self.t_flow += time.time() - t0
            self.n_flow += 1
            d_prop = backward_warp(self.prev_out, fl)

            if self.adaptive and (self.i - self.last_key) >= self.min_key_gap:
                fl_s = self.calc(gray_s, self.prev_gray)
                warp_s = backward_warp(self.prev_gray, fl_s)
                bad = (np.abs(warp_s.astype(np.float32) -
                              gray_s.astype(np.float32)) > 255 * self.res_tau)
                if bad.mean() > self.res_frac:
                    is_key = True
                    self.n_adaptive += 1

        if is_key:
            t0 = time.time()
            d_full = depth_fullres(self.model(frame_bgr), (H, W))
            self.t_infer += time.time() - t0
            self.n_infer += 1
            if self.prev_out is None or d_prop is None:
                out = d_full
            else:
                a, b = align_affine(d_full, d_prop,
                                    np.isfinite(d_prop) & (d_prop > 0))
                d_aligned = a * d_full + b
                out = (self.ema * d_aligned + (1 - self.ema) *
                       np.where(np.isfinite(d_prop), d_prop, d_aligned))
            self.last_key = self.i
        else:
            out = d_prop

        self.prev_out, self.prev_gray, self.i = out, gray_s, self.i + 1
        return out

    def stats(self):
        total = self.t_infer + self.t_flow + self.t_warp
        n = max(self.i, 1)
        return {
            "frames": self.i, "keyframe_interval": self.keyframe,
            "ms_per_frame": round(total * 1000 / n, 2),
            "fps_total": round(n / max(total, 1e-6), 1),
            "ms_infer_avg": round(self.t_infer * 1000 / max(self.n_infer, 1), 2),
            "ms_flow_avg": round(self.t_flow * 1000 / max(self.n_flow, 1), 2),
            "n_infer": self.n_infer, "n_flow": self.n_flow,
            "n_adaptive": self.n_adaptive,
        }


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("engine")
    ap.add_argument("-K", "--keyframe", type=int, default=10)
    ap.add_argument("--flow", default="dis", choices=["dis", "farneback"])
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--max-frames", type=int, default=90)
    ap.add_argument("--dump-dir", default="",
                    help="save per-frame depth as .npy for offline checks")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("cannot open video: " + args.video)
    model = Trt82Backend(args.engine)

    # warmup: first TRT inference pays CUDA context/workspace init (~seconds)
    warm = np.zeros((480, 640, 3), np.uint8)
    for _ in range(3):
        model(warm)

    stream = DepthStream(model, keyframe=args.keyframe, flow=args.flow,
                         adaptive=args.adaptive)
    if args.dump_dir:
        import os
        os.makedirs(args.dump_dir, exist_ok=True)

    it = 0
    while it < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        out = stream.process(frame)
        if args.dump_dir:
            np.save(os.path.join(args.dump_dir, "d_%05d.npy" % it), out)
        it += 1
    print(json.dumps(stream.stats(), indent=1))


if __name__ == "__main__":
    main()
