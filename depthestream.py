"""DepthStream: amortized real-time monocular depth for video.

Keyframe frames run the DepthART relative TRT engine (224); in-between frames
propagate the previous full-res depth with optical flow and re-anchor at the
next keyframe with affine (scale,shift) alignment + EMA smoothing.

Usage:
  python depthestream.py <video> [--keyframe N] [--flow dis|farneback]
      [--out out.mp4] [--dump-dir DIR] [--max-frames K]
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

ENGINE = "/home/cq/depthart_test/depthart_fp16.engine"
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class TrtRelativeDepth:
    def __init__(self, engine_path=ENGINE):
        import torch
        import tensorrt as trt
        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            eng = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = eng.create_execution_context()
        names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        self.inn = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.outn = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.stream = torch.cuda.Stream()

    def __call__(self, frame_bgr):
        torch = self.torch
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = ((img - MEAN) / STD).transpose(2, 0, 1)[None]
        inp = torch.from_numpy(np.ascontiguousarray(img)).cuda()
        out = torch.empty(1, 1, 224, 224, device="cuda")
        self.ctx.set_tensor_address(self.inn, inp.data_ptr())
        self.ctx.set_tensor_address(self.outn, out.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return out[0, 0].cpu().numpy()


def depth_fullres(d224, shape_hw):
    return cv2.resize(d224, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)


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
    """(scale, shift) mapping a -> b by LSQ over the valid mask (ddof-free)."""
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
    shift = float(np.mean(b) - scale * np.mean(a))
    return scale, shift


def colorize(d):
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


class DepthStream:
    """Reusable streaming engine: feed frames one by one, get full-res depth.

    oracle_flow: use provided GT forward flow for propagation (ablation).
    adaptive: re-anchor early when photometric residual says flow failed.
    """

    def __init__(self, model, keyframe=3, flow="dis", flow_width=256,
                 ema=1.0, out_size=None, oracle_flow=False, adaptive=False,
                 min_key_gap=2, res_tau=0.15, res_frac=0.30):
        self.model = model
        self.keyframe = max(1, keyframe)
        self.calc = make_flow_backend(flow)
        self.flow_width = flow_width
        self.ema = ema
        self.out_size = out_size
        self.oracle_flow = oracle_flow
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
        self.flicker = []
        self.n_adaptive = 0

    def process(self, frame_bgr, gt_fwd_flow=None):
        H, W = frame_bgr.shape[:2]
        fw = self.flow_width
        fh = int(round(H / W * fw / 8) * 8)
        gray_s = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (fw, fh))

        due = (self.i - self.last_key) >= self.keyframe
        is_key = due or self.prev_out is None
        d_prop = fl = None

        if not is_key and self.prev_out is not None:
            t0 = time.perf_counter()
            if self.oracle_flow and gt_fwd_flow is not None:
                fl = upsample_flow(
                    cv2.resize(gt_fwd_flow, (fw, fh), interpolation=cv2.INTER_AREA),
                    W / fw, (W, H))
            else:
                fl = upsample_flow(self.calc(gray_s, self.prev_gray),
                                   W / fw, (W, H))
            self.t_flow += time.perf_counter() - t0
            self.n_flow += 1
            t1 = time.perf_counter()
            d_prop = backward_warp(self.prev_out, fl)
            self.t_warp += time.perf_counter() - t1

            if self.adaptive and (self.i - self.last_key) >= self.min_key_gap:
                fl_s = self.calc(gray_s, self.prev_gray)
                warp_s = backward_warp(self.prev_gray, fl_s)
                bad = (np.abs(warp_s.astype(np.float32) - gray_s.astype(np.float32))
                       > 255 * self.res_tau)
                if bad.mean() > self.res_frac:
                    is_key = True
                    self.n_adaptive += 1

        if is_key:
            t0 = time.perf_counter()
            d224 = self.model(frame_bgr)
            self.t_infer += time.perf_counter() - t0
            self.n_infer += 1
            d_full = depth_fullres(d224, (H, W))
            if self.prev_out is None or d_prop is None:
                out = d_full
            else:
                a, b = align_affine(d_full, d_prop,
                                    np.isfinite(d_prop) & (d_prop > 0))
                d_aligned = a * d_full + b
                out = (self.ema * d_aligned + (1 - self.ema) *
                       np.where(np.isfinite(d_prop), d_prop, d_aligned))
            if self.prev_out is not None and d_prop is not None:
                m = np.isfinite(out) & np.isfinite(d_prop)
                if m.sum() > 500:
                    self.flicker.append(
                        float(np.abs(out[m] - d_prop[m]).mean()))
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
            "temporal_flicker": round(float(np.mean(self.flicker)), 5)
            if self.flicker else None,
        }

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
            "temporal_flicker": round(float(np.mean(self.flicker)), 5)
            if self.flicker else None,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--keyframe", type=int, default=3)
    ap.add_argument("--flow", default="dis", choices=["dis", "farneback"])
    ap.add_argument("--flow-width", type=int, default=256)
    ap.add_argument("--out", default="depthstream_out.mp4")
    ap.add_argument("--dump-dir", default="")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.6)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = TrtRelativeDepth()
    stream = DepthStream(model, keyframe=args.keyframe, flow=args.flow,
                         flow_width=args.flow_width, ema=args.ema)

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W * 2, H))
    it = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and it >= args.max_frames):
            break
        out = stream.process(frame)
        if args.dump_dir:
            np.save(os.path.join(args.dump_dir, f"d_{it:05d}.npy"), out)
        vw.write(np.hstack([frame, colorize(out)]))
        it += 1

    cap.release()
    vw.release()
    stats = stream.stats()
    stats["out"] = args.out
    print(json.dumps(stats, indent=1))
    if args.dump_dir:
        with open(os.path.join(args.dump_dir, "manifest.json"), "w") as f:
            json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
