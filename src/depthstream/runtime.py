"""The DepthStream runtime: keyframe scheduling, propagation, re-anchoring."""
import time

import cv2
import numpy as np

from .flow import align_affine, backward_warp, make_flow_backend, upsample_flow
from .utils import depth_fullres


class DepthStream:
    """Training-free amortized video depth.

    Feed frames one by one via :meth:`process`; receive a full-resolution
    depth field per frame. The wrapped ``model`` (any callable
    ``frame_bgr -> (h, w)`` relative-depth array) runs only on keyframes;
    in-between frames reuse the previous anchored field through backward
    optical-flow warping. Fresh keyframe predictions are affine-aligned to the
    propagated field (re-anchoring), and a photometric-residual monitor can
    trigger keyframes early where flow fails.
    """

    def __init__(self, model, keyframe=10, flow="dis", flow_width=256,
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
        self.n_adaptive = 0
        self.reanchor_jumps = []

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
                bad = (np.abs(warp_s.astype(np.float32) -
                              gray_s.astype(np.float32)) > 255 * self.res_tau)
                if bad.mean() > self.res_frac:
                    is_key = True
                    self.n_adaptive += 1

        if is_key:
            t0 = time.perf_counter()
            d_small = self.model(frame_bgr)
            self.t_infer += time.perf_counter() - t0
            self.n_infer += 1
            d_full = depth_fullres(d_small, (H, W))
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
                    self.reanchor_jumps.append(
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
            "frames": self.i,
            "keyframe_interval": self.keyframe,
            "ms_per_frame": round(total * 1000 / n, 2),
            "fps_total": round(n / max(total, 1e-6), 1),
            "ms_infer_avg": round(self.t_infer * 1000 / max(self.n_infer, 1), 2),
            "ms_flow_avg": round(self.t_flow * 1000 / max(self.n_flow, 1), 2),
            "n_infer": self.n_infer,
            "n_flow": self.n_flow,
            "n_adaptive": self.n_adaptive,
            "reanchor_jump_mean": round(float(np.mean(self.reanchor_jumps)), 5)
            if self.reanchor_jumps else None,
        }
