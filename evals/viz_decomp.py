"""Intermediate-visualization A: model/transport/stream error decomposition.

Runs DepthStream (kf10) on one Sintel sequence, capturing per-frame:
  - model output D^phi_t (per-frame inference, no streaming)
  - transported prediction D_hat_t (backward-warp of previous anchored field)
  - streamed output D_t
and produces: error maps at 3 sample ages + mean-error-vs-age curves.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper")
sys.path.insert(0, "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper/evals")
from depthstream import DepthStream  # noqa: E402
from depthstream.backends.trt_gpu import TrtDepth  # noqa: E402
from depthstream.flow import make_flow_backend, upsample_flow, backward_warp  # noqa: E402
from depthstream.utils import depth_fullres  # noqa: E402
import sintel_io  # noqa: E402

ROOT = "/home/cq/depthart_test"
SEQ = ("alley", "alley_1")
N = 30
K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper/paper/figs"
BENCH = "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper/benchmarks"

model = TrtDepth("/home/cq/depthart_test/depthart_fp16.engine")
for _ in range(3):
    model(np.zeros((480, 640, 3), np.uint8))

imgs, gts, flows = sintel_io.load_seq(ROOT, SEQ[0], SEQ[1], N)
H, W = imgs[0].shape[:2]
fw = 256
fh = int(round(H / W * fw / 8) * 8)
calc = make_flow_backend("dis")

# per-frame model output (dense reference, for epsilon_m)
model_outs = [depth_fullres(model(im), (H, W)) for im in imgs]

# stream
st = DepthStream(model, keyframe=K, flow="dis", flow_width=256, ema=1.0)
stream_outs, ages, transports = [], [], []
prev_gray = None
for t, im in enumerate(imgs):
    gray_s = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (fw, fh))
    if t > 0:
        fl = upsample_flow(calc(gray_s, prev_gray), W / fw, (W, H))
        transports.append(backward_warp(stream_outs[t - 1], fl))
    else:
        transports.append(None)
    stream_outs.append(st.process(im))
    prev_gray = gray_s


def err_map(pred, gt, cap=500.0):
    """Relative abs error map vs GT (NaN where invalid), after LSQ align."""
    p, g = pred.astype(np.float64), gt.astype(np.float64)
    m = (g > 0) & (g < cap) & np.isfinite(p) & np.isfinite(g)
    A = np.stack([p[m], np.ones(m.sum())], 1)
    coef, *_ = np.linalg.lstsq(A, g[m], rcond=None)
    p2 = np.zeros_like(pred)
    p2[m] = np.maximum(A @ coef, 1e-6)
    e = np.full(pred.shape, np.nan, np.float32)
    e[m] = np.abs(p2[m] - g[m]) / np.maximum(g[m], 1e-6)
    return e


def align_to(pred, ref):
    p, r = pred.astype(np.float64), ref.astype(np.float64)
    m = np.isfinite(p) & np.isfinite(r)
    if m.sum() < 100:
        return pred, 1.0, 0.0
    A = np.stack([p[m], np.ones(m.sum())], 1)
    coef, *_ = np.linalg.lstsq(A, r[m], rcond=None)
    return coef[0] * pred + coef[1], coef[0], coef[1]


# mean errors per frame
em_t, ew_t, es_t, ages_t = [], [], [], []
for t in range(N):
    gt = gts[t]
    em = err_map(model_outs[t], gt)
    es = err_map(stream_outs[t], gt)
    ages_t.append(t - (t // K) * K if K > 1 else 0)
    if t > 0:
        dhat_aligned, _, _ = align_to(transports[t], model_outs[t])
        ew = err_map(dhat_aligned, gt)
    else:
        ew = em
    em_t.append(float(np.nanmean(em)))
    ew_t.append(float(np.nanmean(ew)))
    es_t.append(float(np.nanmean(es)))

# figure: top row = error maps at 3 ages (0, 5, 9); bottom = curves
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

def heat(e, vmax=0.5):
    v = np.clip(np.nan_to_num(e, nan=0.0) / vmax, 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

fig = plt.figure(figsize=(12, 6.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])
sample_ages = [0, max(1, K // 3), K - 1]
for j, age in enumerate(sample_ages):
    t = min(age, N - 1)
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(cv2.cvtColor(heat(err_map(stream_outs[t], gts[t])), cv2.COLOR_BGR2RGB))
    ax.set_title(f"stream error, age={t % K}", fontsize=10)
    ax.axis("off")
ax = fig.add_subplot(gs[1, :])
ax.plot(range(N), em_t, label="model error $\\epsilon_m$ (per-frame)", color="tab:blue")
ax.plot(range(N), ew_t, label="transport error $\\epsilon_w$", color="tab:orange")
ax.plot(range(N), es_t, label=f"stream error (K={K})", color="tab:green", lw=2)
for k in range(K, N, K):
    ax.axvline(k, color="gray", ls=":", lw=0.8)
ax.set_xlabel("frame t")
ax.set_ylabel("mean relative error")
ax.set_title(f"error decomposition over time ({SEQ[1]}, K={K})", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/decomposition{SUFFIX}.png", dpi=150)
plt.close(fig)

json.dump(
    {"em_mean": round(float(np.mean(em_t)), 4),
     "ew_mean": round(float(np.mean(ew_t)), 4),
     "es_mean": round(float(np.mean(es_t)), 4)},
    open(f"{BENCH}/viz_decomp.json", "w"), indent=1)
print("decomposition figure saved;",
      "em=", round(float(np.mean(em_t)), 4),
      "ew=", round(float(np.mean(ew_t)), 4),
      "es=", round(float(np.mean(es_t)), 4))
