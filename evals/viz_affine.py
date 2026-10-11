"""Intermediate-visualization B: affine re-anchoring mechanics.

Left: per-keyframe (alpha_t, beta_t) sequences over a Sintel clip — the
temporal drift that per-frame inference carries and re-anchoring absorbs.
Right: scatter of propagated vs model output for one keyframe with the
fitted affine line and the identity reference.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper")
sys.path.insert(0, "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper/evals")
from depthstream.backends.trt_gpu import TrtDepth  # noqa: E402
from depthstream.flow import make_flow_backend, upsample_flow, backward_warp  # noqa: E402
from depthstream.utils import depth_fullres  # noqa: E402
import sintel_io  # noqa: E402

ROOT = "/home/cq/depthart_test"
SEQ = ("alley", "alley_1")
N, K = 50, 5
OUT = "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper/paper/figs"

model = TrtDepth("/home/cq/depthart_test/depthart_fp16.engine")
for _ in range(3):
    model(np.zeros((480, 640, 3), np.uint8))

imgs, gts, flows = sintel_io.load_seq(ROOT, SEQ[0], SEQ[1], N)
H, W = imgs[0].shape[:2]
fw = 256
fh = int(round(H / W * fw / 8) * 8)
calc = make_flow_backend("dis")

alphas, betas, key_ts = [], [], []
scatter = None
prev_gray = prev_anchored = None
for t, im in enumerate(imgs):
    d_full = depth_fullres(model(im), (H, W))
    gray_s = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (fw, fh))
    if t > 0 and (t % K) == 0:
        fl = upsample_flow(calc(gray_s, prev_gray), W / fw, (W, H))
        dhat = backward_warp(prev_anchored, fl)
        # fit D^phi -> anchor frame: output = a * dhat + b  (per runtime:
        # fit model output to propagated, then output = alpha*Dphi+beta)
        m = np.isfinite(dhat) & np.isfinite(d_full)
        A = np.stack([d_full[m], np.ones(m.sum())], 1)
        coef, *_ = np.linalg.lstsq(A, dhat[m], rcond=None)
        a_out = float(np.clip(coef[0], 0.25, 4.0))
        b_out = float(np.mean(dhat[m]) - a_out * np.mean(d_full[m]))
        alphas.append(a_out)
        betas.append(b_out)
        key_ts.append(t)
        if t == 20:
            sub = np.random.default_rng(0).choice(
                np.where(m.ravel())[0], 4000, replace=False)
            scatter = (d_full.ravel()[sub], dhat.ravel()[sub], a_out, b_out)
    prev_gray = gray_s
    prev_anchored = d_full if t == 0 or (t % K) == 0 else prev_anchored
    # NOTE: anchored field = the re-anchored keyframe output; approximate by
    # the raw keyframe prediction for drift visualization purposes.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
ax1.plot(key_ts, alphas, "o-", color="tab:blue", ms=5, label=r"$\alpha_t$ (scale)")
ax2b = ax1.twinx()
ax2b.plot(key_ts, betas, "s--", color="tab:red", ms=5, label=r"$\beta_t$ (shift)")
ax1.set_xlabel("keyframe time $t$")
ax1.set_ylabel(r"$\alpha_t$", color="tab:blue")
ax2b.set_ylabel(r"$\beta_t$ (px)", color="tab:red")
ax1.set_title("Affine drift absorbed at each keyframe", fontsize=10)
ax1.grid(alpha=0.3)

if scatter is not None:
    xs, ys, a, b = scatter
    ax2.scatter(xs, ys, s=3, alpha=0.25, color="tab:gray")
    lo, hi = np.nanpercentile(xs, [1, 99])
    line = np.array([lo, hi])
    ax2.plot(line, a * line + b, "-", color="tab:green", lw=2,
             label=f"re-anchor fit $\\alpha$={a:.3f}")
    ax2.plot(line, line, ":", color="black", lw=1, label="identity")
    ax2.set_xlabel("model output $D^{\\phi}_t$")
    ax2.set_ylabel("transported field $\\hat{D}_t$")
    ax2.set_title("One keyframe: what the affine fit absorbs", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(f"{OUT}/affine_drift.png", dpi=150)
print("affine drift figure saved; alphas:", [round(a, 3) for a in alphas])
