"""Optical-flow backends and depth-field warping utilities."""
import math

import cv2
import numpy as np


def make_flow_backend(name: str = "dis"):
    """Return calc(prev_gray, cur_gray) -> flow (H,W,2), flow defined on prev."""
    if name == "dis":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

        def calc(a, b):
            return dis.calc(a, b, None)
        return calc
    if name == "farneback":
        return lambda a, b: cv2.calcOpticalFlowFarneback(
            a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    raise ValueError(f"unknown flow backend: {name}")


def upsample_flow(fl: np.ndarray, up: float, size) -> np.ndarray:
    """Scale a small-resolution flow field up to `size` = (W, H)."""
    fx = cv2.resize(fl[..., 0] * up, size, interpolation=cv2.INTER_LINEAR)
    fy = cv2.resize(fl[..., 1] * up, size, interpolation=cv2.INTER_LINEAR)
    return np.stack([fx, fy], -1)


def backward_warp(img: np.ndarray, flow_bwd: np.ndarray) -> np.ndarray:
    """Sample img at (u + f(u), v + f(v)) per pixel (backward sampling)."""
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow_bwd[..., 0], gy + flow_bwd[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def align_affine(a_depth, b_depth, valid, method="sd", clip=(0.25, 4.0)):
    """Closed-form (scale, shift) mapping a -> b over the valid mask.

    Both fields carry comparable warp/interpolation noise, so a plain OLS
    slope (method="lsq") is attenuated and the reverse slope overshoots;
    iterated over the anchor chain, attenuation collapses the field's
    contrast (measured: 0.44x raw range at K=1).

    method="sd" (default): match the two fields' spread directly,
    alpha = sd(b)/sd(a) with the sign of cov(a,b). Contrast-preserving by
    construction and free of regression attenuation.

    method="deatt": reverse-regression slope var(b)/cov(a,b) (unbiased only
    when all error is in the regressor; overshoots here). Kept for the
    ablation table. Both variants clip the scale to `clip` and fall back to
    identity on degenerate overlaps (<100 px or zero variance).
    """
    a = np.asarray(a_depth)[valid].astype(np.float64)
    b = np.asarray(b_depth)[valid].astype(np.float64)
    if a.size < 100:
        return 1.0, 0.0
    da = a - a.mean()
    db = b - b.mean()
    denom = float(np.sum(da * da))
    denom_b = float(np.sum(db * db))
    cov = float(np.sum(da * db))
    if denom < 1e-9 or denom_b < 1e-9 or abs(cov) < 1e-9:
        return 1.0, 0.0
    if method == "deatt":
        scale = denom_b / cov
    elif method == "sd":
        scale = np.sign(cov) * math.sqrt(denom_b / denom)
    else:
        scale = cov / denom
    sgn = 1.0 if scale >= 0 else -1.0
    scale = sgn * min(max(abs(scale), clip[0]), clip[1])
    shift = float(b.mean() - scale * a.mean())
    return scale, shift
