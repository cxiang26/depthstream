"""Optical-flow backends and depth-field warping utilities."""
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


def align_affine(a_depth, b_depth, valid):
    """Closed-form (scale, shift) mapping a -> b by LSQ over the valid mask.

    Uses explicit centered sums (ddof-free): scale = Σ(Δa·Δb) / Σ(Δa)².
    """
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
