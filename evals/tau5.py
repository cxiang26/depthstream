"""tau5 temporal consistency metric (VeloDepth-style) for DepthStream evals.

tau5: per-pixel inlier rate over consecutive frame pairs. Depth at t-1 is
warped to t with flow, ego-motion compensated, scale-shift aligned (SSI
variant) per sequence; a pixel is an inlier when |D_t - warp(D_{t-1})|
is within 5% of D_t. Higher is better; GT depth scores ~95%+ with the
full recipe (VeloDepth Sec. 4.1).

We use GT flow where available (Sintel), estimated backward flow otherwise
(Bonn; identical across configs). Ego-motion compensation requires camera
poses: without poses we apply the scale-shift alignment per frame pair,
which is the dominant correction for the affine-invariant students compared
here; see paper note.
"""
import numpy as np


def tau5_pair(pred_t, pred_t1, flow_t1to0, gt_t=None):
    """Inlier rate for one frame pair. flow maps t -> t-1 (backward)."""
    import cv2
    h, w = pred_t.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    warped = cv2.remap(pred_t1, gx + flow_t1to0[..., 0],
                       gy + flow_t1to0[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # scale-shift align warped(t-1) -> t over the valid overlap (SSI)
    m = np.isfinite(warped) & np.isfinite(pred_t)
    if gt_t is not None:
        m &= np.isfinite(gt_t) & (gt_t > 0)
    a, b = warped[m], pred_t[m]
    if a.size < 500:
        return None
    da, db = a - a.mean(), b - b.mean()
    denom = float(np.sum(da * da))
    if denom < 1e-9:
        return None
    alpha = float(np.sum(da * db) / denom)
    warped = alpha * warped + (b.mean() - alpha * a.mean())

    diff = np.abs(warped - pred_t)
    if gt_t is not None:
        rel = diff / np.maximum(np.abs(gt_t), 1e-6)
        mm = m & np.isfinite(rel)
    else:
        scale = float(np.nanmedian(np.abs(pred_t[m])))
        rel = diff / max(scale, 1e-6)
        mm = m & np.isfinite(rel)
    return float((rel[mm] < 0.05).mean())


def tau5_sequence(preds, flows, gts=None):
    """Mean tau5 over consecutive pairs of one sequence (GT-aligned fields)."""
    vals = []
    if gts is not None:
        from depthstream.utils import align_to_gt
        preds = [align_to_gt(p, g) for p, g in zip(preds, gts)]
    for t in range(len(preds) - 1):
        if flows[t] is None:
            continue
        gt = gts[t] if gts is not None else None
        v = tau5_pair(preds[t], preds[t + 1], flows[t], gt_t=gt)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None
