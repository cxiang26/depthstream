"""Pilot: homography-decomposed propagation vs flow-only (kf10, Sintel).

Variants:
  flow   - DIS backward flow warp (current DepthStream)
  homo   - pure homography warp from RANSAC static inliers
  hsplit - per-pixel blend: homography where flow matches H, flow elsewhere
"""
import json
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper")
from depthestream import (DepthStream, TrtRelativeDepth, align_affine,  # noqa: E402
                          backward_warp, depth_fullres, make_flow_backend,
                          upsample_flow)
import sintel_io  # noqa: E402


def homo_decompose(prev_gray_s, gray_s, fl_s):
    """RANSAC homography on sparse DIS backward-flow correspondences.
    Returns (H cur->prev, inlier_ratio, static_mask_small) or (None, 0, None)."""
    fh, fw = fl_s.shape[:2]
    ys, xs = np.mgrid[0:fh:4, 0:fw:4]
    p_cur = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    p_prev = p_cur + fl_s[ys.ravel(), xs.ravel()].astype(np.float64)
    Hm, inl = cv2.findHomography(p_cur, p_prev, cv2.RANSAC, 1.5)
    if Hm is None or inl is None:
        return None, 0.0, None
    ratio = float(inl.mean())
    # homography-predicted prev position -> residual vs observed flow
    ones = np.hstack([p_cur, np.ones((len(p_cur), 1))])
    proj = (Hm @ ones.T).T
    pred_prev = proj[:, :2] / proj[:, 2:3]
    resid = np.linalg.norm(pred_prev - p_prev, axis=1).reshape(fh // 4, fw // 4)
    static = (resid < 2.0).astype(np.float32)
    static = cv2.dilate(static, np.ones((3, 3), np.uint8))
    return Hm, ratio, static


def propagate(prev_out, prev_gray_s, gray_s, fl_s, up, size, variant):
    fl_full = upsample_flow(fl_s, up, size)
    if variant == "flow":
        return backward_warp(prev_out, fl_full)
    Hm, ratio, static_s = homo_decompose(prev_gray_s, gray_s, fl_s)
    use_homo = (variant == "homo" and Hm is not None) or \
               (variant == "hsplit" and Hm is not None and ratio > 0.3)
    if not use_homo:
        return backward_warp(prev_out, fl_full)
    out_h = cv2.warpPerspective(prev_out, Hm, size,
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    if variant == "homo":
        return out_h
    mask = cv2.resize(static_s, size, interpolation=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask, (0, 0), 9)
    out_f = backward_warp(prev_out, fl_full)
    return mask * out_h + (1 - mask) * out_f


def run_variant(imgs, gts, flows, model, variant, keyframe=10, flow_width=256):
    calc = make_flow_backend("dis")
    prev_out = prev_gray = None
    last_key = 0
    outs, ages = [], []
    d1s = []
    for t, im in enumerate(imgs):
        H, W = im.shape[:2]
        fw = flow_width
        fh = int(round(H / W * fw / 8) * 8)
        gray_s = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (fw, fh))
        is_key = (t - last_key) >= keyframe or prev_out is None
        if is_key:
            d_full = depth_fullres(model(im), (H, W))
            if prev_out is None:
                out = d_full
            else:
                fl_s = calc(gray_s, prev_gray)
                d_prop = propagate(prev_out, prev_gray, gray_s, fl_s,
                                   W / fw, (W, H), variant)
                a, b = align_affine(d_full, d_prop,
                                    np.isfinite(d_prop) & (d_prop > 0))
                out = a * d_full + b   # ema=1.0
            last_key = t
        else:
            fl_s = calc(gray_s, prev_gray)
            out = propagate(prev_out, prev_gray, gray_s, fl_s,
                            W / fw, (W, H), variant)
        # delta1
        g = gts[t].astype(np.float64)
        p = out.astype(np.float64)
        m = (g > 0) & (g < 500) & np.isfinite(p)   # cap sky/outlier depths
        if m.sum() > 500:
            pm, gm = p[m], g[m]
            A = np.stack([pm, np.ones(pm.shape[0])], 1)
            coef, *_ = np.linalg.lstsq(A, gm, rcond=None)
            p2 = np.maximum(A @ coef, 1e-6)
            ratio = np.maximum(gm / p2, p2 / gm)
            d1s.append(float((ratio < 1.25).mean()))
        prev_out, prev_gray = out, gray_s
    return float(np.mean(d1s))


def main():
    seqs = [s for s in sintel_io.list_depth_seqs()
            if s[1] in ("alley_1", "alley_2", "market_2", "temple_2")][:4]
    if len(seqs) < 3:
        seqs = sintel_io.list_depth_seqs()[:4]
    print("pilot seqs:", seqs)
    model = TrtRelativeDepth()

    rows = {}
    for variant in ["flow", "homo", "hsplit"]:
        per = {}
        t_all = time.perf_counter()
        for scene, seq, _ in seqs:
            imgs, gts, flows = sintel_io.load_seq(sintel_io.SINTEL_ROOT,
                                                  scene, seq, 50)
            d1 = run_variant(imgs, gts, flows, model, variant)
            per[seq] = round(d1, 4)
        ms = (time.perf_counter() - t_all) * 1000 / sum(
            min(s[2], 50) for s in seqs)
        rows[variant] = {"per_seq": per,
                         "mean": round(float(np.mean(list(per.values()))), 4),
                         "ms_per_frame": round(ms, 2)}
        print(variant, rows[variant], flush=True)

    with open("/home/cq/depthart_test/pilot_motsplit.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
