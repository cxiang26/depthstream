"""Rich qualitative comparison figures for DepthStream paper.

Figs:
  A. comparison_strip.png  - RGB|GT|dense|kf3|kf10-adp|DAv2-L depth + error rows (Sintel)
  B. spacetime_slice.png   - depth of one scanline stacked over time: dense vs kf10-adp vs GT-warped
  C. flicker_maps.png      - |D_t - warp(D_{t-1})| heatmaps: dense vs kf10-adp (GT flow warp)
  D. bonn_qual.png         - RGB|dense|kf10-adp|DAv2-L on a real Bonn frame
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper")
from depthestream import DepthStream, TrtRelativeDepth, backward_warp  # noqa: E402
import sintel_io  # noqa: E402
import bonn_io  # noqa: E402
from eval_sintel_v2 import DAv2Baseline, delta1_aligned, err_heat  # noqa: E402

FIG = "/home/cq/depthart_test/paper_figs"
SEQ = ("alley", "alley_1")
SLICE_ROW_FRAC = 0.55      # scanline row for space-time slice
GT_CAP = 500.0


def colorize_gt(d, lo, hi):
    dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    dn[~np.isfinite(dn)] = 0
    return cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def colorize_fixed(d, lo, hi):
    dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    dn[~np.isfinite(dn)] = 0
    return cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def label(canvas, text, y=18):
    cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def stack_h(tiles, h):
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    return np.hstack(tiles)


def main():
    import os
    os.makedirs(FIG, exist_ok=True)
    model = TrtRelativeDepth()
    dav2 = DAv2Baseline()

    # ---------- run configs on one Sintel sequence ----------
    imgs, gts, flows = sintel_io.load_seq(sintel_io.SINTEL_ROOT, *SEQ, 50)
    H, W = imgs[0].shape[:2]

    def stream(name, kf, **kw):
        st = DepthStream(model, keyframe=kf, flow="dis", flow_width=256,
                         ema=1.0, **kw)
        return [st.process(im) for im in imgs]

    print("running configs...", flush=True)
    outs = {
        "dense": stream("dense", 1),
        "kf3": stream("kf3", 3),
        "kf10-adp": stream("kf10-adp", 10, adaptive=True),
    }
    outs["dav2l518"] = [dav2(im) for im in imgs]
    print("done", flush=True)

    # ---------- Fig A: comparison strip at a chosen frame ----------
    t = 24   # mid-sequence, object in view
    gt = np.clip(gts[t], 0, 100)
    vlo, vhi = np.percentile(gt[gt > 0], 2), np.percentile(gt[gt > 0], 98)
    row1 = [label(imgs[t].copy(), "RGB"),
            label(colorize_gt(gts[t], vlo, vhi), "GT depth (cap 100m)")]
    row2 = []

    def panel_aligned(d, gt_ref, lo, hi):
        """Affine-align pred to GT (like delta1) so colors share semantics."""
        p, g = d.astype(np.float64), gt_ref.astype(np.float64)
        m = (g > 0) & np.isfinite(p) & np.isfinite(g)
        A = np.stack([p[m], np.ones(m.sum())], 1)
        coef, *_ = np.linalg.lstsq(A, g[m], rcond=None)
        return colorize_fixed(coef[0] * d + coef[1], lo, hi)

    for name in ["dense", "kf3", "kf10-adp", "dav2l518"]:
        d = outs[name][t]
        row1.append(label(panel_aligned(d, gts[t], vlo, vhi), name))
        row2.append(label(err_heat(d, gts[t]), f"{name} err (d1="
                          f"{delta1_aligned(d, gts[t]):.3f})"))
    COL_W, COL_H = 560, 240

    def tile_fixed(im):
        return cv2.resize(im, (COL_W, COL_H), interpolation=cv2.INTER_AREA)

    while len(row2) < len(row1):                       # pad blanks to equal cols
        row2.append(np.zeros((COL_H, COL_W, 3), np.uint8))
    cols = [np.vstack([tile_fixed(r1), tile_fixed(r2)])
            for r1, r2 in zip(row1, row2)]
    strip = np.hstack(cols)
    cv2.imwrite(f"{FIG}/comparison_strip.png", strip)

    # ---------- Fig B: space-time slice ----------
    row = int(H * SLICE_ROW_FRAC)
    def slice_img(dlist):
        sl = np.stack([d[row] for d in dlist])          # (T, W)
        lo, hi = np.nanpercentile(sl, 2), np.nanpercentile(sl, 98)
        return colorize_fixed(sl, lo, hi)
    # GT slice: warp-free, just GT depths
    gt_slice = np.stack([np.clip(g[t][row], 0, 100) for g, t in
                         [(gts, i) for i in range(len(gts))]])
    tiles = [label(colorize_fixed(gt_slice, np.nanpercentile(gt_slice, 2),
                                  np.nanpercentile(gt_slice, 98)), "GT"),
             label(slice_img(outs["dense"]), "dense (per-frame)"),
             label(slice_img(outs["kf10-adp"]), "kf10-adp (stream)")]
    fig_b = np.vstack(tiles)   # all slices share (T, W) shape
    cv2.imwrite(f"{FIG}/spacetime_slice.png", fig_b)

    # ---------- Fig C: flicker maps at frame t (GT flow warp of prev) ----------
    tiles = []
    for name in ["dense", "kf10-adp"]:
        if flows[t - 1] is None:
            continue
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                             np.arange(H, dtype=np.float32))
        warped = cv2.remap(outs[name][t - 1],
                           gx + flows[t - 1][..., 0],
                           gy + flows[t - 1][..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = np.isfinite(warped) & np.isfinite(outs[name][t])
        diff = np.zeros((H, W), np.float32)
        diff[m] = np.abs(warped[m] - outs[name][t][m])
        dnorm = diff / max(float(np.nanmean(diff[m])), 1e-6)
        vis = cv2.applyColorMap(
            (np.clip(dnorm, 0, 2) / 2 * 255).astype(np.uint8),
            cv2.COLORMAP_INFERNO)
        mean_v = float(diff[m].mean())
        tiles.append(label(vis, f"{name}  |dD| mean={mean_v:.4f}"))
    cv2.imwrite(f"{FIG}/flicker_maps.png", np.hstack(tiles))

    # ---------- Fig D: Bonn real-data comparison ----------
    seq = "balloon2"
    bimgs, bgts = bonn_io.load_seq(seq, 40)
    if len(bimgs) >= 20:
        Hb, Wb = bimgs[0].shape[:2]
        outs_b = {"dense": [], "kf10-adp": []}
        for name, kf, kw in [("dense", 1, {}), ("kf10-adp", 10,
                                                {"adaptive": True})]:
            st = DepthStream(model, keyframe=kf, flow="dis", flow_width=256,
                             ema=1.0, **kw)
            outs_b[name] = [st.process(im) for im in bimgs]
        outs_b["dav2l518"] = [dav2(im) for im in bimgs]
        tb = 20
        g = bgts[tb]
        vlo, vhi = np.nanpercentile(g, 2), np.nanpercentile(g, 98)
        r1 = [label(bimgs[tb].copy(), "RGB (Bonn balloon2)"),
              label(colorize_gt(g, vlo, vhi), "GT depth")]

        def panel_aligned_bonn(d):
            p, g2 = d.astype(np.float64), g.astype(np.float64)
            m = (g2 > 0) & np.isfinite(p) & np.isfinite(g2)
            A = np.stack([p[m], np.ones(m.sum())], 1)
            coef, *_ = np.linalg.lstsq(A, g2[m], rcond=None)
            return colorize_fixed(coef[0] * d + coef[1], vlo, vhi)

        for name in ["dense", "kf10-adp", "dav2l518"]:
            r1.append(label(panel_aligned_bonn(outs_b[name][tb]), name))
        strip_b = stack_h(r1, 240)
        for i, name in enumerate(["dense", "kf10-adp", "dav2l518"]):
            d1 = delta1_aligned(outs_b[name][tb], g)
            cv2.putText(strip_b, f"d1={d1:.3f}",
                        (i * strip_b.shape[1] // 5 + (320 if i else 560), 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.imwrite(f"{FIG}/bonn_qual.png", strip_b)

    print("saved:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
