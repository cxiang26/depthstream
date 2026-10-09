"""depthart_cascade: speculative-depth inference runtime.

Small student (DepthART MNv4-slim TRT, 224) runs on the full image. Cheap
disagreement signals (horizontal-flip self-consistency + edge strength) pick
the top-p% contested blocks; the big teacher (DAv2-L, fp16) runs ONLY on those
crops. Teacher crops are affine-aligned to the student's global frame on the
crop ring and feather-blended in.

Quality proxy: affine-invariant agreement with the full teacher (no GT needed).
Usage:
  python depthart_cascade.py --images img1 [img2 ...] [--budget 0.2] [--full]
"""
import argparse
import glob
import json
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper")
WORK_RES = (504, 378)          # cascade working resolution (w, h), multiples of 14
GRID = 16                      # blocks per side
TEACHER_IN = 518


# ---------- student (TRT 224, from depthestream) ----------
sys.path.insert(0, "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper")
from depthestream import TrtRelativeDepth, depth_fullres  # noqa: E402


# ---------- teacher (DAv2-L fp16) ----------
class DAv2Teacher:
    def __init__(self):
        import torch
        sys.path.insert(0, "/home/cq/depthart_test/Depth-Anything-V2-main")
        from depth_anything_v2.dpt import DepthAnythingV2
        cfg = {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024],
               "use_bn": False, "use_clstoken": False}
        m = DepthAnythingV2(**cfg)
        sd = torch.load("/home/cq/depthart_test/depth_anything_v2_vitl.pth", map_location="cpu")
        m.load_state_dict(sd)
        self.torch = torch
        self.model = m.cuda().eval().half()

    def __call__(self, rgb):
        """rgb float32 [0,1] HxWx3 -> depth HxW (fp32). H,W must be multiples of 14."""
        import torch
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).cuda().half()[None]
        with torch.no_grad():
            d = self.model(x)
        return d.squeeze().float().cpu().numpy()

    def batch(self, rgbs):
        """List of rgb arrays (same size, multiples of 14) -> list of depths."""
        import torch
        x = torch.from_numpy(np.stack([r.transpose(2, 0, 1) for r in rgbs])).cuda().half()
        with torch.no_grad():
            d = self.model(x)
        return [d[i].float().cpu().numpy() for i in range(d.shape[0])]


def teacher_crop(teacher, rgb, x0, y0, x1, y1):
    """Run teacher on a crop; crop size adjusted to multiple of 14 around box."""
    H, W = rgb.shape[:2]
    cw, ch = x1 - x0, y1 - y0
    cw14, ch14 = max(14, (cw // 14) * 14), max(14, (ch // 14) * 14)
    crop = rgb[y0:y0 + ch14, x0:x0 + cw14]
    if crop.shape[0] != ch14 or crop.shape[1] != cw14:
        pad = np.zeros((ch14, cw14, 3), np.float32)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad
    d = teacher(crop)
    return cv2.resize(d, (cw, ch), interpolation=cv2.INTER_LINEAR)


def norm01(d, lo_p=2, hi_p=98):
    lo, hi = np.percentile(d, lo_p), np.percentile(d, hi_p)
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)


def agreement(pred, ref):
    """MiDaS-style affine-invariant delta1 of pred against reference (higher=better)."""
    p = pred.astype(np.float64)
    r = ref.astype(np.float64)
    m = np.isfinite(p) & np.isfinite(r)
    p, r = p[m], r[m]
    if p.size < 500:
        return 0.0
    A = np.stack([p, np.ones_like(p)], 1)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    p2 = np.maximum(A @ coef, 1e-9)
    ratio = np.maximum(r / p2, p2 / r)
    return float((ratio < 1.25).mean())


def cascade(image_bgr, student, teacher, budget=0.2, collect=None):
    t0 = time.perf_counter()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, WORK_RES, interpolation=cv2.INTER_AREA)
    H, W = rgb.shape[:2]

    # 1) student full + flip self-consistency
    d_s = depth_fullres(student(image_bgr), (H, W))
    d_sf = depth_fullres(student(image_bgr[:, ::-1].copy()), (H, W))[:, ::-1]
    disagree = np.abs(norm01(d_s) - norm01(d_sf))

    # 2) edges
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    edges = np.abs(edges) / (edges.max() + 1e-6)

    # 3) block scoring, top-budget selection
    bh, bw = H // GRID, W // GRID
    score = np.zeros((GRID, GRID), np.float32)
    for i in range(GRID):
        for j in range(GRID):
            blk_d = disagree[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            blk_e = edges[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            score[i, j] = 0.7 * blk_d.mean() + 0.3 * blk_e.mean()
    k = max(1, int(round(budget * GRID * GRID)))
    idx = np.dstack(np.unravel_index(np.argsort(score, axis=None)[::-1], score.shape))[0][:k]

    # 4) teacher on selected crops, BATCHED (chunks of 8 at 518²)
    boxes = []
    for i, j in idx:
        y0, x0 = max(i * bh - bh // 2, 0), max(j * bw - bw // 2, 0)
        y1, x1 = min((i + 1) * bh + bh // 2, H), min((j + 1) * bw + bw // 2, W)
        boxes.append((x0, y0, x1, y1))

    crops, keep = [], []
    for (x0, y0, x1, y1) in boxes:
        cw, ch = x1 - x0, y1 - y0
        crop = rgb[y0:y1, x0:x0 + cw]
        r518 = cv2.resize(crop, (TEACHER_IN, TEACHER_IN), interpolation=cv2.INTER_CUBIC)
        crops.append(r518)
        keep.append((x0, y0, x1, y1, cw, ch))

    t_ms = 0.0
    d_ts = []
    t1 = time.perf_counter()
    for c in range(0, len(crops), 8):
        d_ts.extend(teacher.batch(crops[c:c + 8]))
    t_ms = (time.perf_counter() - t1) * 1000

    d_out = d_s.copy()
    for (x0, y0, x1, y1, cw, ch), d518 in zip(keep, d_ts):
        d_t = cv2.resize(d518, (cw, ch), interpolation=cv2.INTER_LINEAR)

        # affine-align teacher crop to student on the RING (outer band)
        ring = np.zeros((y1 - y0, x1 - x0), bool)
        band = max(4, bh // 4)
        ring[:band, :] = ring[-band:, :] = True
        ring[:, :band] = ring[:, -band:] = True
        a, b = 1.0, 0.0
        sub_s = d_s[y0:y1, x0:x1]
        v = ring & np.isfinite(d_t) & np.isfinite(sub_s)
        if v.sum() > 200:
            aa, bb = sub_s[v], d_t[v]
            var = float(np.var(aa))
            if var > 1e-9:
                a = float(np.clip(np.cov(aa, bb)[0, 1] / var, 0.25, 4.0))
                b = float(np.mean(bb) - a * np.mean(aa))

        # feather blend inside inner region
        inner = np.zeros_like(ring)
        inner[band:-band, band:-band] = True
        alpha = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), band / 2.5)
        blended = a * d_t + b
        d_out[y0:y1, x0:x1] = (alpha * blended + (1 - alpha) * d_out[y0:y1, x0:x1])

    total_ms = (time.perf_counter() - t0) * 1000
    out = {"ms_total": round(total_ms, 2), "ms_teacher": round(t_ms, 1),
           "n_crops": len(idx)}
    if collect is not None:
        collect.append(d_out)
    return d_out, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    args = ap.parse_args()

    files = []
    for p in args.images:
        files.extend(sorted(glob.glob(p)))
    files = files[:12]
    print(f"{len(files)} images")

    student = TrtRelativeDepth()
    teacher = DAv2Teacher()

    rows = []
    refs = []
    for f in files:
        bgr = cv2.imread(f)
        bgr = cv2.resize(bgr, WORK_RES, interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        d_t = teacher(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        t_ref = (time.perf_counter() - t0) * 1000
        refs.append(d_t)
        row = {"img": f.split("/")[-1], "teacher_full_ms": round(t_ref, 1)}

        t0 = time.perf_counter()
        d_s = depth_fullres(student(bgr), (bgr.shape[0], bgr.shape[1]))
        row["student_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        row["student_agree"] = round(agreement(d_s, d_t), 4)

        for p in args.budgets:
            d_c, info = cascade(bgr, student, teacher, budget=p)
            row[f"casc{int(p*100)}_ms"] = info["ms_total"]
            row[f"casc{int(p*100)}_agree"] = round(agreement(d_c, d_t), 4)
        rows.append(row)
        print(json.dumps(row))

    def avg(key):
        vals = [r[key] for r in rows if key in r]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {k: avg(k) for k in
               ["teacher_full_ms", "student_ms", "student_agree"] +
               [f"casc{int(p*100)}_{m}" for p in args.budgets for m in ["ms", "agree"]]}
    print("SUMMARY", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
