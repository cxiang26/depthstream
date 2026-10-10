"""Generate paper figures from eval_v2_result.json + qual/ dumps."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "/home/cq/depthart_test"
FIG = f"{OUT}/paper_figs"
import os  # noqa: E402
os.makedirs(FIG, exist_ok=True)

r = json.load(open(f"{OUT}/eval_v2_result.json"))
S = {k: v for k, v in r.items() if not k.startswith("_")}

# ---------- Fig 1: Pareto (latency vs accuracy) ----------
plt.figure(figsize=(6, 4))
pts = {
    "DAv2-L 518": ("dav2l518", "red", "^"),
    "DAv2-L 336": ("dav2l336", "salmon", "v"),
    "dense": ("dense", "tab:blue", "o"),
    "kf3": ("kf3", "tab:cyan", "s"),
    "kf10-adp": ("kf10-adp", "tab:green", "D"),
    "kf10": ("kf10", "tab:olive", "P"),
}
for label, (key, c, m) in pts.items():
    if key not in S:
        continue
    plt.scatter(S[key]["ms_per_frame"], S[key]["delta1"], c=c, marker=m, s=70,
                label=label, zorder=3)

# Jetson Nano 4GB (640x480 synthetic clip, no GT -> latency-only, MAXN):
# dense / kf3 / kf10 / kf10-adp. Plotted at the Sintel mean delta1 of the
# same config for vertical positioning; dashed line = the Nano curve.
NANO = {"dense": 47.2, "kf3": 20.9, "kf10": 11.2, "kf10-adp": 11.2}
nano_pts = [(NANO[k], S[k]["delta1"]) for k in NANO if k in S]
if nano_pts:
    xs, ys = zip(*nano_pts)
    plt.plot(xs, ys, color="tab:purple", lw=1.2, ls="--", alpha=0.6, zorder=2)
    plt.scatter(xs, ys, c="tab:purple", marker="*", s=110, zorder=3,
                label="Jetson Nano 4GB (latency only)")
plt.xscale("log")
plt.xlabel("latency (ms/frame, log)  -  lower is better")
plt.ylabel("delta1 (aligned)  -  higher is better")
plt.title("DepthStream: latency vs accuracy on Sintel (6 seqs)")
plt.grid(alpha=0.3)
plt.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(f"{FIG}/pareto.png", dpi=150)
plt.close()

# ---------- Fig 2: propagation error accumulation ----------
plt.figure(figsize=(6, 4))
for name, label, c in [("kf10", "kf10 fixed", "tab:olive"),
                       ("kf10-adp", "kf10 adaptive", "tab:green"),
                       ("kf10-oracle", "kf10 fwd-GT-flow", "tab:gray")]:
    b = r["_accum"].get(name)
    if not b:
        continue
    ks = sorted(int(k) for k in b)
    plt.plot(ks, [b[str(k)] for k in ks], marker="o", ms=4, label=label, color=c)
plt.xlabel("frames since last keyframe")
plt.ylabel("delta1 (aligned)")
plt.title("Propagation error accumulation (kf10)")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG}/accum.png", dpi=150)
plt.close()

# ---------- Fig 3: per-sequence grouped bars ----------
ps = r["_per_seq"]
seqs = sorted(ps["dense"].keys())
sel = ["dense", "kf3", "kf10-adp", "dav2l518"]
x = np.arange(len(seqs))
w = 0.2
plt.figure(figsize=(8, 4))
for i, name in enumerate(sel):
    vals = [ps[name].get(s, np.nan) for s in seqs]
    plt.bar(x + (i - 1.5) * w, vals, w, label=name)
plt.xticks(x, seqs, rotation=20, fontsize=8)
plt.ylabel("delta1 (aligned)")
plt.title("Per-sequence accuracy breakdown")
plt.grid(axis="y", alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG}/perseq.png", dpi=150)
plt.close()

# ---------- Fig 4: qualitative strip ----------
import cv2  # noqa: E402
rows = ["rgb", "dense_depth", "kf10_depth", "kf10_err"]
tiles = []
for name in rows:
    p = f"{OUT}/qual/{name}.png"
    if os.path.exists(p):
        tiles.append(cv2.imread(p))
if tiles:
    h = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    strip = np.hstack(tiles)
    labels = ["RGB", "dense depth", "kf10 depth", "kf10 error (red=bad)"]
    y0 = 22
    canvas = np.zeros((strip.shape[0] + y0, strip.shape[1], 3), np.uint8)
    canvas[y0:] = strip
    tw = strip.shape[1] // len(tiles)
    for i, lb in enumerate(labels):
        cv2.putText(canvas, lb, (i * tw + 8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(f"{FIG}/qualitative.png", canvas)

print("figures saved to", FIG)
for f in sorted(os.listdir(FIG)):
    print(" -", f)
