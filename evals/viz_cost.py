"""Intermediate-visualization C: validate the cost model Eq.(cost) against
the measured K-sweep latencies. Fits c_m, c_f+c_w from the same data and
plots predicted vs measured with R^2."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ks = json.load(open("/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper/benchmarks/eval_ksweep.json"))
Ks = np.array(sorted(int(k) for k in ks))
ms = np.array([ks[str(k)]["ms_per_frame"] for k in Ks])

# c(K) = c_m/K + (1 - 1/K) * c_p,  c_p = flow + warp per-frame cost
# linear in x = 1/K:  c = c_p + (c_m - c_p) * x
x = 1.0 / Ks
A = np.stack([np.ones_like(x), x], 1)
coef, *_ = np.linalg.lstsq(A, ms, rcond=None)
c_p, c_m = coef
pred = c_p + (c_m - c_p) * x
ss_res = float(np.sum((ms - pred) ** 2))
ss_tot = float(np.sum((ms - ms.mean()) ** 2))
r2 = 1 - ss_res / ss_tot

FIG = "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper/paper/figs"
plt.figure(figsize=(5.4, 3.6))
plt.plot(Ks, ms, "o", ms=6, label="measured (TRT fp16, 1024$\\times$436)")
kk = np.linspace(1, 20, 100)
plt.plot(kk, c_p + (c_m - c_p) / kk, "-", lw=1.6,
         label=f"Eq. cost model  ($R^2$={r2:.3f})")
plt.axhline(c_p, color="gray", ls=":", lw=1)
plt.text(12.5, c_p + 0.12, f"asymptote $c_p$={c_p:.2f} ms (flow+warp)", fontsize=8,
         color="gray")
plt.xlabel("keyframe interval $K$")
plt.ylabel("ms / frame")
plt.title("Cost model: $c(K)=c_p+(c_m-c_p)/K$", fontsize=10)
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG}/costmodel.png", dpi=150)
print(json.dumps({"c_m": round(c_m, 2), "c_p": round(c_p, 2), "R2": round(r2, 4)}))
