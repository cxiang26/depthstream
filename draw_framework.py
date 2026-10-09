"""Draw the DepthStream framework figure (PNG 300dpi + PDF)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_W, FIG_H = 13.2, 7.4
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 104)
ax.set_ylim(0, 62)
ax.axis("off")

C_KEY = "#cfe2f3"      # keyframe (model) path
C_KEY_E = "#2f6db5"
C_PROP = "#fde5cd"     # propagation path
C_PROP_E = "#d97a29"
C_FUSE = "#d5efce"     # fusion/output
C_FUSE_E = "#3e8e41"
C_GRAY = "#ececec"
C_GRAY_E = "#888888"
C_ADPT = "#f6d4dd"
C_ADPT_E = "#c2456b"


def box(x, y, w, h, text, fc, ec, fs=9, ls="-", tc="#222222", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.35,rounding_size=0.9",
                                fc=fc, ec=ec, lw=1.4, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.45)


def arrow(x1, y1, x2, y2, color="#555555", ls="-", lw=1.6, label=None,
          lx=0, ly=0, fs=8.5,Mut=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, color=color, lw=lw,
                                 linestyle=ls, zorder=1))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=fs,
                color=color, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.5), zorder=4)


# ---------------- boxes ----------------
box(1, 27, 11, 8, "frame $I_t$\n(video)", C_GRAY, C_GRAY_E, fs=9.5)

box(16, 25, 13, 12, "$t - t_k \\geq K$ ?\n(adaptive:\nresidual trigger)",
    "#fff7e6", "#c9971d", fs=9)

box(34, 42, 22, 10, "frozen student $\\phi$ (6.05M)\nDepthART MNv4-slim-SPF\nTensorRT fp16 @ 224$^2$",
    C_KEY, C_KEY_E, fs=9)
box(60, 44, 13, 7.5, "bilinear $\\uparrow$\n224$^2$ $\\to$ full res", C_KEY, C_KEY_E, fs=9)

box(34, 8, 18, 10, "DIS backward flow\n$F_t$ (est., @256)\n$F_t$: $I_t \\to I_{t-1}$",
    C_PROP, C_PROP_E, fs=9)
box(56, 8, 15, 10, "backward warp\n$\\hat{D}_t = \\mathcal{W}(D_{t-1}, F_t)$",
    C_PROP, C_PROP_E, fs=9)

box(75, 25, 16, 12, "affine re-anchor\n$(\\alpha,\\beta)=\\arg\\min\\,"
    "\\|\\alpha\\hat{D}_t+\\beta-D_t^{\\phi}\\|^2$\nfuse $D_t \\leftarrow \\alpha D_t^{\\phi}+\\beta$",
    C_FUSE, C_FUSE_E, fs=8.6)

box(94, 26, 9, 10, "output\n$D_t$\n(full res)", C_FUSE, C_FUSE_E, fs=9, bold=True)

box(75, 8, 16, 8, "previous depth\n$D_{t-1}$ (feedback)", C_GRAY, C_GRAY_E, fs=9)

box(33, -4, 26, 7, "photometric residual monitor\n"
    "$\\|\\mathcal{W}(I_{t-1},F_t)-I_t\\|>\\theta$ px-frac $>\\eta$",
    C_ADPT, C_ADPT_E, fs=8.6, ls="--")

# ---------------- arrows ----------------
arrow(12, 31, 16, 31)
arrow(29, 33.5, 34, 45, label="YES", ly=1.5, color=C_KEY_E)     # to student
arrow(29, 28, 34, 13, label="NO", ly=-1.5, color=C_PROP_E)      # to flow
arrow(56, 47, 60, 47, color=C_KEY_E)
arrow(73, 47, 77, 37.2, color=C_KEY_E)                          # upsample -> reanchor
arrow(52, 13, 56, 13, color=C_PROP_E)
arrow(71, 13, 77, 25, color=C_PROP_E, label="$\\hat{D}_t$", lx=3, ly=2)
arrow(12, 27, 34, 12, color=C_PROP_E, ls=(0, (4, 2)), lw=1.2)   # frame -> flow (gray input)
arrow(83, 8, 60, 8, color=C_PROP_E)                             # Dprev -> warp (feed)
arrow(94, 26, 83, 16, color=C_PROP_E, ls=(0, (5, 3)), lw=1.3,
      label="$t \\leftarrow t{+}1$", lx=6, ly=-1)
arrow(46, 3, 25, 24, color=C_ADPT_E, ls="--", lw=1.5, label="trigger", lx=-1, ly=14)

# path labels
ax.text(41, 56, "keyframe path (model inference, every $K$ frames)",
        fontsize=9.5, color=C_KEY_E, ha="left", fontweight="bold")
ax.text(34, 21, "propagation path (in-between frames)",
        fontsize=9.5, color=C_PROP_E, ha="left", fontweight="bold")

# legend / footer
ax.text(1, 59, "DepthStream: training-free amortized depth streaming "
        "(no learned propagation components)", fontsize=11,
        fontweight="bold", color="#333333")
ax.text(1, -7.5,
        "all components are training-free and deployable (TensorRT / ONNX Runtime); "
        "color: blue = model path, orange = propagation, green = fusion, "
        "pink dashed = adaptive re-anchoring",
        fontsize=8.5, color="#666666")

plt.tight_layout()
fig.savefig("/home/cq/depthart_test/paper_figs/framework.png", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig("/home/cq/depthart_test/paper_figs/framework.pdf",
            bbox_inches="tight", facecolor="white")
print("framework figure saved")
