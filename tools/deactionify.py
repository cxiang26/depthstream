"""One-off: neutralize marketing headings and lunch vocabulary in main.tex."""
p = "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper/paper/main.tex"
s = open(p, encoding="utf-8").read()

subs = [
    ("\\subsection{A free lunch at $K{=}3$: strict dominance}",
     "\\subsection{Main results: streaming matches per-frame accuracy}"),
    ("\\subsection{Why the lunch is free: mechanism}",
     "\\subsection{Ablations: the improvement is structural}"),
    ("\\subsection{Where the lunch ends: the cadence boundary}",
     "\\subsection{Cadence sweep}"),
    ("\\subsection{Purchasing power: platforms and backend invariance}",
     "\\subsection{Platforms and backend invariance}"),
    # conclusion rewrite (paper voice)
    ("""For a depth model distilled to its last parameter, the remaining efficiency
is temporal --- and it can be harvested for free. DepthStream runs a frozen
6\\,M student on keyframes, transports the anchored depth field with
backward flow, and re-anchors fresh predictions with a closed-form affine
fit; a photometric monitor guards the schedule. The free lunch is real and
strict: at $K{=}3$ the stream matches per-frame accuracy exactly while
cutting latency and raising temporally consistent pixels from 55\\% to 73\\%,
on synthetic and real benchmarks alike. The lunch has charted boundaries ---
a cadence knee at $K^*{\\approx}10$, occlusion-dense penalties for blending,
and a distillation condition --- and its purchasing power flips the
deployment map: 90\\,FPS stable depth on a 2019 \\$99 board, 249\\,FPS on a
CPU, identical accuracy on either backend. For practitioners the rule is
one sentence: propagate --- do not correct; and re-anchor --- do not
smooth.""",
     """Streaming depth, formulated as anchored-field maintenance, changes what
a tiny model can do: at $K{=}3$ the stream matches per-frame accuracy
exactly while lowering latency and raising temporally consistent pixels
from 55\\% to 73\\%, on synthetic and real benchmarks alike. The error
decomposition explains both the gain and its limits --- the cadence knee at
$K^*{\\approx}10$, the occlusion bound on transport error, and the
distillation closure that leaves no room for learned correction on
distilled students. Operationally, the formulation moves real-time video
depth onto hardware where the per-frame formulation cannot go: 90\\,FPS on
a 2019 \\$99 board, 249\\,FPS on a CPU, identical accuracy on either
backend. The practical rule follows from the decomposition: propagate ---
do not correct; re-anchor --- do not smooth."""),
]
for old, new in subs:
    assert old in s, "MISSING: " + old[:70]
    s = s.replace(old, new)

# sweep remaining lunch/harvest/purchasing vocabulary in body
import re
for w in ["lunch", "harvest", "purchasing", "squeezed dry", "found it dry"]:
    s = re.sub(r"(?i)\b" + w + r"\b", "%%LUNCH%%", s) if False else s

open(p, "w", encoding="utf-8", newline="\n").write(s)
import subprocess
EOF_MARKER = None
print("retitled + conclusion rewritten")
