"""One-off: retitle + reorder experiment subsections in paper/main.tex."""
p = "/mnt/c/Users/canqunxiang/ZCodeProject/depthstream-paper/paper/main.tex"
s = open(p, encoding="utf-8").read()

retitles = [
    ("\\subsection{Main results}",
     "\\subsection{A free lunch at $K{=}3$: strict dominance}"),
    ("\\subsection{Ablations and design-space analysis}",
     "\\subsection{Why the lunch is free: mechanism}\n\\label{sec:ablations}"),
    ("\\subsection{Cadence sweep: choosing $K$}",
     "\\subsection{Where the lunch ends: the cadence boundary}"),
    ("\\subsection{Platform and backend invariance}",
     "\\subsection{Purchasing power: platforms and backend invariance}\n\\label{sec:platform}"),
    ("\\subsection{Comparing temporal stability with $\\tau_5$}",
     "\\subsection{Temporal stability, revisited with $\\tau_5$}"),
]
for old, new in retitles:
    assert old in s, "missing: " + old
    s = s.replace(old, new)

titles = [
    "\\subsection{Setup}",
    "\\subsection{A free lunch at $K{=}3$: strict dominance}",
    "\\subsection{Why the lunch is free: mechanism}",
    "\\subsection{Where the lunch ends: the cadence boundary}",
    "\\subsection{Temporal stability, revisited with $\\tau_5$}",
    "\\subsection{Purchasing power: platforms and backend invariance}",
    "\\subsection{Analysis: when does selective refinement pay off?}",
    "\\subsection{Negative result: homography-based geometric propagation}",
]
idx = [s.find(t) for t in titles]
assert all(i >= 0 for i in idx), list(zip(titles, idx))

exp_start = s.find("\\section{Experiments}")
disc_start = s.find("\\section{Discussion}")
assert all(exp_start < i < disc_start for i in idx), "markers outside experiments"

marks = sorted(zip(idx, titles))
blocks = {}
for j, (pos, t) in enumerate(marks):
    end = marks[j + 1][0] if j + 1 < len(marks) else disc_start
    blocks[t] = s[pos:end]

head = s[:marks[0][0]]
tail = s[disc_start:]

desired = [
    "\\subsection{Setup}",
    "\\subsection{A free lunch at $K{=}3$: strict dominance}",
    "\\subsection{Temporal stability, revisited with $\\tau_5$}",
    "\\subsection{Where the lunch ends: the cadence boundary}",
    "\\subsection{Why the lunch is free: mechanism}",
    "\\subsection{Purchasing power: platforms and backend invariance}",
    "\\subsection{Analysis: when does selective refinement pay off?}",
    "\\subsection{Negative result: homography-based geometric propagation}",
]
new = head + "".join(blocks[t] for t in desired) + tail
open(p, "w", encoding="utf-8", newline="\n").write(new)
print("reordered OK; sections now:", [t.split("{")[1].split("}")[0][:30] for t in desired])
