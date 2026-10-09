# DepthStream — Training-Free Amortized Depth Streaming

Companion code for the paper
*"DepthStream: Training-Free Amortized Streaming for Real-Time, Temporally
Consistent Video Depth"*.

DepthStream wraps a frozen 6M-parameter monocular depth student
(DepthART MobileNetV4-M-slim-SPF, TensorRT fp16) in a completely
training-free streaming runtime:

- **keyframe inference** — the student runs every `K` frames;
- **backward-flow propagation** — in-between frames reuse the previous
  full-resolution depth field through DIS optical flow;
- **affine re-anchoring** — fresh keyframe predictions are aligned to the
  propagated field by closed-form least squares (no EMA blending);
- **photometric-residual adaptive scheduling** — flow failure triggers early
  re-anchoring.

Result highlights (Sintel 23 seq / Bonn RGB-D): kf3 matches per-frame accuracy
exactly while cutting latency 20% and temporal error 15%; at 1/6.7–1/13.7 of
the DAv2-L teacher's latency it retains 93.6–98.2% accuracy; accuracy is
backend-invariant (TensorRT fp16 GPU vs ONNX-Runtime fp32 CPU, |Δδ1| ≤ 1e-4);
CPU-only deployment reaches 249 FPS end-to-end.

## Repository layout

```
depthestream.py        core runtime (DepthStream class + TRT student backend)
depthart_cascade.py    speculative-cascade runtime (negative-result analysis, §4.6)
photoruler.py          single-photo real-world measurement app (EXIF → metric depth)
eval_sintel_v2.py      main Sintel evaluation (6 configs + DAv2-L baselines)
eval_bonn.py           Bonn RGB-D evaluation
bench_cpu.py           CPU platform benchmark (thread sweep, INT8, E2E, backend agreement)
sintel_io.py           Sintel readers (.dpt PIEH depth, .flo flow)
bonn_io.py             Bonn RGB-D readers (timestamp pairing)
qual_figs.py           qualitative figures (comparison strip, space-time slice, flicker maps)
plots.py               statistical figures (Pareto, accumulation, per-sequence)
draw_framework.py      framework diagram (Fig. 1)
convert_fp16.py        ONNX fp32 → fp16 conversion
build_engine.py        TensorRT engine build + benchmark
tools/                 dataset fetchers, one-off diagnostics, superseded experiments
docs/                  paper outline + consolidated results
paper/                 LaTeX draft (main.tex) + figures
results/               evaluation result JSONs + rendered figures
```

## Environment

- Linux (tested in WSL2 Ubuntu 20.04) + NVIDIA GPU (tested RTX 3060 Laptop 6GB)
- Python 3.11, PyTorch 2.7.1+cu126 — see `requirements.txt`
- TensorRT (tested 11.3 via pip wheels); ONNX Runtime 1.26 for CPU path

```bash
pip install -r requirements.txt
```

## Models

| Role | Artifact |
|---|---|
| student (relative depth) | DepthART MNv4-M-slim-SPF @224 — official ONNX → TRT fp16 engine |
| teacher (baseline) | Depth Anything V2-Large, fp16 @518 |
| metric head (PhotoRuler) | DepthART metric TinyViM-S @448 (indoor / outdoor) |

## Reproduce the paper

```bash
# Sintel main table (23 seqs, needs Sintel complete + depth packages)
python eval_sintel_v2.py --max-seqs 23 --frames 50
# Bonn (needs 3 dynamic sequences)
python eval_bonn.py --frames 60
# CPU platform (accuracy invariance + latency)
python bench_cpu.py
# figures
python plots.py && python qual_figs.py && python draw_framework.py
```

Result JSONs land in `results/` (mirrored copies committed under `results/`).

## Notes

- Student/teacher weights inherit CC BY-NC 4.0 terms from DepthART and
  Depth Anything V2 — non-commercial use of the weights; the runtime itself
  carries no such restriction.
- Known pitfalls handled in the code: Sintel GT sky outliers (capped at
  500 m), Bonn rgb/depth timestamp pairing, PIEH `.dpt` depth format.
