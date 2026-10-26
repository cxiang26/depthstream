# DepthStream

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)]()

**Training-free amortized video depth streaming for tiny monocular depth
models.**

DepthStream wraps a frozen 6M-parameter monocular depth student
(DepthART MobileNetV4-M-slim-SPF, TensorRT fp16) in a completely
training-free streaming runtime:

- **keyframe inference** — the student runs every `K` frames;
- **backward-flow propagation** — in-between frames reuse the previous
  full-resolution depth field through DIS optical flow;
- **spread-matching affine re-anchoring** — fresh keyframe predictions are
  aligned to the propagated field with a contrast-preserving
  spread-matching fit (sign-of-covariance scale), robust under
  errors-in-variables where plain least squares collapses contrast;
- **photometric-residual adaptive scheduling** — flow failure triggers
  early re-anchoring.

At `K=1` the runtime degenerates to per-frame inference exactly; with
`K>1` per-frame accuracy is matched or exceeded while end-to-end latency
drops 2–3.5× and frame-to-frame temporal error drops ~30%.

## Measured performance (640×480)

| platform | backend | per-frame | kf10 streaming |
|---|---|---|---|
| RTX 3060 Laptop 6 GB | TensorRT fp16 | 12.33 ms | **5.74 ms (174 FPS)** |
| desktop CPU | ONNX Runtime fp32 | 27.53 ms | **7.39 ms (135 FPS)** |
| Jetson Nano 4 GB | TensorRT fp16 | 77.65 ms | **22.23 ms (45 FPS)** |

Accuracy on Sintel (δ1 after GT alignment): kf3 = per-frame exactly
(0.461); kf10 0.441; temporal alignment error kf10 0.103 vs per-frame
0.151. Numbers are backend-invariant (TRT fp16 GPU vs ORT fp32 CPU,
|Δδ1| ≤ 1e-4).

## Repository layout

```
src/depthstream/       runtime package: DepthStream scheduler, DIS flow,
                       spread-matching re-anchoring, TRT/ORT backends
evals/                 dataset evaluation: Sintel, Bonn RGB-D, τ5/TAE
                       temporal metrics (canonical ORT backend), DAv2 baselines,
                       adaptive-trigger statistics
tools/                 bench_unified.py  unified 3-backend timing harness
                       build_trt8_fp16.py  TensorRT 8 fp16 engine builder (Nano)
                       ds_nano.py  self-contained Jetson Nano deployment (py3.6)
                       nano_slam_fix/  DeepStream C++ depth integration sources
                       fetch_bonn*.py, extract_frames.py  dataset preparation
tests/                 runtime unit tests (re-anchoring, blending, regression)
depthestream.py        standalone runtime entry point
depthart_cascade.py    speculative-cascade runtime variant
photoruler.py          single-photo real-world measurement app (EXIF → metric)
eval_sintel_v2.py      standalone Sintel evaluation (6 configs + DAv2-L baseline)
eval_bonn.py           standalone Bonn evaluation
bench_cpu.py           CPU platform benchmark (thread sweep, INT8, E2E)
sintel_io.py, bonn_io.py  dataset readers (.dpt PIEH depth, .flo flow)
convert_fp16.py, build_engine.py, build_metric_engine.py  ONNX/TRT tooling
```

Evaluation result JSONs are written to `benchmarks/` (git-ignored; not
committed).

## Environment

- Linux (tested in WSL2 Ubuntu 20.04) + NVIDIA GPU (tested RTX 3060 Laptop 6 GB);
  Jetson Nano path tested on JetPack 4.6.1 (Python 3.6, TensorRT 8.2)
- Python 3.11, PyTorch 2.7.1+cu126 — see `requirements.txt`
- TensorRT (pip wheels on desktop; 8.2.1 on Nano); ONNX Runtime 1.26 for
  the CPU path

```bash
pip install -r requirements.txt
python -m pytest tests/          # runtime unit tests
python tools/bench_unified.py --help
```

Each eval/tool script is runnable as-is; see its module docstring for
arguments and required data paths.

## Notes

- Code licensed Apache-2.0. Student/teacher weights inherit CC BY-NC 4.0
  terms from DepthART and Depth Anything V2 — non-commercial use of the
  weights; the runtime itself carries no such restriction.
- Known pitfalls handled in the code: Sintel GT sky outliers (capped at
  500 m), Bonn rgb/depth timestamp pairing, PIEH `.dpt` depth format,
  cross-process TensorRT tactic nondeterminism (canonical metrics use a
  deterministic ORT backend).
