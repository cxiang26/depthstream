# DepthStream

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()

**Training-free amortized video depth streaming for tiny monocular depth
models.**

Run a 6M-parameter monocular depth model as a *temporally stable,
full-resolution video depth stream* at up to 249 FPS end-to-end — on a
laptop GPU **or a plain CPU** — with zero training and zero custom
training pipelines.

```
pip install -e .
depthstream run video.mp4 -K 10 -o out.mp4        # CPU, auto-downloads weights
```

## How it works

Depth fields change slowly under camera and object motion: most of the
signal at frame *t* is already present at frame *t−1*, displaced by motion.
DepthStream exploits this with three training-free components:

1. **Keyframe inference** — a frozen 6M student (DepthART MobileNetV4-slim,
   TensorRT fp16 or ONNX Runtime) runs every *K* frames;
2. **Backward-flow propagation** — in-between frames reuse the previous
   depth field through DIS optical-flow warping;
3. **Affine re-anchoring** — fresh keyframe predictions are aligned to the
   propagated field by closed-form scale–shift fitting, which transfers
   temporal continuity without suppressing detail. A photometric-residual
   monitor triggers early re-anchoring where flow fails (occlusions, fast
   motion).

## Results

| Config | Sintel δ1↑ | Sintel TAE↓ | ms/frame | Bonn δ1↑ |
|---|---|---|---|---|
| DAv2-L @518 per-frame | 0.4930 | 0.1844 | 56.4 | 0.9811 |
| student per-frame | 0.4614 | 0.1585 | 10.6 | 0.9655 |
| **DepthStream kf3** | **0.4614** | 0.1343 | **8.4** | 0.9637 |
| **DepthStream kf10-adp** | 0.4434 | **0.1054** | 7.6 | 0.9484 |

- kf3 matches per-frame accuracy **exactly** while cutting temporal error
  15% and latency 20% — the per-frame baseline is strictly dominated.
- Streaming retains 93.6–98.2% of the $6.7$–$13.7\times$ slower teacher's
  accuracy with **better temporal stability than the teacher itself**.
- Accuracy is backend-invariant: TensorRT fp16 GPU and ONNX-Runtime fp32 CPU
  agree to |Δδ1| ≤ 1e-4 over all configurations.

See `benchmarks/` for full result JSONs and the paper draft for details.

## Install

```bash
# CPU-only (default): numpy + OpenCV + ONNX Runtime
pip install -e .

# GPU: add TensorRT + torch for the TensorRT backend
pip install -e .[trt]
```

Python ≥ 3.9, Linux/Windows. On first run the student ONNX (24 MB) is
downloaded from the official
[DepthART release](https://huggingface.co/Fengxue93/DepthART) into
`~/.cache/depthstream/` (override with `DEPTHSTREAM_CACHE`).

## Usage

```bash
depthstream run video.mp4 -K 10 -o out.mp4              # CPU backend
depthstream run video.mp4 -K 10 --backend trt \
    --engine student_fp16.engine                        # TensorRT backend
depthstream run video.mp4 -K 10 --adaptive              # failure-aware scheduling
```

Python API:

```python
from depthstream import DepthStream
from depthstream.backends.ort_cpu import OnnxDepth

stream = DepthStream(OnnxDepth("student.onnx"), keyframe=10)
for frame in video_frames:
    depth = stream.process(frame)     # full-resolution, temporally stable
```

## Benchmarks

`evals/` contains the Sintel / Bonn / CPU evaluation pipelines used in the
paper (each a single command; see `--help`). Result JSONs are committed
under `benchmarks/`.

## Notes on weights

The student ONNX is downloaded from the official DepthART release and
inherits **CC BY-NC 4.0** terms from DepthART / Depth Anything V2
(non-commercial use of the weights). The DepthStream runtime code itself is
Apache-2.0.

## Citing

If you use DepthStream, please also cite the underlying model:

```bibtex
@article{depthart2026,
  title  = {DepthART: Scaling Foundation Monocular Depth to Tiny Models},
  author = {Xue, Feng and others},
  journal= {arXiv:2607.17099},
  year   = {2026}
}
```
