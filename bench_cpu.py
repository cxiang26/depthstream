"""CPU benchmark for DepthStream student (ONNX Runtime, MobileNetV4-slim 224).

Measures:
  1. model latency sweep: threads x {fp32, int8-dynamic}
  2. end-to-end DepthStream on CPU (best config) on test_video
  3. agreement with GPU TensorRT output on 3 Sintel frames
"""
import json
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

from depthstream import DepthStream  # noqa: E402
from depthstream.utils import colorize  # noqa: E402

ONNX32 = os.environ.get(
    "DEPTHSTREAM_ONNX",
    os.path.join(os.path.expanduser("~"), ".cache", "depthstream",
                 "depthart_mnv4m_slim_spf_224.onnx"))
ONNX8 = ONNX32.replace(".onnx", "_int8.onnx")
ENGINE = os.environ.get("DEPTHSTREAM_ENGINE", "")
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def preprocess(frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = ((img - MEAN) / STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(img)


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class OnnxCpuDepth:
    def __init__(self, path, threads=8):
        self.sess = make_session(path, threads)
        self.inn = self.sess.get_inputs()[0].name

    def __call__(self, frame_bgr):
        out = self.sess.run(None, {self.inn: preprocess(frame_bgr)})[0]
        return out.squeeze()


def bench_latency(fn, warmup=10, iters=50):
    dummy = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    for _ in range(warmup):
        fn(dummy)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(dummy)
    return (time.perf_counter() - t0) * 1000 / iters


def main():
    # 0) INT8 dynamic quantization (PTQ, train-free)
    import os
    if not os.path.exists(ONNX8):
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(ONNX32, ONNX8, weight_type=QuantType.QInt8)
        print("quantized ->", ONNX8, os.path.getsize(ONNX8) // 1024 // 1024, "MB")

    # 1) latency sweep
    print(f"{'threads':>8} {'fp32 ms':>10} {'int8 ms':>10}")
    sweep = {}
    for th in (2, 4, 8, 16):
        s32 = make_session(ONNX32, th)
        dummy = np.random.randint(0, 255, (480, 640, 3), np.uint8)
        inp = {s32.get_inputs()[0].name: preprocess(dummy)}
        for _ in range(5):
            s32.run(None, inp)
        t0 = time.perf_counter()
        for _ in range(30):
            s32.run(None, inp)
        m32 = (time.perf_counter() - t0) * 1000 / 30

        s8 = make_session(ONNX8, th)
        inp8 = {s8.get_inputs()[0].name: preprocess(dummy)}
        for _ in range(5):
            s8.run(None, inp8)
        t0 = time.perf_counter()
        for _ in range(30):
            s8.run(None, inp8)
        m8 = (time.perf_counter() - t0) * 1000 / 30
        sweep[th] = (round(m32, 2), round(m8, 2))
        print(f"{th:>8} {m32:>10.2f} {m8:>10.2f}")

    best_th = min(sweep, key=lambda k: min(sweep[k]))
    best_path = ONNX8 if sweep[best_th][1] <= sweep[best_th][0] else ONNX32
    print(f"best: threads={best_th} {'int8' if best_path == ONNX8 else 'fp32'}")

    # 2) end-to-end DepthStream on CPU
    cap = cv2.VideoCapture(os.environ.get(
    "DEPTHSTREAM_TEST_VIDEO", ""))
    model = OnnxCpuDepth(best_path, threads=best_th)
    st = DepthStream(model, keyframe=10, flow="dis", flow_width=256)
    vw = cv2.VideoWriter("ds_cpu.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 480))
    n = 0
    while n < 90:
        ok, frame = cap.read()
        if not ok:
            break
        out = st.process(frame)
        vw.write(np.hstack([frame, colorize(out)]))
        n += 1
    cap.release()
    vw.release()
    print("E2E CPU:", json.dumps(st.stats()))

    # 3) agreement with GPU TensorRT output
    from depthstream.backends.trt_gpu import TrtDepth  # noqa: E402
    gpu = TrtDepth(os.environ.get(
        "DEPTHSTREAM_ENGINE",
        os.path.join(os.path.expanduser("~"), ".cache", "depthstream",
                     "depthart_fp16.engine")))
    agree = []
    for seq in ("alley_1", "alley_2", "market_2"):
        for t in (5, 20, 40):
            root = os.environ.get("DEPTHSTREAM_SINTEL", "")
            p = f"{root}/training/clean/{seq}/frame_{t+1:04d}.png"
            im = cv2.imread(p)
            if im is None:
                continue
            d_cpu = model(im)
            d_gpu = gpu(im)
            a, b = d_cpu.astype(np.float64).ravel(), d_gpu.astype(np.float64).ravel()
            c = np.corrcoef(a, b)[0, 1]
            agree.append(float(c))
    print(f"CPU-vs-GPU output Pearson r: mean={np.mean(agree):.4f} "
          f"min={np.min(agree):.4f} over {len(agree)} frames")


if __name__ == "__main__":
    main()
