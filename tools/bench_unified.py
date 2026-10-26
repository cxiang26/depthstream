"""Unified timing protocol (reviewer S6): one session, one table.

Components (p50/p95 over 100 calls): model = preprocess+H2D+infer+D2H,
flow = DIS optical flow at 256 px, warp = full-res remap.
E2E: dense (model every frame) and kf10 stream over 300 frames of a
640x480 synthetic clip; per-frame wall-clock percentiles p50/p95/p99;
stream = the paper runtime's logic inlined (sd spread-matching anchor).
Peak host RSS (+ CUDA reserved when applicable). Optional INA3221 power
sampling on Jetson.

Self-contained: needs only cv2/numpy + one of (tensorrt+torch | onnxruntime).
Usage:
  python bench_unified.py --backend trt  --engine E --out JSON
  python bench_unified.py --backend ort  --onnx O   --out JSON
"""
import argparse
import json
import math
import resource
import threading
import time

import cv2
import numpy as np

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def preprocess(frame_bgr, size=224):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = ((img - MEAN) / STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(img)


def make_model(backend, path, size=224):
    if backend == "ort":
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name

        def call(frame_bgr):
            x = preprocess(frame_bgr, size)
            return sess.run(None, {inp: x})[0][0, 0]
        return call

    import torch
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f:
        eng = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    major = int(trt.__version__.split(".")[0])
    if major >= 10:
        names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        inn = next(n for n in names
                   if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        outn = next(n for n in names
                    if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        stream = torch.cuda.Stream()

        def call(frame_bgr):
            x = torch.from_numpy(preprocess(frame_bgr, size)).cuda()
            out = torch.empty(1, 1, size, size, device="cuda")
            ctx.set_tensor_address(inn, x.data_ptr())
            ctx.set_tensor_address(outn, out.data_ptr())
            ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            return out[0, 0].cpu().numpy()
    else:                                   # TRT 8.x (JetPack 4.6)
        i_idx = next(i for i in range(eng.num_bindings)
                     if eng.binding_is_input(i))
        o_idx = next(i for i in range(eng.num_bindings)
                     if not eng.binding_is_input(i))

        def call(frame_bgr):
            x = torch.from_numpy(preprocess(frame_bgr, size)).cuda()
            out = torch.empty(1, 1, size, size, device="cuda")
            ctx.execute_v2([x.data_ptr(), out.data_ptr()])
            torch.cuda.synchronize()
            return out[0, 0].cpu().numpy()
    return call


def make_clip(n=300, hw=(480, 640)):
    rng = np.random.RandomState(0)
    base = rng.randint(0, 255, (hw[0], hw[1], 3), np.uint8)
    frames = []
    moves = [(3, 1), (-2, 2), (1, -3)]
    for t in range(n):
        f = base.copy()
        for k, (dx, dy) in enumerate(moves):
            x = (80 + k * 180 + dx * t) % 520
            y = (100 + k * 100 + dy * t) % 380
            cv2.rectangle(f, (x, y), (x + 90, y + 90),
                          (255 * (k % 2), 200, 60), -1)
        frames.append(f)
    return frames


def pctl(xs, p):
    return round(float(np.percentile(xs, p)), 3)


def sd_scale(a, b):
    da, db = a - a.mean(), b - b.mean()
    da, db = da[np.isfinite(da)], db[np.isfinite(db)]
    denom_a = float(np.sum(da * da))
    denom_b = float(np.sum(db * db))
    cov = float(np.sum(da * db))
    if denom_a < 1e-9 or denom_b < 1e-9 or abs(cov) < 1e-9:
        return 1.0, 0.0
    sgn = 1.0 if cov >= 0 else -1.0
    alpha = sgn * min(max(math.sqrt(denom_b / denom_a), 0.25), 4.0)
    return alpha, float(b.mean() - alpha * a.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["trt", "ort"])
    ap.add_argument("--engine", default="")
    ap.add_argument("--onnx", default="")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default="bench_unified.json")
    ap.add_argument("--power", action="store_true",
                    help="sample INA3221 power rails (Jetson)")
    args = ap.parse_args()

    model = make_model(args.backend, args.engine or args.onnx)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    def calc(a, b):
        return dis.calc(a, b, None)
    clip = make_clip(args.frames)
    H, W = clip[0].shape[:2]
    fw, fh = 256, int(round(H / W * 256 / 8) * 8)

    pwr_stop = threading.Event()
    pwr_vals = []

    def pwr_thread():
        paths = []
        for root, dirs, files in os.walk("/sys/bus/i2c/devices"):
            for f in files:
                if f == "power1" or f.startswith("in_power"):
                    paths.append(os.path.join(root, f))
        while not pwr_stop.is_set():
            for p in paths:
                try:
                    with open(p) as fh:
                        pwr_vals.append(int(fh.read().strip()))
                except OSError:
                    pass
            time.sleep(0.05)

    if args.power:
        threading.Thread(target=pwr_thread, daemon=True).start()

    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in clip]
    gsmall = [cv2.resize(g, (fw, fh)) for g in gray]

    # warmup
    for i in range(20):
        model(clip[i])
        calc(gsmall[i + 1], gsmall[i]) if i + 1 < len(gsmall) else None

    res = {}

    def timed(fn, n=100):
        ts = []
        for i in range(n):
            t0 = time.perf_counter()
            fn(i)
            ts.append((time.perf_counter() - t0) * 1000)
        return {"p50": pctl(ts, 50), "p95": pctl(ts, 95),
                "mean": round(float(np.mean(ts)), 3)}

    res["model"] = timed(lambda i: model(clip[i % len(clip)]))
    res["flow"] = timed(
        lambda i: calc(gsmall[(i + 1) % len(gsmall)], gsmall[i % len(gsmall)]))
    warp_dst = np.empty((H, W), np.float32)
    warp_src = np.random.rand(H, W).astype(np.float32)
    flow_up = np.random.rand(H, W, 2).astype(np.float32)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))

    def warp_call(i):
        cv2.remap(warp_src, gx + flow_up[..., 0], gy + flow_up[..., 1],
                  cv2.INTER_LINEAR, dst=warp_dst,
                  borderMode=cv2.BORDER_REPLICATE)
    res["warp"] = timed(warp_call)

    def gray_call(i):
        cv2.cvtColor(clip[i % len(clip)], cv2.COLOR_BGR2GRAY)
        cv2.resize(gsmall[i % len(gsmall)], (W, H))
    res["gray"] = timed(gray_call)

    fa = np.random.rand(H, W).astype(np.float32)
    fb = np.random.rand(H, W).astype(np.float32)
    mask = np.ones((H, W), bool)

    def fit_call(i):
        a = fa[mask]
        b = fb[mask]
        da, db = a - a.mean(), b - b.mean()
        cov = float(np.sum(da * db))
        math.sqrt(float(np.sum(db * db)) /
                  max(float(np.sum(da * da)), 1e-9))
    res["anchor_fit"] = timed(fit_call)

    # E2E: dense and kf10 stream (runtime logic inlined, sd anchor)
    def stream(kf):
        prev_out = prev_gray = None
        last_key, i = 0, 0
        per_frame = []
        t_all0 = time.perf_counter()
        for f_bgr, g, gs in zip(clip, gray, gsmall):
            t0 = time.perf_counter()
            due = (i - last_key) >= kf
            is_key = due or prev_out is None
            d_prop = None
            if prev_out is not None:
                fl = cv2.resize(calc(gs, prev_gray), (W, H),
                                interpolation=cv2.INTER_LINEAR)
                fl = fl * np.array([W / fw, H / fh], np.float32)
                d_prop = cv2.remap(prev_out, gx + fl[..., 0], gy + fl[..., 1],
                                   cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
            if is_key:
                d_small = model(f_bgr)
                d_full = cv2.resize(d_small, (W, H),
                                    interpolation=cv2.INTER_LINEAR)
                if d_prop is None:
                    out = d_full
                else:
                    a, b = sd_scale(d_full, d_prop)
                    out = a * d_full + b
                last_key = i
            else:
                out = d_prop
            prev_out, prev_gray = out.astype(np.float32), gs
            per_frame.append((time.perf_counter() - t0) * 1000)
            i += 1
        return {"mean": round(float(np.mean(per_frame)), 2),
                "p50": pctl(per_frame, 50), "p95": pctl(per_frame, 95),
                "p99": pctl(per_frame, 99),
                "fps": round(len(per_frame) /
                             (time.perf_counter() - t_all0), 1)}

    res["e2e_dense"] = stream(1)
    res["e2e_kf10"] = stream(10)
    res["e2e_kf3"] = stream(3)
    res["peak_rss_mb"] = round(resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    try:
        import torch
        if torch.cuda.is_available():
            res["cuda_reserved_mb"] = round(
                torch.cuda.max_memory_reserved() / 2**20, 1)
    except Exception:
        pass
    if args.power and pwr_vals:
        pwr_stop.set()
        time.sleep(0.2)
        res["power_mw"] = {"mean": int(np.mean(pwr_vals)),
                           "max": int(np.max(pwr_vals)),
                           "n": len(pwr_vals), "rails": "all INA3221 channels"}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
