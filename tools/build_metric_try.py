"""Try multiple build configs for the metric engine."""
import ctypes
import time

import torch  # noqa: F401  (plugin .so needs libtorch loaded first)
import tensorrt as trt

ctypes.CDLL("/home/cq/depthart_test/plugin/build/depthart_selective_scan_trt.so",
            mode=ctypes.RTLD_GLOBAL)

CASES = [
    ("/home/cq/depthart_test/metric_s_448_kv.onnx", "metric_s_448_kv_fp32.engine", True),
    ("/home/cq/depthart_test/metric_s_448_fused.onnx", "metric_s_448_fp32.engine", False),
    ("/home/cq/depthart_test/metric_s_448_kv.onnx", "metric_s_448_kv_fp32.engine", False),
]

logger = trt.Logger(trt.Logger.WARNING)
for onnx_path, engine_path, strongly in CASES:
    tag = f"{onnx_path.split('/')[-1]} strongly={strongly}"
    try:
        builder = trt.Builder(logger)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly else 0
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print(f"[{tag}] PARSE FAIL: {parser.get_error(0).desc()}")
                continue
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        t0 = time.time()
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            print(f"[{tag}] BUILD FAIL (None) after {time.time()-t0:.0f}s")
            continue
        with open(engine_path, "wb") as f:
            f.write(plan)
        print(f"[{tag}] OK -> {engine_path} ({time.time()-t0:.0f}s, {len(plan)//1024//1024}MB)")
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[{tag}] EXC: {repr(e)[:150]}")
print("ALL_CONFIGS_FAILED")
