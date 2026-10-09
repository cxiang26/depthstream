"""Compile the DepthART SelectiveScan TRT plugin, then build the metric engine."""
import os
import time

import torch
from torch.utils.cpp_extension import load

PLUGIN_DIR = "/home/cq/depthart_test/plugin"
os.makedirs(f"{PLUGIN_DIR}/build", exist_ok=True)
ONNX = "/home/cq/depthart_test/metric_s_448_fused.onnx"
ENGINE = "/home/cq/depthart_test/metric_s_448_fp32.engine"

print("torch:", torch.__version__, "abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
t0 = time.time()
mod = load(
    name="depthart_selective_scan_trt",
    sources=[
        f"{PLUGIN_DIR}/trt_plugin/selective_scan_plugin.cpp",
        f"{PLUGIN_DIR}/vendor/selective_scan_oflex.cpp",
        f"{PLUGIN_DIR}/vendor/selective_scan_core_fwd.cu",
        f"{PLUGIN_DIR}/vendor/selective_scan_core_bwd.cu",
    ],
    extra_include_paths=[
        f"{PLUGIN_DIR}/trt_include",
        f"{PLUGIN_DIR}/vendor",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=[
        "/home/cq/miniconda3/lib/python3.11/site-packages/tensorrt_libs/libnvinfer.so.11",
        "-Wl,-rpath,/home/cq/miniconda3/lib/python3.11/site-packages/tensorrt_libs",
    ],
    build_directory=f"{PLUGIN_DIR}/build",
    verbose=False,
)
print(f"plugin compiled+loaded in {time.time()-t0:.0f}s")

import ctypes
import glob

so = glob.glob(f"{PLUGIN_DIR}/build/**/depthart_selective_scan_trt*.so", recursive=True)
print("plugin .so:", so)

import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
parser = trt.OnnxParser(network, logger)
with open(ONNX, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(min(parser.num_errors, 4)):
            print("PARSE ERR:", parser.get_error(i).desc())
        raise SystemExit("metric ONNX parse failed")
print("metric ONNX parsed ok")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
t0 = time.time()
plan = builder.build_serialized_network(network, config)
if plan is None:
    raise SystemExit("metric engine BUILD FAILED")
print(f"metric engine built in {time.time()-t0:.0f}s")
with open(ENGINE, "wb") as f:
    f.write(plan)
print("ENGINE_OK:", ENGINE)
