"""Build a TensorRT 8.x fp16 engine on Jetson Nano from the student ONNX.

TRT 8 API (BuilderFlag.FP16 still exists here, unlike TRT 11). Layers that
cannot run in fp16 fall back to fp32 automatically (mixed engine).
"""
import sys
import time

import tensorrt as trt

onnx_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ds/mnv4m_slim_spf_224.onnx"
out_path = sys.argv[2] if len(sys.argv) > 2 else "/home/ds/depthart_fp16.engine"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit(1)

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.max_workspace_size = 1 << 28
if builder.platform_has_fast_fp16:
    print("platform_has_fast_fp16: True")
else:
    print("WARNING: platform reports no fast fp16")

t0 = time.time()
plan = builder.build_serialized_network(network, config)
if plan is None:
    print("BUILD FAILED")
    sys.exit(1)
with open(out_path, "wb") as f:
    f.write(plan)
print(f"built {out_path} in {time.time()-t0:.0f}s")
