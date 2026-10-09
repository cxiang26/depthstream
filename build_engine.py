import sys
import time
import torch
import tensorrt as trt

onnx_path = sys.argv[1] if len(sys.argv) > 1 else "/home/cq/depthart_test/mnv4m_slim_spf_224_fp16.onnx"
engine_path = sys.argv[2] if len(sys.argv) > 2 else "/home/cq/depthart_test/depthart_fp16.engine"
in_h, in_w = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (224, 224)

print("TensorRT:", trt.__version__)
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
parser = trt.OnnxParser(network, logger)

with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print("PARSE ERR:", parser.get_error(i))
        raise SystemExit("ONNX parse failed")
print("parsed ok, inputs:", [network.get_input(i).name for i in range(network.num_inputs)])

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

t0 = time.time()
plan = builder.build_serialized_network(network, config)
if plan is None:
    raise SystemExit("BUILD FAILED (returned None)")
print(f"engine built in {time.time()-t0:.1f}s, saving {engine_path}")
with open(engine_path, "wb") as f:
    f.write(plan)

engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
ctx = engine.create_execution_context()
names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
in_name = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
out_name = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
print("io:", in_name, engine.get_tensor_shape(in_name), "->", out_name, engine.get_tensor_shape(out_name))

inp = torch.randn(1, 3, in_h, in_w, device="cuda")
if -1 in engine.get_tensor_shape(in_name):
    ctx.set_input_shape(in_name, (1, 3, in_h, in_w))
oshape = tuple(ctx.get_tensor_shape(out_name))
out = torch.empty(oshape, device="cuda")
ctx.set_tensor_address(in_name, inp.data_ptr())
ctx.set_tensor_address(out_name, out.data_ptr())
stream = torch.cuda.Stream()

stream.synchronize()
t0 = time.perf_counter()
for _ in range(300):
    ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
thr_ms = (time.perf_counter() - t0) / 300 * 1000

lat = []
for _ in range(300):
    t0 = time.perf_counter()
    ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    lat.append((time.perf_counter() - t0) * 1000)
lat.sort()
mean = sum(lat) / len(lat)
print(f"output shape: {oshape}, sample: {out.flatten()[:4].tolist()}")
print(f"throughput (pipelined): {thr_ms:.3f} ms/iter -> {1000/thr_ms:.0f} FPS")
print(f"latency (synced) ms  mean={mean:.3f}  p50={lat[150]:.3f}  p99={lat[297]:.3f}")
print("BENCH_OK")
