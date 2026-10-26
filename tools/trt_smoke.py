import torch
import tensorrt as trt

print("TensorRT version:", trt.__version__)
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(0)
config = builder.create_builder_config()

x = network.add_input("x", trt.DataType.FLOAT, (1, 8))
idn = network.add_identity(x)
y = idn.get_output(0)
y.name = "y"
network.mark_output(y)

serialized = builder.build_serialized_network(network, config)
assert serialized is not None, "engine build FAILED"
engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
assert engine is not None, "deserialize FAILED"
ctx = engine.create_execution_context()

inp = torch.full((1, 8), 3.0, device="cuda")
out = torch.zeros(1, 8, device="cuda")
ctx.set_tensor_address("x", inp.data_ptr())
ctx.set_tensor_address("y", out.data_ptr())
stream = torch.cuda.Stream()
ok = ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()

print("execute_async_v3 returned:", ok)
print("input :", inp.tolist())
print("output:", out.tolist())
print("SMOKE TEST:", "PASS" if torch.equal(inp, out) else "FAIL")
