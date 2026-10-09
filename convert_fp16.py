import sys
import onnx
from onnxconverter_common import float16

src = sys.argv[1] if len(sys.argv) > 1 else "/home/cq/depthart_test/mnv4m_slim_spf_224.onnx"
dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".onnx", "_fp16.onnx")

m = onnx.load(src)
m16 = float16.convert_float_to_float16(m, keep_io_types=True)
onnx.save(m16, dst)

io = [(i.name, i.type.tensor_type.elem_type) for i in onnx.load(dst).graph.input]
print("converted:", dst)
print("input elem_types (1=float32):", io)
