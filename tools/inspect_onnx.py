import sys
import onnx

path = sys.argv[1] if len(sys.argv) > 1 else "/home/cq/depthart_test/metric_s_448_fused.onnx"
m = onnx.load(path)

ELEM = {1: "fp32", 10: "fp16", 7: "int64", 6: "int32"}
print("=== INPUTS ===")
for i in m.graph.input:
    t = i.type.tensor_type
    dims = [d.dim_value if d.HasField("dim_value") else f"?{d.dim_param}" for d in t.shape.dim]
    print(f"  {i.name}: dtype={ELEM.get(t.elem_type, t.elem_type)} shape={dims}")
print("=== OUTPUTS ===")
for o in m.graph.output:
    t = o.type.tensor_type
    dims = [d.dim_value if d.HasField("dim_value") else f"?{d.dim_param}" for d in t.shape.dim]
    print(f"  {o.name}: dtype={ELEM.get(t.elem_type, t.elem_type)} shape={dims}")
