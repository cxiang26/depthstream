import collections
import onnx

m = onnx.load("/home/cq/depthart_test/mnv4m_slim_spf_224.onnx")
ops = collections.Counter(n.op_type for n in m.graph.node)
print("total nodes:", sum(ops.values()))
for op, c in ops.most_common():
    print(f"{op}: {c}")
