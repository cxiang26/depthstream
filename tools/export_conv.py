import torch
import torch.nn as nn

m = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(8, 8, 3, padding=1),
).eval()

dummy = torch.randn(1, 3, 64, 64)
torch.onnx.export(m, dummy, "/home/cq/depthart_test/conv_tiny.onnx",
                  input_names=["x"], output_names=["y"], opset_version=17)
print("exported ok")
