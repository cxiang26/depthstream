"""TensorRT fp16 GPU backend."""
import numpy as np

from ..utils import preprocess_imagenet


class TrtDepth:
    """engine_path: a TensorRT engine built for the student ONNX
    (static shape 1x3x224x224, fp16 recommended)."""

    def __init__(self, engine_path, size=224):
        import torch
        import tensorrt as trt
        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            eng = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = eng.create_execution_context()
        names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        self.inn = next(n for n in names
                        if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.outn = next(n for n in names
                         if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.stream = torch.cuda.Stream()
        self.size = size

    def __call__(self, frame_bgr):
        torch = self.torch
        inp = torch.from_numpy(
            preprocess_imagenet(frame_bgr, self.size)).cuda()
        out = torch.empty(1, 1, self.size, self.size, device="cuda")
        self.ctx.set_tensor_address(self.inn, inp.data_ptr())
        self.ctx.set_tensor_address(self.outn, out.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return out[0, 0].cpu().numpy()
