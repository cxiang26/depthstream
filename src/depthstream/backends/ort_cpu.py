"""ONNX Runtime CPU backend (default; no GPU required)."""
import numpy as np

from ..utils import preprocess_imagenet


class OnnxDepth:
    def __init__(self, onnx_path, threads=4, size=224):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path_or_bytes=onnx_path,
                                         sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.inn = self.sess.get_inputs()[0].name
        self.size = size

    def __call__(self, frame_bgr):
        out = self.sess.run(None, {self.inn: preprocess_imagenet(frame_bgr, self.size)})
        return out[0].squeeze()
