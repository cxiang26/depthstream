"""Depth model backends. Each backend is a callable frame_bgr -> depth (h,w)."""
from .ort_cpu import OnnxDepth

__all__ = ["OnnxDepth"]

try:  # TensorRT/torch are optional
    from .trt_gpu import TrtDepth
    __all__.append("TrtDepth")
except Exception:  # pragma: no cover
    pass


def make_backend(name="ort", **kwargs):
    if name == "ort":
        return OnnxDepth(**kwargs)
    if name == "trt":
        return TrtDepth(**kwargs)
    raise ValueError(f"unknown backend: {name}")
