"""Small shared helpers."""
import cv2
import numpy as np

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def preprocess_imagenet(frame_bgr, size=224):
    """BGR uint8 frame -> normalized (1,3,size,size) float32 (ImageNet stats)."""
    import cv2
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = ((img - MEAN) / STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(img)


def depth_fullres(d_small, shape_hw):
    import cv2
    return cv2.resize(d_small, (shape_hw[1], shape_hw[0]),
                      interpolation=cv2.INTER_LINEAR)


def colorize(d, lo=None, hi=None):
    lo = float(np.percentile(d, 2)) if lo is None else lo
    hi = float(np.percentile(d, 98)) if hi is None else hi
    dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    dn[~np.isfinite(dn)] = 0
    return cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
