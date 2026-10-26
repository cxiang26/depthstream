"""PhotoRuler: single-photo real-world measurement with DepthART-Metric.

Pipeline: EXIF intrinsics -> DepthART-Metric (TinyViM-S, PyTorch, real K input)
-> two-point 3D distance via pinhole back-projection.

Usage:
  python photoruler.py photo.jpg u1 v1 u2 v2        # measure two pixel points
  python photoruler.py photo.jpg --click            # interactive pick (GUI)
  python photoruler.py photo.jpg --save-depth       # also dump depth map png
  python photoruler.py --selftest                   # math check, no model
  optional: --domain outdoor   (default indoor)
"""
import math
import os
import sys

import numpy as np

REPO = "/home/cq/depthart_test/DepthART-main/metric"
CKPT = "/home/cq/depthart_test/depthart_metric_{domain}_s_448.pth"
sys.path.insert(0, REPO)


# ---------- intrinsics from EXIF ----------
def exif_intrinsics(img):
    w, h = img.size
    cx, cy = w / 2.0, h / 2.0
    tags = {}
    from PIL.ExifTags import TAGS
    _exif = img.getexif() or {}
    for k, v in _exif.items():
        tags[TAGS.get(k, k)] = v
    try:  # Exif SubIFD holds FocalLength / FocalLengthIn35mmFilm
        for k, v in _exif.get_ifd(0x8769).items():
            tags.setdefault(TAGS.get(k, k), v)
    except Exception:
        pass
    f35 = tags.get("FocalLengthIn35mmFilm")
    focal = tags.get("FocalLength")
    fpres = tags.get("FocalPlaneXResolution")
    fpunit = tags.get("FocalPlaneResolutionUnit", 2)
    if f35:
        f = w * float(f35) / 36.0
        return f, f, cx, cy, f"35mm-equiv f={f35}mm"
    if focal and fpres:
        mmpp = (25.4 if int(fpunit) == 2 else 10.0 if int(fpunit) == 3 else 1.0) / float(fpres)
        sensor_w = w * mmpp
        f = w * float(focal) / sensor_w
        return f, f, cx, cy, f"focal-plane f={float(focal):.1f}mm sensor={sensor_w:.1f}mm"
    return 0.8 * w, 0.8 * w, cx, cy, "DEFAULT fx=0.8W (no usable EXIF)"


# ---------- DepthART-Metric via author code path ----------
_model_cache = {}


def load_metric(domain):
    if domain in _model_cache:
        return _model_cache[domain]
    sys.path.insert(0, REPO)
    from model import load_model  # noqa: E402  (needs REPO on sys.path)
    ckpt = CKPT.format(domain=domain)
    assert os.path.exists(ckpt), f"checkpoint missing: {ckpt}"
    model = load_model(ckpt, "S", domain)
    _model_cache[domain] = model
    return model


def predict_depth(img_pil, K4, domain):
    import cv2
    import torch
    from common import make_K, preprocess
    bgr = cv2.cvtColor(np.asarray(img_pil)[:, :, ::-1].copy(), cv2.COLOR_RGB2BGR)
    K = make_K(*K4)
    tw, th = (448, 448) if domain == "outdoor" else (640, 480)
    tensor, Kt = preprocess(bgr, K, tw, th)
    model = load_metric(domain)
    with torch.no_grad():
        pred = model(tensor.cuda(), Kt.cuda())[:, None]
    H, W = bgr.shape[:2]
    depth = torch.nn.functional.interpolate(
        pred, size=(H, W), mode="bilinear", align_corners=True)
    return depth[0, 0].float().cpu().numpy()


# ---------- measurement ----------
def backproject(u, v, z, fx, fy, cx, cy):
    return np.asarray([(u - cx) * z / fx, (v - cy) * z / fy, z])


def main():
    argv = sys.argv[1:]
    flags = {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    domain = "outdoor" if "--domain outdoor" in argv or "--outdoor" in flags else "indoor"

    if "--selftest" in flags:
        fx = 500.0
        cx, cy, z = 320.0, 240.0, 2.0
        p1 = backproject(320, 240, z, fx, fx, cx, cy)
        p2 = backproject(570, 240, z, fx, fx, cx, cy)
        d = float(np.linalg.norm(p1 - p2))
        ok = abs(d - 250 * z / fx) < 1e-6
        print(f"selftest: measured={d:.4f}m expected={250 * z / fx:.4f}m -> "
              f"{'PASS' if ok else 'FAIL'}")
        return

    img_path, coords = args[0], args[1:5]
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    fx, fy, cx, cy, src = exif_intrinsics(img)
    print(f"intrinsics: fx=fy={fx:.1f}px c=({cx:.0f},{cy:.0f}) [{src}]")

    if "--click" in flags or len(coords) < 4:
        import matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(img)
        print("click two points on the image window...")
        pts = plt.ginput(2, timeout=0)
        plt.close(fig)
        coords = [c for p in pts for c in p]

    depth = predict_depth(img, (fx, fy, cx, cy), domain)
    print(f"[{domain}] depth map {depth.shape}  "
          f"range {float(np.nanmin(depth)):.2f}-{float(np.nanmax(depth)):.2f} m")

    u1, v1, u2, v2 = map(float, coords)
    z1 = float(depth[int(np.clip(v1, 0, depth.shape[0] - 1)),
                     int(np.clip(u1, 0, depth.shape[1] - 1))])
    z2 = float(depth[int(np.clip(v2, 0, depth.shape[0] - 1)),
                     int(np.clip(u2, 0, depth.shape[1] - 1))])
    p1 = backproject(u1, v1, z1, fx, fy, cx, cy)
    p2 = backproject(u2, v2, z2, fx, fy, cx, cy)
    d = float(np.linalg.norm(p1 - p2))
    print(f"P1=({u1:.0f},{v1:.0f}) z={z1:.3f}m | P2=({u2:.0f},{v2:.0f}) z={z2:.3f}m")
    print(f"3D distance: {d:.3f} m")

    if "--save-depth" in flags:
        import matplotlib.pyplot as plt
        out = os.path.splitext(img_path)[0] + "_depth.png"
        plt.imsave(out, depth, cmap="viridis")
        print("depth map saved:", out)


if __name__ == "__main__":
    main()
