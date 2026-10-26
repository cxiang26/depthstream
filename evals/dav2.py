"""Depth Anything V2-Large per-frame baseline (optional; needs the DAv2 repo)."""
import sys

import numpy as np


class DAv2Baseline:
    def __init__(self, repo_dir, weights, encoder="vitl"):
        import torch
        sys.path.insert(0, repo_dir)
        from depth_anything_v2.dpt import DepthAnythingV2
        cfg = {"encoder": encoder, "features": 256,
               "out_channels": [256, 512, 1024, 1024],
               "use_bn": False, "use_clstoken": False}
        m = DepthAnythingV2(**cfg)
        m.load_state_dict(torch.load(weights, map_location="cpu"))
        self.torch = torch
        self.model = m.cuda().eval().half()

    def __call__(self, frame_bgr, long_side=518):
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]
        s = long_side / max(h, w)
        nw, nh = int(round(w * s / 14)) * 14, int(round(h * s / 14)) * 14
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        x = self.torch.from_numpy(rgb.transpose(2, 0, 1)).cuda().half()[None]
        with self.torch.no_grad():
            d = self.model(x)
        d = d.squeeze().float().cpu().numpy()
        if d.shape != (h, w):
            import cv2
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        return d
