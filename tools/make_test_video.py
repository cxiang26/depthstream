"""Synthetic camera-motion test video for DepthStream: pan + zoom + moving object."""
import cv2
import numpy as np

W, H, FPS, SEC = 640, 480, 30, 6
scene = cv2.imread("/home/depthart_test/test_photo.jpg")
if scene is None:
    scene = cv2.imread("/home/cq/depthart_test/test_photo.jpg")
sh, sw = scene.shape[:2]

vw = cv2.VideoWriter("/home/cq/depthart_test/test_video.mp4",
                     cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
n = FPS * SEC
for t in range(n):
    ph = t / n
    # camera: slow pan (sine) + zoom in-out
    zoom = 1.15 + 0.1 * np.sin(2 * np.pi * ph)
    cx = sw / 2 + 120 * np.sin(2 * np.pi * ph)
    cy = sh / 2 + 60 * np.cos(2 * np.pi * ph)
    src_w, src_h = sw / zoom, sh / zoom
    x0, y0 = cx - src_w / 2, cy - src_h / 2
    M = np.float32([[zoom, 0, -x0 * zoom], [0, zoom, -y0 * zoom]])
    frame = cv2.warpAffine(scene, M, (W, H))
    # moving object: bouncing ball with shadow
    bx = int(W / 2 + 200 * np.sin(2 * np.pi * 2 * ph))
    by = int(H / 2 + 120 * np.cos(2 * np.pi * 2 * ph))
    r = 24
    cv2.circle(frame, (bx, by), r + 6, (40, 40, 40), -1)
    cv2.circle(frame, (bx, by), r, (50, 160, 235), -1)
    cv2.circle(frame, (bx - 7, by - 7), 7, (255, 255, 255), -1)
    vw.write(frame)

vw.release()
print(f"saved test_video.mp4  {n} frames  {W}x{H}@{FPS}")
