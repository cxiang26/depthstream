"""Extract benchmark frames from the test video."""
import cv2

cap = cv2.VideoCapture("/home/cq/depthart_test/test_video.mp4")
i = 0
n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % 15 == 0 and n < 12:
        cv2.imwrite(f"/home/cq/depthart_test/casc_frame_{n:02d}.jpg", frame)
        n += 1
    i += 1
print("extracted", n, "frames")
