"""Generate a test photo with EXIF for PhotoRuler E2E check."""
import numpy as np
from PIL import Image, ImageDraw
import piexif

W, H = 1000, 750
img = Image.new("RGB", (W, H))
d = ImageDraw.Draw(img)
d.rectangle([0, 0, W, 300], fill=(180, 190, 205))       # wall
d.rectangle([0, 300, W, H], fill=(120, 100, 85))         # floor
d.rectangle([80, 80, 320, 380], fill=(70, 110, 160))     # window
d.rectangle([600, 120, 900, 480], fill=(90, 140, 90))    # board
d.rectangle([420, 300, 560, H], fill=(60, 60, 65))       # doorway
for i in range(0, W, 50):                                # floor lines
    d.line([i, 300, i, H], fill=(100, 85, 70), width=2)

exif = piexif.dump({
    "Exif": {
        piexif.ExifIFD.FocalLengthIn35mmFilm: 24,
        piexif.ExifIFD.FocalLength: (400, 100),        # 4.0mm
    }
})
img.save("/home/cq/depthart_test/test_photo.jpg", exif=exif, quality=92)
print("saved test_photo.jpg 1000x750, EXIF f35=24mm")
