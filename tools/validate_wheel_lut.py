"""
Compares the GUI's painted wheel (nss.color_gui.wheel_lut.wheel_image) with Lightroom's active wheel (the handle-free composite of
tests/test_data/lr_wheel_<H>.png) and writes a side-by-side PNG: composite | ours | |difference| x 8.

    python -m tools.validate_wheel_lut out.png
Reports CIE76 dE over the disc without the outer 2 px (antialiased edge) and, separately, without Lightroom's grey centre dot.
"""
import sys

import cv2
import numpy as np
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication
from skimage.color import deltaE_cie76, rgb2lab

from nss.color_gui.wheel_lut import wheel_image
from tools.make_wheel_tables import C0, N, composite

BACKGROUND = 71.0           # Lightroom's panel grey


def painted(radius: float) -> np.ndarray:
    """Our wheel as RGB 0..255 on the composite's canvas (centre C0), over the panel grey."""
    size = 2 * int(round(radius))                          # even size: centre (size - 1) / 2 lands on C0 with an integer offset
    im = wheel_image(size).convertToFormat(QImage.Format.Format_RGBA8888)
    rgba = np.frombuffer(im.constBits().asstring(im.sizeInBytes()), np.uint8).reshape(size, size, 4).astype(float)
    a = rgba[..., 3:] / 255.0
    over = rgba[..., :3] / np.maximum(a, 1e-6) * a + BACKGROUND * (1 - a)
    canvas = np.full((N, N, 3), BACKGROUND)
    o = int(round(C0 - (size - 1) / 2.0))
    canvas[o:o + size, o:o + size] = over
    return canvas


def main() -> None:
    """Prints the dE statistics and writes the side-by-side PNG to sys.argv[1]."""
    app = QApplication.instance() or QApplication([])
    comp, radius = composite()
    ours = painted(radius)
    dE = deltaE_cie76(rgb2lab(comp / 255.0), rgb2lab(ours / 255.0))
    yy, xx = np.mgrid[0:N, 0:N]
    rpx = np.hypot(xx - C0, yy - C0)
    disc = rpx < radius - 2
    no_dot = disc & (rpx > 6)
    for name, m in (("disc (outer 2 px excluded)", disc), ("disc without the grey centre dot", no_dot)):
        print(f"{name:34s} dE mean {dE[m].mean():5.2f}  p95 {np.percentile(dE[m], 95):5.2f}  max {dE[m].max():5.2f}")
    worst = np.unravel_index(np.argmax(np.where(no_dot, dE, 0.0)), dE.shape)
    print(f"worst pixel (no dot): r/R {rpx[worst] / radius:.3f}, angle {np.degrees(np.arctan2(C0 - worst[0], worst[1] - C0)) % 360:.1f}")
    diff = np.clip(np.abs(ours - comp) * 8.0, 0, 255)
    gap = np.full((N, 6, 3), 255.0)
    side = np.concatenate([comp, gap, ours, gap, diff], 1)
    cv2.imwrite(sys.argv[1], cv2.cvtColor(np.rint(side).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()
