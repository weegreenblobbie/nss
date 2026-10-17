"""
Lightroom's Color Grading wheel for the GUI: the hue convention, the measured painting, and the geometry shared by the wheel and the Hue sliders.

Convention (measured on Lightroom's wheel: the handle for hue H sits at screen angle H + 0.04 degrees): the hue number IS the screen angle,
0 at 3 o'clock, counter-clockwise.  The painting is warped (green is painted near 154 degrees, not 120); that only affects what is drawn,
never the hue number sent to Grading.apply.  Saturation is linear in the distance from the centre, Sat 100 at SAT_FULL_RADIUS of the
painted radius; the painted disc itself extends to 1.0.
"""
import math

import numpy as np
from numpy.typing import ArrayLike
from PyQt6.QtGui import QImage

from nss.color_gui.wheel_lut_tables import BLEND_G, BLEND_R, CENTRE_GREY, RIM_RGB

SAT_FULL_RADIUS = 0.907         # r / R of the handle at Sat 100 in Lightroom; Sat 0 is the centre


def hue_from_angle(angle: ArrayLike) -> np.ndarray | float:
    """Hue number (0..360) at a screen angle in degrees (0 at 3 o'clock, counter-clockwise)."""
    return np.mod(angle, 360.0)


def angle_from_hue(hue: ArrayLike) -> np.ndarray | float:
    """Screen angle in degrees of a hue number; the inverse of hue_from_angle."""
    return np.mod(hue, 360.0)


def rim_color(angle: ArrayLike) -> np.ndarray:
    """Display RGB 0..1 (..., 3) painted at the rim at a screen angle: RIM_RGB interpolated linearly, wrapping at 360."""
    a = np.mod(np.asarray(angle, float), 360.0)
    i = np.floor(a).astype(int)
    f = (a - i)[..., None]
    return RIM_RGB[i % 360] * (1.0 - f) + RIM_RGB[(i + 1) % 360] * f


def wheel_color(angle: ArrayLike, radius: ArrayLike = 1.0) -> np.ndarray:
    """Display RGB 0..1 (..., 3) painted at a screen angle and a radius (fraction of the painted radius, 0 centre .. 1 rim)."""
    g = np.interp(np.asarray(radius, float), BLEND_R, BLEND_G)[..., None]
    return np.clip(CENTRE_GREY + g * (rim_color(angle) - CENTRE_GREY), 0.0, 1.0)


def wheel_image(size: int) -> QImage:
    """size x size ARGB image of the painted wheel (disc diameter = size, antialiased edge, transparent outside)."""
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    dx, dy = xx - c, c - yy
    d = np.hypot(dx, dy)
    radius = size / 2.0
    rgb = wheel_color(np.degrees(np.arctan2(dy, dx)), np.minimum(d / radius, 1.0))
    alpha = np.clip(radius - d + 0.5, 0.0, 1.0)                  # pixel coverage of the disc edge
    argb = np.empty((size, size, 4), np.uint8)
    argb[..., :3] = np.rint(rgb * alpha[..., None] * 255.0)      # premultiplied, B G R A byte order below
    argb[..., 3] = np.rint(alpha * 255.0)
    bgra = np.ascontiguousarray(argb[..., [2, 1, 0, 3]])
    return QImage(bgra.data, size, size, 4 * size, QImage.Format.Format_ARGB32_Premultiplied).copy()


def hue_sat_at(dx: float, dy: float, radius: float) -> tuple[float, float]:
    """(hue, sat 0..1) of a point at (dx, dy) screen pixels from the centre (dy down) of a wheel painted with `radius` pixels."""
    hue = float(hue_from_angle(math.degrees(math.atan2(-dy, dx))))
    sat = min(1.0, math.hypot(dx, dy) / (SAT_FULL_RADIUS * radius))
    return hue, sat


def offset_at(hue: float, sat: float, radius: float) -> tuple[float, float]:
    """(dx, dy) screen pixels from the centre (dy down) where the handle for hue / sat 0..1 is drawn on a wheel of `radius` pixels."""
    a = math.radians(float(angle_from_hue(hue)))
    d = sat * SAT_FULL_RADIUS * radius
    return d * math.cos(a), -d * math.sin(a)


def pure_hue_rgb(hue: ArrayLike) -> np.ndarray:
    """Display RGB 0..1 (..., 3) of HSV (hue, 1, 1), the colour Lightroom shows for a hue outside the wheel."""
    h = np.mod(np.asarray(hue, float), 360.0) / 60.0
    return np.clip(np.stack([np.abs(h - 3.0) - 1.0, 2.0 - np.abs(h - 2.0), 2.0 - np.abs(h - 4.0)], -1), 0.0, 1.0)


def hue_groove_stylesheet() -> str:
    """Style sheet giving a horizontal QSlider over hue 0..360 a pure-hue groove; HSV (h, 1, 1) is linear between the 60-degree stops."""
    stops = ", ".join(f"stop:{h / 360.0:.4f} rgb({r}, {g}, {b})" for h in range(0, 361, 60)
                      for r, g, b in [np.rint(pure_hue_rgb(h) * 255).astype(int)])
    return ("QSlider::groove:horizontal { height: 8px; border: 1px solid #8c8c8c; border-radius: 3px; "
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops}); }}"
            "QSlider::handle:horizontal { width: 10px; margin: -4px 0; border: 1px solid #5c5c5c; border-radius: 3px; background: #f0f0f0; }")
