"""Tests of nss.color_gui.color_wheel: colour state, hue = screen angle for clicks and the drawn handle, saturation radius, widget size."""
import math

import numpy as np
import pytest
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPointF
from nss.color_gui.color_wheel import ColorWheel
from nss.color_gui.wheel_lut import SAT_FULL_RADIUS, offset_at

def test_color_wheel() -> None:
    """set_color stores the hue modulo 360 and clamps the saturation to 0..1."""
    app = QApplication.instance() or QApplication([])
    
    wheel = ColorWheel()
    
    # Test setting color
    wheel.set_color(180.0, 0.5)
    assert wheel.hue == 180.0
    assert wheel.sat == 0.5

    # Test clamping sat
    wheel.set_color(400.0, 1.5)
    assert wheel.hue == 40.0
    assert wheel.sat == 1.0

    wheel.set_color(-30.0, -0.5)
    assert wheel.hue == 330.0
    assert wheel.sat == 0.0


def _click(wheel: ColorWheel, angle: float, r: float) -> None:
    """Clicks at screen angle `angle` (degrees, 0 at 3 o'clock, counter-clockwise) and r times the painted radius."""
    cx, cy, radius = wheel.wheel_geometry()
    a = math.radians(angle)
    wheel._update_color_from_mouse(QPointF(cx + r * radius * math.cos(a), cy - r * radius * math.sin(a)))


@pytest.mark.parametrize("angle", np.arange(0.0, 360.0, 15.0))
def test_click_at_angle_gives_that_hue(angle: float) -> None:
    """The hue number is the screen angle of the click, exactly as on Lightroom's wheel."""
    app = QApplication.instance() or QApplication([])
    wheel = ColorWheel()
    wheel.resize(240, 240)
    _click(wheel, angle, 0.5)
    assert abs((wheel.hue - angle + 180.0) % 360.0 - 180.0) < 1e-6


def test_click_distance_gives_saturation() -> None:
    """Sat is linear in the distance from the centre, 1.0 at SAT_FULL_RADIUS of the painted radius, clamped beyond; the centre keeps the hue."""
    app = QApplication.instance() or QApplication([])
    wheel = ColorWheel()
    wheel.resize(240, 240)
    _click(wheel, 200.0, SAT_FULL_RADIUS)
    assert wheel.sat == pytest.approx(1.0)
    _click(wheel, 200.0, SAT_FULL_RADIUS * 0.35)
    assert wheel.sat == pytest.approx(0.35)
    _click(wheel, 200.0, 1.0)
    assert wheel.sat == 1.0
    cx, cy, _ = wheel.wheel_geometry()
    wheel._update_color_from_mouse(QPointF(cx, cy))
    assert wheel.sat == 0.0 and wheel.hue == pytest.approx(200.0)


def _handle_centre(img: np.ndarray) -> tuple[float, float]:
    """Centroid (x, y) of the handle's black ring; the painted wheel has no near-black pixels."""
    ys, xs = np.nonzero(img[..., :3].max(-1) < 60)
    return float(xs.mean()), float(ys.mean())


@pytest.mark.parametrize("hue", np.arange(0.0, 360.0, 15.0))
def test_handle_drawn_at_the_hue_angle(hue: float) -> None:
    """Rendered offscreen, the handle for (hue, Sat 100) is within 1 px of angle `hue` at SAT_FULL_RADIUS, and clicking there returns the hue."""
    app = QApplication.instance() or QApplication([])
    wheel = ColorWheel()
    wheel.resize(240, 240)
    wheel.set_color(hue, 1.0)
    shot = wheel.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    img = np.frombuffer(shot.constBits().asstring(shot.sizeInBytes()), np.uint8).reshape(shot.height(), shot.bytesPerLine() // 4, 4)
    x, y = _handle_centre(img)
    cx, cy, radius = wheel.wheel_geometry()
    dx, dy = offset_at(hue, 1.0, radius)
    assert math.hypot(x + 0.5 - (cx + dx), y + 0.5 - (cy + dy)) <= 1.0     # pixel i covers [i, i + 1]: its centre is i + 0.5
    wheel._update_color_from_mouse(QPointF(cx + dx, cy + dy))
    assert abs((wheel.hue - hue + 180.0) % 360.0 - 180.0) < 0.5


def test_color_wheel_increased_size() -> None:
    """The wheel is 160 to 240 px square."""
    app = QApplication.instance() or QApplication([])
    
    wheel = ColorWheel()
    min_size = wheel.minimumSize()
    max_size = wheel.maximumSize()
    
    # Check that the sizes are 33% larger (minimum 160 and maximum 240)
    assert min_size.width() == 160
    assert min_size.height() == 160
    assert max_size.width() == 240
    assert max_size.height() == 240
