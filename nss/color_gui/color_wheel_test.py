import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPointF
from nss.color_gui.color_wheel import ColorWheel

def test_color_wheel() -> None:
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


def test_color_wheel_mouse_coordinate_mapping() -> None:
    app = QApplication.instance() or QApplication([])
    
    wheel = ColorWheel()
    
    cx = wheel.width() / 2.0
    cy = wheel.height() / 2.0
    
    # Top-Left quadrant (Green area: dx = -30, dy = -30)
    # angle_rad = atan2(-dy, dx) = atan2(30, -30) = 135 degrees (Green)
    pos_top_left = QPointF(cx - 30.0, cy - 30.0)
    wheel._update_color_from_mouse(pos_top_left)
    assert abs(wheel.hue - 135.0) < 1.0
    
    # Bottom-Left quadrant (Blue area: dx = -30, dy = 30)
    # angle_rad = atan2(-dy, dx) = atan2(-30, -30) = -135 degrees = 225 degrees (Blue)
    pos_bottom_left = QPointF(cx - 30.0, cy + 30.0)
    wheel._update_color_from_mouse(pos_bottom_left)
    assert abs(wheel.hue - 225.0) < 1.0


def test_color_wheel_increased_size() -> None:
    app = QApplication.instance() or QApplication([])
    
    wheel = ColorWheel()
    min_size = wheel.minimumSize()
    max_size = wheel.maximumSize()
    
    # Check that the sizes are 33% larger (minimum 160 and maximum 240)
    assert min_size.width() == 160
    assert min_size.height() == 160
    assert max_size.width() == 240
    assert max_size.height() == 240
