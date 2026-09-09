import sys
import pytest
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QMouseEvent
from nss.color_gui import ColorWheel, ColorSwatch, MainWindow, ClickableSwatchLabel

class MockQSettings:
    _store = {}

    def __init__(self, organization: str, application: str) -> None:
        pass

    def value(self, key: str, default: any = None) -> any:
        return self._store.get(key, default)

    def setValue(self, key: str, value: any) -> None:
        self._store[key] = value

@pytest.fixture(autouse=True)
def mock_qsettings_gui():
    MockQSettings._store.clear()
    with patch('nss.color_gui.QSettings', MockQSettings):
        yield

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

def test_color_swatch() -> None:
    app = QApplication.instance() or QApplication([])
    
    swatch = ColorSwatch(120.0, 0.4)
    assert swatch.hue == 120.0
    assert swatch.sat == 0.4

    swatch.update_color(60.0, 0.8)
    assert swatch.hue == 60.0
    assert swatch.sat == 0.8

def test_main_window_color_methods() -> None:
    app = QApplication.instance() or QApplication([])
    
    window = MainWindow()
    
    # Test default history lists
    sh_colors = window.load_custom_colors("shadows")
    assert len(sh_colors) == 16
    assert sh_colors[0] == [240.0, 0.0]

    mid_colors = window.load_custom_colors("midtones")
    assert len(mid_colors) == 16
    assert mid_colors[0] == [120.0, 0.0]

    hi_colors = window.load_custom_colors("highlights")
    assert len(hi_colors) == 16
    assert hi_colors[0] == [60.0, 0.0]

    # Test adding to custom color history
    window.add_custom_color("shadows", 210.0, 0.25)
    new_sh_colors = window.load_custom_colors("shadows")
    assert new_sh_colors[0] == [210.0, 0.25]
    # Check deduplication / limit
    assert len(new_sh_colors) == 16

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

def test_color_math_boundary_interpolation_and_white_desaturation() -> None:
    # Set up QApplication if not already created
    app = QApplication.instance() or QApplication([])

    # 1. Test white desaturation formula on swatches
    swatch = ColorSwatch(180.0, 0.0)
    # Color should be pure white when saturation is 0.0 (lightness = 1.0)
    from PyQt6.QtGui import QColor
    color = QColor.fromHslF(180.0 / 360.0, 0.0, 1.0)
    assert color.name() in swatch.styleSheet()

    # 2. Test 0/360 circular boundary interpolation in apply_grading
    # Set up a simple 1x1 image
    import numpy as np
    from nss.color_math import apply_grading, create_default_state
    img = np.zeros((1, 1, 3), dtype=np.float32)
    img[0, 0] = [0.5, 0.5, 0.5] # gray
    
    # State with shadows at 359° and highlights at 1°
    state = create_default_state("Monochromatic")
    state["shadow_hue"] = 359.0
    state["shadow_sat"] = 0.5
    state["midtone_sat"] = 0.0
    state["highlight_hue"] = 1.0
    state["highlight_sat"] = 0.5
    state["balance"] = 0.0 # Equal weight
    state["blending"] = 0.5
    
    graded = apply_grading(img, state)
    import cv2
    hls_graded = cv2.cvtColor(graded, cv2.COLOR_RGB2HLS)
    hue = hls_graded[0, 0, 0]
    
    # Direct average would be 180 (Cyan). Correct circular average is ~0.0 (Red).
    # Since 359° and 1° are symmetric, the result must be very close to 0.0 / 360.0.
    assert hue < 2.0 or hue > 358.0

def test_color_swatch_drag_and_drop() -> None:
    app = QApplication.instance() or QApplication([])

    # Create a draggable ClickableSwatchLabel
    active_swatch = ClickableSwatchLabel()
    active_swatch.hue = 300.0
    active_swatch.sat = 0.75

    # Create a drop-accepting ColorSwatch
    grid_swatch = ColorSwatch(120.0, 0.5)
    grid_swatch.index = 3

    # Connect to check overridden signals
    overwritten_data = []
    grid_swatch.overwritten.connect(lambda idx, h, s: overwritten_data.append((idx, h, s)))

    # Simulate Drop
    from PyQt6.QtCore import QMimeData
    from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QDrag
    
    # Setup Mime Data
    mime_data = QMimeData()
    mime_data.setText("300.0,0.75")

    # Simulate drag enter
    # On PyQt6 we can create mock events or call dropEvent directly to test the behavior
    # Calling dropEvent with a mock event carrying mime_data is extremely clean
    class MockEvent:
        def __init__(self, mime):
            self._mime = mime
            self.proposed_accepted = False

        def mimeData(self):
            return self._mime

        def acceptProposedAction(self):
            self.proposed_accepted = True

    mock_event = MockEvent(mime_data)
    grid_swatch.dropEvent(mock_event)

    # Verify that the color swatch is updated with the dragged color
    assert grid_swatch.hue == 300.0
    assert grid_swatch.sat == 0.75
    assert mock_event.proposed_accepted
    assert overwritten_data == [(3, 300.0, 0.75)]

