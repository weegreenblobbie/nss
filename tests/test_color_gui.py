import sys
import pytest
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QMouseEvent
from nss.color_gui import ColorWheel, ColorSwatch, MainWindow

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
    assert len(sh_colors) == 8
    assert sh_colors[0] == [240.0, 0.0]

    mid_colors = window.load_custom_colors("midtones")
    assert len(mid_colors) == 8
    assert mid_colors[0] == [120.0, 0.0]

    hi_colors = window.load_custom_colors("highlights")
    assert len(hi_colors) == 8
    assert hi_colors[0] == [60.0, 0.0]

    # Test adding to custom color history
    window.add_custom_color("shadows", 210.0, 0.25)
    new_sh_colors = window.load_custom_colors("shadows")
    assert new_sh_colors[0] == [210.0, 0.25]
    # Check deduplication / limit
    assert len(new_sh_colors) == 8
