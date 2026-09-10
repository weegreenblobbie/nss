import pytest
from unittest.mock import patch
from PyQt6.QtWidgets import QApplication
from nss.color_gui.main_window import MainWindow, ShadowsZoneWidget, MidtonesZoneWidget, HighlightsZoneWidget

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
    with patch('nss.color_gui.main_window.QSettings', MockQSettings):
        yield


def test_main_window_memento_flow() -> None:
    app = QApplication.instance() or QApplication([])
    
    window = MainWindow()
    
    # Verify unique class-level STATE_KEY registrations
    registered = window.state_manager.get_registered_widgets()
    assert "shadows_zone" in registered
    assert "midtones_zone" in registered
    assert "highlights_zone" in registered
    assert "master_zone" in registered
    assert "explore_widget" in registered
    
    # Push initial state snapshot
    snapshot1 = window.state_manager.capture_snapshot()
    window.state_manager.push_state(snapshot1)
    
    # Modify slider value on mid_zone
    window.mid_zone.sat_slider.setValue(80)
    window.mid_zone.light_slider.setValue(45)
    
    # Capture & push new state snapshot
    snapshot2 = window.state_manager.capture_snapshot()
    window.state_manager.push_state(snapshot2)
    
    # Verify the slider is changed in state
    assert snapshot2["midtones_zone"]["sat"] == 0.8
    assert snapshot2["midtones_zone"]["light"] == 0.45
    
    # Undo
    prev_snapshot = window.state_manager.undo()
    assert prev_snapshot is not None
    window.state_manager.restore_snapshot(prev_snapshot)
    
    # Mid zone values should have reverted to original state (defaults: 0.0, 0.0)
    assert window.mid_zone.sat_slider.value() == 0
    assert window.mid_zone.light_slider.value() == 0
