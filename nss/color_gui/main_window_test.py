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


def test_manual_slider_changed_no_crash() -> None:
    import numpy as np
    app = QApplication.instance() or QApplication([])
    
    window = MainWindow()
    # Set non-square dummy images so on_manual_slider_changed doesn't return early
    dummy_img = np.zeros((15, 12, 3), dtype=np.float32)
    window.master_image = dummy_img
    window.proxy_image = dummy_img
    
    # Set non-zero, distinct values to verify they flow correctly through the logic
    window.sh_zone.wheel.set_color(120.5, 0.15)
    window.sh_zone.sat_slider.setValue(15)
    
    # Simulate trigger of manual slider change
    window.on_manual_slider_changed()
    
    # Ensure swatch_lbl received the values and set values on it correctly
    assert window.sh_zone.swatch_lbl.hue == 120.5
    assert window.sh_zone.swatch_lbl.sat == 0.15


def test_explore_mode_randomize_locks_center() -> None:
    import numpy as np
    from unittest.mock import patch
    app = QApplication.instance() or QApplication([])
    
    window = MainWindow()
    # Set non-square dummy images so operations don't return early
    dummy_img = np.zeros((15, 12, 3), dtype=np.float32)
    window.master_image = dummy_img
    window.proxy_image = dummy_img
    
    # Establish a baseline state
    window.sh_zone.wheel.set_color(120.5, 0.15)
    window.sh_zone.sat_slider.setValue(15)
    
    # Enable Explore Mode
    window.explore_checkbox.setChecked(True)
    
    # Read the established center state
    snapshot = window.state_manager.capture_snapshot()
    baseline_state = window.rebuild_state_node_from_memento(snapshot)
    assert baseline_state["shadow_hue"] == 120.5
    assert baseline_state["shadow_sat"] == 0.15
    
    # Clicking on the randomize/generate button should lock/keep the center
    # and only regenerate the 8 outer mutations
    with patch('nss.color_gui.main_window.GradingWorker.start') as mock_start:
        window.on_randomize_clicked()
        
    assert window.grid_states is not None
    assert len(window.grid_states) == 9
    
    # Center state (index 4) must be exactly identical to our baseline state
    assert window.grid_states[4]["shadow_hue"] == 120.5
    assert window.grid_states[4]["shadow_sat"] == 0.15
    
    # Outer states must be randomly mutated (so at least some of them differ from the center)
    outer_hues = [window.grid_states[i]["shadow_hue"] for i in range(9) if i != 4]
    assert any(h != 120.5 for h in outer_hues)


def test_slider_label_updates_value_flow() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    # --- Shadows Tab ---
    window.sh_zone.sat_slider.setValue(45)
    assert window.sh_zone.sat_lbl.text() == "Sat: 0.45"

    window.sh_zone.light_slider.setValue(-35)
    assert window.sh_zone.light_lbl.text() == "Luma: -0.35"

    # --- Midtones Tab ---
    window.mid_zone.sat_slider.setValue(75)
    assert window.mid_zone.sat_lbl.text() == "Sat: 0.75"

    window.mid_zone.light_slider.setValue(15)
    assert window.mid_zone.light_lbl.text() == "Luma: +0.15"

    # --- Highlights Tab ---
    window.hi_zone.sat_slider.setValue(12)
    assert window.hi_zone.sat_lbl.text() == "Sat: 0.12"

    window.hi_zone.light_slider.setValue(85)
    assert window.hi_zone.light_lbl.text() == "Luma: +0.85"

    # --- Master Zone ---
    window.master_zone.blending_slider.setValue(65)
    assert window.master_zone.blending_lbl.text() == "Blending: 0.65"

    window.master_zone.balance_slider.setValue(-45)
    assert window.master_zone.balance_lbl.text() == "Balance: -0.45"

    # --- Intensity Slider ---
    window.intensity_slider.setValue(85)
    assert window.intensity_label.text() == " Intensity: 0.85x "


def test_global_rotation_behavior() -> None:
    import numpy as np
    from unittest.mock import patch
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    # 1. Establish base hues on the wheel
    window.sh_zone.wheel.set_color(240.0, 0.0)
    window.mid_zone.wheel.set_color(120.0, 0.0)
    window.hi_zone.wheel.set_color(60.0, 0.0)

    # Move the Rotation slider to +45 degrees
    window.master_zone.rotation_slider.setValue(45)

    # Retrieve current StateNode
    snapshot = window.state_manager.capture_snapshot()
    state = window.rebuild_state_node_from_memento(snapshot)

    # The actual applied/rebuilt state should have rotation = 45.0
    assert state["rotation"] == 45.0

    # Ensure label text updated correctly
    assert window.master_zone.rotation_lbl.text() == "Rotation: 45°"

    # Verify that the effective hues shown on labels and wheels are correctly updated
    assert window.sh_zone.hue_lbl.text() == "Hue: 285°"
    assert window.mid_zone.hue_lbl.text() == "Hue: 165°"
    assert window.hi_zone.hue_lbl.text() == "Hue: 105°"
    
    assert window.sh_zone.wheel.rotation_offset == 45.0
    assert window.mid_zone.wheel.rotation_offset == 45.0
    assert window.hi_zone.wheel.rotation_offset == 45.0

    # Test applying grading on a dummy non-square array with the offset
    dummy_img = np.zeros((15, 12, 3), dtype=np.float32)
    dummy_img[:, :, 0] = 1.0  # pure red
    from nss.color_math import apply_grading
    graded = apply_grading(dummy_img, state)
    assert graded.shape == (15, 12, 3)

    # 2. Test that clicking on "Generate New Random Look" resets Rotation to 0
    # Set dummy images so randomize doesn't return early
    window.master_image = dummy_img
    window.proxy_image = dummy_img
    
    # Trigger Generate / Randomize click
    window.on_randomize_clicked()

    # The Rotation slider and StateNode rotation parameter must be 0
    assert window.master_zone.rotation_slider.value() == 0
    snapshot_after = window.state_manager.capture_snapshot()
    state_after = window.rebuild_state_node_from_memento(snapshot_after)
    assert state_after["rotation"] == 0.0


def test_saturation_slider_propagates_to_wheel() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    # Move the Shadows saturation slider to 65% (represents 0.65 saturation)
    window.sh_zone.sat_slider.setValue(65)

    # Re-verify that the Shadows color wheel's internal state sat matches exactly
    assert window.sh_zone.wheel.sat == 0.65

    # Move Midtones saturation slider to 38%
    window.mid_zone.sat_slider.setValue(38)
    assert window.mid_zone.wheel.sat == 0.38

    # Move Highlights saturation slider to 84%
    window.hi_zone.sat_slider.setValue(84)
    assert window.hi_zone.wheel.sat == 0.84





