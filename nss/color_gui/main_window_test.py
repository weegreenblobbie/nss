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
    window.sh_zone.hue_slider.setValue(120)
    window.sh_zone.wheel.set_color(120.0, 0.15)
    window.sh_zone.sat_slider.setValue(15)
    
    # Simulate trigger of manual slider change
    window.on_manual_slider_changed()
    
    # Ensure swatch_lbl received the values and set values on it correctly
    assert window.sh_zone.swatch_lbl.hue == 120.0
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
    window.sh_zone.hue_slider.setValue(120)
    window.sh_zone.wheel.set_color(120.0, 0.15)
    window.sh_zone.sat_slider.setValue(15)
    
    # Enable Explore Mode
    with patch('nss.color_gui.main_window.GradingWorker.start') as mock_start:
        window.explore_checkbox.setChecked(True)
    
    # Read the established center state
    snapshot = window.state_manager.capture_snapshot()
    baseline_state = window.rebuild_state_node_from_memento(snapshot)
    assert baseline_state["shadow_hue"] == 120.0
    assert baseline_state["shadow_sat"] == 0.15
    
    # Clicking on the randomize/generate button should lock/keep the center
    # and only regenerate the 8 outer mutations
    with patch('nss.color_gui.main_window.GradingWorker.start') as mock_start:
        window.on_randomize_clicked()
        
    assert window.grid_states is not None
    assert len(window.grid_states) == 9
    
    # Center state (index 4) must be exactly identical to our baseline state
    assert window.grid_states[4]["shadow_hue"] == 120.0
    assert window.grid_states[4]["shadow_sat"] == 0.15
    
    # Outer states must be randomly mutated (so at least some of them differ from the center)
    outer_hues = [window.grid_states[i]["shadow_hue"] for i in range(9) if i != 4]
    assert any(h != 120.0 for h in outer_hues)


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


def test_zone_hue_sliders_and_rotation_interaction() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    # 1. Simulate dragging the Shadows Hue slider to 145 degrees
    window.sh_zone.hue_slider.setValue(145)

    # By default, Rotation is 0, so effective hue is 145
    assert window.sh_zone.hue_lbl.text() == "Hue: 145°"
    assert window.sh_zone.wheel.hue == 145.0
    assert window.sh_zone.wheel.rotation_offset == 0.0

    # 2. Move global Rotation slider to +30 degrees
    window.master_zone.rotation_slider.setValue(30)

    # Re-verify that effective hue label reflects (145 + 30) % 360 = 175°
    assert window.sh_zone.hue_lbl.text() == "Hue: 175°"
    assert window.sh_zone.wheel.hue == 145.0
    assert window.sh_zone.wheel.rotation_offset == 30.0

    # 3. Drag Midtones Hue slider to 350 degrees, and set Rotation to +20 degrees
    window.mid_zone.hue_slider.setValue(350)
    window.master_zone.rotation_slider.setValue(20)

    # Re-verify that Midtones effective hue is (350 + 20) % 360 = 10°
    assert window.mid_zone.hue_lbl.text() == "Hue: 10°"
    assert window.mid_zone.wheel.hue == 350.0
    assert window.mid_zone.wheel.rotation_offset == 20.0


def test_explore_mode_click_halves_variation_strength() -> None:
    import numpy as np
    from unittest.mock import patch
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    # Use non-square dummy images as per guidelines
    dummy_img = np.zeros((15, 12, 3), dtype=np.float32)
    window.master_image = dummy_img
    window.proxy_image = dummy_img

    # Initialize a baseline grid states to simulate click drill-down
    with patch('nss.color_gui.main_window.GradingWorker.start') as mock_start:
        window.explore_checkbox.setChecked(True)
    assert window.grid_states is not None
    assert len(window.grid_states) == 9

    # Set variation slider to 80 (represents 0.8 strength)
    window.explore_widget.var_slider.setValue(80)
    assert window.explore_widget.var_slider.value() == 80
    assert window.explore_widget.variation_strength == 0.8

    # Simulate left-click on an outer slot (container 0)
    with patch('nss.color_gui.main_window.GradingWorker.start') as mock_start:
        window.on_explore_container_clicked(0)

    # Slider value must be programmatically cut in half (from 80 to 40)
    assert window.explore_widget.var_slider.value() == 40
    assert window.explore_widget.variation_strength == 0.4


def test_multiview_tone_toggle_and_wheel_rendering() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    # 1. Default show_all_tones should be False
    assert window.show_all_tones is False
    assert window.tabs.currentIndex() == 0  # Shadows is active by default

    # 2. Re-clicking the already active tab (index 0) should toggle show_all_tones to True
    window.on_tab_bar_clicked(0)
    assert window.show_all_tones is True

    # 3. Re-clicking it again should toggle show_all_tones back to False
    window.on_tab_bar_clicked(0)
    assert window.show_all_tones is False

    # 4. Toggle back to True and set specific, non-zero values for all three zones
    window.on_tab_bar_clicked(0)
    assert window.show_all_tones is True

    window.sh_zone.hue_slider.setValue(145)
    window.sh_zone.sat_slider.setValue(25)  # 0.25 sat

    window.mid_zone.hue_slider.setValue(210)
    window.mid_zone.sat_slider.setValue(55)  # 0.55 sat

    window.hi_zone.hue_slider.setValue(70)
    window.hi_zone.sat_slider.setValue(85)  # 0.85 sat

    # Set global Rotation slider to +30 degrees
    window.master_zone.rotation_slider.setValue(30)

    # Trigger real-time callback
    window.on_manual_slider_changed()

    # Verify that the active Shadows wheel gets show_all_tones = True and all_tones_data
    assert window.sh_zone.wheel.show_all_tones is True
    assert window.sh_zone.wheel.rotation_offset == 30.0
    
    # Verify that the inactive wheels have show_all_tones = False
    assert window.mid_zone.wheel.show_all_tones is False
    assert window.hi_zone.wheel.show_all_tones is False

    # Verify that all_tones_data contains exact non-zero base hues and saturations
    assert window.sh_zone.wheel.all_tones_data['S'] == (145.0, 0.25)
    assert window.sh_zone.wheel.all_tones_data['M'] == (210.0, 0.55)
    assert window.sh_zone.wheel.all_tones_data['H'] == (70.0, 0.85)

    # Double check that effective hues compute correctly
    # Shadows: (145.0 + 30.0) % 360.0 = 175.0
    # Midtones: (210.0 + 30.0) % 360.0 = 240.0
    # Highlights: (70.0 + 30.0) % 360.0 = 100.0
    sh_eff = (window.sh_zone.wheel.all_tones_data['S'][0] + window.sh_zone.wheel.rotation_offset) % 360.0
    mid_eff = (window.sh_zone.wheel.all_tones_data['M'][0] + window.sh_zone.wheel.rotation_offset) % 360.0
    hi_eff = (window.sh_zone.wheel.all_tones_data['H'][0] + window.sh_zone.wheel.rotation_offset) % 360.0

    assert sh_eff == 175.0
    assert mid_eff == 240.0
    assert hi_eff == 100.0

    # 5. Switching tabs (e.g. to Midtones tab at index 1) must PRESERVE the show_all_tones state!
    # Simulate tab switch to Midtones
    window.tabs.setCurrentIndex(1)
    
    # State must still be True
    assert window.show_all_tones is True
    
    # Midtones wheel (now active) should have show_all_tones = True
    assert window.mid_zone.wheel.show_all_tones is True
    assert window.sh_zone.wheel.show_all_tones is False
    assert window.hi_zone.wheel.show_all_tones is False
    
    # Clicking the already active Midtones tab (index 1) should toggle it back to False
    window.on_tab_bar_clicked(1)
    assert window.show_all_tones is False
    assert window.mid_zone.wheel.show_all_tones is False


def test_layout_fixes() -> None:
    from PyQt6.QtWidgets import QScrollArea, QToolButton
    from PyQt6.QtCore import Qt
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    # 1. Verify QScrollArea integration
    scroll_areas = window.findChildren(QScrollArea)
    assert len(scroll_areas) == 1
    sa = scroll_areas[0]
    
    # Check that scroll_area wraps the inspector_panel
    assert sa.widget() is window.inspector_panel
    assert sa.widgetResizable() is True
    assert sa.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff

    # 2. Verify Undo/Redo corner widget integration
    menu_bar = window.menuBar()
    corner_widget = menu_bar.cornerWidget(Qt.Corner.TopRightCorner)
    assert corner_widget is not None
    
    # Check that back_btn and forward_btn are children of the corner widget
    assert window.back_btn.parent() is corner_widget
    assert window.forward_btn.parent() is corner_widget
    assert window.history_label.parent() is corner_widget
    
    # Check they are QToolButtons
    assert isinstance(window.back_btn, QToolButton)
    assert isinstance(window.forward_btn, QToolButton)
    
    # Verify their actions remain fully functional
    assert window.back_btn.defaultAction() is window.back_action
    assert window.forward_btn.defaultAction() is window.forward_action









