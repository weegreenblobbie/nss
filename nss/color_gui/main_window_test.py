"""Tests of nss.color_gui.main_window: memento flow, wheels and master controls, the Global tab, threaded rendering, presets and TIFF IO."""
import json
import threading
from pathlib import Path
from typing import Iterator
import numpy as np
import pytest
from unittest.mock import patch
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from nss.color_gui.main_window import MainWindow, ShadowsZoneWidget, MidtonesZoneWidget, HighlightsZoneWidget, GlobalZoneWidget, RENDER_DEBOUNCE_MS
from nss.color_grading.grading import Grading
from nss.color_mixer import ColorMixer
from nss.tiff_file import TiffFile
from nss.icc import gamma_icc_bytes
from plugin import lr_bridge

ICC = gamma_icc_bytes("Adobe RGB (1998)", [(0.60974, 0.31111, 0.01947), (0.20528, 0.62567, 0.06087), (0.14919, 0.06322, 0.74457)], 563 / 256)


def test_main_window_memento_flow() -> None:
    """Widgets register unique STATE_KEYs and undo restores the previous slider values."""
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


def test_explore_mode_randomize_locks_center() -> None:
    """In Explore Mode, Random Shift keeps the centre state and re-rolls only the 8 outer images."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()

    # Establish a baseline state
    window.sh_zone.hue_slider.setValue(120)
    window.sh_zone.wheel.set_color(120.0, 0.15)
    window.sh_zone.sat_slider.setValue(15)
    window.harmony_panel.add_sample(6, 3, np.array([0.6, 0.3, 0.2]), 20.0)

    # Enable Explore Mode
    with patch('nss.color_gui.main_window.GradingWorker.start') as mock_start:
        window.explore_checkbox.setChecked(True)
    first_grid = list(window.grid_states)

    # Read the established center state
    snapshot = window.state_manager.capture_snapshot()
    baseline_state = window.rebuild_state_node_from_memento(snapshot)
    assert baseline_state["shadow_hue"] == 120.0
    assert baseline_state["shadow_sat"] == 0.15

    # Random Shift keeps the center and only regenerates the 8 outer images
    with patch('nss.color_gui.main_window.GradingWorker.start') as mock_start:
        window.on_randomize_clicked()

    assert window.grid_states is not None
    assert len(window.grid_states) == 9

    # Center state (index 4) must be exactly identical to our baseline state
    assert window.grid_states[4] == baseline_state

    # Outer states are new random shifts of the center
    outer = [window.grid_states[i] for i in range(9) if i != 4]
    assert all(s != baseline_state for s in outer)
    assert outer != [first_grid[i] for i in range(9) if i != 4]


def test_slider_label_updates_value_flow() -> None:
    """Slider moves update every zone's and the master's labels."""
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
    """Rotation shifts the effective hues on labels and wheels, and Generate resets it to 0."""
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


def test_saturation_slider_propagates_to_wheel() -> None:
    """Each zone's Sat slider sets its wheel's saturation."""
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
    """Hue labels show hue + Rotation while the wheel keeps the base hue."""
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
    """Promoting an outer mutation halves the Variation Strength."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    window.harmony_panel.add_sample(6, 3, np.array([0.6, 0.3, 0.2]), 20.0)

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


def test_random_shift_enabled_only_with_a_sample_and_a_checked_panel() -> None:
    """Random Shift needs a Color Harmony sample and Mixer or Grading checked."""
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.randomize_btn.text() == "Random Shift" and not window.randomize_btn.isEnabled()
    window.harmony_panel.add_sample(6, 3, np.array([0.6, 0.3, 0.2]), 20.0)
    assert window.randomize_btn.isEnabled()
    window.explore_widget.mixer_checkbox.setChecked(False)
    assert window.randomize_btn.isEnabled()
    window.explore_widget.grading_checkbox.setChecked(False)
    assert not window.randomize_btn.isEnabled()
    window.explore_widget.mixer_checkbox.setChecked(True)
    assert window.randomize_btn.isEnabled()
    window.harmony_panel.clear_samples()
    assert not window.randomize_btn.isEnabled()


def test_random_shift_changes_only_checked_panels_from_the_current_values() -> None:
    """Mixer only: the wheels stay and each click is random_shift of the sliders as they are now (so clicks accumulate), one undo step each;
    Grading only: the mixer stays."""
    from nss.color_explore.random_shift import random_shift
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    window.harmony_panel.add_sample(6, 3, np.array([0.6, 0.3, 0.2]), 20.0)
    window.explore_widget.var_slider.setValue(50)
    window.explore_widget.grading_checkbox.setChecked(False)
    current = lambda: window.rebuild_state_node_from_memento(window.state_manager.capture_snapshot())
    before = current()
    window.on_randomize_clicked()
    first = current()
    assert first["mixer_hue"] != before["mixer_hue"]
    assert {k: v for k, v in first.items() if not k.startswith("mixer_")} == {k: v for k, v in before.items() if not k.startswith("mixer_")}

    expected = random_shift(first, window.selected_colors(first), 0.5, True, False, np.random.default_rng(9))
    window.rng = np.random.default_rng(9)
    steps = len(window.state_manager.undo_stack)
    window.on_randomize_clicked()
    assert current() == expected and len(window.state_manager.undo_stack) == steps + 1

    window.explore_widget.mixer_checkbox.setChecked(False)
    window.explore_widget.grading_checkbox.setChecked(True)
    window.on_randomize_clicked()
    after = current()
    assert all(after[f"mixer_{kind}"] == expected[f"mixer_{kind}"] for kind in ("hue", "sat", "lum"))
    assert after != expected
    window.render_timer.stop()


def test_explore_mode_without_a_sample_shows_the_centre_only() -> None:
    """Without a sample Explore Mode grades only the centre, says why, and the empty outer cells cannot be promoted or edited."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    with patch('nss.color_gui.main_window.GradingWorker.start'):
        window.explore_checkbox.setChecked(True)
        assert window.grid_states[4] is not None and all(window.grid_states[i] is None for i in range(9) if i != 4)
        assert window.statusBar().currentMessage().startswith("Explore Mode needs a colour sampled in Color Harmony")
        window.explore_widget.var_slider.setValue(40)
        window.on_explore_container_clicked(0)
        window.on_explore_edit_requested(0)
    assert window.explore_widget.var_slider.value() == 40 and window.explore_checkbox.isChecked()


def test_multiview_tone_toggle_and_wheel_rendering() -> None:
    """Re-clicking the active tab toggles the all-tones view, which survives tab switches."""
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
    """The inspector sits in one scroll area and Undo / Redo live in the menu-bar corner widget."""
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











# --- Global wheel ---

_WINDOWS = []      # windows made by _window_with_image; their render threads are finished before the window can be destroyed


@pytest.fixture(autouse=True)
def finish_render_threads() -> Iterator[None]:
    """After each test, waits for the render threads of the windows in _WINDOWS while the QApplication is still alive."""
    app = QApplication.instance() or QApplication([])      # held until the threads below are finished
    yield
    while _WINDOWS:
        window = _WINDOWS.pop()
        window.render_timer.stop()
        _finish_renders(window)
        if window.save_worker is not None:
            window.save_worker.wait()


def _finish_renders(window: MainWindow) -> None:
    """Waits for the preview worker, and the newest request it queued, to finish and deliver their images."""
    while window.preview_worker is not None and (window.preview_worker.isRunning() or window.pending_preview_state is not None):
        window.preview_worker.wait()
        QApplication.processEvents()


def _window_with_image(icc: bytes | None = ICC) -> MainWindow:
    """A MainWindow showing a small image tagged with `icc`; the caller holds the QApplication."""
    window = MainWindow()
    img = np.random.default_rng(6).random((15, 12, 3)).astype(np.float32)
    window.tiff_obj = TiffFile.from_array(img, icc=icc)
    window.master_image = img
    window.proxy_image = img
    _WINDOWS.append(window)
    return window


def _set_global(window: MainWindow, hue: int, sat: int, light: int) -> None:
    """Moves the Global tab'"'"'s three sliders (slider units)."""
    window.gl_zone.hue_slider.setValue(hue)
    window.gl_zone.sat_slider.setValue(sat)
    window.gl_zone.light_slider.setValue(light)


def test_global_tab_defaults_and_ranges() -> None:
    """The Global tab is the 4th tab, registered, starts at 0 / 0 / 0 and clamps to its slider ranges."""
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert isinstance(window.gl_zone, GlobalZoneWidget)
    assert window.tabs.count() == 4 and window.tabs.tabText(3) == "Global"
    assert "global_zone" in window.state_manager.get_registered_widgets()
    gz = window.gl_zone
    assert (gz.hue_slider.value(), gz.sat_slider.value(), gz.light_slider.value()) == (0, 0, 0)
    assert (gz.hue_slider.minimum(), gz.hue_slider.maximum()) == (0, 360)
    assert (gz.sat_slider.minimum(), gz.sat_slider.maximum()) == (0, 100)
    assert (gz.light_slider.minimum(), gz.light_slider.maximum()) == (-100, 100)
    _set_global(window, 500, 150, -300)
    assert (gz.hue_slider.value(), gz.sat_slider.value(), gz.light_slider.value()) == (360, 100, -100)


def test_global_sliders_reach_grading_apply_debounced() -> None:
    """Three Global slider changes cause one debounced Grading.apply call with the image'"'"'s profile and the gl_* values."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    calls = []
    real_apply = Grading.apply
    with patch.object(Grading, "apply", autospec=True, side_effect=lambda self, img, icc, **kw: calls.append((icc, kw)) or real_apply(self, img, icc, **kw)):
        _set_global(window, 200, 35, -20)
        assert calls == []                                  # nothing graded while the sliders are still moving
        assert window.render_timer.isActive()
        QTest.qWait(RENDER_DEBOUNCE_MS + 200)
        _finish_renders(window)
    assert len(calls) == 1                                  # three slider changes, one render
    icc, kw = calls[0]
    assert icc == ICC
    assert (kw["gl_h"], kw["gl_s"], kw["gl_l"]) == pytest.approx((200.0, 35.0, -20.0))


def test_release_renders_immediately_and_cancels_pending() -> None:
    """A slider release renders at once (no debounce) and cancels the pending timer."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    with patch.object(Grading, "apply", autospec=True, side_effect=lambda self, img, icc, **kw: img) as apply:
        window.gl_zone.sat_slider.setValue(40)
        window.gl_zone.parent_slider_released()             # discrete action: render now, not after the debounce
        assert not window.render_timer.isActive()
        _finish_renders(window)
        assert apply.call_count == 1
        assert apply.call_args.kwargs["gl_s"] == pytest.approx(40.0)


def test_preview_renders_off_the_gui_thread() -> None:
    """The GUI call returns while Grading.apply is still running; requests made meanwhile collapse to the newest one."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    release = threading.Event()
    calls = []

    def slow_apply(self: Grading, img: np.ndarray, icc: bytes, **kw: float) -> np.ndarray:
        """Records the call and blocks until the test releases it."""
        calls.append(kw)
        release.wait(5.0)
        return img

    with patch.object(Grading, "apply", autospec=True, side_effect=slow_apply):
        state = window.rebuild_state_node_from_memento(window.state_manager.capture_snapshot())
        for sat in (0.1, 0.2, 0.3, 0.4):
            window.request_preview(dict(state, global_sat=sat))
        assert window.preview_worker.isRunning()            # still grading the first request, yet we got here
        assert window.pending_preview_state["global_sat"] == 0.4
        release.set()
        _finish_renders(window)
    assert [kw["gl_s"] for kw in calls] == pytest.approx([10.0, 40.0])     # the first, then only the newest


def test_no_icc_profile_leaves_image_ungraded() -> None:
    """Without an ICC profile nothing is graded and no render thread starts."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image(icc=None)
    state = window.rebuild_state_node_from_memento(window.state_manager.capture_snapshot())
    state["global_sat"] = 0.5
    with patch.object(Grading, "apply", autospec=True) as apply:
        assert window.graded(window.proxy_image, state) is window.proxy_image
        window.request_preview(state)
        assert window.preview_worker is None                # nothing to grade, so no thread
        apply.assert_not_called()


def test_reset_all_resets_global_and_wheels() -> None:
    """Reset Grading returns every wheel and master slider to its default as one undo step."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    _set_global(window, 120, 40, 25)
    window.sh_zone.sat_slider.setValue(30)
    window.master_zone.blending_slider.setValue(80)
    window.master_zone.rotation_slider.setValue(30)
    pushes = len(window.state_manager.undo_stack)
    window.master_zone.reset_all_btn.click()
    gz = window.gl_zone
    assert (gz.hue_slider.value(), gz.sat_slider.value(), gz.light_slider.value()) == (0, 0, 0)
    assert window.sh_zone.sat_slider.value() == 0
    assert window.master_zone.blending_slider.value() == 50
    assert window.master_zone.rotation_slider.value() == 0
    assert len(window.state_manager.undo_stack) == pushes + 1


def test_global_double_click_reset() -> None:
    """Double-clicking the Global labels resets their sliders."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    _set_global(window, 120, 40, 25)
    window.gl_zone.sat_lbl.doubleClicked.emit()
    window.gl_zone.light_lbl.doubleClicked.emit()
    window.gl_zone.hue_lbl.doubleClicked.emit()
    assert (window.gl_zone.hue_slider.value(), window.gl_zone.sat_slider.value(), window.gl_zone.light_slider.value()) == (0, 0, 0)


def test_global_in_undo_snapshots() -> None:
    """Global values are part of undo / redo."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    window.on_discrete_action()
    _set_global(window, 90, 20, 10)
    window.on_discrete_action()
    window.on_undo_clicked()
    assert window.gl_zone.sat_slider.value() == 0
    window.on_redo_clicked()
    assert (window.gl_zone.hue_slider.value(), window.gl_zone.sat_slider.value(), window.gl_zone.light_slider.value()) == (90, 20, 10)


def test_legacy_snapshot_without_global_defaults_to_zero() -> None:
    """A snapshot without global_zone rebuilds to Global 0."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    snapshot = window.state_manager.capture_snapshot()
    del snapshot["global_zone"]
    state = window.rebuild_state_node_from_memento(snapshot)
    assert (state["global_hue"], state["global_sat"], state["global_light"]) == (0.0, 0.0, 0.0)


def test_preset_save_load_round_trip(tmp_path: Path) -> None:
    """Save Preset then Load Preset restores the Global and master values."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    path = str(tmp_path / "look.json")
    _set_global(window, 210, 45, -15)
    window.master_zone.balance_slider.setValue(-40)
    with patch("nss.color_gui.main_window.QFileDialog.getSaveFileName", return_value=(path, "")):
        window.on_save_preset()
    window.master_zone.reset_all_btn.click()
    assert window.gl_zone.sat_slider.value() == 0
    with patch("nss.color_gui.main_window.QFileDialog.getOpenFileName", return_value=(path, "")):
        window.on_load_preset()
    assert (window.gl_zone.hue_slider.value(), window.gl_zone.sat_slider.value(), window.gl_zone.light_slider.value()) == (210, 45, -15)
    assert window.master_zone.balance_slider.value() == -40


def test_load_legacy_preset_without_global(tmp_path: Path) -> None:
    """Loading a preset without global_* keys sets Global to 0 and applies its other values."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    _set_global(window, 210, 45, -15)
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"version": 1, "state": {"shadow_hue": 200.0, "shadow_sat": 0.3, "blending": 0.6}}))
    with patch("nss.color_gui.main_window.QFileDialog.getOpenFileName", return_value=(str(path), "")):
        window.on_load_preset()
    assert (window.gl_zone.hue_slider.value(), window.gl_zone.sat_slider.value(), window.gl_zone.light_slider.value()) == (0, 0, 0)
    assert window.sh_zone.sat_slider.value() == 30


def test_open_and_save_keep_values_and_icc(tmp_path: Path) -> None:
    """Open keeps the stored values; Save runs in a SaveWorker and writes the graded image with its ICC profile."""
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    _WINDOWS.append(window)
    window.mru_manager.add_path = lambda path: None         # keep the user's recent-directories list untouched
    raw = np.linspace(2000, 40000, 9 * 7 * 3).astype(np.uint16).reshape(9, 7, 3)
    src, dst = str(tmp_path / "in.tif"), str(tmp_path / "out.tif")
    TiffFile.from_array(raw / 65535.0, icc=ICC).saveas(src)
    with patch("nss.color_gui.main_window.QFileDialog.getOpenFileName", return_value=(src, "")):
        window.open_file(start_dir=str(tmp_path))
    assert np.allclose(window.master_image, raw / 65535.0, atol=1e-6)    # stored values, no min/max stretch
    _set_global(window, 30, 50, 0)
    with patch("nss.color_gui.main_window.QFileDialog.getSaveFileName", return_value=(dst, "")):
        window.save_file()
    assert not window.save_action.isEnabled()              # saving runs in a SaveWorker
    window.save_worker.wait()
    QApplication.processEvents()
    assert window.save_action.isEnabled()
    assert window.statusBar().currentMessage().startswith("Successfully saved")
    out = TiffFile().read(dst, dtype=np.float64)
    assert out.iccprofile == ICC
    assert out.array.shape == (9, 7, 3)
    assert np.abs(out.array - raw / 65535.0).max() > 1e-3


def test_mixer_runs_before_grading_debounced() -> None:
    """Mixer slider changes cause one debounced ColorMixer.apply with the image's profile and levels; Grading.apply gets the mixer's output."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    assert "color_mixer" in window.state_manager.get_registered_widgets()
    mixer_calls, graded_inputs = [], []
    with patch.object(ColorMixer, "apply", autospec=True, side_effect=lambda self, img, icc, hue, sat, lum: mixer_calls.append((icc, hue, sat, lum)) or img * 0.5), \
         patch.object(Grading, "apply", autospec=True, side_effect=lambda self, img, icc, **kw: graded_inputs.append(img) or img):
        window.mixer_panel.sliders["hue"][1].setValue(35)
        window.mixer_panel.sliders["lum"][4].setValue(-15)
        assert mixer_calls == []
        QTest.qWait(RENDER_DEBOUNCE_MS + 200)
        _finish_renders(window)
    assert len(mixer_calls) == 1
    icc, hue, sat, lum = mixer_calls[0]
    assert icc == ICC
    assert hue == (0.0, 35.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert sat == (0.0,) * 8
    assert lum == (0.0, 0.0, 0.0, 0.0, -15.0, 0.0, 0.0, 0.0)
    assert np.allclose(graded_inputs[-1], window.proxy_image * 0.5)


def test_mixer_drag_renders_drag_proxy_then_release_renders_proxy() -> None:
    """While a mixer slider is held the preview renders the drag proxy; the release renders the full proxy and records an undo step."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    window.drag_proxy = window.proxy_image[::2, ::2].copy()
    shapes = []
    with patch.object(ColorMixer, "apply", autospec=True, side_effect=lambda self, img, icc, hue, sat, lum: shapes.append(img.shape) or img):
        slider = window.mixer_panel.sliders["sat"][6]
        slider.setSliderDown(True)
        slider.setValue(45)
        QTest.qWait(RENDER_DEBOUNCE_MS + 200)
        _finish_renders(window)
        assert shapes == [(8, 6, 3)]
        undo_len = len(window.state_manager.undo_stack)
        slider.setSliderDown(False)                         # emits sliderReleased
        _finish_renders(window)
    assert shapes == [(8, 6, 3), (15, 12, 3)]
    assert len(window.state_manager.undo_stack) == undo_len + 1
    assert window.state_manager.undo_stack[-1]["color_mixer"]["sat"][6] == 45


def test_mixer_release_at_unchanged_value_still_replaces_drag_preview() -> None:
    """A drag released where it started records no undo step but still renders the full proxy after the debounce."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    window.drag_proxy = window.proxy_image[::2, ::2].copy()
    window.state_manager.push_state(window.state_manager.capture_snapshot())
    shapes = []
    with patch.object(ColorMixer, "apply", autospec=True, side_effect=lambda self, img, icc, hue, sat, lum: shapes.append(img.shape) or img):
        window.mixer_panel.sliders["hue"][3].setValue(20)     # not dragging: a keyboard-like change, then back
        window.on_discrete_action()
        _finish_renders(window)
        shapes.clear()
        slider = window.mixer_panel.sliders["hue"][3]
        slider.setSliderDown(True)
        slider.setValue(-10)
        slider.setValue(20)
        QTest.qWait(RENDER_DEBOUNCE_MS + 200)
        _finish_renders(window)
        assert shapes == [(8, 6, 3)]                        # the drag preview
        assert window.preview_worker.image is window.drag_proxy
        undo_len = len(window.state_manager.undo_stack)
        slider.setSliderDown(False)
        QTest.qWait(RENDER_DEBOUNCE_MS + 200)
        _finish_renders(window)
    assert len(window.state_manager.undo_stack) == undo_len
    assert window.preview_worker.image is window.proxy_image   # its mixer result is reused from the cache
    assert shapes == [(8, 6, 3)]


def test_mixer_undo_restores_sliders() -> None:
    """Undo after a mixer change restores the previous mixer values."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    with patch.object(ColorMixer, "apply", autospec=True, side_effect=lambda self, img, icc, hue, sat, lum: img):
        window.on_discrete_action()
        window.mixer_panel.sliders["lum"][2].setValue(55)
        window.on_discrete_action()
        window.on_undo_clicked()
        _finish_renders(window)
    assert window.mixer_panel.sliders["lum"][2].value() == 0


def test_save_runs_mixer_on_full_image(tmp_path: Path) -> None:
    """Save runs the mixer on the full-resolution master (not the proxy) and does not keep it in the preview cache."""
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    window.proxy_image = window.master_image[::3, ::3].copy()
    shapes = []
    with patch.object(ColorMixer, "apply", autospec=True, side_effect=lambda self, img, icc, hue, sat, lum: shapes.append((img.shape, hue)) or img):
        window.mixer_panel.sliders["hue"][0].setValue(-25)
        window.render_timer.stop()
        _finish_renders(window)
        with patch("nss.color_gui.main_window.QFileDialog.getSaveFileName", return_value=(str(tmp_path / "out.tif"), "")):
            window.save_file()
        window.save_worker.wait()
        QApplication.processEvents()
    assert shapes == [((15, 12, 3), (-25.0,) + (0.0,) * 7)]
    assert id(window.master_image) not in window.mixer_cache.entries


def test_mixer_panel_sits_between_harmony_and_wheel_tabs_with_wide_sliders() -> None:
    """Explore & Harmony, then Color Harmony, then the Color Mixer, then the Color Grading group; in the 300 px panel the mixer sliders span >= 90 % of the tab."""
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    layout = window.inspector_panel.layout()
    order = [layout.itemAt(i).widget() for i in range(layout.count()) if layout.itemAt(i).widget() is not None]
    i = order.index(window.mixer_panel)
    assert order[i - 2].title() == "EXPLORE & HARMONY"
    assert order[i - 1] is window.harmony_panel and window.harmony_panel.title() == "COLOR HARMONY"
    assert order[i + 1] is window.grading_box
    window.show()
    QApplication.processEvents()
    page_width = window.mixer_panel.tabs.currentWidget().width()
    assert page_width > 200
    assert all(slider.width() >= 0.9 * page_width for slider in window.mixer_panel.sliders["hue"])
    window.close()


def test_color_grading_group_holds_wheel_tabs_and_master_controls() -> None:
    """A COLOR GRADING group frames the wheel tabs then the master controls; no MASTER DOCK CONTROLS title; the reset button reads Reset Grading."""
    from PyQt6.QtWidgets import QLabel
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    box = window.grading_box
    assert box.title() == "COLOR GRADING"
    layout = box.layout()
    assert [layout.itemAt(i).widget() for i in range(layout.count())] == [window.tabs, window.master_zone]
    assert window.master_zone.reset_all_btn.text() == "Reset Grading"
    assert all("MASTER DOCK" not in label.text() for label in window.findChildren(QLabel))


def test_harmony_sampling_round_trip() -> None:
    """An eyedropper pick adds a sample at the proxy pixel with the displayed median colour and the pre-mixer hue, places its circle on the preview,
    keeps its hue when a mixer Hue slider moves, and undo / redo remove and restore it."""
    from nss.color_explore.harmony import median_rgb
    from nss.color_mixer.hue_remap import native_hue
    app = QApplication.instance() or QApplication([])
    window = _window_with_image()
    window.render_center_preview()
    _finish_renders(window)
    window.on_discrete_action()
    window.harmony_panel.eyedropper_btn.setChecked(True)
    assert window.single_preview.sampling
    window.single_preview.sample_picked.emit(0.5, 0.2)            # (15, 12) proxy: column 6, row 3
    sample, = window.harmony_panel.samples
    assert (sample.slot, sample.x, sample.y) == (1, 6, 3)
    assert np.allclose(sample.rgb, median_rgb(window.single_preview.image_array, 6, 3))
    assert abs(sample.hue - native_hue(median_rgb(window.proxy_image, 6, 3), ICC)[0]) < 1e-9
    assert window.single_preview.markers == [(1, 6.5 / 12, 3.5 / 15)]

    window.mixer_panel.sliders["hue"][0].setValue(80)
    window.on_mixer_released()
    assert window.harmony_panel.hue_levels[0] == 80.0 and window.harmony_panel.samples[0].hue == sample.hue

    window.on_undo_clicked()                                     # before the mixer change: the sample stays, the wheel follows the mixer back
    assert window.harmony_panel.hue_levels[0] == 0.0 and window.harmony_panel.samples == [sample]
    window.on_undo_clicked()                                     # before the sample
    assert window.harmony_panel.samples == [] and window.single_preview.markers == []
    window.on_redo_clicked()
    assert window.harmony_panel.samples == [sample] and window.single_preview.markers == [(1, 6.5 / 12, 3.5 / 15)]
    window.render_timer.stop()


def _lightroom_request(tmp_path: Path) -> str:
    """Writes a TIFF and a request file with distinct non-zero mixer and wheel values; returns the request's path."""
    raw = np.linspace(2000, 40000, 9 * 7 * 3).astype(np.uint16).reshape(9, 7, 3)
    TiffFile.from_array(raw / 65535.0, icc=ICC).saveas(str(tmp_path / "in.tif"))
    mixer = {"hue": [3, -8, -12, -18, 10, 6, 1, 2], "sat": [4, 12, 18, 22, 15, 30, 5, 6], "lum": [7, 8, 5, -12, 14, -22, 9, 11]}
    wheels = {"shadows": (200, 30, -4), "midtones": (110, 12, 6), "highlights": (40, 25, -5), "global": (300, 15, 8)}
    lines = lr_bridge.result_lines(mixer, wheels, 70, -20) + [f"tiff={tmp_path / 'in.tif'}", f"result={tmp_path / 'result.txt'}"]
    path = tmp_path / "request.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_lightroom_request_sets_sliders_and_apply_writes_them_with_rotation(tmp_path: Path) -> None:
    """The request's mixer, four wheels, blending and balance reach the widgets; Apply writes them back (hues + Rotation) and closes."""
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    _WINDOWS.append(window)
    with patch("nss.mru.MruManager.add_path"):
        window.open_lightroom_request(lr_bridge.read_request(_lightroom_request(tmp_path)))
    assert window.master_image.shape == (9, 7, 3) and window.tiff_obj.iccprofile == ICC
    assert [s.value() for s in window.mixer_panel.sliders["lum"]] == [7, 8, 5, -12, 14, -22, 9, 11]
    assert (window.sh_zone.hue_slider.value(), window.sh_zone.sat_slider.value(), window.sh_zone.light_slider.value()) == (200, 30, -4)
    assert (window.gl_zone.hue_slider.value(), window.gl_zone.sat_slider.value(), window.gl_zone.light_slider.value()) == (300, 15, 8)
    assert (window.master_zone.blending_slider.value(), window.master_zone.balance_slider.value()) == (70, -20)
    window.master_zone.rotation_slider.setValue(70)
    _finish_renders(window)
    window.apply_to_lightroom()
    assert window.lightroom_applied and not window.isVisible()
    result = dict(line.split("=") for line in (tmp_path / "result.txt").read_text(encoding="utf-8").split())
    assert result["HueAdjustmentOrange"] == "-8" and result["SaturationAdjustmentBlue"] == "30" and result["LuminanceAdjustmentMagenta"] == "11"
    assert result["SplitToningShadowHue"] == "270" and result["ColorGradeMidtoneHue"] == "180" and result["ColorGradeGlobalHue"] == "10"
    assert result["SplitToningHighlightSaturation"] == "25" and result["ColorGradeGlobalLum"] == "8"
    assert result["ColorGradeBlending"] == "70" and result["SplitToningBalance"] == "-20"


def test_lightroom_close_without_apply_writes_nothing(tmp_path: Path) -> None:
    """Closing the window any other way leaves no result file."""
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    _WINDOWS.append(window)
    with patch("nss.mru.MruManager.add_path"):
        window.open_lightroom_request(lr_bridge.read_request(_lightroom_request(tmp_path)))
    window.close()
    assert not window.lightroom_applied and not (tmp_path / "result.txt").exists()
