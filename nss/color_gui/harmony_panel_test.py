"""Tests of nss.color_gui.harmony_panel: slot numbering, the 16-sample limit, the mode fit, live warping, reset and the undo snapshots."""
import numpy as np
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QLabel

from nss.color_explore.harmony import HARMONY_MODES
from nss.color_mixer.hue_remap import hue_remap_forward, hue_remap_inverse, hue_remap_lut
from nss.color_gui.harmony_panel import (MAX_SAMPLES, MIXER_TO_WHEEL, HarmonyPanel, Sample, SampleInfoDialog, color_fields, lowest_free_slot, wheel_angle,
                                         wheel_image, wheel_remap_lut)
from nss.color_gui.mixer_panel import band_color
from nss.color_gui.state_manager import StateManager
from nss.color_gui.wheel_lut import wheel_color, wheel_image as grading_wheel_image
from nss.color_gui.wheel_lut_tables import RIM_RGB


def mixer_hue_at(angle: float) -> float:
    """The mixer hue that sits at a wheel angle (g^-1)."""
    return float(hue_remap_inverse(MIXER_TO_WHEEL, angle)[0])


def _panel(current=None) -> tuple[HarmonyPanel, list]:
    """A HarmonyPanel and the list its undo-step callback appends to."""
    released = []
    return HarmonyPanel(lambda: released.append(1), current or (lambda x, y: None)), released


def _add(panel: HarmonyPanel, hue: float, x: int = 3, y: int = 7) -> None:
    """Adds a sample at hue with a reddish colour."""
    panel.add_sample(x, y, np.array([0.6, 0.3, 0.2]), hue)


def test_lowest_free_slot_fills_gaps() -> None:
    """The lowest unused number 1..16; None when all are used."""
    s = lambda slot: Sample(slot, 0, 0, (0.0, 0.0, 0.0), 0.0)
    assert lowest_free_slot([]) == 1
    assert lowest_free_slot([s(1), s(2), s(4)]) == 3
    assert lowest_free_slot([s(k) for k in range(1, 17)]) is None


def test_slots_fill_gaps_without_renumbering() -> None:
    """Removing sample 2 of 1..3 keeps 1 and 3; the next sample takes 2; each add and remove records an undo step."""
    app = QApplication.instance() or QApplication([])
    panel, released = _panel()
    for hue in (10.0, 20.0, 30.0):
        _add(panel, hue)
    panel.remove_sample(2)
    assert [s.slot for s in panel.samples] == [1, 3]
    _add(panel, 40.0, x=11, y=5)
    assert sorted((s.slot, s.hue, s.x, s.y) for s in panel.samples) == [(1, 10.0, 3, 7), (2, 40.0, 11, 5), (3, 30.0, 3, 7)]
    assert len(released) == 5


def test_eyedropper_disables_at_sixteen_and_returns_after_a_removal() -> None:
    """Filling the 16th slot stops sampling and disables the eyedropper; removing one enables it again (still off)."""
    app = QApplication.instance() or QApplication([])
    panel, _ = _panel()
    sampling = []
    panel.sampling_changed.connect(sampling.append)
    panel.eyedropper_btn.setChecked(True)
    for k in range(MAX_SAMPLES):
        _add(panel, 20.0 * k)
    assert not panel.eyedropper_btn.isEnabled() and not panel.eyedropper_btn.isChecked()
    assert sampling == [True, False]
    _add(panel, 5.0)
    assert len(panel.samples) == MAX_SAMPLES
    panel.remove_sample(9)
    assert panel.eyedropper_btn.isEnabled() and not panel.eyedropper_btn.isChecked()


def test_mode_change_fits_rotation_once() -> None:
    """Complementary over samples at wheel angles 37 and 217 sets the rotation to 37 in one undo step; dragging the slider does not refit."""
    app = QApplication.instance() or QApplication([])
    panel, released = _panel()
    _add(panel, mixer_hue_at(37.0))
    _add(panel, mixer_hue_at(217.0))
    released.clear()
    panel.mode_dropdown.setCurrentText("Complementary")
    assert panel.rotation == 37.0 and panel.rotation_slider.value() == 74 and len(released) == 1
    assert panel.readout_lbl.text() == "2 / 2 samples in harmony"
    panel.rotation_slider.setValue(200)
    assert panel.rotation == 100.0 and len(released) == 1
    assert panel.readout_lbl.text() == "0 / 2 samples in harmony"


def test_hue_sliders_warp_but_markers_stay() -> None:
    """A Green Hue shift changes the mapping (and the fit readout) but not the samples' input hues."""
    app = QApplication.instance() or QApplication([])
    panel, released = _panel()
    _add(panel, 95.0)
    panel.mode_dropdown.setCurrentText("Monochromatic")
    angle = float(wheel_angle(95.0))
    assert abs(panel.rotation - angle) <= 0.25 and panel.readout_lbl.text() == "1 / 1 samples in harmony"
    released.clear()
    rotation = panel.rotation
    panel.set_hue_levels((0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 0.0))
    assert panel.samples[0].hue == 95.0 and panel.rotation == rotation
    assert float(hue_remap_forward(panel.lut, angle)) > angle + 15.0
    assert panel.readout_lbl.text() == "0 / 1 samples in harmony"
    assert panel.wheel.lut is panel.lut and released == []


def test_reset_clears_harmony_only() -> None:
    """Reset Harmony: no samples, mode None, rotation 0, one undo step; the mixer mapping stays."""
    app = QApplication.instance() or QApplication([])
    panel, released = _panel()
    panel.set_hue_levels((10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0))
    _add(panel, 50.0)
    panel.mode_dropdown.setCurrentText("Triadic")
    panel.rotation_slider.setValue(33)
    released.clear()
    panel.reset_btn.click()
    assert (panel.samples, panel.mode, panel.rotation, panel.rotation_slider.value()) == ([], "None", 0.0, 0)
    assert panel.hue_levels == (10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0)
    assert len(released) == 1


def test_undo_redo_round_trip() -> None:
    """Snapshots of add, mode fit, rotation and remove restore exactly through the StateManager."""
    app = QApplication.instance() or QApplication([])
    manager = StateManager()
    panel, _ = _panel()
    panel.on_released_slot = lambda: manager.push_state(manager.capture_snapshot())
    manager.register_widget(panel)
    manager.push_state(manager.capture_snapshot())
    _add(panel, 37.0)
    _add(panel, 217.0)
    panel.mode_dropdown.setCurrentText("Complementary")
    panel.set_rotation(12.5)
    panel.on_released_slot()
    panel.remove_sample(1)
    final = manager.capture_snapshot()
    history = list(manager.undo_stack)
    for expected in reversed(history[:-1]):
        manager.restore_snapshot(manager.undo())
        assert manager.capture_snapshot() == expected
    assert panel.samples == [] and panel.mode == "None"
    while manager.can_redo():
        manager.restore_snapshot(manager.redo())
    assert manager.capture_snapshot() == final
    assert [s.slot for s in panel.samples] == [2] and panel.mode == "Complementary" and panel.rotation == 12.5


def test_info_popup_shows_original_and_current() -> None:
    """The popup lists both colours; current comes from the callback at the sample's pixel."""
    app = QApplication.instance() or QApplication([])
    asked = []
    panel, _ = _panel(lambda x, y: (asked.append((x, y)), np.array([0.2, 0.4, 0.8]))[1])
    panel.add_sample(11, 4, np.array([1.0, 0.0, 0.0]), 0.0)
    panel.show_sample_info(1)
    texts = [label.text() for label in panel.info_dialog.findChildren(QLabel)]
    assert asked == [(11, 4)]
    assert color_fields((1.0, 0.0, 0.0)) in texts and color_fields((0.2, 0.4, 0.8)) in texts
    assert color_fields((1.0, 0.0, 0.0)) == "Hue 0°  Sat 100%  Lum 50%  #ff0000"
    assert color_fields((0.5, 0.5, 0.5)).startswith("Hue -")
    panel.info_dialog.close()


def pixels(img: QImage) -> np.ndarray:
    """(h, w, 4) uint8 copy of an ARGB32 image (B, G, R, A bytes)."""
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    return np.frombuffer(ptr, np.uint8).reshape(img.height(), img.width(), 4).copy()


def test_mixer_to_wheel_is_monotonic_and_matches_band_color() -> None:
    """g rises strictly over the mixer hues and lands within 1.5 degrees of the rim colour band_color picks (its nearest whole degree)."""
    closed = np.append(MIXER_TO_WHEEL, MIXER_TO_WHEEL[0] + 360.0)
    assert np.diff(closed).min() > 0.0
    h = np.arange(0.0, 360.0, 7.0)
    picked = np.array([np.flatnonzero((RIM_RGB == c).all(axis=1))[0] for c in band_color(h)])
    assert np.abs((wheel_angle(h) - picked + 180.0) % 360.0 - 180.0).max() < 1.5
    assert abs(float(wheel_angle(90.0)) - 136.0) < 1.5 and abs(float(wheel_angle(240.0)) - 214.0) < 1.5     # far from the identity


def test_neutral_wheel_is_the_color_grading_wheel() -> None:
    """With all Hue sliders at 0, F is the identity and the harmony wheel is painted exactly like the Color Grading wheel (the image is a disc, so square)."""
    app = QApplication.instance() or QApplication([])
    lut = wheel_remap_lut(hue_remap_lut(np.zeros(8)))
    assert np.abs(lut - np.arange(360.0)).max() < 1e-2
    diff = pixels(wheel_image(lut, 101)).astype(int) - pixels(grading_wheel_image(101)).astype(int)
    assert np.abs(diff).max() <= 1


def test_warped_wheel_paints_the_output_colour() -> None:
    """Red +100: the rim at 3 o'clock shows the grading wheel's colour at F(0), not at 0."""
    app = QApplication.instance() or QApplication([])
    lut = wheel_remap_lut(hue_remap_lut(np.array([100.0, 0, 0, 0, 0, 0, 0, 0])))
    target = float(hue_remap_forward(lut, 0.0))
    assert abs(((target + 180.0) % 360.0) - 180.0) > 10.0
    px = pixels(wheel_image(lut, 101))[50, 97, 2::-1] / 255.0                  # R, G, B at angle 0, radius 47 / 50.5
    assert np.abs(px - wheel_color(target, 47.0 / 50.5)).max() < 0.02
    assert np.abs(px - wheel_color(0.0, 47.0 / 50.5)).max() > 0.05
