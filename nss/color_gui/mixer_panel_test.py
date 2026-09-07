"""Tests of nss.color_gui.mixer_panel: the Color Mixer tabs and sliders, their snapshots and resets, and the mixed-image cache."""
from unittest.mock import patch

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel

from nss.color_mixer import ColorMixer
from nss.color_mixer.bands import CENTRES
from nss.color_gui.mixer_panel import (BAND_NAMES, HUE_TRACK_STOPS, KINDS, MixerCache, MixerPanel, ResetSlider, band_color, band_hues, mixer_levels,
                                       screen_hue, track_colors, track_stylesheet)
from nss.color_gui.wheel_lut_tables import RIM_RGB
from nss.icc import gamma_icc_bytes

ICC = gamma_icc_bytes("Adobe RGB (1998)", [(0.60974, 0.31111, 0.01947), (0.20528, 0.62567, 0.06087), (0.14919, 0.06322, 0.74457)], 563 / 256)


def _panel() -> tuple[MixerPanel, list, list]:
    """A MixerPanel and the lists its changed / released callbacks append to."""
    changed, released = [], []
    return MixerPanel(lambda: changed.append(1), lambda: released.append(1)), changed, released


def test_tabs_ranges_and_defaults() -> None:
    """Hue / Saturation / Luminance tabs, each with the 8 band sliders (-100..100, starting at 0) under tinted band names and +0 values."""
    app = QApplication.instance() or QApplication([])
    panel, _, _ = _panel()
    assert [panel.tabs.tabText(i) for i in range(panel.tabs.count())] == ["Hue", "Saturation", "Luminance"]
    for i, kind in enumerate(KINDS):
        page = panel.tabs.widget(i)
        assert page.findChildren(ResetSlider) == panel.sliders[kind]
        texts = [label.text() for label in page.findChildren(QLabel)]
        assert texts == [t for band in BAND_NAMES for t in (band, "+0")]
        r, g, b = np.rint(band_color(CENTRES[0])[0] * 255).astype(int)
        assert f"rgb({r}, {g}, {b})" in page.findChildren(QLabel)[0].styleSheet()     # Red's name is tinted with its band colour
        for slider in panel.sliders[kind]:
            assert (slider.minimum(), slider.maximum(), slider.value(), slider.singleStep()) == (-100, 100, 0, 1)


def test_sliders_span_nearly_the_whole_tab_width() -> None:
    """At the 270 px the inspector panel leaves, every slider is at least 90 % as wide as its tab page."""
    app = QApplication.instance() or QApplication([])
    panel, _, _ = _panel()
    panel.resize(270, 900)
    panel.show()
    for i, kind in enumerate(KINDS):
        panel.tabs.setCurrentIndex(i)
        QApplication.processEvents()
        page_width = panel.tabs.widget(i).width()
        assert page_width > 200
        assert all(slider.width() >= 0.9 * page_width for slider in panel.sliders[kind])
    panel.close()


def test_snapshot_round_trip_and_value_labels() -> None:
    """get_state returns the slider values; restore_state sets them (and the value labels) without emitting changes."""
    app = QApplication.instance() or QApplication([])
    panel, changed, _ = _panel()
    panel.sliders["hue"][0].setValue(25)
    panel.sliders["sat"][3].setValue(-40)
    panel.sliders["lum"][7].setValue(60)
    assert changed == [1, 1, 1]
    state = panel.get_state()
    assert state["color_mixer"]["hue"] == [25, 0, 0, 0, 0, 0, 0, 0]
    assert state["color_mixer"]["sat"] == [0, 0, 0, -40, 0, 0, 0, 0]
    assert state["color_mixer"]["lum"] == [0, 0, 0, 0, 0, 0, 0, 60]
    assert panel.value_labels["sat"][3].text() == "-40"

    other, changed2, _ = _panel()
    other.restore_state(state)
    assert other.get_state() == state
    assert changed2 == []
    assert other.value_labels["lum"][7].text() == "+60"
    assert other.value_labels["hue"][0].text() == "+25"
    other.restore_state({"master_zone": {}})                # snapshots without the mixer leave it unchanged
    assert other.get_state() == state


def test_double_click_and_reset_all() -> None:
    """Double-clicking a slider resets it to 0; Reset Mixer zeroes all 24; each records one undo step."""
    app = QApplication.instance() or QApplication([])
    panel, _, released = _panel()
    panel.sliders["hue"][2].setValue(30)
    panel.sliders["lum"][5].setValue(-70)
    QTest.mouseDClick(panel.sliders["hue"][2], Qt.MouseButton.LeftButton)
    assert panel.sliders["hue"][2].value() == 0
    assert panel.sliders["lum"][5].value() == -70
    assert released[-1:] == [1]
    n = len(released)
    panel.reset_btn.click()
    assert all(slider.value() == 0 for kind in KINDS for slider in panel.sliders[kind])
    assert len(released) == n + 1


def test_mixer_levels_defaults_and_values() -> None:
    """mixer_levels reads mixer_hue / sat / lum as float tuples; missing keys are all zero."""
    assert mixer_levels({}) == ((0.0,) * 8,) * 3
    state = {"mixer_hue": [10, 0, 0, 0, 0, 0, 0, -5], "mixer_sat": [0] * 8, "mixer_lum": [0, 0, 33, 0, 0, 0, 0, 0]}
    assert mixer_levels(state) == ((10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0), (0.0,) * 8, (0.0, 0.0, 33.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_cache_reuses_result_until_levels_or_image_change() -> None:
    """The cache returns ColorMixer.apply's result, runs it once per (image, levels) and again when either changes."""
    img = np.random.default_rng(3).random((13, 12, 3)).astype(np.float32)
    levels = ((20.0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, -30.0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 40.0, 0, 0))
    cache = MixerCache()
    expected = ColorMixer().apply(img, ICC, *levels)
    real_apply = ColorMixer.apply
    with patch.object(ColorMixer, "apply", autospec=True, side_effect=real_apply) as apply:
        first = cache.mixed(img, ICC, levels)
        assert cache.mixed(img, ICC, levels) is first
        assert apply.call_count == 1
        assert np.array_equal(first, expected)
        assert np.abs(first - img).max() > 1e-3                 # the mixer did change the image
        cache.mixed(img, ICC, ((0.0,) * 8, (0.0,) * 8, (10.0,) + (0.0,) * 7))
        cache.mixed(img.copy(), ICC, levels)
        assert apply.call_count == 3
        cache.clear()
        cache.mixed(img, ICC, levels)
        assert apply.call_count == 4


def test_band_colors_follow_the_bands_around_the_screen_hue_circle() -> None:
    """Each band centre is painted with a wheel rim colour whose screen hue is where the band's name says (Green green, Blue blue, ...)."""
    hues = screen_hue(band_color(CENTRES))
    expected = {"Red": (340, 360), "Orange": (15, 40), "Yellow": (40, 65), "Green": (85, 135), "Aqua": (150, 190), "Blue": (195, 245),
                "Purple": (265, 300), "Magenta": (310, 345)}
    for name, hue in zip(BAND_NAMES, hues):
        low, high = expected[name]
        assert low <= hue <= high, (name, hue)
    assert all(any(np.allclose(c, rim) for rim in RIM_RGB) for c in band_color(CENTRES))       # painted with the wheel's own colours


def test_hue_track_runs_through_the_hues_the_band_moves_to() -> None:
    """A Hue track has HUE_TRACK_STOPS stops: band_color of band_hues at -100 .. +100, with the band colour in the middle; Red runs magenta -> orange."""
    for k in range(8):
        stops = track_colors("hue", k)
        assert stops.shape == (HUE_TRACK_STOPS, 3)
        assert np.allclose(stops, band_color(band_hues(k, np.linspace(-100.0, 100.0, HUE_TRACK_STOPS))))
        assert np.allclose(stops[HUE_TRACK_STOPS // 2], band_color(CENTRES[k])[0])
    h_minus, h_plus = band_hues(0, [-100.0, 100.0])
    assert h_minus == pytest.approx(-30.0, abs=1.0) and h_plus == pytest.approx(30.0, abs=1.0)    # Red moves toward magenta / orange
    left, right = screen_hue(track_colors("hue", 0)[[0, -1]])
    assert 280 <= left <= 340 and 15 <= right <= 45


def test_sat_and_lum_tracks() -> None:
    """Saturation runs from a grey of the band colour's brightness to the band colour; Luminance from dark to light band colour."""
    for k in range(8):
        color = band_color(CENTRES[k])[0]
        sat = track_colors("sat", k)
        assert sat.shape == (2, 3) and np.ptp(sat[0]) == 0.0 and np.allclose(sat[1], color)
        assert sat[0][0] == pytest.approx(np.dot(color, (0.2126, 0.7152, 0.0722)))
        lum = track_colors("lum", k)
        assert lum.shape == (3, 3) and np.allclose(lum[1], color)
        assert lum[0].sum() < color.sum() < lum[2].sum()
        assert np.all(lum[2] <= 1.0)


def test_sliders_are_painted_with_their_tracks() -> None:
    """Every slider's style sheet is the gradient of its own track; band names carry the band colour."""
    app = QApplication.instance() or QApplication([])
    panel, _, _ = _panel()
    for kind in KINDS:
        for k, slider in enumerate(panel.sliders[kind]):
            assert slider.styleSheet() == track_stylesheet(track_colors(kind, k))
    sheet = track_stylesheet(track_colors("hue", 5))
    assert sheet.count("stop:") == HUE_TRACK_STOPS and "stop:0.0000" in sheet and "stop:1.0000" in sheet
    r, g, b = np.rint(band_color(CENTRES[3])[0] * 255).astype(int)
    assert f"rgb({r}, {g}, {b})" in panel.tabs.widget(0).findChildren(QLabel)[6].styleSheet()      # Green's name
