"""Tests of nss.color_gui.wheel_lut: the hue = angle convention, the measured painting tables, wheel_image and the wheel geometry."""
import colorsys
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from nss.color_gui.wheel_lut import (SAT_FULL_RADIUS, angle_from_hue, hue_from_angle, hue_groove_stylesheet, hue_sat_at, offset_at,
                                     pure_hue_rgb, rim_color, wheel_color, wheel_image)
from nss.color_gui.wheel_lut_tables import BLEND_G, BLEND_R, CENTRE_GREY, RIM_RGB


def _painted_hue(rgb: np.ndarray) -> float:
    """HSV hue in degrees of a display RGB 0..1 colour."""
    return colorsys.rgb_to_hsv(*rgb)[0] * 360.0


def test_hue_is_the_screen_angle() -> None:
    """hue_from_angle and angle_from_hue are the identity modulo 360, also on arrays."""
    a = np.array([[0.0, 15.5, 359.9], [360.0, -30.0, 725.0]])
    assert_allclose(hue_from_angle(a), [[0.0, 15.5, 359.9], [0.0, 330.0, 5.0]])
    assert_allclose(angle_from_hue(hue_from_angle(a)), np.mod(a, 360.0))


def test_tables_shapes_and_ranges() -> None:
    """Rim table (360, 3), blend profile on 0..1 starting at 0, centre grey a light grey; all finite and within 0..1."""
    assert RIM_RGB.shape == (360, 3) and np.all(np.isfinite(RIM_RGB))
    assert RIM_RGB.min() >= 0.0 and RIM_RGB.max() <= 1.0
    assert BLEND_R[0] == 0.0 and BLEND_R[-1] == 1.0 and np.all(np.diff(BLEND_R) > 0)
    assert BLEND_G[0] == 0.0 and np.all(np.isfinite(BLEND_G))
    assert 0.9 < CENTRE_GREY < 1.0


def test_blend_profile_is_monotone() -> None:
    """The rim colour's weight grows from the centre outwards and reaches about 1 at the rim."""
    assert np.all(np.diff(BLEND_G) >= 0.0)
    assert BLEND_G[-1] == pytest.approx(1.0, abs=0.05)


def test_rim_is_continuous_and_wraps() -> None:
    """Neighbouring degrees differ little, including 359 -> 0."""
    steps = np.abs(np.diff(np.vstack([RIM_RGB, RIM_RGB[:1]]), axis=0)).max()
    assert steps < 0.03


def test_painted_hue_turns_once_counter_clockwise() -> None:
    """The painted HSV hue increases monotonically with the screen angle and makes exactly one turn."""
    u = np.degrees(np.unwrap(np.radians([_painted_hue(c) for c in RIM_RGB])))
    assert np.all(np.diff(u) > 0.0)
    assert u[-1] - u[0] + (u[1] - u[0]) == pytest.approx(360.0, abs=1.0)


@pytest.mark.parametrize("hue, angle", [(60, 59.8), (120, 154.0), (180, 180.2), (240, 239.1), (300, 309.0)])
def test_painting_is_warped_like_lightroom(hue: float, angle: float) -> None:
    """Primaries are painted where Lightroom paints them (green near 154 degrees), independent of the hue numbers."""
    a = np.arange(360)
    painted = np.array([_painted_hue(c) for c in RIM_RGB])
    nearest = a[np.argmin(np.abs((painted - hue + 180.0) % 360.0 - 180.0))]
    assert abs((nearest - angle + 180.0) % 360.0 - 180.0) <= 2.0


def test_rim_color_interpolates_and_wraps() -> None:
    """Between degrees the colour is interpolated linearly, and 360 equals 0."""
    assert_allclose(rim_color(10.5), (RIM_RGB[10] + RIM_RGB[11]) / 2.0)
    assert_allclose(rim_color(359.5), (RIM_RGB[359] + RIM_RGB[0]) / 2.0)
    assert_allclose(rim_color(360.0), RIM_RGB[0])
    assert rim_color(np.zeros((4, 2))).shape == (4, 2, 3)


def test_wheel_color_blends_to_the_centre_grey() -> None:
    """Radius 0 is the centre grey, radius 1 is the rim colour (up to the fitted profile), half way is in between."""
    assert_allclose(wheel_color(77.0, 0.0), CENTRE_GREY)
    assert_allclose(wheel_color(77.0, 1.0), np.clip(CENTRE_GREY + BLEND_G[-1] * (rim_color(77.0) - CENTRE_GREY), 0, 1))
    mid = wheel_color(77.0, 0.5)
    assert np.all(np.minimum(rim_color(77.0), CENTRE_GREY) - 1e-9 <= mid) and np.all(mid <= np.maximum(rim_color(77.0), CENTRE_GREY) + 1e-9)
    assert wheel_color(np.zeros((3, 5)), np.full((3, 5), 0.3)).shape == (3, 5, 3)


def test_wheel_image_paints_the_tables() -> None:
    """wheel_image is a square ARGB image: transparent corners, the centre grey in the middle, wheel_color near the rim."""
    app = QApplication.instance() or QApplication([])
    size = 121
    im = wheel_image(size)
    assert (im.width(), im.height()) == (size, size)
    assert im.format() == QImage.Format.Format_ARGB32_Premultiplied
    assert (im.pixel(0, 0) >> 24) == 0
    c = (size - 1) // 2
    assert_allclose(np.array(im.pixelColor(c, c).getRgbF()[:3]), CENTRE_GREY, atol=1.5 / 255)
    for a in (0, 90, 154, 240):
        r = 0.9 * size / 2.0
        x, y = c + r * math.cos(math.radians(a)), c - r * math.sin(math.radians(a))
        got = np.array(im.pixelColor(round(x), round(y)).getRgbF()[:3])
        assert_allclose(got, wheel_color(math.degrees(math.atan2(c - round(y), round(x) - c)), math.hypot(round(x) - c, c - round(y)) / (size / 2.0)), atol=1.5 / 255)


def test_saturation_radius() -> None:
    """Sat 100 is drawn at SAT_FULL_RADIUS (0.907) of the painted radius, linear to the centre; clicks beyond it clamp to Sat 100."""
    dx, dy = offset_at(30.0, 1.0, 200.0)
    assert math.hypot(dx, dy) == pytest.approx(SAT_FULL_RADIUS * 200.0)
    assert SAT_FULL_RADIUS == 0.907
    assert hue_sat_at(SAT_FULL_RADIUS * 100.0, 0.0, 200.0)[1] == pytest.approx(0.5)
    assert hue_sat_at(0.0, -199.0, 200.0) == (90.0, 1.0)


@pytest.mark.parametrize("hue", np.arange(0.0, 360.0, 15.0))
def test_handle_position_round_trip(hue: float) -> None:
    """The handle offset for (hue, sat) maps back to the same hue and sat; the hue is the screen angle (y up)."""
    dx, dy = offset_at(hue, 0.6, 150.0)
    assert math.degrees(math.atan2(-dy, dx)) % 360.0 == pytest.approx(hue, abs=1e-9)
    h, s = hue_sat_at(dx, dy, 150.0)
    assert (h - hue + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=1e-9)
    assert s == pytest.approx(0.6)


def test_pure_hue_rgb() -> None:
    """HSV (h, 1, 1) at the six anchors and in between."""
    assert_allclose(pure_hue_rgb([0, 60, 120, 180, 240, 300, 360]), [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0]])
    assert_allclose(pure_hue_rgb(30.0), [1.0, 0.5, 0.0])


def test_hue_groove_stylesheet() -> None:
    """The Hue slider groove runs through the seven 60-degree pure-hue stops from red back to red."""
    css = hue_groove_stylesheet()
    assert css.count("stop:") == 7
    assert "stop:0.0000 rgb(255, 0, 0)" in css and "stop:0.3333 rgb(0, 255, 0)" in css and "stop:1.0000 rgb(255, 0, 0)" in css
