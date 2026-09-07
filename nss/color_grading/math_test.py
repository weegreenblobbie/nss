"""Tests of nss.color_grading.math: the Lab conversion and every Color Grading formula and constant."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nss.color_grading.math import (rgb_to_lab_pure, C0, WLUMA, PROPHOTO, SAT_Q, LUM_Q, GAMMA, P0, P1,
                                    logit, inv_logit, odds_gain, hue_vector, global_c, balance_shift, cap_shift, sat_mult)
from nss.color_grading.global_wheel import SATMULT_U, SATMULT_POS, SATMULT_NEG

TABLES = (SATMULT_U, SATMULT_POS, SATMULT_NEG)

HEXAGON = [(0, (1, -1, -1)), (60, (1, 1, -1)), (120, (-1, 1, -1)), (180, (-1, 1, 1)), (240, (-1, -1, 1)), (300, (1, -1, 1))]


# --- rgb_to_lab_pure ---

def test_rgb_to_lab_pure_black() -> None:
    """Black maps to Lab (0, 0, 0) as float64, shape preserved."""
    lab = rgb_to_lab_pure(np.zeros((7, 5, 3)))
    assert lab.shape == (7, 5, 3)
    assert lab.dtype == np.float64
    assert_allclose(lab, 0.0, atol=1e-5)


def test_rgb_to_lab_pure_white() -> None:
    """White maps to Lab (100, 0, 0)."""
    lab = rgb_to_lab_pure(np.ones((7, 5, 3), dtype=np.float32))
    assert lab.dtype == np.float64
    assert_allclose(lab[..., 0], 100.0, atol=0.01)
    assert_allclose(lab[..., 1:], 0.0, atol=0.01)


def test_rgb_to_lab_pure_red() -> None:
    """sRGB red in D65 Lab is (53.24, 80.09, 67.20)."""
    red = np.zeros((9, 6, 3))
    red[..., 0] = 1.0
    lab = rgb_to_lab_pure(red)
    assert lab.shape == (9, 6, 3)
    assert_allclose(lab[..., 0], 53.24, atol=0.05)
    assert_allclose(lab[..., 1], 80.09, atol=0.05)
    assert_allclose(lab[..., 2], 67.20, atol=0.05)


# --- constants ---

def test_c0_is_global_sat_100() -> None:
    """C0 = 2 atanh(0.28), the Global Sat 100 shift."""
    assert C0 == pytest.approx(0.575364, abs=1e-6)
    assert C0 == pytest.approx(global_c(0, 100, 0)[0])


def test_wluma_is_prophoto_luminance_row() -> None:
    """WLUMA sums to 1 and matches the Y row of the ProPhoto matrix."""
    assert WLUMA.sum() == pytest.approx(1.0)
    assert_allclose(WLUMA, PROPHOTO[1], atol=1e-3)


def test_prophoto_white_is_d50() -> None:
    """ProPhoto RGB white maps to the D50 white point."""
    assert_allclose(PROPHOTO.sum(1), [0.9642, 1.0, 0.8249], atol=1e-3)


def test_slider_constants() -> None:
    """The slider scale and Balance constants keep their fitted values."""
    assert (SAT_Q, LUM_Q) == (0.0028, 0.0024)
    assert (GAMMA, P0, P1) == (2.4, 0.5261, -0.00379)


# --- logit / inv_logit ---

def test_logit_values() -> None:
    """logit at known points."""
    assert_allclose(logit(np.array([0.5, 0.75, 0.1])), [0.0, np.log(3.0), -np.log(9.0)])


def test_logit_round_trip() -> None:
    """inv_logit and logit invert each other."""
    l = np.linspace(1e-6, 1 - 1e-6, 4 * 3 * 2).reshape(4, 3, 2)
    assert_allclose(inv_logit(logit(l)), l, rtol=1e-9)
    u = np.linspace(-16.0, 4.0, 15).reshape(5, 3)
    assert_allclose(logit(inv_logit(u)), u, atol=1e-9)


def test_inv_logit_is_odds_gain_of_half() -> None:
    """inv_logit(u) equals the odds gain of 0.5 by u."""
    u = np.linspace(-8.0, 8.0, 21)
    assert_allclose(inv_logit(u), odds_gain(0.5, u))


# --- odds_gain ---

def test_odds_gain_identity_at_zero() -> None:
    """c = 0 leaves every level unchanged."""
    l = np.linspace(0.0, 1.0, 13 * 3).reshape(13, 3)
    assert_allclose(odds_gain(l, 0.0), l)


def test_odds_gain_fixes_black_and_white() -> None:
    """0 and 1 stay fixed for any c."""
    for c in (-3.0, -0.4, 0.7, 5.0):
        assert odds_gain(0.0, c) == 0.0
        assert odds_gain(1.0, c) == 1.0


def test_odds_gain_adds_c_in_logit() -> None:
    """logit(odds_gain(l, c)) = logit(l) + c."""
    l = np.array([0.02, 0.3, 0.8])
    assert_allclose(logit(odds_gain(l, 0.37)), logit(l) + 0.37)


def test_odds_gain_monotonic_in_c() -> None:
    """A larger c gives a larger level."""
    out = odds_gain(0.3, np.linspace(-4.0, 4.0, 41))
    assert np.all(np.diff(out) > 0)


def test_odds_gain_broadcasts_per_channel_c() -> None:
    """A per-channel c (3,) broadcasts over an image (..., 3)."""
    l = np.full((7, 2, 3), 0.4)
    out = odds_gain(l, np.array([-0.5, 0.0, 0.5]))
    assert out.shape == (7, 2, 3)
    assert out[0, 0, 0] < 0.4
    assert out[0, 0, 1] == 0.4
    assert out[0, 0, 2] > 0.4


# --- hue_vector ---

@pytest.mark.parametrize("hue, vec", HEXAGON)
def test_hue_vector_corners(hue: float, vec: tuple[int, int, int]) -> None:
    """The six primary and secondary hues give the hexagon corners."""
    assert_allclose(hue_vector(hue), vec)


def test_hue_vector_range_and_shape() -> None:
    """Components stay in [-1, 1] and an array of hues gives (..., 3)."""
    v = hue_vector(np.arange(0.0, 360.0, 0.5).reshape(144, 5))
    assert v.shape == (144, 5, 3)
    assert v.min() == -1.0 and v.max() == 1.0


def test_hue_vector_wraps() -> None:
    """Hues wrap modulo 360."""
    assert_allclose(hue_vector(360), hue_vector(0))
    assert_allclose(hue_vector(-60), hue_vector(300))


def test_hue_vector_between_corners() -> None:
    """Hues between corners interpolate linearly (orange, azure)."""
    assert_allclose(hue_vector(30), [1, 0, -1])
    assert_allclose(hue_vector(210), [-1, 0, 1])


# --- global_c ---

def test_global_c_pinned_values() -> None:
    """2 atanh(0.0028 * 100) and 2 atanh(0.0024 * 50)."""
    assert global_c(240, 100, 0)[2] == pytest.approx(0.575364, abs=1e-6)
    assert_allclose(global_c(0, 0, 50), 0.241162, atol=1e-6)


def test_global_c_sat_follows_hue_vector_sign() -> None:
    """The Sat part has the hue vector's signs and magnitude 2 atanh(0.0028 Sat)."""
    c = global_c(60, 40, 0)
    assert np.all(np.sign(c) == hue_vector(60))
    assert_allclose(np.abs(c), 2.0 * np.arctanh(SAT_Q * 40))


def test_global_c_sat_and_lum_add() -> None:
    """The Sat and Lum parts add."""
    assert_allclose(global_c(200, 35, -20), global_c(200, 35, 0) + global_c(200, 0, -20))


def test_global_c_zero() -> None:
    """Sat 0 and Lum 0 give no shift."""
    assert_allclose(global_c(123, 0, 0), 0.0)


# --- balance_shift ---

def test_balance_shift_zero_at_zero() -> None:
    """Balance 0 does not shift the masks."""
    assert balance_shift(0) == 0.0


def test_balance_shift_monotonic_and_sign() -> None:
    """The shift grows with Balance and has Balance's sign."""
    d = [balance_shift(b) for b in range(-100, 101, 25)]
    assert np.all(np.diff(d) > 0)
    assert balance_shift(-50) < 0 < balance_shift(50)


# --- cap_shift ---

def test_cap_shift_inside_window_is_mask_times_f() -> None:
    """Inside the window and without a cap the result is M * f."""
    M = np.array([[0.5, 1.0, 2.0], [1.5, 0.25, 1.0]])
    f = np.array([[0.2, -0.1, 0.3], [-0.4, 0.0, 0.1]])
    assert_allclose(cap_shift(M, f, np.array([0.5, 0.5]), np.array([-0.5, -0.5]), 0.0), M * f)


def test_cap_shift_clamps_max_to_window() -> None:
    """A maximum above fh moves all three channels down by the excess."""
    M = np.ones((1, 3))
    f = np.array([[0.9, 0.1, -0.2]])
    y = cap_shift(M, f, np.array([0.6]), np.array([-1.0]), 0.0)
    assert_allclose(y, f - 0.3)


def test_cap_shift_clamps_min_to_window() -> None:
    """A minimum below fl moves all three channels up by the shortfall."""
    M = np.full((1, 3), 2.0)
    f = np.array([[0.1, -0.7, 0.3]])
    y = cap_shift(M, f, np.array([1.0]), np.array([-0.5]), 0.0)
    assert_allclose(y, 2.0 * (f + 0.2))


def test_cap_shift_soft_caps_negatives_only() -> None:
    """Negatives become A tanh(y / A); positives are untouched."""
    M = np.ones((2, 3))
    f = np.array([[-0.8, 0.5, -0.1], [0.3, -2.0, 0.0]])
    A = 0.6
    y = cap_shift(M, f, np.full(2, 5.0), np.full(2, -5.0), A)
    assert_allclose(y[f >= 0], f[f >= 0])
    assert_allclose(y[f < 0], A * np.tanh(f[f < 0] / A))
    assert np.all(y > -A)


# --- sat_mult ---

def test_sat_mult_is_one_above_table_top() -> None:
    """No boost at u >= -4.25."""
    l_max = inv_logit(np.array([-4.25, -4.0, -2.0, 0.0, 3.0]))
    csat = np.array([0.4, -0.2, -0.2])
    assert_allclose(sat_mult(l_max, csat, *TABLES), 1.0)


def test_sat_mult_is_one_where_csat_is_zero() -> None:
    """Channels with csat == 0 get 1 at every level."""
    l_max = inv_logit(np.linspace(-13.0, -3.0, 9))
    m = sat_mult(l_max, np.array([0.0, 0.5, -0.5]), *TABLES)
    assert m.shape == (9, 3)
    assert_allclose(m[:, 0], 1.0)


def test_sat_mult_boosts_raised_channels_near_black() -> None:
    """Near black, raised channels use the pos table (> 1.5) and lowered ones the neg table."""
    m = sat_mult(np.array([inv_logit(-12.0)]), np.array([0.3, -0.1, -0.1]), *TABLES)
    assert m[0, 0] > 1.5
    assert m[0, 0] == pytest.approx(np.interp(-12.0, SATMULT_U, SATMULT_POS))
    assert m[0, 1] == pytest.approx(np.interp(-12.0, SATMULT_U, SATMULT_NEG))


def test_sat_mult_uses_the_brightest_channel() -> None:
    """A pixel (0.001, 0, 0) gets the multiplier of 0.001 on every channel, not that of each channel's own (zero) level."""
    l = np.array([[0.001, 0.0, 0.0]])
    m = sat_mult(l.max(-1), np.array([0.2, 0.2, 0.2]), *TABLES)
    assert_allclose(m, np.interp(logit(0.001), SATMULT_U, SATMULT_POS))
    assert m[0, 1] != pytest.approx(np.interp(logit(1e-7), SATMULT_U, SATMULT_POS))


def test_sat_mult_clips_black_and_white() -> None:
    """Levels 0 and 1 are clipped before logit, so the result stays finite."""
    m = sat_mult(np.array([0.0, 1.0]), np.array([0.2, -0.2, 0.0]), *TABLES)
    assert np.all(np.isfinite(m))
    assert_allclose(m[0, :2], [SATMULT_POS[0], SATMULT_NEG[0]])
    assert_allclose(m[1], 1.0)
