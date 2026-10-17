"""Tests of nss.color_grading.tonal: TonalWheel masks and shifts, and grade_tonal including the Global wheel."""
from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nss.color_grading.math import WLUMA, logit, balance_shift
from nss.color_grading.tonal import UGRID, BL_KNOTS, TonalWheel, grade_tonal
from nss.color_grading.shadows import Shadows
from nss.color_grading.midtones import Midtones
from nss.color_grading.highlights import Highlights

WHEELS = [Shadows, Midtones, Highlights]


@pytest.fixture(scope="module", params=WHEELS, ids=lambda w: w.__name__)
def wheel(request: pytest.FixtureRequest) -> TonalWheel:
    """Each of the three tonal wheels in turn."""
    return request.param()


def test_mask_non_negative(wheel: TonalWheel) -> None:
    """The mask is never negative for any Balance and Blending."""
    for balance in (-100, 0, 100):
        for blending in (0, 10, 50, 90, 100):
            assert wheel.mask(UGRID, balance, blending).min() >= 0.0


@pytest.mark.parametrize("k, blending", list(enumerate(BL_KNOTS)))
def test_mask_reproduces_table_rows(wheel: TonalWheel, k: int, blending: int) -> None:
    """At the Blending knots the mask is exactly the table row."""
    assert_allclose(wheel.mask(UGRID, 0, blending), wheel.MASK[k], atol=1e-12)


def test_balance_shifts_the_curve(wheel: TonalWheel) -> None:
    """M_b(u) = M_0(u + d(b)): the Balance curve sampled at u - d equals the Balance 0 curve at u."""
    u = np.linspace(-10.0, 0.0, 37)
    for balance in (-60, 40):
        d = balance_shift(balance)
        assert_allclose(wheel.mask(u - d, balance, 35), wheel.mask(u, 0, 35))
        assert not np.allclose(wheel.mask(u, balance, 35), wheel.mask(u, 0, 35))


def _levels(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(l, u, uY) for n pixels: linear levels (n, 3), their logit and the logit of their luminance."""
    l = np.linspace(0.01, 0.95, n * 3).reshape(n, 3)
    lc = np.clip(l, 1e-6, 1 - 1e-6)
    Y = l @ WLUMA
    return l, logit(lc), logit(Y)


@pytest.mark.parametrize("sat, lum", [(40, 0), (0, -30), (25, 20)])
def test_shift_shape(wheel: TonalWheel, sat: float, lum: float) -> None:
    """shift() returns a finite, non-zero c of shape (N, 3)."""
    l, u, uY = _levels(7)
    c = wheel.shift(l, u, uY, 200, sat, lum, balance=0, blending=50)
    assert c.shape == (7, 3)
    assert np.all(np.isfinite(c))
    assert np.any(c != 0.0)


def test_shift_zero_without_sat_or_lum(wheel: TonalWheel) -> None:
    """Sat 0 and Lum 0 give c = 0."""
    l, u, uY = _levels(5)
    assert_allclose(wheel.shift(l, u, uY, 200, 0, 0), 0.0)


def test_grade_tonal_without_wheels_round_trips(adobe_icc: bytes) -> None:
    """No wheels: the ICC round trip returns the input within 1e-6."""
    enc = np.linspace(0.0, 1.0, 4 * 2 * 3).reshape(4, 2, 3)
    out = grade_tonal(enc, adobe_icc, [])
    assert out.shape == (4, 2, 3)
    assert_allclose(out, enc, atol=1e-6)


def test_grade_tonal_applies_wheels(adobe_icc: bytes) -> None:
    """Active wheels change the image and keep it in 0..1."""
    enc = np.linspace(0.05, 0.95, 5 * 2 * 3).reshape(5, 2, 3)
    out = grade_tonal(enc, adobe_icc, [(Shadows(), 200, 50, 0), (Highlights(), 30, 40, 10)], balance=-20, blending=60)
    assert out.shape == (5, 2, 3)
    assert np.all((out >= 0.0) & (out <= 1.0))
    assert np.abs(out - enc).max() > 1e-3


def test_grade_tonal_global_only(adobe_icc: bytes) -> None:
    """A red Global tint alone raises red."""
    enc = np.linspace(0.05, 0.95, 3 * 4 * 3).reshape(3, 4, 3)
    out = grade_tonal(enc, adobe_icc, [], glob=(0, 30, 0))
    assert np.all(out[..., 0] >= enc[..., 0])
    assert np.abs(out - enc).max() > 1e-3


def test_grade_tonal_chunks_match(adobe_icc: bytes) -> None:
    """Processing in chunks gives the same result as in one piece."""
    enc = np.random.default_rng(3).random((9, 4, 3))
    wheels = [(Midtones(), 120, 30, -10)]
    assert_allclose(grade_tonal(enc, adobe_icc, wheels, chunk=5), grade_tonal(enc, adobe_icc, wheels))


def _no_boost(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Global wheel without the dark-end Sat boost (the model before it)."""
    monkeypatch.setattr("nss.color_grading.tonal.sat_mult", lambda l_max, csat, *tables: np.ones(np.shape(l_max) + (3,)))


@pytest.mark.parametrize("lum", [0, 35, -60])
def test_global_sat_zero_is_bit_identical_to_no_boost(adobe_icc: bytes, ramp: Callable[..., np.ndarray], monkeypatch: pytest.MonkeyPatch, lum: float) -> None:
    """With Global Sat 0 the dark-end boost changes nothing, bit for bit."""
    enc = np.stack([ramp((1, 1, 1), 300), ramp((0.9, 0.2, 0.05), 300)])
    boosted = grade_tonal(enc, adobe_icc, [], glob=(70, 0, lum))
    _no_boost(monkeypatch)
    assert np.array_equal(grade_tonal(enc, adobe_icc, [], glob=(70, 0, lum)), boosted)


def test_global_sat_boost_raises_dark_end(adobe_icc: bytes, ramp: Callable[..., np.ndarray], monkeypatch: pytest.MonkeyPatch) -> None:
    """Global alone, Sat 100 red on a dark neutral ramp: the raised (red) channel gains more with the boost than without."""
    enc = ramp((1, 1, 1), 1024)[None, :100]                 # encoded 0 .. ~0.1
    boosted = grade_tonal(enc, adobe_icc, [], glob=(0, 100, 0))
    _no_boost(monkeypatch)
    plain = grade_tonal(enc, adobe_icc, [], glob=(0, 100, 0))
    gain_boosted, gain_plain = boosted[..., 0] - enc[..., 0], plain[..., 0] - enc[..., 0]
    assert gain_boosted[0, 1:].mean() > gain_plain[0, 1:].mean() > 0.0
