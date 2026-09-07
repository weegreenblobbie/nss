import numpy as np
import pytest
from nss.color_mixer import ColorMixer, bands
from nss.color_mixer.mixer import prophoto_matrix, hexagon_hue
from nss.icc import icc_to_linear, gamma_icc_bytes

ICC = gamma_icc_bytes("Adobe RGB (1998)", [(0.60974, 0.31111, 0.01947), (0.20528, 0.62567, 0.06087), (0.14919, 0.06322, 0.74457)], 563 / 256)
NAMES = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
MIX = ColorMixer()


def pp(img):
    """linear ProPhoto channels of encoded pixels (n, 3)."""
    return icc_to_linear(np.asarray(img, float).reshape(-1, 3), ICC) @ prophoto_matrix(ICC).T


def hue_of(img):
    p = pp(img)
    return hexagon_hue(p, p.max(axis=1), p.max(axis=1) - p.min(axis=1))


def band(k, v):
    a = np.zeros(8)
    a[k] = v
    return a


def px(*rgb):
    return np.array(rgb, float).reshape(1, 1, 3)


def test_no_sliders_is_identity():
    img = np.random.default_rng(0).uniform(0, 1, (8, 8, 3))
    assert MIX.apply(img, ICC) is img
    assert MIX.apply(img, ICC, hue=np.zeros(8), sat=np.zeros(8), lum=np.zeros(8)) is img


def test_requires_profile_and_eight_values():
    with pytest.raises(ValueError):
        MIX.apply(np.zeros((2, 2, 3)), None, sat=[10] * 8)
    with pytest.raises(ValueError):
        MIX.apply(np.zeros((2, 2, 3)), ICC, sat=[10] * 3)
    with pytest.raises(ValueError):
        MIX.apply(np.zeros((2, 2, 4)), ICC, sat=[10] * 8)


def test_dtype_shape_range_preserved():
    img = np.random.default_rng(1).uniform(0, 1, (5, 7, 3)).astype(np.float32)
    out = MIX.apply(img, ICC, hue=[20] * 8, sat=[-30, 40, 0, 10, -50, 60, 0, 25], lum=[10, -20, 30, 0, 50, -60, 0, 5])
    assert out.shape == img.shape and out.dtype == np.float32 and np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_extreme_pixels_stay_finite():
    img = np.array([[[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]], float)
    for kw in (dict(sat=[100] * 8), dict(sat=[-100] * 8), dict(lum=[100] * 8), dict(lum=[-100] * 8), dict(hue=[100] * 8), dict(hue=[-100] * 8)):
        out = MIX.apply(img, ICC, **kw)
        assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_grays_never_move():
    ramp = np.repeat(np.linspace(0.0, 1.0, 101)[:, None], 3, axis=1)[None]
    for kw in (dict(sat=[100] * 8), dict(sat=[-100] * 8), dict(lum=[100] * 8), dict(lum=[-100] * 8), dict(hue=[100] * 8), dict(hue=[-60] * 8)):
        out = MIX.apply(ramp, ICC, **kw)
        assert np.abs(out - ramp).max() < 1e-3, kw             # the test profile's rounded primaries leave ~1e-4 colour on 'gray'


def test_sat_minus_100_desaturates():
    img = np.array([[[0.7, 0.3, 0.2], [0.2, 0.6, 0.3], [0.3, 0.4, 0.8], [0.8, 0.7, 0.2]]])
    out = MIX.apply(img, ICC, sat=[-100] * 8)
    p = pp(out)
    assert ((p.max(axis=1) - p.min(axis=1)) / p.max(axis=1)).max() < 0.02


def test_sat_plus_raises_saturation_and_minus_lowers_it():
    img = px(0.55, 0.35, 0.30)
    s = lambda o: (lambda p: (p.max() - p.min()) / p.max())(pp(o)[0])
    assert s(MIX.apply(img, ICC, sat=[50] * 8)) > s(img) > s(MIX.apply(img, ICC, sat=[-50] * 8))


def test_lum_moves_saturated_colours_brighter_and_darker_but_not_gray():
    img = px(0.8, 0.3, 0.2)
    y = lambda o: pp(o)[0] @ np.array([0.288, 0.712, 0.0])
    assert y(MIX.apply(img, ICC, lum=[60] * 8)) > y(img) > y(MIX.apply(img, ICC, lum=[-60] * 8))
    g = px(0.4, 0.4, 0.4)
    assert np.abs(MIX.apply(g, ICC, lum=[100] * 8) - g).max() < 1e-3


def test_hue_slider_rotates_the_hexagon_hue_by_the_band_table():
    img = np.array([[[0.6, 0.4, 0.4], [0.7, 0.55, 0.35], [0.45, 0.6, 0.45], [0.4, 0.5, 0.65]]])      # inside the gamut also after the rotation
    h0 = hue_of(img)
    for k in range(8):
        for lv in (100, -100, 40):
            h1 = hue_of(MIX.apply(img, ICC, hue=band(k, lv)))
            want = bands.hue_shift(h0, band(k, lv))
            got = (h1 - h0 + 180.0) % 360.0 - 180.0
            assert np.abs(got - want).max() < 0.05, (k, lv, got, want)


def test_band_isolation():
    green, red = px(0.2, 0.7, 0.2), px(0.8, 0.2, 0.2)
    for kind in ("hue", "sat", "lum"):
        out_red = MIX.apply(red, ICC, **{kind: band(3, 100)})
        assert np.abs(out_red - red).max() < 1e-6, kind                      # a green slider leaves red alone
        out_green = MIX.apply(green, ICC, **{kind: band(3, 100)})
        assert np.abs(out_green - green).max() > 1e-3, kind                  # but moves green


def test_levels_are_ordered():
    img = px(0.7, 0.35, 0.25)
    sat = [pp(MIX.apply(img, ICC, sat=[v] * 8))[0] for v in (-100, -50, 0, 50, 100)]
    s = [(p.max() - p.min()) / p.max() for p in sat]
    assert s == sorted(s)
    lum = [pp(MIX.apply(img, ICC, lum=[v] * 8))[0].max() for v in (-100, -50, 0, 50, 100)]
    assert lum == sorted(lum)


def test_two_d_and_flat_agree():
    img = np.random.default_rng(2).uniform(0, 1, (6, 5, 3))
    a = MIX.apply(img, ICC, sat=[30] * 8, lum=[-20] * 8, hue=[10] * 8)
    b = MIX.apply(img.reshape(-1, 3), ICC, sat=[30] * 8, lum=[-20] * 8, hue=[10] * 8).reshape(img.shape)
    assert np.allclose(a, b)
