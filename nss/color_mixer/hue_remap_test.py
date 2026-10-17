"""Tests of nss.color_mixer.hue_remap: the mixer's hue mapping table, its inverse and the native hue of encoded pixels."""
import numpy as np

from nss.color_mixer.bands import hue_shift
from nss.color_mixer.hue_remap import hue_remap_forward, hue_remap_inverse, hue_remap_lut, native_hue
from nss.color_mixer.mixer import hsv_to_rgb
from nss.icc import gamma_icc_bytes, icc_from_linear

ICC = gamma_icc_bytes("Adobe RGB (1998)", [(0.60974, 0.31111, 0.01947), (0.20528, 0.62567, 0.06087), (0.14919, 0.06322, 0.74457)], 563 / 256)
H = np.arange(360.0)


def band(k: int, v: float) -> np.ndarray:
    """Eight Hue levels, band k at v, the others 0."""
    a = np.zeros(8)
    a[k] = v
    return a


def test_neutral_sliders_give_the_identity() -> None:
    """All Hue sliders at 0: f(h) = h at every node, also at a non-default table size."""
    assert np.abs(hue_remap_lut(np.zeros(8)) - H).max() < 1e-3
    assert np.abs(hue_remap_lut(np.zeros(8), n=90) - np.arange(0.0, 360.0, 4.0)).max() < 1e-3


def test_table_is_the_mixer_hue_shift() -> None:
    """The strip run through ColorMixer.apply shifts every hue by bands.hue_shift (the image code path)."""
    levels = np.array([20.0, 100.0, -100.0, 0.0, 40.0, 0.0, -70.0, 15.0])
    assert np.abs(hue_remap_lut(levels) - H - hue_shift(H, levels)).max() < 1e-2


def test_single_band_moves_only_its_range() -> None:
    """Green +60 moves hues between Yellow (55) and Aqua (160); its slope terms reach one band further, beyond Orange (30) and Blue (230) nothing moves."""
    shift = hue_remap_lut(band(3, 60.0)) - H
    assert shift[95] > 30.0
    assert np.abs(shift[(H <= 30.0) | (H >= 230.0)]).max() < 1e-2
    assert np.abs(shift[(H > 30.0) & (H <= 55.0)]).max() < 0.5 and np.abs(shift[(H >= 160.0) & (H < 230.0)]).max() < 0.5


def test_inverse_of_forward_is_identity_when_monotonic() -> None:
    """For a moderate setting f is monotonic: f^-1(f(h)) is the single preimage h."""
    lut = hue_remap_lut(np.array([30.0, -20.0, 10.0, 40.0, -30.0, 20.0, 0.0, -10.0]))
    assert np.diff(np.append(lut, lut[0] + 360.0)).min() > 0.0
    for h in (0.0, 13.7, 95.0, 181.25, 359.5):
        pre = hue_remap_inverse(lut, float(hue_remap_forward(lut, h)))
        assert len(pre) == 1 and abs(pre[0] - h) < 1e-6


def test_fold_over_returns_every_preimage() -> None:
    """Orange +100 with Yellow -100 folds f over: one output hue has three input hues, each mapping back to it."""
    lut = hue_remap_lut(band(1, 100.0) + band(2, -100.0))
    pre = hue_remap_inverse(lut, 50.0)
    assert len(pre) == 3
    assert np.allclose(hue_remap_forward(lut, pre), 50.0, atol=1e-6)


def test_native_hue_of_encoded_pixels() -> None:
    """Pixels built at known linear ProPhoto hues read back those hues through the file's profile."""
    from nss.color_mixer.mixer import prophoto_matrix
    hues = np.array([[10.0, 75.0], [150.0, 200.0], [260.0, 340.0]])           # (3, 2) image
    p = hsv_to_rgb(hues.ravel(), np.full(6, 0.4), np.full(6, 0.6))
    enc = icc_from_linear(p @ np.linalg.inv(prophoto_matrix(ICC)).T, ICC).reshape(3, 2, 3)
    assert np.abs(native_hue(enc, ICC) - hues.ravel()).max() < 0.2
