"""
The Color Mixer's hue mapping f (mixer-input hue -> mixer-output hue, degrees of the linear ProPhoto hexagon hue) as a lookup table, and its inverse.

The table is measured by running a strip of fully saturated colours (S = 1, V = STRIP_VALUE) through the mixer's own per-pixel transform
(ColorMixer._linear, the step ColorMixer.apply runs before clipping), so it follows the image code path.  The strip is encoded in linear ProPhoto, the
mixer's own space, where an S = 1 colour stays inside the gamut after any hue shift.
Only the eight Hue sliders change f; Saturation and Luminance never move the hue.
"""
import numpy as np
from numpy.typing import ArrayLike

from nss.color_mixer.mixer import ColorMixer, PROPHOTO_XYZ, hexagon_hue, hsv_to_rgb, prophoto_matrix
from nss.icc import gamma_icc_bytes, icc_to_linear

LINEAR_PROPHOTO = gamma_icc_bytes("linear ProPhoto RGB", [tuple(column) for column in PROPHOTO_XYZ.T], 1.0)
STRIP_VALUE = 0.5           # V of the synthetic strip


def native_hue(rgb_enc: ArrayLike, icc: bytes) -> np.ndarray:
    """Mixer-input hue in degrees (n,) of encoded pixels (..., 3) in the profile icc: the hexagon hue of their linear ProPhoto channels."""
    p = icc_to_linear(np.asarray(rgb_enc, np.float64).reshape(-1, 3), icc) @ prophoto_matrix(icc).T
    v = p.max(axis=1)
    return hexagon_hue(p, v, v - p.min(axis=1))


def hue_remap_lut(hue_levels: ArrayLike, n: int = 360) -> np.ndarray:
    """f at the n input hues 360 * i / n (i = 0 .. n - 1) for the eight Hue slider levels (red .. magenta, -100..100), unwrapped: f(h) - h is the shift in degrees."""
    h = 360.0 * np.arange(n) / n
    strip = hsv_to_rgb(h, np.ones(n), np.full(n, STRIP_VALUE))
    m = prophoto_matrix(LINEAR_PROPHOTO)
    still = np.zeros(8)
    lin = ColorMixer._linear(strip, LINEAR_PROPHOTO, m, np.linalg.inv(m), np.asarray(hue_levels, np.float64), still, still)
    p = lin @ m.T
    v = p.max(axis=1)
    out = hexagon_hue(p, v, v - p.min(axis=1))
    return h + (out - h + 180.0) % 360.0 - 180.0


def _closed(lut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) nodes of the table closed at 360: x = 0 .. 360, y ends at f(0) + 360."""
    n = len(lut)
    return 360.0 * np.arange(n + 1) / n, np.append(lut, lut[0] + 360.0)


def hue_remap_forward(lut: np.ndarray, h_in: ArrayLike) -> np.ndarray:
    """f(h_in) in degrees 0..360, linear between the table nodes."""
    x, y = _closed(lut)
    return np.interp(np.mod(h_in, 360.0), x, y) % 360.0


def hue_remap_inverse(lut: np.ndarray, h_out: float) -> np.ndarray:
    """Every input hue (sorted, degrees 0..360) that f maps to h_out; more than one where f folds over."""
    x, y = _closed(lut)
    y0, y1 = y[:-1], y[1:]
    found = []
    for target in float(h_out) % 360.0 + 360.0 * np.arange(-1, 3):      # y spans less than [-360, 720)
        hit = ((y0 <= target) & (target < y1)) | ((y1 < target) & (target <= y0))      # each crossing counted on one segment only
        found.append(x[:-1][hit] + (target - y0[hit]) / (y1[hit] - y0[hit]) * (x[1:][hit] - x[:-1][hit]))
    return np.sort(np.concatenate(found) % 360.0)
