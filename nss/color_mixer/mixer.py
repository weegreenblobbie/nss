"""
Lightroom Color Mixer (HSL mode: Hue / Saturation / Luminance of eight bands) in the image's own working space.

Everything is computed on LINEAR ProPhoto RGB (Lightroom's mixer space), whatever the working space of the file (the ICC profile is required; sRGB is never assumed):

    p = linear ProPhoto channels;  V = max, S = (max - min) / max, H = hexagon hue (degrees), m = folded position of the middle channel
    every slider of every band is read at the INPUT hue H (bands.py: smoothstep weights, eight bands, sum 1) and added there
    Hue:  H' = H + bands.hue_shift(H)
    Lum:  V' = V (1 + a_V)^2,  S' = S exp(a_S)         a_V, a_S read from the tables of tables.py at (V, S, m) and the slider level (Lum+ and Lum- separately)
    Sat+: V' = V (1 + tabulated),  S' = S exp(tabulated);   Sat-: V' = V (1 + tabulated),  S' = S (1 - a) V / V'  with a = -Sat / 100
    Lum and Sat ratios multiply; the result is hsv (H', S', V') of linear ProPhoto, converted back to the working space and clipped.
"""
import numpy as np
from nss.icc import icc_to_linear, icc_from_linear, _profile
from nss.color_mixer import bands, lookup as lk
from nss.color_mixer import tables as tb

PROPHOTO_XYZ = np.array([[0.7976749, 0.1351917, 0.0313534],
                         [0.2880402, 0.7118741, 0.0000857],
                         [0.0, 0.0, 0.8252100]])
CHUNK = 400000
LEVELS = (25.0, 50.0, 75.0, 100.0)


def prophoto_matrix(icc):
    """3x3 from the file's linear RGB to linear ProPhoto (through D50 XYZ)."""
    return np.linalg.inv(PROPHOTO_XYZ) @ _profile(icc)["M"]


def hexagon_hue(p, v, c):
    """HSV hue in degrees of channels p (n, 3) with max v and range c."""
    d = np.maximum(c, 1e-12)
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    h = np.where(v == r, ((g - b) / d) % 6.0, np.where(v == g, (b - r) / d + 2.0, (r - g) / d + 4.0))
    return (h * 60.0) % 360.0


def hsv_to_rgb(h, s, v):
    h6 = (h % 360.0) / 60.0
    k = np.stack([(5.0 + h6) % 6.0, (3.0 + h6) % 6.0, (1.0 + h6) % 6.0], axis=-1)
    return v[:, None] * (1.0 - s[:, None] * np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0))


def _levels(vals, name):
    a = np.zeros(8) if vals is None else np.asarray(vals, np.float64)
    if a.shape != (8,):
        raise ValueError(f"{name}: expected eight values (red, orange, yellow, green, aqua, blue, purple, magenta), got shape {a.shape}")
    return np.clip(a, -100.0, 100.0)


def lum_ratios(cell, level):
    """(V ratio, S ratio) of the Lum slider (signed, -100..100 per pixel); `cell` = lookup.Cell of the pixels."""
    vr, sr = np.ones(level.shape), np.ones(level.shape)
    for sign, tv, ts in ((1.0, tb.LUM_POS_V, tb.LUM_POS_S), (-1.0, tb.LUM_NEG_V, tb.LUM_NEG_S)):
        m = sign * level > 0
        if m.any():
            c, lv = cell.take(m), np.abs(level[m])
            vr[m] = (1.0 + c.sample_levels(tv, LEVELS, lv)) ** 2
            sr[m] = np.exp(c.sample_levels(ts, LEVELS, lv))
    return vr, sr


def sat_ratios(cell, level):
    """(V ratio, S ratio) of the Saturation slider (signed, -100..100 per pixel)."""
    vr, sr = np.ones(level.shape), np.ones(level.shape)
    m = level > 0
    if m.any():
        c, lv = cell.take(m), level[m]
        vr[m] = 1.0 + c.sample_levels(tb.SAT_POS_V, LEVELS, lv)
        sr[m] = np.exp(c.sample_levels(tb.SAT_POS_S, LEVELS, lv))
    m = level < 0
    if m.any():
        c, lv = cell.take(m), -level[m]
        vr[m] = 1.0 + c.sample_levels(tb.SAT_NEG_V, LEVELS, lv)
        sr[m] = (1.0 - lv / 100.0) / np.maximum(vr[m], 1e-12)         # the range gain of every channel is exactly 1 - a
    return vr, sr


class ColorMixer:
    """Lightroom Color Mixer, HSL mode.

    `img_enc` is the picture as stored (encoded 0..1, shape (..., 3)) and `icc` the ICC profile bytes it is tagged with (TiffFile.iccprofile).  hue / sat / lum are
    the eight band sliders (red, orange, yellow, green, aqua, blue, purple, magenta), each -100..100; omitted = all 0.  Returns the same shape and dtype.
    """

    def apply(self, img_enc, icc, hue=None, sat=None, lum=None):
        if icc is None:
            raise ValueError("an ICC profile is required: the mixer is defined in the image's own working space, sRGB is never assumed")
        img_enc = np.asarray(img_enc)
        if img_enc.shape[-1] != 3:
            raise ValueError(f"expected (..., 3) RGB, got shape {img_enc.shape}")
        hue, sat, lum = _levels(hue, "hue"), _levels(sat, "sat"), _levels(lum, "lum")
        if not (hue.any() or sat.any() or lum.any()):
            return img_enc
        M = prophoto_matrix(icc)
        Minv = np.linalg.inv(M)
        flat = img_enc.reshape(-1, 3).astype(np.float64)
        out = np.empty_like(flat)
        for a in range(0, len(flat), CHUNK):
            out[a:a + CHUNK] = self._chunk(flat[a:a + CHUNK], icc, M, Minv, hue, sat, lum)
        out = out.reshape(img_enc.shape)
        return out.astype(img_enc.dtype) if img_enc.dtype.kind == "f" else out

    @staticmethod
    def _linear(enc, icc, M, Minv, hue, sat, lum):
        """the mixed picture as linear working-space RGB before clipping (negative / above 1 where the result leaves the gamut)."""
        p = icc_to_linear(enc, icc) @ M.T
        v, mn = p.max(axis=1), p.min(axis=1)
        c = v - mn
        s = c / np.maximum(v, 1e-12)
        h = hexagon_hue(p, v, c)
        vr, sr = np.ones(len(v)), np.ones(len(v))
        if lum.any() or sat.any():
            mid = np.sort(p, axis=1)[:, 1]
            cell = lk.Cell(v, s, lk.fold((mid - mn) / np.maximum(c, 1e-12)))
        if lum.any():
            a, b = lum_ratios(cell, bands.slider_at(h, lum))
            vr, sr = vr * a, sr * b
        if sat.any():
            a, b = sat_ratios(cell, bands.slider_at(h, sat))
            vr, sr = vr * a, sr * b
        h2 = h + bands.hue_shift(h, hue) if hue.any() else h
        rgb = hsv_to_rgb(h2, np.clip(s * sr, 0.0, 1.0), v * vr)
        return rgb @ Minv.T

    @staticmethod
    def _chunk(enc, icc, M, Minv, hue, sat, lum):
        return icc_from_linear(np.clip(ColorMixer._linear(enc, icc, M, Minv, hue, sat, lum), 0.0, 1.0), icc)
    