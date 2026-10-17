"""
Band geometry of Lightroom's Color Mixer (HSL mode), measured on linear ProPhoto RGB.

Hue axis: hexagon (HSV) hue of the linear ProPhoto channels, degrees, 0 = red.  Eight bands with centres CENTRES (red, orange, yellow, green, aqua, blue,
purple, magenta).  Between two neighbouring centres a and b a colour at hue h (t = (h - a) / (b - a)) belongs to band a with weight 1 - s(t) and to band b
with weight s(t), s(t) = 3 t^2 - 2 t^3 (smoothstep): the weights of the eight bands always sum to 1.  Every slider (Hue, Saturation, Luminance) of all bands is
read at the hue of the INPUT pixel and added there.

    slider_at(h, levels)    effective slider of one kind at hue h:  (1 - s) * level_a + s * level_b
    hue_shift(h, levels)    degrees to add to h for the eight Hue sliders.  The eight centres move by d_k = (level_k / 100) * HUE_GAP (forward gap for a positive level, the gap
                            behind the band for a negative one); between centres the shift is the cubic Hermite curve through the d_k with node slopes m_k: zero where the
                            neighbouring secant slopes differ in sign, else sign * min(3 * smaller |secant|, SLOPE_CAP * |3-point slope|), plus small per-band terms COEF
                            (the peak of a single band is slightly rounded).  Bands that move the same way therefore interact (Hue Yellow + Green together are NOT the sum
                            of the two single-band curves); bands that move in opposite directions add exactly.
"""
import numpy as np

NAMES = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
CENTRES = np.array([0.0, 30.0, 55.0, 95.0, 160.0, 230.0, 275.0, 330.0])
# Hue +100 on band k moves its centre forward by HUE_GAP[k] degrees (toward band k+1); Hue -100 on band k moves it back by HUE_GAP[k - 1] (toward band k-1).
# Measured peaks of the single-band responses: 29.95 29.93 35.96 58.50 62.98 40.50 49.47 29.93 (+100) and the same numbers shifted by one band for -100.
HUE_GAP = np.array([30.0, 30.0, 36.0, 58.5, 63.0, 40.5, 49.5, 30.0])
SLOPE_CAP = 1.4
# node-slope terms per unit of d_k, columns: node k - 1, node k, node k + 1 (fit on the single-band +-100 curves)
COEF = np.array([[0.000408, -0.00519, 0.001075], [0.000655, -0.005888, 0.000789], [0.002456, -0.003824, 0.000106], [0.001876, -0.001501, 0.000122],
                 [0.000591, -0.000812, 0.000453], [0.000153, -0.001295, 0.000932], [0.000448, -0.001767, 0.000244], [0.000233, -0.001735, 0.002843]])
_EDGES = np.append(CENTRES, 360.0)
_WIDTH = np.diff(_EDGES)


def locate(h):
    """(k, s): band index of the lower neighbouring centre and smoothstep weight of band k + 1, for hue h in degrees."""
    h = np.asarray(h, np.float64) % 360.0
    k = np.clip(np.searchsorted(_EDGES, h, side="right") - 1, 0, 7)
    t = (h - _EDGES[k]) / (_EDGES[k + 1] - _EDGES[k])
    return k, t * t * (3.0 - 2.0 * t)


def slider_at(h, levels):
    """effective value of a band slider kind at hue h; `levels` = the eight slider values (-100..100)."""
    levels = np.asarray(levels, np.float64)
    k, s = locate(h)
    return (1.0 - s) * levels[k] + s * levels[(k + 1) % 8]


def hue_nodes(levels):
    """(d, m): displacement (deg) and slope of the hue-shift curve at the eight centres for the eight Hue sliders."""
    lv = np.asarray(levels, np.float64) / 100.0
    d = lv * np.where(lv >= 0, HUE_GAP, np.roll(HUE_GAP, 1))
    sec = (np.roll(d, -1) - d) / _WIDTH                     # secant slope from node k to k + 1
    sl, sr = np.roll(sec, 1), sec                           # left / right of node k
    hl, hr = np.roll(_WIDTH, 1), _WIDTH
    same = sl * sr > 0.0
    p = (hr * sl + hl * sr) / (hl + hr)
    m = np.where(same, np.sign(sl) * np.minimum(3.0 * np.minimum(np.abs(sl), np.abs(sr)), SLOPE_CAP * np.abs(p)), 0.0)
    m = m + COEF[:, 1] * d + np.roll(COEF[:, 0] * d, -1) + np.roll(COEF[:, 2] * d, 1)
    return d, m


def hue_shift(h, levels):
    """degrees to add to hue h for the eight Hue sliders (see the module docstring)."""
    d, m = hue_nodes(levels)
    h = np.asarray(h, np.float64) % 360.0
    k = np.clip(np.searchsorted(_EDGES, h, side="right") - 1, 0, 7)
    j = (k + 1) % 8
    w = _WIDTH[k]
    t = (h - _EDGES[k]) / w
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * d[k] + (t3 - 2 * t2 + t) * w * m[k] + (-2 * t3 + 3 * t2) * d[j] + (t3 - t2) * w * m[j])
