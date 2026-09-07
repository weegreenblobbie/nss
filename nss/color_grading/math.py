"""Shared, side-effect-free maths of the color grading package: CIE Lab conversion and the Lightroom Color Grading model's formulas and constants."""
import numpy as np
from numpy.typing import ArrayLike

D65 = [0.95047, 1.00000, 1.08883]

SRGB_TO_XYZ_D65 = np.array([
    [0.4124564, 0.2126729, 0.0193339],
    [0.3575761, 0.7151522, 0.1191920],
    [0.1804375, 0.0721750, 0.9503041],
])

# --- ACTIVE COLOR SPACE SELECTION ---
WHITE = D65
MATRIX_RGB_TO_XYZ = SRGB_TO_XYZ_D65


def rgb_to_lab_pure(rgb_img: np.ndarray) -> np.ndarray:
    """
    Converts a [0.0, 1.0] RGB image array to float64 CIE L*a*b* color space.
    
    The conversion performs standard IEC 61966-2-1 sRGB linearization (gamma expansion),
    maps to CIE 1931 XYZ tristimulus values using standard sRGB transformation matrix,
    normalizes against the standard D65 illuminant white point (Xn=0.95047, Yn=1.00000, Zn=1.08883),
    and applies standard CIE L*a*b* cube-root non-linear mapping.

    Args:
        rgb_img (np.ndarray): Input RGB array scaled to [0.0, 1.0].
            Can be 2D slice or 3D image of shape (..., 3).

    Returns:
        np.ndarray: Converted array in CIE L*a*b* space with dtype np.float64,
            with L* in [0, 100], a* in [-128, 127], and b* in [-128, 127].
    """
    # Assume input is already scaled to 0.0 - 1.0
    rgb = rgb_img.astype(np.float64)
    
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    
    xyz = np.dot(rgb, MATRIX_RGB_TO_XYZ)
    xyz /= np.array(WHITE)
    
    mask = xyz > 0.008856
    f_xyz = np.empty_like(xyz)
    f_xyz[mask] = np.cbrt(xyz[mask])
    f_xyz[~mask] = (7.787 * xyz[~mask]) + (16.0 / 116.0)
    
    L = (116.0 * f_xyz[..., 1]) - 16.0
    a = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
    
    return np.stack([L, a, b], axis=-1)


# --- Color Grading model: per channel log-odds shifts in linear ProPhoto light ---
C0 = float(2.0 * np.arctanh(0.28))                 # c of a tonal wheel at mask 1 and f 1
WLUMA = np.array([0.288, 0.712, 0.0])              # ProPhoto luma row
PROPHOTO = np.array([[0.7977, 0.1352, 0.0313], [0.2880, 0.7119, 0.0001], [0.0, 0.0, 0.8249]])      # linear -> XYZ D50
SAT_Q, LUM_Q = 0.0028, 0.0024                      # Global wheel Sat / Lum per slider unit
GAMMA, P0, P1 = 2.4, 0.5261, -0.00379              # Balance -> centre level of the masks


def logit(l: ArrayLike) -> np.ndarray | float:
    """u = log(l / (1 - l)), the level axis of the masks."""
    return np.log(l / (1.0 - l))


def inv_logit(u: ArrayLike) -> np.ndarray | float:
    """l with logit(l) = u."""
    return 1.0 / (1.0 + np.exp(-u))


def odds_gain(l: ArrayLike, c: ArrayLike) -> np.ndarray | float:
    """l' with logit(l') = logit(l) + c; l is clipped to 0..1, so 0 and 1 stay fixed."""
    l = np.clip(l, 0.0, 1.0)
    e = np.exp(c)
    return l * e / (1.0 - l + l * e)


def hue_vector(hue: ArrayLike) -> np.ndarray:
    """+-1 hexagon vector of a hue (degrees): red 0, green 120, blue 240."""
    x = (np.asarray(hue, float) / 60.0) % 6.0
    r = np.clip(np.abs(x - 3.0) - 1.0, 0, 1)
    g = np.clip(2.0 - np.abs(x - 2.0), 0, 1)
    b = np.clip(2.0 - np.abs(x - 4.0), 0, 1)
    return 2.0 * np.stack([r, g, b], -1) - 1.0


def global_c(hue: ArrayLike, sat: float, lum: float) -> np.ndarray:
    """Per-channel log-odds shift (3,) of the Global wheel."""
    return 2.0 * np.arctanh(SAT_Q * sat * hue_vector(hue)) + 2.0 * np.arctanh(LUM_Q * lum)


def balance_shift(b: float) -> float:
    """d(b): M_b(u) = M_0(u + d)."""
    uc = lambda x: logit((P0 + P1 * x) ** GAMMA)
    return float(uc(0.0) - uc(b))


def cap_shift(M: np.ndarray, f: np.ndarray, fh: np.ndarray, fl: np.ndarray, A: float) -> np.ndarray:
    """y (N,3), units of C0: f is moved by ONE common shift into the window [fl, fh] (f units, per pixel), multiplied by the per-channel mask M, and
    negatives are soft-capped with A (0 = none)."""
    mx, mn = f.max(-1), f.min(-1)
    f = f + np.where(mx > fh, fh - mx, np.where(mn < fl, fl - mn, 0.0))[:, None]
    y = M * f
    if A:
        y = np.where(y < 0, A * np.tanh(y / A), y)
    return y


def sat_mult(l_max: np.ndarray, csat: np.ndarray, u_tab: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Dark-end factor (N,3) on the Global wheel's Sat part of c.  Looked up at u = logit of each pixel's brightest linear ProPhoto channel l_max (N,):
    the `pos` table where csat > 0, `neg` where csat < 0 (both on the level axis u_tab, 1 above it), 1 where csat == 0."""
    u = logit(np.clip(l_max, 1e-7, 1 - 1e-7))[..., None]
    return np.where(csat > 0, np.interp(u, u_tab, pos, right=1.0), np.where(csat < 0, np.interp(u, u_tab, neg, right=1.0), 1.0))
