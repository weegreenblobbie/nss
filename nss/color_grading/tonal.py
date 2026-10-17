"""
Lightroom Color Grading (Shadows / Midtones / Highlights wheels, Balance, Blending, optional Global) as a native-space model.
Everything is done per channel in LINEAR ProPhoto light, whatever the image's own working space (the ICC profile is required; sRGB is never assumed).

    l      = linear ProPhoto channel level            u = logit(l)
    c_w    = C0 * M_w(u + d(Balance); Blending) * f_w     (per channel; M_w tabulated per wheel, Blending interpolated with PCHIP)
    l'     = l e^c / (1 - l + l e^c)                       with c = c_global + sum of the wheels' c
    Shadows / Highlights Sat: f = Sat/100 (v - w.v), then one common shift moves the three channels into a luminance-dependent window,
                              negatives soft-capped (tables in shadows.py / highlights.py);   Midtones: f = Sat/100 v
    Lum of a tonal wheel: common shift  2 atanh(0.0024 Lum) * ratio(u + d) * M_blend / M_50;  Shadows Lum > 0 also lifts black (additive).
    Global: c_global = c_sat * sat_mult(logit(max channel)) + 2 atanh(0.0024 Lum),  c_sat = 2 atanh(0.0028 Sat v)  (Sat boosted near black).

The tables live inline in shadows.py / midtones.py / highlights.py / global_wheel.py (generated from the measurements by tests/test_data/make_production.py).
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from nss.icc import icc_to_linear, icc_from_linear, _profile
from nss.color_grading.math import C0, WLUMA, PROPHOTO, LUM_Q, logit, odds_gain, hue_vector, global_c, balance_shift, cap_shift, sat_mult
from nss.color_grading.global_wheel import SATMULT_U, SATMULT_POS, SATMULT_NEG

UGRID = np.linspace(-16.0, 4.0, 801)               # level axis of every table
BL_KNOTS = (0, 25, 50, 75, 100)                    # Blending values of the mask rows


class TonalWheel:
    """Base of Shadows / Midtones / Highlights; the subclasses only carry the tables.
    MASK (5, 801): M on UGRID at Blending 0/25/50/75/100, Balance 0.   LUM (801,): Lum ratio on UGRID.
    CAP_HI, CAP_LO (801,) in units of C0, CAP_A float (None: no window, e.g. Midtones).   LIFT_L, LIFT_Y: additive black lift of Lum 100 (Shadows only)."""
    NAME = ""
    CHROMA_ONLY = True          # f = v - w.v (Shadows, Highlights);  False: f = v (Midtones)
    MASK = LUM = CAP_HI = CAP_LO = LIFT_L = LIFT_Y = None
    CAP_A = 0.0

    def __init__(self) -> None:
        """Builds the Blending interpolator and, for wheels with a cap, the window in f units."""
        self._rows = {}
        self._pchip = PchipInterpolator(BL_KNOTS, np.asarray(self.MASK), axis=0)
        ref = np.asarray(self.MASK)[2]
        self._ref = ref
        if self.CAP_HI is not None:                      # window in f units = Hi / M_ref; where M_ref ~ 0 the ratio is noise -> hold the nearest reliable value
            ok = ref > 0.05 * ref.max()
            r = lambda a: np.interp(UGRID, UGRID[ok], (np.asarray(a) / np.maximum(ref, 1e-6))[ok])
            self._fh, self._fl = r(self.CAP_HI), r(self.CAP_LO)

    def _row(self, blending: float) -> np.ndarray:
        """Mask curve on UGRID at this Blending (Balance 0), cached per value."""
        k = float(blending)
        if k not in self._rows:
            self._rows[k] = np.asarray(self._pchip(k))
        return self._rows[k]

    def mask(self, u: np.ndarray, balance: float = 0, blending: float = 50) -> np.ndarray:
        """Mask strength M at levels u for this Balance and Blending."""
        return np.interp(u + balance_shift(balance), UGRID, self._row(blending))

    def shift(self, l: np.ndarray, u: np.ndarray, uY: np.ndarray, hue: float, sat: float, lum: float, balance: float = 0, blending: float = 50) -> np.ndarray:
        """c (N,3) of this wheel.  l, u: (N,3) linear ProPhoto level / logit; uY: (N,) logit of the pixel luminance + Balance shift."""
        c = np.zeros_like(l)
        d = balance_shift(balance)
        if lum:
            Mc = self.mask(u, balance, blending)
            M5 = np.interp(u + d, UGRID, self._ref)
            R = np.where(M5 > 0.02 * np.asarray(self.MASK).max(), Mc / np.maximum(M5, 1e-9), 1.0)
            c += 2.0 * np.arctanh(LUM_Q * lum) * np.interp(u + d, UGRID, self.LUM) * R
        if sat:
            v = hue_vector(hue)
            f = np.broadcast_to((sat / 100.0) * ((v - v @ WLUMA) if self.CHROMA_ONLY else v), l.shape).copy()
            M = self.mask(u, balance, blending)
            if self.CAP_HI is not None:
                c += C0 * cap_shift(M, f, np.interp(uY, UGRID, self._fh), np.interp(uY, UGRID, self._fl), self.CAP_A)
            else:
                c += C0 * M * f
        return c

    def lift(self, l: np.ndarray, lum: float) -> np.ndarray | float:
        """Additive linear lift of black (Shadows Lum > 0); zero for the other wheels."""
        if self.LIFT_L is None or lum <= 0:
            return 0.0
        return (np.arctanh(LUM_Q * lum) / np.arctanh(0.24)) * np.interp(l, self.LIFT_L, self.LIFT_Y)


def working_matrix(icc: bytes) -> np.ndarray:
    """3x3 from the profile's linear RGB to linear ProPhoto."""
    return np.linalg.inv(PROPHOTO) @ _profile(icc)["M"]


def grade_tonal(enc: np.ndarray, icc: bytes, wheels: list[tuple[TonalWheel, float, float, float]], balance: float = 0, blending: float = 50,
                glob: tuple[float, float, float] | None = None, chunk: int = 400000) -> np.ndarray:
    """enc: (...,3) encoded 0..1 in the profile `icc` -> graded, same shape.  wheels: [(wheel object, hue, sat, lum), ...]; glob: optional (hue, sat, lum)."""
    Cm = working_matrix(icc)
    Ci = np.linalg.inv(Cm).T
    flat = np.asarray(enc, np.float64).reshape(-1, 3)
    out = np.empty_like(flat)
    db = balance_shift(balance)
    for i in range(0, len(flat), chunk):
        l = (icc_to_linear(flat[i:i + chunk], icc) @ Cm.T).clip(0.0, 1.0)
        Y = np.clip(l @ WLUMA, 1e-6, 1 - 1e-6)
        uY = logit(Y) + db
        lc = np.clip(l, 1e-6, 1 - 1e-6)
        u = logit(lc)
        c = np.zeros_like(l)
        if glob is not None:
            gh, gs, gl = glob
            csat = global_c(gh, gs, 0)
            c += csat * sat_mult(l.max(-1), csat, SATMULT_U, SATMULT_POS, SATMULT_NEG) + global_c(gh, 0, gl)
        for w, h, s, lum in wheels:
            c += w.shift(l, u, uY, h, s, lum, balance, blending)
        lin = odds_gain(l, c)
        for w, h, s, lum in wheels:
            lin = lin + w.lift(l, lum)
        out[i:i + chunk] = icc_from_linear(np.clip(lin @ Ci, 0.0, 1.0), icc)
    return out.reshape(np.shape(enc))