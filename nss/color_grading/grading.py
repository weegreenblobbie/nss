"""Public API of the color grading package: `Grading().apply(...)`, Lightroom Color Grading in the image's own working space."""
import numpy as np

from nss.color_grading.shadows import Shadows
from nss.color_grading.midtones import Midtones
from nss.color_grading.highlights import Highlights
from nss.color_grading.tonal import grade_tonal


class Grading:
    """Lightroom Color Grading in the image's own working space.

    `img_enc` is the picture as stored (encoded 0..1, shape (..., 3)) and `icc` the ICC profile bytes it is tagged with (TiffFile.iccprofile);
    the grade is computed in linear ProPhoto light and returned in the same encoding.  Hue 0 = red, 60 yellow, 120 green, 180 cyan, 240 blue,
    300 magenta; sat 0..100; lum -100..100; balance -100..100; blend 0..100.  The optional Global wheel (gl_*) adds to the tonal wheels.
    Best validated: Balance 0 +-50, Blending 25..75, Sat <= 50 (about 430 DN of 65535 RMS over the 71 Lightroom exports it was fitted to).
    """

    def __init__(self) -> None:
        """Creates the three tonal wheels; their tables are loaded once and reused by every apply()."""
        self.shadows = Shadows()
        self.midtones = Midtones()
        self.highlights = Highlights()

    def apply(self, img_enc: np.ndarray, icc: bytes, balance: float = 0, blend: float = 50,
              sh_h: float = 0.0, sh_s: float = 0.0, sh_l: float = 0.0,
              mi_h: float = 0.0, mi_s: float = 0.0, mi_l: float = 0.0,
              hi_h: float = 0.0, hi_s: float = 0.0, hi_l: float = 0.0,
              gl_h: float = 0.0, gl_s: float = 0.0, gl_l: float = 0.0) -> np.ndarray:
        """img_enc graded with these slider values (see the class docstring for units); returns the input itself when every wheel is off."""
        if icc is None:
            raise ValueError("an ICC profile is required: the grade is defined in the image's own working space, sRGB is never assumed")
        img_enc = np.asarray(img_enc)
        if img_enc.shape[-1] != 3:
            raise ValueError(f"expected (..., 3) RGB, got shape {img_enc.shape}")
        wheels = [(w, h, s, l) for w, h, s, l in ((self.shadows, sh_h, sh_s, sh_l), (self.midtones, mi_h, mi_s, mi_l), (self.highlights, hi_h, hi_s, hi_l)) if s or l]
        glob = (gl_h, gl_s, gl_l) if gl_s or gl_l else None
        if not wheels and glob is None:
            return img_enc
        out = grade_tonal(img_enc, icc, wheels, balance, blend, glob)
        return out.astype(img_enc.dtype) if img_enc.dtype.kind == "f" else out