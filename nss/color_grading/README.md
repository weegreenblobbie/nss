# nss/color_grading - Lightroom Color Grading model

A native-space model of Adobe Lightroom's Color Grading panel: Shadows, Midtones, Highlights and Global wheels (Hue, Sat, Lum), Balance and
Blending.  No GUI code lives here (see `nss/color_gui`).

## Public API

```python
from nss.color_grading.grading import Grading

out = Grading().apply(img_enc, icc, balance=0, blend=50,
                      sh_h=210, sh_s=60, sh_l=-15, mi_h=30, mi_s=40, mi_l=10,
                      hi_h=45, hi_s=25, hi_l=25, gl_h=30, gl_s=20, gl_l=5)
```

* `img_enc`: the image as stored, encoded 0..1, shape `(..., 3)`; float32 / float64 dtype is preserved.
* `icc`: the ICC profile bytes the image is tagged with (`nss.tiff_file.TiffFile.iccprofile`).  It is required: the grade is computed in the image's
  own working space and sRGB is never assumed (`ValueError` without one).
* Units are Lightroom's: Hue 0..360 (0 red, 60 yellow, 120 green, 180 cyan, 240 blue, 300 magenta), Sat 0..100, Lum -100..100, Balance -100..100,
  Blending 0..100.  Wheels with Sat 0 and Lum 0 are skipped; with nothing active the input is returned unchanged.

## Model

Per channel, in linear ProPhoto light `l` with `u = logit(l)`, every wheel adds a log-odds shift `c`, and `l' = l e^c / (1 - l + l e^c)`.

* Tonal wheels: `c_w = C0 * M_w(u + d(Balance); Blending) * f_w`, `C0 = 2 atanh(0.28)`.  `M_w` is a measured mask table per wheel, interpolated in
  Blending (PCHIP over 0/25/50/75/100); Balance is a pure shift `d` of `u`.  Shadows / Highlights: `f = Sat/100 (v - w.v)` moved into a
  luminance-dependent window and soft-capped; Midtones: `f = Sat/100 v` (`v` = `hue_vector(Hue)`).  Lum: `2 atanh(0.0024 Lum)` times a measured
  ratio; Shadows Lum > 0 also lifts black.
* Global wheel: `c = c_sat * sat_mult(logit(brightest channel)) + 2 atanh(0.0024 Lum)` with `c_sat = 2 atanh(0.0028 Sat v)`; it is added to the
  tonal wheels' `c` in the same odds gain.  `sat_mult` boosts the Sat part near black and is 1 above `u = -4.25`.

## Files

* `grading.py` - `Grading`, the public API.
* `tonal.py` - the engine: `TonalWheel` (base of the three tonal wheels), `grade_tonal`, `working_matrix`.
* `math.py` - shared side-effect-free maths and constants (`logit`, `odds_gain`, `hue_vector`, `global_c`, `balance_shift`, `cap_shift`,
  `sat_mult`, `C0`, `PROPHOTO`, ...).  Research scripts in `tests/test_data` import from here too.
* `shadows.py`, `midtones.py`, `highlights.py`, `global_wheel.py` - **generated** tables, see below.
* `*_test.py` next to each module; shared fixtures (synthetic Adobe RGB profile, ramp builder) in `conftest.py`.

## Generated tables

The table modules are written by

    python -m tests.test_data.make_production            # default output: nss/color_grading

from the measurement CSVs in `tests/test_data` (`balance_blend.csv`, `cap_tables.csv`, `lum_tables.csv`, `dark_tables.csv`,
`global_dark_tables.csv`).  Never edit them by hand and never put tests in them: refit the CSV, then regenerate.

## Validation

Lightroom exports of `tests/test_data/synthetic_v2.tif` (neutral ramp, black / gray / white -> colour ramps, hue ribbon), compared as RMS in
16-bit DN (baselines on 2026-10-11):

    python -m tests.test_data.tone_model                 # tonal wheels over w_*.tif: overall about 432 DN
    python -m tests.test_data.global_check               # Global over g_*.tif / gx_*.tif; column 'prod' = production Grading.apply
    python -m tests.test_data.run_cinematic_validation   # one full grade: 741.5 DN
    python -m tests.test_data.make_production --selftest # production modules vs tone_model: must print PASSED
    python -m tests.test_data.tone_model --selftest      # mask extraction plumbing: must print PASSED
    pytest -m lightroom                                  # slow regression guard on the exports (overall RMS < 500 DN)

Any change to the maths or tables must leave these numbers unchanged unless the change is a deliberate, validated model update.
