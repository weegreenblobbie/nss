# Lightroom Color Grading: Reference Exports and Reproduction Guide

This document explains how the Lightroom reference images behind `nss/color_grading` were produced, what each family of files measures, and how to repeat the whole process (for example for a new Lightroom version, or to improve the weakest parts of the model). It is written so an AI agent can follow it with a human doing the Lightroom steps.

Contents: 1. Overview · 2. The input image · 3. Lightroom setup · 4. Export catalog · 5. How the exports became code · 6. Reproduction procedure · 7. Pitfalls · 8. Known weak spots

---

## 1. Overview

Goal: reproduce Adobe Lightroom's **Color Grading** panel (Shadows / Midtones / Highlights wheels, Global wheel, Balance, Blending, per-wheel Lum) in Python.

Method: black-box measurement.
1. Build one synthetic 16-bit image (`generate_synthetic_v2.py` -> `synthetic_v2.tif`) that contains every kind of pixel we need: neutral ramp, black->colour, gray->colour and white->colour gradients, a hue ribbon and anchor patches.
2. Import it into Lightroom, change **only** the Color Grading controls, export each setting as a 16-bit TIFF.
3. Compare each export with the input pixel by pixel, extract curves/tables, fit a compact model, and check it against held-out combinations.

Everything is measured on the same image, so any difference between input and export is caused by exactly one slider change.

## 2. The input image

`synthetic_v2.tif`: 2048 x 1200, 16-bit, **embedded ICC profile (Adobe RGB)**. Produced by `generate_synthetic_v2.py`; regenerate it rather than editing it.

| Rows | Band | Used for |
|---|---|---|
| 0-200 | neutral ramp (black -> white, gray) | masks, Balance/Blending, Lum, caps, Global dark end |
| 200-440 | black -> colour (pure R/G/B etc.) | tint on dark coloured pixels, clipping behaviour |
| 440-680 | gray -> colour | saturation scaling |
| 680-920 | white -> colour | highlight behaviour |
| 920-1050 | hue ribbon | hue mapping (hue -> per-channel gain) |
| 1050-1200 | anchors | patches at known levels (e.g. 0.50 gray midtone anchor) |

The band table is `BANDS` in the analysis code (`lr_io.py` in the archived scripts).

## 3. Lightroom setup (identical for every export)

- Import `synthetic_v2.tif`. Process version current, Profile **Adobe Color / Linear** untouched.
- **Everything else at default/0**: Exposure, Contrast, Tone Curve, HSL, Calibration, Optics, Detail, Effects, masks. Only the Color Grading panel changes.
- Color Grading defaults: Shadows/Midtones/Highlights/Global Sat 0, Lum 0; Blending 50; Balance 0. Wheels not under test must read exactly Sat 0, Lum 0.
- Export: **TIFF, 16-bit, uncompressed, no resize, no sharpening, colour space Adobe RGB (same as the input)**. The exported file must carry the same embedded profile as the input. Never assume sRGB; the code always converts through the embedded ICC.
- Before each export, **read the sliders back** (including Global Lum and Sat) rather than trusting the last state. Stale values from the previous file were the main source of bad data (see Pitfalls).
- Filename = the settings (convention below), so the analysis can parse them.

## 4. Export catalog

Files live next to the input (`tests/test_data/`). Hue is in degrees (Red 0, Green 120, Blue 240), Sat 0-100, Lum -100..100.

### 4.1 `w_*` — single tonal wheel on `synthetic_v2.tif`

`w_<wheel>_h<H>_s<S>[_l<L>][_bal<neg|pos><N>][_bl<N>].tif`, wheel = `sh` (Shadows), `mid`, `hi`. Other wheels off, Balance 0 and Blending 50 unless in the name.

- **Hue sweep at Sat 100**: h0, h60, h120, h180, h240, h300 for each wheel (hue -> per-channel gain map).
- **Saturation scaling**: Sat 25/50/75 at h240 and h120 for each wheel (confirms how effect scales with Sat).
- **Lum**: `w_<wheel>_h0_s0_l100` for each wheel (and Lum -100/-50/+50 sets for the lum tables).
- **Balance**: `_balneg100 / neg50 / neg25 / pos50 / pos75 / pos100` at h240 s25 (moves the mask centre).
- **Blending**: `_bl0 / bl25 / bl75 / bl100` at h240 s25 (changes mask width/shape).
- **Combinations (hold-out)**: `w_all_a.tif` = Shadows 240/100/0, Midtones 0/100/0, Highlights 120/100/0. `w_all_b.tif` = Shadows 60/50/+30, Midtones 300/75/-40, Highlights 180/100/+20.

### 4.2 Early gradient sets (`set_a_* ... set_e_*`, Phase 1-3)

First-generation measurements on a plain grayscale gradient (`input_synthetic.tif`, superseded by `synthetic_v2.tif`). They established the structure and are kept only for history:

- Set A: hue sweeps of each wheel at Sat 100 (`set_a_{sh,mid,hi}_h{0..300}`).
- Set B: Sat scaling at hue 0 and 240 (`set_b_mid_*`).
- Set C: Balance and Blending sweeps per wheel (`set_c_{sh,mid,hi}_bal*`, `_blend*`, plus the four corners Balance +-100 x Blending 0/100). Hue 240 (blue) was used as the "clip-free measuring stick" because it never clips in these tests.
- Set D: Lum sliders per wheel (`set_d_*_lum_*`) and midtone colour/Lum checks (`set_d_color_*`).
- Set E: ten random 3-wheel combinations with Blending/Balance (`set_e_rand_01..10`), settings recorded in the list in section 9. Used as a validation crucible.

### 4.3 `g_*` — Global wheel alone

`g_h<H>_s<S>_l<L>.tif`; all tonal wheels off.

- Hue at Sat 25: h0, 15, 30, 45, 60, 90, 105, 120, 150, 180, 195, 210, 240, 270, 300, 330.
- Sat scale at h240: s10/25/50/75/100; Sat 100 at h0 and h120.
- Lum linearity: `g_h0_s0_l{-100,-50,-25,25,50,100}`.
- Additivity check: `g_h240_s50_l50`.

### 4.4 `gx_*` — Global combined with a tonal wheel

Tests that Global and tonal wheels add (they do; see section 5).

- `gx_h0_s0_l50_{sh,mid,hi}_h240_s50` (Global Lum +50 only).
- `gx_h0_s50_{sh,mid,hi}_h120_s50` (Global hue 0, Sat 50).
- `gx_h240_s50_{sh,mid,hi}_h120_s50` (Global hue 240, Sat 50).
- `gx_all_a.tif`: Global 30/40/+10, Shadows 210/50/-10, Midtones 60/30/0, Highlights 40/40/+15, Balance 0, Blending 50.

Name pattern for the tonal part: `_<wheel>_h<H>_s<S>`; the Global part is the leading `h/s/l` group.

Note: the `gx_h240_s50_*` set was exported with a stale Global Lum and is slightly off; re-export with Global Lum 0 for a clean set.

## 5. How the exports became code

The final model lives in `nss/color_grading/` (tables inline; `make_production.py`, if kept, regenerates them from CSVs). Summary of what each family fed, and of the maths:

**Space.** Convert encoded RGB to linear light through the embedded ICC, then to linear ProPhoto (the space where Lightroom's grading behaves like a per-channel gain). Work with level `l` in 0..1 and `u = logit(l)`.

**Core operator ("odds gain").** For each channel, `l' = l * e^c / (1 - l + l * e^c)`. Gains compose by adding `c`. This keeps 0 and 1 fixed (no clipping) and matches Lightroom's behaviour.

**Tonal wheels.** `c = C0 * M(u + d(Balance); Blending) * f`, `C0 = 2*atanh(0.28) = 0.5763`.
- `M` = mask (Shadows/Midtones/Highlights) as a function of level. Measured from the neutral ramp with hue 240 at Sat 100 (`set_c_*` / `w_*_bal*_bl*`); Balance shifts it, Blending changes its shape (table over a Balance x Blending grid, interpolated).
- `f` = hue vector x Sat/100. Hue vector from the 6-point hue sweeps (hexagon, R 0, G 120, B 240, +-1). Shadows/Highlights use `f = Sat/100 * (v - w.v)` with `w = (0.288, 0.712, 0)` (luminance weighting); Midtones use `Sat/100 * v`.
- **Caps**: Shadows/Highlights saturate at high Sat; a window/soft-cap table (`cap_tables`) from the Sat 25/50/75/100 series, with highlights using `A*tanh(y/A)`, `A ~ 1.554`.
- **Lum**: ratio table x mask (`lum_tables`), from the `*_s0_l*` files. Shadows Lum > 0 also adds a small black lift.
- **Dark end** (`dark_tables`): corrections near black from the dark parts of the ramps.

**Global wheel.** `c = 2*atanh(0.0028*Sat*v(hue)) + 2*atanh(0.0024*Lum)`, added to the tonal wheels' `c` in the same odds gain (verified: fitted weights a = b = 1.000, residual ~0 on `gx_h0_s0_l50_*`). Sat boost near black: the raised channel is multiplied by `m(u)` (~2.3 near black falling to 1.0 by u ~ -6; separate tables for raised and lowered channels), looked up at the pixel's **brightest channel**. A small additive Lum residual was measured but not used (it only helped the neutral ramp and hurt coloured dark pixels).

**Accuracy reached** (RMS, 16-bit DN; 257 DN = 1 eight-bit code): Global alone ~130 DN overall (~240 DN worst case in the darkest pixels); validation sets ~430 DN overall; weakest cases listed in section 8.

## 6. Reproduction procedure (for an AI agent + a human operator)

Agent: run analysis and write code. Human: operate Lightroom.

1. **Generate the input.** `python generate_synthetic_v2.py` -> `synthetic_v2.tif`. Check the profile (Adobe RGB) and the band rows.
2. **Import and verify Lightroom state.** Human imports the TIFF, resets all develop settings, confirms Color Grading is at defaults.
3. **Export in batches, in this order** (each batch validates before the next):
   1. One wheel, one hue (e.g. `w_mid_h240_s100`) to check the pipeline: output must differ from the input only in the tint.
   2. Neutral-ramp families: hue sweeps -> Balance/Blending sweeps -> Lum.
   3. Sat scaling and caps (`s25/50/75/100`).
   4. Global alone (`g_*`).
   5. Combinations (`gx_*`, `w_all_*`) **last, as hold-out**; never fit on them.
   The agent supplies the exact filename list for each batch (ideally as a checklist file with each slider setting), and the human ticks them off. The agent should ask the human to re-export any file whose measured tint contradicts its filename.
4. **Load pairs with ICC conversion only.** Read input and export as float, convert both through the embedded profile to linear ProPhoto; refuse to compare if the profiles differ.
5. **Extract per-channel `c`** from `logit(l_out) - logit(l_in)` on the pixels where no channel is clipped; use the neutral ramp for masks and caps, the hue ribbon for hue maps, black->colour and gray->colour bands to confirm channel independence.
6. **Fit tables on the full 801-point `u` grid** (`linspace(-16, 4, 801)`); coarser grids caused hundreds of DN of error in the shadows. Fit in **encoded output space** (least squares) wherever the export clips a channel (Adobe RGB clips lowered channels near 0 and 1).
7. **Validate** on the hold-out files; report overall RMS and a per-band breakdown (ramp, black->colour, gray->colour, white->colour, ribbon). A change is accepted only if it does not worsen the whole image, even if one band improves.
8. **Generate the production tables** and regression tests against a handful of exports; commit.

## 7. Pitfalls we hit

- **Stale slider values.** Six `gx_*` files were wrong because a previous Global Lum +50 or Sat 0 stayed set. Diagnose by looking for a constant leftover `c` in the residuals. Always read all sliders before exporting.
- **Clipped channels.** Adobe RGB export clips lowered channels near black/white; their measured `c` is unreliable. Fit in encoded space or use only unclipped channels. Hue 240 is a good clip-free probe for tonal wheels.
- **Corrections measured on gray do not transfer to coloured dark pixels** (one channel at 0, another varying). Always check a correction on the black->colour band before keeping it.
- **Window ratio blow-ups.** Dividing a window by a mask near zero created 3000 DN jumps ("black chunks"). Hold the nearest reliable ratio where the mask is below 5% of its maximum.
- **Grid resolution.** Use the full 801-point grid; do not downsample.
- **Never assume sRGB**, and never let the TIFF reader modify pixels.
- Evaluate "is this visible" in 8-bit codes (DN / 257): below ~1 code is invisible, above ~5 shows in flat dark areas.

## 8. Known weak spots

Where the model is least accurate (candidates for another measurement round):

- Shadows at Sat >= 50: 500-1400 DN.
- `w_all_b` (strong Lum with cross-wheel mixing): ~1400 DN, mostly dark end and high end.
- Highlights at hue 120 in some bands.
- Balance/Blending effect on the caps and the black lift (not independently measured).
- Global Sat dark end for lowered channels (about 100-360 DN at enc 0.02-0.05 at Sat 100).
- Lum scaling checked at only a few values; more Lum values per wheel would tighten it.

## 9. Appendix: Phase 1 random validation set (Set E)

Hue / Sat / Lum per wheel, plus Blending and Balance. Exports `set_e_rand_01..10.tif`.

| # | Blend | Bal | Shadows H/S/L | Midtones H/S/L | Highlights H/S/L |
|---|---|---|---|---|---|
| 01 | 81 | -72 | 12/94/-30 | 125/28/-65 | 52/86/+89 |
| 02 | 69 | -78 | 302/54/-92 | 15/11/-45 | 119/64/+54 |
| 03 | 3 | +43 | 112/57/+50 | 359/69/+7 | 101/91/+66 |
| 04 | 35 | -99 | 81/89/+8 | 174/35/-61 | 110/97/-14 |
| 05 | 13 | -77 | 194/12/-9 | 176/77/-33 | 22/93/+17 |
| 06 | 68 | -69 | 193/10/+41 | 150/80/+58 | 185/73/-51 |
| 07 | 90 | -83 | 23/84/-42 | 148/10/-41 | 51/48/-29 |
| 08 | 58 | +62 | 186/20/-6 | 181/26/+71 | 136/89/+74 |
| 09 | 82 | -82 | 311/81/-57 | 273/93/-38 | 83/59/-3 |
| 10 | 34 | +63 | 352/71/-44 | 350/41/+96 | 28/29/-92 |