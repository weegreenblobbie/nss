# Lightroom Color Grading: Final Extraction Protocol

**Input Image:** The 16-bit `tests/test_data/input_synthetic.tif` (2048x1200).
**Export Format:** 16-bit TIFF, Uncompressed, sRGB or ProPhoto RGB (ensure it exactly matches the input color space).
**Prerequisites:**

* Ensure **ALL** other Lightroom Develop settings (Exposure, Contrast, Profile, Tone Curve, Calibration, Optics) are strictly at `0` or disabled.
* Only manipulate the Color Grading panel.
* Unless explicitly stated in a section, assume inactive wheels have `Sat = 0, Lum = 0`, `Blending = 50`, and `Balance = 0`.

---

## Set A: The Chrominance Matrix (18 Images)

**Purpose:** Maps the 3x2 matrix conversion from UI polar coordinates (Hue) to exact $\Delta RGB$ shifts for each individual tonal mask.

### 1. Shadows Wheel Sweep

* **Base Settings:** Midtones Off, Highlights Off. Blending = 50, Balance = 0.
* **Shadows Settings:** Saturation = 100, Luminance = 0.
* **Adjust:** Shadows Hue slider.
* `set_a_sh_h0.tif` (Hue = 0)
* `set_a_sh_h60.tif` (Hue = 60)
* `set_a_sh_h120.tif` (Hue = 120)
* `set_a_sh_h180.tif` (Hue = 180)
* `set_a_sh_h240.tif` (Hue = 240)
* `set_a_sh_h300.tif` (Hue = 300)

### 2. Midtones Wheel Sweep

* **Base Settings:** Shadows Off, Highlights Off. Blending = 50, Balance = 0.
* **Midtones Settings:** Saturation = 100, Luminance = 0.
* **Adjust:** Midtones Hue slider.
* `set_a_mid_h0.tif` (Hue = 0)
* `set_a_mid_h60.tif` (Hue = 60)
* `set_a_mid_h120.tif` (Hue = 120)
* `set_a_mid_h180.tif` (Hue = 180)
* `set_a_mid_h240.tif` (Hue = 240)
* `set_a_mid_h300.tif` (Hue = 300)

### 3. Highlights Wheel Sweep

* **Base Settings:** Shadows Off, Midtones Off. Blending = 50, Balance = 0.
* **Highlights Settings:** Saturation = 100, Luminance = 0.
* **Adjust:** Highlights Hue slider.
* `set_a_hi_h0.tif` (Hue = 0)
* `set_a_hi_h60.tif` (Hue = 60)
* `set_a_hi_h120.tif` (Hue = 120)
* `set_a_hi_h180.tif` (Hue = 180)
* `set_a_hi_h240.tif` (Hue = 240)
* `set_a_hi_h300.tif` (Hue = 300)

---

## Set B: Saturation Scaling & Gamut Protection (6 Images)

**Purpose:** Determines if UI Saturation scales linearly, and measures how Lightroom attenuates the tint on pixels that are already highly saturated (Gamut roll-off).

### 1. Midtones Red Saturation

* **Base Settings:** Shadows Off, Highlights Off. Blending = 50, Balance = 0.
* **Midtones Settings:** Hue = 0, Luminance = 0.
* **Adjust:** Midtones Saturation slider.
* `set_b_mid_red_s25.tif` (Sat = 25)
* `set_b_mid_red_s50.tif` (Sat = 50)
* `set_b_mid_red_s75.tif` (Sat = 75)

### 2. Midtones Blue Saturation

* **Base Settings:** Shadows Off, Highlights Off. Blending = 50, Balance = 0.
* **Midtones Settings:** Hue = 240, Luminance = 0.
* **Adjust:** Midtones Saturation slider.
* `set_b_mid_blue_s25.tif` (Sat = 25)
* `set_b_mid_blue_s50.tif` (Sat = 50)
* `set_b_mid_blue_s75.tif` (Sat = 75)

---

## Set C: Blending and Balance Deformation (12 Images)

**Purpose:** Mathematically defines how the Blending slider alters mask width (standard deviation) and how the Balance slider shifts the mask center point ($\mu$).

### 1. Blending Sweep

* **Base Settings:** Shadows Off, Highlights Off. Balance = 0.
* **Midtones Settings:** Hue = 240, Saturation = 100, Luminance = 0.
* **Adjust:** Global Blending slider.
* `set_c_mid_blend_0.tif` (Blending = 0)
* `set_c_mid_blend_25.tif` (Blending = 25)
* `set_c_mid_blend_75.tif` (Blending = 75)
* `set_c_mid_blend_100.tif` (Blending = 100)

### 2. Balance Sweep (Midtones)

* **Base Settings:** Shadows Off, Highlights Off. Blending = 50.
* **Midtones Settings:** Hue = 240, Saturation = 100, Luminance = 0.
* **Adjust:** Global Balance slider.
* `set_c_mid_bal_neg100.tif` (Balance = -100)
* `set_c_mid_bal_neg50.tif` (Balance = -50)
* `set_c_mid_bal_pos50.tif` (Balance = +50)
* `set_c_mid_bal_pos100.tif` (Balance = +100)

### 3. Balance Edge Cases (Shadows)

* **Base Settings:** Midtones Off, Highlights Off. Blending = 50.
* **Shadows Settings:** Hue = 240, Saturation = 100, Luminance = 0.
* **Adjust:** Global Balance slider.
* `set_c_sh_bal_neg100.tif` (Balance = -100)
* `set_c_sh_bal_neg50.tif` (Balance = -50)
* `set_c_sh_bal_pos25.tif` (Balance = +25)
* `set_c_sh_bal_pos50.tif` (Balance = +50)
* `set_c_sh_bal_pos60.tif` (Balance = +60)
* `set_c_sh_bal_pos70.tif` (Balance = +70)
* `set_c_sh_bal_pos80.tif` (Balance = +80)
* `set_c_sh_bal_pos90.tif` (Balance = +90)
* `set_c_sh_bal_pos100.tif` (Balance = +100)

### 4. Balance Edge Cases (Highlights)

* **Base Settings:** Shadows Off, Midtones Off. Blending = 50.
* **Highlights Settings:** Hue = 240, Saturation = 100, Luminance = 0.
* **Adjust:** Global Balance slider.
* `set_c_hi_bal_neg100.tif` (Balance = -100)
* `set_c_hi_bal_neg50.tif` (Balance = -50)
* `set_c_hi_bal_pos50.tif` (Balance = +50)
* `set_c_hi_bal_pos100.tif` (Balance = +100)

### 5. Blending Sweep (Shadows)

* **Base Settings:** Midtones Off, Highlights Off. Balance = 0.
* **Midtones Settings:** Hue = 240, Saturation = 100, Luminance = 0.
* **Adjust:** Global Blending slider.
* `set_c_sh_blend_0.tif` (Blending = 0)
* `set_c_sh_blend_5.tif` (Blending = 5)
* `set_c_sh_blend_10.tif` (Blending = 10)
* `set_c_sh_blend_15.tif` (Blending = 15)
* `set_c_sh_blend_25.tif` (Blending = 25)
* `set_c_sh_blend_75.tif` (Blending = 75)
* `set_c_sh_blend_100.tif` (Blending = 100)

---

## Set D: Wheel Luminance Sliders (6 Images)

**Purpose:** Maps the specific luminosity curve added/subtracted when adjusting the lightness of a specific tonal zone.

### 1. Shadows Luminance

* **Base Settings:** Midtones Off, Highlights Off. Blending = 50, Balance = 0.
* **Shadows Settings:** Hue = 0, Saturation = 0.
* **Adjust:** Shadows Luminance slider.
* `set_d_sh_lum_neg100.tif` (Luminance = -100)
* `set_d_sh_lum_neg50.tif` (Luminance = -50)
* `set_d_sh_lum_pos50.tif` (Luminance = +50)
* `set_d_sh_lum_pos100.tif` (Luminance = +100)

### 2. Midtones Luminance

* **Base Settings:** Shadows Off, Highlights Off. Blending = 50, Balance = 0.
* **Midtones Settings:** Hue = 0, Saturation = 0.
* **Adjust:** Midtones Luminance slider.
* `set_d_mid_lum_neg100.tif` (Luminance = -100)
* `set_d_mid_lum_neg50.tif` (Luminance = -50)
* `set_d_mid_lum_pos50.tif` (Luminance = +50)
* `set_d_mid_lum_pos100.tif` (Luminance = +100)

### 3. Highlights Luminance

* **Base Settings:** Shadows Off, Midtones Off. Blending = 50, Balance = 0.
* **Highlights Settings:** Hue = 0, Saturation = 0.
* **Adjust:** Highlights Luminance slider.
* `set_d_hi_lum_neg100.tif` (Luminance = -100)
* `set_d_hi_lum_neg50.tif` (Luminance = -50)
* `set_d_hi_lum_pos50.tif` (Luminance = +50)
* `set_d_hi_lum_pos100.tif` (Luminance = +100)

---

## Set E: The Validation Crucible (10 Images)

**Purpose:** Proves our final mathematical model against random, complex combinations. Prevents overfitting to isolated variables.

### Random Combinations

* **Instructions:** Use a random number generator to pick values for Hue (0-360), Saturation (0-100), and Luminance (-100 to +100) across all 3 wheels simultaneously, along with random Blending (0-100) and Balance (-100 to +100).
* **Important:** Save a JSON or text file alongside these images documenting the exact settings used for each export.
* **Exports:**
* `set_e_rand_01.tif`
* `set_e_rand_02.tif`
* `set_e_rand_03.tif`
* `set_e_rand_04.tif`
* `set_e_rand_05.tif`
* `set_e_rand_06.tif`
* `set_e_rand_07.tif`
* `set_e_rand_08.tif`
* `set_e_rand_09.tif`
* `set_e_rand_10.tif`

 #### 1. set_e_rand_01.tif
  * Global Controls: Blending = 81, Balance = -72
  * Shadows: Hue = 12, Saturation = 94, Luminance = -30
  * Midtones: Hue = 125, Saturation = 28, Luminance = -65
  * Highlights: Hue = 52, Saturation = 86, Luminance = +89

  #### 2. set_e_rand_02.tif
  * Global Controls: Blending = 69, Balance = -78
  * Shadows: Hue = 302, Saturation = 54, Luminance = -92
  * Midtones: Hue = 15, Saturation = 11, Luminance = -45
  * Highlights: Hue = 119, Saturation = 64, Luminance = +54

  #### 3. set_e_rand_03.tif
  * Global Controls: Blending = 3, Balance = +43
  * Shadows: Hue = 112, Saturation = 57, Luminance = +50
  * Midtones: Hue = 359, Saturation = 69, Luminance = +7
  * Highlights: Hue = 101, Saturation = 91, Luminance = +66

  #### 4. set_e_rand_04.tif
  * Global Controls: Blending = 35, Balance = -99
  * Shadows: Hue = 81, Saturation = 89, Luminance = +8
  * Midtones: Hue = 174, Saturation = 35, Luminance = -61
  * Highlights: Hue = 110, Saturation = 97, Luminance = -14

  #### 5. set_e_rand_05.tif
  * Global Controls: Blending = 13, Balance = -77
  * Shadows: Hue = 194, Saturation = 12, Luminance = -9
  * Midtones: Hue = 176, Saturation = 77, Luminance = -33
  * Highlights: Hue = 22, Saturation = 93, Luminance = +17

  #### 6. set_e_rand_06.tif

  * Global Controls: Blending = 68, Balance = -69
  * Shadows: Hue = 193, Saturation = 10, Luminance = +41
  * Midtones: Hue = 150, Saturation = 80, Luminance = +58
  * Highlights: Hue = 185, Saturation = 73, Luminance = -51

  #### 7. set_e_rand_07.tif

  * Global Controls: Blending = 90, Balance = -83
  * Shadows: Hue = 23, Saturation = 84, Luminance = -42
  * Midtones: Hue = 148, Saturation = 10, Luminance = -41
  * Highlights: Hue = 51, Saturation = 48, Luminance = -29

  #### 8. set_e_rand_08.tif

  * Global Controls: Blending = 58, Balance = +62
  * Shadows: Hue = 186, Saturation = 20, Luminance = -6
  * Midtones: Hue = 181, Saturation = 26, Luminance = +71
  * Highlights: Hue = 136, Saturation = 89, Luminance = +74

  #### 9. set_e_rand_09.tif

  * Global Controls: Blending = 82, Balance = -82
  * Shadows: Hue = 311, Saturation = 81, Luminance = -57
  * Midtones: Hue = 273, Saturation = 93, Luminance = -38
  * Highlights: Hue = 83, Saturation = 59, Luminance = -3

  #### 10. set_e_rand_10.tif

  * Global Controls: Blending = 34, Balance = +63
  * Shadows: Hue = 352, Saturation = 71, Luminance = -44
  * Midtones: Hue = 350, Saturation = 41, Luminance = +96
  * Highlights: Hue = 28, Saturation = 29, Luminance = -92