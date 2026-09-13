# Color Grading Math Validation Plan

To validate our custom color grading math against an industry-standard engine like Adobe Lightroom, we need a controlled test image to eliminate the unpredictable variables of real photographs (like noise and mixed lighting).

## 1. The Ideal Test Image
Generate a purely mathematical, lossless test image (save it as a 16-bit PNG or uncompressed TIFF to prevent compression artifacts from skewing the math). 

The image must contain:
* **A Continuous Grayscale Ramp:** A smooth gradient from pure black (0) to pure white (255). This is critical for visualizing exactly where the Shadows, Midtones, and Highlights zones start, peak, and fade out.
* **A 50% Neutral Gray Block:** To cleanly measure hue and saturation injections without existing color interference.
* **Pure RGB & CMY Blocks:** Solid patches of Red, Green, Blue, Cyan, Magenta, and Yellow (at 100% saturation and 50% luminance). This validates that the color rotation math shifts existing colors accurately.

## 2. Lightroom Setup & Mapping
Lightroom’s "Color Grading" panel maps perfectly to our UI. Before running the tests, ensure your baseline matches Lightroom's default state:
* **Wheels:** Shadows, Midtones, Highlights (Hue and Saturation).
* **Global Controls:** Blending (how smoothly the zones overlap) and Balance (shifting the center point of what is considered a shadow vs. a highlight). 

---

## 3. The Validation Test Cases

Export the test image from Lightroom with the following specific settings applied, then run the Python engine with the matching UI parameters and compare the output arrays.

### Test Case 1: Tonal Isolation (The Zone Check)
* **Goal:** Verify that our Shadow, Midtone, and Highlight mathematical masks match standard tonal definitions.
* **LR Settings:** 
  * Shadows: Hue = 240 (Blue), Saturation = 100 (or 1.0)
  * Midtones & Highlights: Saturation = 0
  * Blending = Default (usually 50)
  * Balance = Default (ususally 0)
* **Validation:** Look at the grayscale ramp. The blue should peak in the dark grays and smoothly fade out to 0 before hitting the middle of the ramp. Repeat this exclusively for Midtones (peaks at 50% gray) and Highlights (peaks in the light grays).
  * reference_color_grading_input.tiff (original generated input image)
  * case_1_shadows.tif
  * case_1_midtones.tif
  * case_2_highlights.tif

### Test Case 2: Saturation Linearity (The Bug Check)
* **Goal:** Validate that our recent saturation fix scales exactly like a commercial engine.
* **LR Settings:**
  * Apply to Midtones only. Hue = 0 (Red).
  * Export 3 versions: Saturation at 10% (0.10), 50% (0.50), and 100% (1.0).
* **Validation:** Measure the pixel values of the 50% Gray block. The RGB values should shift away from neutral gray linearly across the three exports.
  * reference_color_grading_input.tiff (original generated input image)
  * case_2_10.tif
  * case_2_50.tif
  * case_2_100.tif

### Test Case 3: Global Rotation Math
* **Goal:** Verify the new offset slider rotates palettes accurately.
* **LR Settings:**
  * Set Shadows to Hue = 240 (Blue), Saturation = 50.
  * Set Midtones to Hue = 0 (Red), Saturation = 50.
  * *Note:* Since Lightroom doesn't have a single "Rotate" slider, manually calculate the offset: change Shadows to Hue = 330 (which is 240 + 90), and Midtones to Hue = 90 (which is 0 + 90).
  * *Note:* Since Lightroom doesn't have a single "Rotate" slider, manually calculate the offset: change Shadows to Hue = 60 (which is (330 + 90) % 360), and Midtones to Hue = 180 (which is 90 + 90).  
* **Validation:** Input the original (0 and 240) into the Python UI, then set the Rotation slider to +90. The output should mathematically match the Lightroom export.
  * reference_color_grading_input.tiff (original generated input image)
  * case_3_rot_0.tif
  * case_3_rot_90.tif
  * case_3_rot_180.tif
 
### Test Case 4: The Harmony Clash (Opposing Zones)
* **Goal:** Ensure pixels don't clip, invert, or produce artifacts when zones overlap with opposing colors (Complementary harmony).
* **LR Settings:**
  * Shadows: Hue = 30 (Orange), Sat = 50.
  * Highlights: Hue = 210 (Teal), Sat = 50.
  * Midtones: Sat = 0.
* **Validation:** Check the grayscale ramp around the 40-60% brightness mark. The transition from Orange to Teal should seamlessly cross through neutral gray without creating harsh bands, color noise, or sudden brightness dips.
  * reference_color_grading_input.tiff (original generated input image)
  * case_4.tif

### Test Case 5: Balance and Blending Sweeps
* **Goal:** Validate the global sliders we isolated from the randomizer.
* **LR Settings:**
  * Set a heavy color cast (e.g., Shadows: Hue = 240, Saturation = 100, Midtones: Hue = 120, Saturation = 100, Highlights: Hue = 0, Saturation = 100).
  * Sent Balance to the default (usually 0)
  * Export one image with Blending = 0, 33, 66, 100 (hard masking, soft, really soft, extra soft)
  * Reset Blending to the default (usually 50).
  * Export one image with Balance = -100, -50, 50, 100 (pushes Midtones into the shadows), and one with Balance = +100 (pushes Midtones into the highlights).
* **Validation:** The color cast on the grayscale ramp should physically shift left and right, proving the crossover points update correctly based on the global sliders. 
  * reference_color_grading_input.tiff (original generated input image)
  * case_5_blend_0.tif
  * case_5_blend_33.tif
  * case_5_blend_66.tif
  * case_5_blend_100.tif
  * case_5_bal_neg100.tif
  * case_5_bal_neg50.tif
  * case_5_bal_pos50.tif
  * case_5_bal_pos100.tif

*Success Criteria: If our non-square numpy arrays can process these exact UI parameters and return arrays with a minimal delta (e.g., within 1-2 integer values out of 255 due to standard float rounding) compared to the Lightroom exports, our custom engine's math is validated.*