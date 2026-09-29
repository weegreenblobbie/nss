# The Ultimate Color Grading Training & Validation Curriculum

To permanently prevent overfitting and ensure our custom color grading math perfectly mimics Adobe Lightroom, this curriculum forces the Genetic Algorithm (GA) to map the entire color wheel (solving the 6-point hue spline) and uses holdout images to verify the math generalizes to real-world pixel distributions.

## Phase 1: The Base Images (The Anchors)
We need three base images. The GA will train heavily on the synthetic image, but we will use the real images to verify it generalizes to real-world photography (where midtones and shadows blend unpredictably).

*Note: All images and exports MUST be 16-bit uncompressed TIFFs to prevent compression artifacts from skewing the RMSE.*

*   **`input_synthetic.tif`**: A purely mathematical chart containing:
    *   A smooth grayscale ramp (0 to 255).
    *   A 50% neutral gray block.
    *   Pure RGB & CMY blocks (Red, Green, Blue, Cyan, Magenta, Yellow at 100% saturation, 50% luminance).
*   **`input_portrait.tif`**: A high-quality photo of a person (tests complex skin tones, warm midtones, and smooth highlight falloff).
*   **`input_landscape.tif`**: A high-dynamic-range photo (tests deep crushed shadows in trees/rocks, bright sky, high contrast).

---

## Phase 2: The Training Set (Fed to the GA)
These images are actively evaluated by the GA to find the structural parameters and hue weights. 

### Test Suite A: The Spline Calibration (18 Exports)
*Goal: Lock in the structural curves (Centers, Widths, Gains) and the exact Hue Multipliers for all three zones across the entire color wheel.*

**LR Settings:** Blending = `50`, Balance = `0`. 
Apply the following to `input_synthetic.tif`:

**Shadows Isolated (Sat = 100, Mids/Highs Sat = 0)**
*   `train_sh_0.tif` (Hue = 0 / Red)
*   `train_sh_60.tif` (Hue = 60 / Yellow)
*   `train_sh_120.tif` (Hue = 120 / Green)
*   `train_sh_180.tif` (Hue = 180 / Cyan)
*   `train_sh_240.tif` (Hue = 240 / Blue)
*   `train_sh_300.tif` (Hue = 300 / Magenta)

**Midtones Isolated (Sat = 100, Shadows/Highs Sat = 0)**
*   `train_mid_0.tif` to `train_mid_300.tif` (Same 6 hues as above)

**Highlights Isolated (Sat = 100, Shadows/Mids Sat = 0)**
*   `train_hi_0.tif` to `train_hi_300.tif` (Same 6 hues as above)

*(Execution Note: The GA will load all 6 images for a given zone at once, evaluate the parameters against the whole color wheel simultaneously, and return the average RMSE. This eliminates mathematical loopholes).*

### Test Suite B: Global Modifiers Calibration (36 Exports)
*Goal: Train Blending and Balance. We sweep each zone in complete isolation to watch how the global sliders stretch and shift their specific curves. Finally, we turn all three on to solve the additive overlap.*

**All exports in this suite are applied to: `input_synthetic.tif`**

**1. Midtone-Only Balance Sweep**
*Goal: Tracks how Balance shifts the Midtone center of gravity.*
*   **Locked Settings:** 
    *   Shadows: Hue = 0, Saturation = 0
    *   Midtones: Hue = 0, Saturation = 100
    *   Highlights: Hue = 0, Saturation = 0
    *   Blending = 50
*   **Exports:**
    *   `train_bal_mid_neg100.tif` (Balance = -100)
    *   `train_bal_mid_neg50.tif` (Balance = -50)
    *   `train_bal_mid_pos50.tif` (Balance = +50)
    *   `train_bal_mid_pos100.tif` (Balance = +100)

**2. Midtone-Only Blending Sweep**
*Goal: Tracks how Blending stretches or narrows the Midtone width.*
*   **Locked Settings:** 
    *   Shadows: Hue = 0, Saturation = 0
    *   Midtones: Hue = 0, Saturation = 100
    *   Highlights: Hue = 0, Saturation = 0
    *   Balance = 0
*   **Exports:**
    *   `train_blend_mid_0.tif` (Blending = 0)
    *   `train_blend_mid_25.tif` (Blending = 25)
    *   `train_blend_mid_75.tif` (Blending = 75)
    *   `train_blend_mid_100.tif` (Blending = 100)

**3. Shadow-Only Balance Sweep**
*Goal: Tracks how Balance shifts the Shadow center of gravity.*
*   **Locked Settings:** 
    *   Shadows: Hue = 240, Saturation = 100
    *   Midtones: Hue = 0, Saturation = 0
    *   Highlights: Hue = 0, Saturation = 0
    *   Blending = 50
*   **Exports:**
    *   `train_bal_sh_neg100.tif` (Balance = -100)
    *   `train_bal_sh_neg50.tif` (Balance = -50)
    *   `train_bal_sh_pos50.tif` (Balance = +50)
    *   `train_bal_sh_pos100.tif` (Balance = +100)

**4. Shadow-Only Blending Sweep**
*Goal: Tracks how Blending stretches or narrows the Shadow width.*
*   **Locked Settings:** 
    *   Shadows: Hue = 240, Saturation = 100
    *   Midtones: Hue = 0, Saturation = 0
    *   Highlights: Hue = 0, Saturation = 0
    *   Balance = 0
*   **Exports:**
    *   `train_blend_sh_0.tif` (Blending = 0)
    *   `train_blend_sh_25.tif` (Blending = 25)
    *   `train_blend_sh_75.tif` (Blending = 75)
    *   `train_blend_sh_100.tif` (Blending = 100)

**5. Highlight-Only Balance Sweep**
*Goal: Tracks how Balance shifts the Highlight center of gravity.*
*   **Locked Settings:** 
    *   Shadows: Hue = 0, Saturation = 0
    *   Midtones: Hue = 0, Saturation = 0
    *   Highlights: Hue = 60, Saturation = 100
    *   Blending = 50
*   **Exports:**
    *   `train_bal_hi_neg100.tif` (Balance = -100)
    *   `train_bal_hi_neg50.tif` (Balance = -50)
    *   `train_bal_hi_pos50.tif` (Balance = +50)
    *   `train_bal_hi_pos100.tif` (Balance = +100)

**6. Highlight-Only Blending Sweep**
*Goal: Tracks how Blending stretches or narrows the Highlight width.*
*   **Locked Settings:** 
    *   Shadows: Hue = 0, Saturation = 0
    *   Midtones: Hue = 0, Saturation = 0
    *   Highlights: Hue = 60, Saturation = 100
    *   Balance = 0
*   **Exports:**
    *   `train_blend_hi_0.tif` (Blending = 0)
    *   `train_blend_hi_25.tif` (Blending = 25)
    *   `train_blend_hi_75.tif` (Blending = 75)
    *   `train_blend_hi_100.tif` (Blending = 100)

**7. The Trinity Overlap (Final Overlap Calibration)**
*Goal: The ultimate test of all zones interacting at maximum saturation. By using perfectly equidistant RGB primaries (0, 120, 240), we stress-test the mathematical symmetry of the blending overlaps.*
*   **Locked Settings:** 
    *   Shadows: Hue = 240, Saturation = 100
    *   Midtones: Hue = 0, Saturation = 100
    *   Highlights: Hue = 120, Saturation = 100
*   **Exports:**
    *   `train_trinity_blend_0.tif` (Blending = 0, Balance = 0)
    *   `train_trinity_blend_100.tif` (Blending = 100, Balance = 0)
    *   `train_trinity_bal_neg100.tif` (Blending = 50, Balance = -100)
    *   `train_trinity_bal_pos100.tif` (Blending = 50, Balance = +100)

---

## Phase 3: The Generalization Set (The True Test)
**These images are NEVER seen by the Genetic Algorithm.** 
Once the GA has output its final Python dictionary of optimized parameters, we plug them into our engine and process the base images. If our RMSE is extremely low compared to these Lightroom exports, the math is officially solved.

### Validation 1: The "Cinematic Teal & Orange"
*Goal: Test complex skin-tone preservation and heavy shadow pushing.*
*   **LR Settings:** 
    *   Shadows: Hue = 220, Sat = 60
    *   Midtones: Hue = 35, Sat = 45
    *   Highlights: Sat = 0
    *   Blending = 60, Balance = -15
*   **Exports:**
    *   `val_cinematic_synthetic.tif` (Applied to `input_synthetic.tif`)
    *   `val_cinematic_portrait.tif` (Applied to `input_portrait.tif`)
    *   `val_cinematic_landscape.tif` (Applied to `input_landscape.tif`)

### Validation 2: The "Vintage Pastel"
*Goal: Test highlight coloration and high-luminance overlaps.*
*   **LR Settings:**
    *   Shadows: Hue = 320 (Magenta), Sat = 20
    *   Midtones: Sat = 0
    *   Highlights: Hue = 50 (Warm Yellow), Sat = 50
    *   Blending = 100 (Maximum overlap), Balance = +25
*   **Exports:**
    *   `val_vintage_synthetic.tif` (Applied to `input_synthetic.tif`)
    *   `val_vintage_portrait.tif` (Applied to `input_portrait.tif`)
    *   `val_vintage_landscape.tif` (Applied to `input_landscape.tif`)

### Validation 3: The "Toxic Wash" (Edge Case Stress Test)
*Goal: Test extreme non-complementary colors. Verifies our math doesn't clip, tear, or break RGB bounds when maxed out with hard boundaries.*
*   **LR Settings:**
    *   Shadows: Hue = 120 (Green), Sat = 100
    *   Midtones: Hue = 270 (Purple), Sat = 100
    *   Highlights: Hue = 0 (Red), Sat = 100
    *   Blending = 0 (Hard mathematical boundaries), Balance = 0
*   **Exports:**
    *   `val_toxic_synthetic.tif` (Applied to `input_synthetic.tif`)
    *   `val_toxic_portrait.tif` (Applied to `input_portrait.tif`)
    *   `val_toxic_landscape.tif` (Appied to `input_landscape.tif`)
