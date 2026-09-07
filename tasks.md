# Tasks: Custom Lightroom-Style Color Wheel & Swatch History

## Objective
Implement a custom PyQt6 color wheel widget for the Shadows, Midtones, and Highlights grading zones, complete with dedicated sliders and persistent recent custom color history grids.

---

## Task Breakdown

### 1. Custom Color Wheel Widget (`nss/color_gui.py` or `nss/widgets/color_wheel.py`)
- [ ] Subclass `QtWidgets.QWidget` to create a custom circular color wheel.
- [ ] Implement `paintEvent` using `QConicalGradient` for hue and `QRadialGradient` for saturation.
- [ ] Implement mouse event handlers (`mousePressEvent`, `mouseMoveEvent`) so clicking or dragging inside the wheel updates **only** the Hue and Saturation parameters.
- [ ] Ensure clean signal emission when a new color is selected from the wheel.

### 2. Zone Control Panel Layout (`nss/color_gui.py`)
- [ ] For each grading zone (Shadows, Midtones, Highlights), create a dedicated control group containing:
  - The custom Color Wheel widget (controlling Hue & Saturation).
  - A **Saturation** slider (fine-tuning the saturation value).
  - A **Luminance** slider (ranging from -1.0 to 1.0, defaulting to 0.0).
- [ ] Synchronize wheel interactions bi-directionally with the Saturation slider and the underlying `StateNode` parameters.

### 3. Persistent 2x4 Custom Color History Grid
- [ ] Maintain a rolling history of up to **8 custom color picks** for each zone (Shadows, Midtones, Highlights).
- [ ] Display these 8 recent colors in a compact **2x4 grid** (2 rows by 4 columns) beneath each zone's controls.
- [ ] Clicking any swatch in the 2x4 grid instantly applies that color/saturation to the respective zone, updating the wheel, sliders, and center image preview.
- [ ] Persist these 8 custom color slots across application sessions (e.g., using `QSettings` or config storage).
