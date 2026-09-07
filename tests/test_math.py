import numpy as np
import pytest
import cv2
from nss.color_math import (
    create_default_state,
    apply_grading,
    generate_mutations,
    StateNode,
)

def test_default_state_creation() -> None:
    state = create_default_state("Monochromatic")
    assert state["harmony_mode"] == "Monochromatic"
    assert state["hue_shift"] == 0.0
    assert state["sat_shift"] == 0.0
    assert state["light_shift"] == 0.0

def test_mutation_generators() -> None:
    # 1. Monochromatic
    center = create_default_state("Monochromatic")
    center["hue_shift"] = 50.0
    center["sat_shift"] = 0.5
    center["light_shift"] = -0.5
    
    mono_states = generate_mutations(center, "Monochromatic")
    assert len(mono_states) == 9
    assert mono_states[4] == center
    for s in mono_states:
        assert s["harmony_mode"] == "Monochromatic"
        assert s["hue_shift"] == 50.0
        assert -1.0 <= s["sat_shift"] <= 1.0
        assert -1.0 <= s["light_shift"] <= 1.0

    # 2. Analogous
    center_analogous = create_default_state("Analogous")
    center_analogous["hue_shift"] = 50.0
    center_analogous["sat_shift"] = 0.5
    center_analogous["light_shift"] = -0.5
    
    analogous_states = generate_mutations(center_analogous, "Analogous")
    assert len(analogous_states) == 9
    assert analogous_states[4] == center_analogous
    for s in analogous_states:
        assert s["harmony_mode"] == "Analogous"
        assert 0.0 <= s["hue_shift"] < 360.0

    # 3. Complementary
    center_comp = create_default_state("Complementary")
    center_comp["highlight_hue"] = 60.0
    center_comp["shadow_hue"] = 240.0
    
    comp_states = generate_mutations(center_comp, "Complementary")
    assert len(comp_states) == 9
    assert comp_states[4] == center_comp
    for s in comp_states:
        assert s["harmony_mode"] == "Complementary"
        # Shadow hue must be highlight_hue + 180 modulo 360
        assert np.isclose(s["shadow_hue"], (s["highlight_hue"] + 180.0) % 360.0)

def test_apply_grading_basic() -> None:
    # A simple red float32 pixel (R=1.0, G=0.0, B=0.0)
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[:, :, 0] = 1.0  # Pure Red
    
    # Apply standard empty shift
    state = create_default_state("Monochromatic")
    graded = apply_grading(img, state)
    
    # Graded image should have identical shape, dtype, and values should be float32 in [0, 1]
    assert graded.shape == img.shape
    assert graded.dtype == np.float32
    assert np.all(graded >= 0.0) and np.all(graded <= 1.0)

def test_luminosity_masks_complementary() -> None:
    # Create an image with saturated colors to preserve hue:
    # Left half: dark saturated red shadows (R=0.1, G=0.0, B=0.0 => L=0.05 < 0.3, Sat=1.0)
    # Right half: bright saturated red/yellow highlights (R=0.9, G=0.8, B=0.8 => L=0.85 > 0.7, Sat=0.33)
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[:, :5, 0] = 0.1  # Shadows
    
    img[:, 5:, 0] = 0.9  # Highlights
    img[:, 5:, 1] = 0.8
    img[:, 5:, 2] = 0.8
    
    state = create_default_state("Complementary")
    state["highlight_hue"] = 120.0  # Green highlights, shadow hue = 120+180 = 300
    
    graded = apply_grading(img, state)
    
    # Convert back to HLS in test to verify hues
    hls_graded = cv2.cvtColor(graded, cv2.COLOR_RGB2HLS)
    H = hls_graded[:, :, 0]
    
    # Left half (Shadows) should have shadow hue (120 + 180) = 300
    # Allow small tolerance due to float32 precision and color space rounding
    assert np.all(np.isclose(H[:, :5], 300.0, atol=3.0))
    # Right half (Highlights) should have highlight hue = 120
    assert np.all(np.isclose(H[:, 5:], 120.0, atol=3.0))

def test_mutation_axes() -> None:
    center = create_default_state("Monochromatic")
    center["hue_shift"] = 50.0
    center["sat_shift"] = 0.5
    center["light_shift"] = -0.5
    
    # Test Saturation Axis (lightness stays locked)
    sat_states = generate_mutations(center, "Monochromatic", axis="Saturation")
    assert len(sat_states) == 9
    for s in sat_states:
        assert s["light_shift"] == center["light_shift"]
        assert s["hue_shift"] == center["hue_shift"]

    # Test Luminance Axis (saturation stays locked)
    lum_states = generate_mutations(center, "Monochromatic", axis="Luminance")
    assert len(lum_states) == 9
    for s in lum_states:
        assert s["sat_shift"] == center["sat_shift"]
        assert s["hue_shift"] == center["hue_shift"]

