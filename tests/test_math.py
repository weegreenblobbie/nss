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
    img = np.zeros((15, 12, 3), dtype=np.float32)
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
    img = np.zeros((15, 12, 3), dtype=np.float32)
    img[:, :6, 0] = 0.1  # Shadows
    
    img[:, 6:, 0] = 0.9  # Highlights
    img[:, 6:, 1] = 0.8
    img[:, 6:, 2] = 0.8
    
    state = create_default_state("Complementary")
    state["highlight_hue"] = 120.0  # Green highlights, shadow hue = 120+180 = 300
    
    graded = apply_grading(img, state)
    
    # Convert back to HLS in test to verify hues
    hls_graded = cv2.cvtColor(graded, cv2.COLOR_RGB2HLS)
    H = hls_graded[:, :, 0]
    
    # Left half (Shadows) should have shadow hue (120 + 180) = 300, blended
    # Allow small tolerance due to float32 precision and color space rounding
    assert np.all(np.isclose(H[:, :6], 352.0, atol=3.0))
    # Right half (Highlights) should have highlight hue = 120, blended
    assert np.all(np.isclose(H[:, 6:], 9.5, atol=3.0))

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


def test_explore_mode_constraints() -> None:
    from nss.color_math import generate_random_harmony_state, generate_explore_mutations
    
    # 1. Test generate_random_harmony_state
    for mode in ["Complementary", "Analogous", "Triadic", "Monochromatic"]:
        state = generate_random_harmony_state(mode)
        # All zone lightness values should be 0.0 (the neutral baseline value)
        assert state["shadow_light"] == 0.0
        assert state["midtone_light"] == 0.0
        assert state["highlight_light"] == 0.0
        # Saturation must be constrained to [0.0, 0.50]
        assert 0.0 <= state["shadow_sat"] <= 0.50
        assert 0.0 <= state["midtone_sat"] <= 0.50
        assert 0.0 <= state["highlight_sat"] <= 0.50

    # 2. Test generate_explore_mutations
    center = create_default_state("Monochromatic")
    # Set non-zero light values to ensure they are passed through
    center["shadow_light"] = -0.3
    center["midtone_light"] = 0.5
    center["highlight_light"] = 0.8
    # Set high saturation values to ensure they get clamped down during mutation
    center["shadow_sat"] = 0.5
    center["midtone_sat"] = 0.7
    center["highlight_sat"] = 0.9
    
    mutations = generate_explore_mutations(center, "Monochromatic", variation_strength=1.5)
    assert len(mutations) == 9
    for i, s in enumerate(mutations):
        # All mutations must preserve the center's exact lightness values
        assert s["shadow_light"] == -0.3
        assert s["midtone_light"] == 0.5
        assert s["highlight_light"] == 0.8
        # Outer mutations (all except index 4) must clamp saturation to [0.0, 0.50]
        if i == 4:
            assert s["shadow_sat"] == 0.5
            assert s["midtone_sat"] == 0.7
            assert s["highlight_sat"] == 0.9
        else:
            assert 0.0 <= s["shadow_sat"] <= 0.50
            assert 0.0 <= s["midtone_sat"] <= 0.50
            assert 0.0 <= s["highlight_sat"] <= 0.50


def test_apply_grading_rotation_offset() -> None:
    import cv2
    img = np.zeros((15, 12, 3), dtype=np.float32)
    img[:, :6, 0] = 0.1  # Shadows
    img[:, 6:, 0] = 0.9  # Highlights
    img[:, 6:, 1] = 0.8
    img[:, 6:, 2] = 0.8
    
    state = create_default_state("Complementary")
    state["highlight_hue"] = 120.0  # Green highlights
    state["rotation"] = 30.0  # Shift hues by +30 degrees
    # Highlight hue rotated: 120 + 30 = 150
    # Shadow hue rotated: 120 + 180 = 300; 300 + 30 = 330
    
    graded = apply_grading(img, state)
    hls = cv2.cvtColor(graded, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    
    # Shadows (left half) should have shadow hue = 355.6
    assert np.all(np.isclose(H[:, :6], 355.6, atol=3.0))
    # Highlights (right half) should have highlight hue = 5.9
    assert np.all(np.isclose(H[:, 6:], 5.9, atol=3.0))


def test_saturation_scaling_linearity() -> None:
    # Use a non-square numpy array as per guidelines
    img = np.zeros((15, 12, 3), dtype=np.float32)
    img[:, :, 0] = 0.5  # Saturated red base
    img[:, :, 1] = 0.2
    img[:, :, 2] = 0.2

    # Case A: Saturation = 0.00
    state_0 = create_default_state("Monochromatic")
    state_0["shadow_sat"] = 0.00
    state_0["shadow_hue"] = 240.0
    graded_0 = apply_grading(img, state_0)

    # Case B: Saturation = 0.01 (tiny 1% tint)
    state_1 = create_default_state("Monochromatic")
    state_1["shadow_sat"] = 0.01
    state_1["shadow_hue"] = 240.0
    graded_1 = apply_grading(img, state_1)

    # Assert that the maximum absolute pixel difference is very small (< 0.02)
    max_diff = np.max(np.abs(graded_0 - graded_1))
    assert max_diff < 0.02, f"Discontinuity detected! Max diff is {max_diff}"




