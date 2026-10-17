import numpy as np
import pytest
from nss.color_math import (
    create_default_state,
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
