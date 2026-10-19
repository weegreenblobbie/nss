"""The colour GUI's state: a State Node is a dictionary of every grading slider value; the Color Mixer's sliders travel in it too, as
mixer_hue / mixer_sat / mixer_lum (eight values each, -100..100)."""
from typing import TypedDict


class StateNode(TypedDict):
    """The grading sliders: sat 0..1, light and balance -1..1, blending 0..1, hues and rotation in degrees."""
    # Master controls for 3-way blending
    blending: float         # zone overlap / transition softness (0.0 to 1.0)
    balance: float          # bias toward shadows (-1.0) or highlights (+1.0)
    rotation: float         # global rotation offset (-180 to 180)
    # Shadows zone
    shadow_hue: float       # target shadow tint hue (0 - 360)
    shadow_sat: float       # shadow tint strength / saturation (0.0 to 1.0)
    shadow_light: float     # shadow zone lightness adjustment (-1.0 to 1.0)
    # Midtones zone
    midtone_hue: float      # target midtone tint hue (0 - 360)
    midtone_sat: float      # midtone tint strength / saturation (0.0 to 1.0)
    midtone_light: float    # midtone zone lightness adjustment (-1.0 to 1.0)
    # Highlights zone
    highlight_hue: float    # target highlight tint hue (0 - 360)
    highlight_sat: float    # highlight tint strength / saturation (0.0 to 1.0)
    highlight_light: float  # highlight zone lightness adjustment (-1.0 to 1.0)
    # Global wheel (whole tonal range)
    global_hue: float       # global tint hue (0 - 360)
    global_sat: float       # global tint strength / saturation (0.0 to 1.0)
    global_light: float     # global lightness adjustment (-1.0 to 1.0)


def create_default_state() -> StateNode:
    """The neutral State Node: every wheel at Sat 0 and Light 0 (hues at their default positions), Blending 0.5, Balance and Rotation 0."""
    return {
        "blending": 0.5,        # Overlap softness
        "balance": 0.0,         # Shadow/Highlight bias
        "rotation": 0.0,        # Global rotation
        # Shadows defaults
        "shadow_hue": 240.0,    # Default blue shadows
        "shadow_sat": 0.0,      # Default no tint strength
        "shadow_light": 0.0,
        # Midtones defaults
        "midtone_hue": 120.0,   # Default green midtones
        "midtone_sat": 0.0,
        "midtone_light": 0.0,
        # Highlights defaults
        "highlight_hue": 60.0,   # Default warm yellow/gold highlights
        "highlight_sat": 0.0,
        "highlight_light": 0.0,
        # Global defaults
        "global_hue": 0.0,
        "global_sat": 0.0,
        "global_light": 0.0,
    }
