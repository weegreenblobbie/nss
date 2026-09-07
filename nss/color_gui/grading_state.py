"""
Bridge from the GUI's StateNode (sat 0..1, light / balance -1..1, blending 0..1, rotation in degrees) to nss.color_grading.Grading.apply
(sat 0..100, lum / balance -100..100, blend 0..100).  Rotation turns every wheel's hue, Global included.
"""
import numpy as np

from nss.color_grading.grading import Grading
from nss.color_math import StateNode

WHEEL_KEYS = (("sh", "shadow"), ("mi", "midtone"), ("hi", "highlight"), ("gl", "global"))


def grading_kwargs(state: StateNode) -> dict:
    """Grading.apply keyword arguments of a state.  Keys missing from an older state count as their neutral value."""
    rot = float(state.get("rotation", 0.0))
    kw = {"balance": 100.0 * float(state.get("balance", 0.0)), "blend": 100.0 * float(state.get("blending", 0.5))}
    for short, name in WHEEL_KEYS:
        kw[f"{short}_h"] = (float(state.get(f"{name}_hue", 0.0)) + rot) % 360.0
        kw[f"{short}_s"] = 100.0 * float(state.get(f"{name}_sat", 0.0))
        kw[f"{short}_l"] = 100.0 * float(state.get(f"{name}_light", 0.0))
    return kw


def grade(grading: Grading, img: np.ndarray, icc: bytes, state: StateNode) -> np.ndarray:
    """The encoded image `img` (..., 3) in the profile `icc`, graded by `state`."""
    return grading.apply(img, icc, **grading_kwargs(state))
