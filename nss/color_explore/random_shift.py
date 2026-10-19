"""Random Shift: random slider shifts toward the colours sampled in Color Harmony (the Explore & Harmony controls and the Explore grid)."""
from typing import List, NamedTuple

import numpy as np

from nss.icc import icc_to_linear
from nss.color_grading.grading import Grading
from nss.color_grading.math import WLUMA, logit
from nss.color_grading.tonal import working_matrix
from nss.color_mixer.bands import slider_at
from nss.color_explore.state import StateNode

ZONES = ("shadow", "midtone", "highlight")      # StateNode prefixes of the tonal wheels, numbered as tonal_zone numbers them
DAMPING_LEVEL = 0.1                             # HSV saturation and value below which a colour's mixer shift fades out (to zero at 0)


class ShiftColors(NamedTuple):
    """The selected colours random_shift aims at, one entry per colour."""
    mixer_hue: np.ndarray       # mixer-input hue in degrees (the nss.color_mixer.bands hue axis)
    wheel_hue: np.ndarray       # the same colour's hue as a Color Grading wheel hue, degrees
    sat: np.ndarray             # HSV saturation 0..1
    value: np.ndarray           # HSV value 0..1
    zone: np.ndarray            # index into ZONES


def tonal_zone(grading: Grading, rgb_enc: np.ndarray, icc: bytes, balance: float, blending: float) -> np.ndarray:
    """Index into ZONES (n,) of the tonal wheel whose mask is strongest at the linear ProPhoto luminance of each encoded colour (n, 3) in the
    profile icc, for Balance -1..1 and Blending 0..1 (StateNode units)."""
    l = (icc_to_linear(np.asarray(rgb_enc, np.float64).reshape(-1, 3), icc) @ working_matrix(icc).T).clip(0.0, 1.0)
    u = logit(np.clip(l @ WLUMA, 1e-6, 1 - 1e-6))
    wheels = (grading.shadows, grading.midtones, grading.highlights)
    return np.argmax([wheel.mask(u, 100.0 * balance, 100.0 * blending) for wheel in wheels], axis=0)


def shift_colors(mixer_hue: np.ndarray, wheel_hue: np.ndarray, rgb_enc: np.ndarray, icc: bytes, grading: Grading, balance: float, blending: float) -> ShiftColors:
    """ShiftColors of n colours: their mixer and wheel hues (n,) and, from their encoded colours (n, 3), the HSV saturation, value and tonal_zone."""
    rgb = np.asarray(rgb_enc, np.float64).reshape(-1, 3)
    value = rgb.max(axis=1)
    sat = (value - rgb.min(axis=1)) / np.maximum(value, 1e-12)
    zone = tonal_zone(grading, rgb, icc, balance, blending)
    return ShiftColors(np.asarray(mixer_hue, np.float64), np.asarray(wheel_hue, np.float64), sat, value, zone)


def circular_median(hues: np.ndarray) -> float:
    """Median of hues in degrees on the circle: the plain median after cutting the circle at the widest gap between neighbouring hues."""
    h = np.sort(np.mod(hues, 360.0))
    start = (np.argmax(np.diff(np.append(h, h[0] + 360.0))) + 1) % len(h)
    unrolled = np.roll(h, -start) + 360.0 * (np.arange(len(h)) >= len(h) - start)
    return float(np.median(unrolled) % 360.0)


def mixer_weights(colors: ShiftColors) -> np.ndarray:
    """Weight (8,) of each mixer band: the largest, over the colours, of the band's weight at the colour's hue times the colour's damping
    clip(sat / DAMPING_LEVEL, 0, 1) * clip(value / DAMPING_LEVEL, 0, 1)."""
    band = np.stack([slider_at(colors.mixer_hue, level) for level in 100.0 * np.eye(8)], axis=-1) / 100.0       # (n, 8)
    damping = np.clip(colors.sat / DAMPING_LEVEL, 0.0, 1.0) * np.clip(colors.value / DAMPING_LEVEL, 0.0, 1.0)
    return (band * damping[:, None]).max(axis=0, initial=0.0)


def random_shift(state: StateNode, colors: ShiftColors, strength: float, mixer: bool, grading: bool, rng: np.random.Generator) -> StateNode:
    """A copy of state shifted at random toward colors, rounded to the sliders' steps; strength 0..1 is Variation Strength / 100 and every r is
    uniform in [-1, 1], drawn from rng.
    mixer: each band's Hue, Sat and Lum get r * strength * 100 * its mixer_weights, clamped to -100..100.
    grading: each tonal wheel with colours in its zone gets hue = circular median of their wheel hues + r * strength * 360 (wrapped),
    sat = median of their sat + r * strength (clamped 0..1) and light = its own light + r * strength (clamped -1..1); other wheels are kept."""
    out = state.copy()
    if mixer:
        weights = mixer_weights(colors)
        for kind in ("hue", "sat", "lum"):
            shifted = np.asarray(state[f"mixer_{kind}"]) + rng.uniform(-1.0, 1.0, 8) * strength * 100.0 * weights
            out[f"mixer_{kind}"] = [float(v) for v in np.clip(np.rint(shifted), -100.0, 100.0)]
    if grading:
        for z, prefix in enumerate(ZONES):
            members = colors.zone == z
            if not members.any():
                continue
            r_hue, r_sat, r_light = rng.uniform(-1.0, 1.0, 3)
            out[f"{prefix}_hue"] = float(np.rint(circular_median(colors.wheel_hue[members]) + r_hue * strength * 360.0) % 360.0)
            out[f"{prefix}_sat"] = float(np.clip(np.rint(100.0 * (np.median(colors.sat[members]) + r_sat * strength)), 0.0, 100.0) / 100.0)
            out[f"{prefix}_light"] = float(np.clip(np.rint(100.0 * (state[f"{prefix}_light"] + r_light * strength)), -100.0, 100.0) / 100.0)
    return out


def explore_states(center: StateNode, colors: ShiftColors, strength: float, mixer: bool, grading: bool, rng: np.random.Generator) -> List[StateNode]:
    """The 3x3 explore grid: a copy of center at index 4, around it 8 independent random_shift(center, ...)."""
    return [center.copy() if i == 4 else random_shift(center, colors, strength, mixer, grading, rng) for i in range(9)]
