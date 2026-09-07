import numpy as np
import random
import matplotlib.colors as mcolors
from typing import TypedDict, Literal, List, Tuple, Sequence, Optional, NamedTuple
from nss.icc import icc_to_linear
from nss.color_grading.grading import Grading
from nss.color_grading.math import WLUMA, logit
from nss.color_grading.tonal import working_matrix
from nss.color_mixer.bands import slider_at

# Type alias for mutation constraint axis
MutationAxis = Literal["All", "Hue", "Saturation", "Luminance"]


class StateNode(TypedDict):
    harmony_mode: Literal["Monochromatic", "Analogous", "Complementary"]
    # Global adjustments
    hue_shift: float        # overall hue shift (0 - 360)
    sat_shift: float        # overall saturation shift (-1.0 to 1.0)
    light_shift: float      # overall lightness shift (-1.0 to 1.0)
    step_size: float        # global mutation intensity / step size (0.1 to 2.0)
    # Master controls for 3-way blending
    blending: float         # zone overlap / transition softness (0.0 to 1.0)
    balance: float          # bias toward shadows (-1.0) or highlights (+1.0)
    rotation: float         # global rotation offset (-180 to 180)
    # Shadows zone (L < 0.3)
    shadow_hue: float       # target shadow tint hue (0 - 360)
    shadow_sat: float       # shadow tint strength / saturation (0.0 to 1.0)
    shadow_light: float     # shadow zone lightness adjustment (-1.0 to 1.0)
    # Midtones zone (0.3 <= L <= 0.7)
    midtone_hue: float      # target midtone tint hue (0 - 360)
    midtone_sat: float      # midtone tint strength / saturation (0.0 to 1.0)
    midtone_light: float    # midtone zone lightness adjustment (-1.0 to 1.0)
    # Highlights zone (L > 0.7)
    highlight_hue: float    # target highlight tint hue (0 - 360)
    highlight_sat: float    # highlight tint strength / saturation (0.0 to 1.0)
    highlight_light: float  # highlight zone lightness adjustment (-1.0 to 1.0)
    # Global wheel (whole tonal range)
    global_hue: float       # global tint hue (0 - 360)
    global_sat: float       # global tint strength / saturation (0.0 to 1.0)
    global_light: float     # global lightness adjustment (-1.0 to 1.0)

def create_default_state(mode: Literal["Monochromatic", "Analogous", "Complementary"] = "Monochromatic") -> StateNode:
    """
    Creates a baseline StateNode dictionary with full 3-way color grading and blend/balance support.
    """
    return {
        "harmony_mode": mode,
        "hue_shift": 0.0,
        "sat_shift": 0.0,
        "light_shift": 0.0,
        "step_size": 0.2,
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

def generate_monochromatic_mutations(center: StateNode, axis: MutationAxis = "All", step_size: float = 0.2) -> List[StateNode]:
    """
    Monochromatic: Lock hue; mutate only saturation and lightness.
    Varies zone-specific saturations and light offsets to drive rich tonal changes.
    Scaled and perturbed by the step_size/intensity.
    """
    states: List[StateNode] = []
    offsets = [
        (-0.2, -0.2), (0.0, -0.2), (0.2, -0.2),
        (-0.2,  0.0), (0.0,  0.0), (0.2,  0.0),
        (-0.2,  0.2), (0.0,  0.2), (0.2,  0.2)
    ]

    for i, (ds, dl) in enumerate(offsets):
        if i == 4:
            node = center.copy()
            node["harmony_mode"] = "Monochromatic"
            states.append(node)
        else:
            # Scale and randomize/perturb the step offsets
            scaled_ds = ds * step_size
            scaled_dl = dl * step_size
            
            perturb_ds = random.uniform(-0.04, 0.04) * step_size
            perturb_dl = random.uniform(-0.04, 0.04) * step_size

            actual_ds = (scaled_ds + perturb_ds) if axis in ("All", "Saturation") else 0.0
            actual_dl = (scaled_dl + perturb_dl) if axis in ("All", "Luminance") else 0.0

            # Locked base hue
            base_hue = center["hue_shift"]

            states.append({
                "harmony_mode": "Monochromatic",
                "hue_shift": base_hue,
                "sat_shift": float(np.clip(center["sat_shift"] + actual_ds, -1.0, 1.0)),
                "light_shift": float(np.clip(center["light_shift"] + actual_dl, -1.0, 1.0)),
                "step_size": center["step_size"],
                "blending": center["blending"],
                "balance": center["balance"],
                # Mutate zone strengths slightly for rich monochromatic variety with baseline saturation
                "shadow_hue": base_hue,
                "shadow_sat": float(np.clip(center["shadow_sat"] + 0.15 + actual_ds * 0.4, 0.01, 1.0)),
                "shadow_light": float(np.clip(center["shadow_light"] + actual_dl * 0.5, -1.0, 1.0)),
                
                "midtone_hue": base_hue,
                "midtone_sat": float(np.clip(center["midtone_sat"] + 0.10 + actual_ds * 0.2, 0.01, 1.0)),
                "midtone_light": float(np.clip(center["midtone_light"] + actual_dl * 0.5, -1.0, 1.0)),
                
                "highlight_hue": base_hue,
                "highlight_sat": float(np.clip(center["highlight_sat"] + 0.20 + actual_ds * 0.4, 0.01, 1.0)),
                "highlight_light": float(np.clip(center["highlight_light"] + actual_dl * 0.5, -1.0, 1.0)),
            })
    return states

def generate_analogous_mutations(center: StateNode, axis: MutationAxis = "All", step_size: float = 0.2) -> List[StateNode]:
    """
    Analogous: Mutate hue within a narrow adjacent band (e.g., ±25 degrees).
    Distributes analogous hue offsets across Shadows (-25°), Midtones (0°), and Highlights (+25°).
    Scaled and perturbed by the step_size/intensity.
    """
    states: List[StateNode] = []
    offsets = [
        (-25.0, -0.15), (0.0, -0.15), (25.0, -0.15),
        (-25.0,  0.0),  (0.0,  0.0),  (25.0,  0.0),
        (-25.0,  0.15), (0.0,  0.15), (25.0,  0.15)
    ]

    for i, (dh, ds) in enumerate(offsets):
        if i == 4:
            node = center.copy()
            node["harmony_mode"] = "Analogous"
            states.append(node)
        else:
            scaled_dh = dh * step_size
            scaled_ds = ds * step_size
            
            perturb_dh = random.uniform(-4.0, 4.0) * step_size
            perturb_ds = random.uniform(-0.03, 0.04) * step_size

            actual_dh = (scaled_dh + perturb_dh) if axis in ("All", "Hue") else 0.0
            actual_ds = (scaled_ds + perturb_ds) if axis in ("All", "Saturation") else 0.0
            
            actual_dl = 0.0
            if axis == "Luminance":
                actual_dl = ds * step_size  # Map 3x3 variation onto Lightness shift

            # Analogous hue mapping across zones for beautifully separated warm/cool gradients
            base_midtone = (center["hue_shift"] + actual_dh) % 360.0
            analogous_shadow = (base_midtone - 25.0 * step_size) % 360.0
            analogous_highlight = (base_midtone + 25.0 * step_size) % 360.0

            states.append({
                "harmony_mode": "Analogous",
                "hue_shift": base_midtone,
                "sat_shift": float(np.clip(center["sat_shift"] + actual_ds, -1.0, 1.0)),
                "light_shift": float(np.clip(center["light_shift"] + actual_dl, -1.0, 1.0)),
                "step_size": center["step_size"],
                "blending": center["blending"],
                "balance": center["balance"],
                # Active analogous color injection with non-zero baseline saturation
                "shadow_hue": analogous_shadow,
                "shadow_sat": float(np.clip(center["shadow_sat"] + 0.15 + actual_ds * 0.4, 0.05, 1.0)),
                "shadow_light": float(np.clip(center["shadow_light"] + actual_dl * 0.3, -1.0, 1.0)),

                "midtone_hue": base_midtone,
                "midtone_sat": float(np.clip(center["midtone_sat"] + 0.10 + actual_ds * 0.2, 0.05, 1.0)),
                "midtone_light": float(np.clip(center["midtone_light"] + actual_dl * 0.3, -1.0, 1.0)),

                "highlight_hue": analogous_highlight,
                "highlight_sat": float(np.clip(center["highlight_sat"] + 0.20 + actual_ds * 0.4, 0.05, 1.0)),
                "highlight_light": float(np.clip(center["highlight_light"] + actual_dl * 0.3, -1.0, 1.0)),
            })
    return states

def generate_complementary_mutations(center: StateNode, axis: MutationAxis = "All", step_size: float = 0.2) -> List[StateNode]:
    """
    Complementary: Force highlight hues to a target, and shadow hues to target + 180 degrees.
    Varies Complementary highlight/shadow hues and saturation/lightness across zones.
    Scaled and perturbed by the step_size/intensity.
    """
    states: List[StateNode] = []
    offsets = [-120.0, -80.0, -40.0, -20.0, 0.0, 20.0, 40.0, 80.0, 120.0]

    for i, dh in enumerate(offsets):
        if i == 4:
            node = center.copy()
            node["harmony_mode"] = "Complementary"
            states.append(node)
        else:
            new_highlight = center["highlight_hue"]
            new_sat = center["sat_shift"]
            new_light = center["light_shift"]
            
            zone_sat_offset = 0.0
            zone_light_offset = 0.0

            if axis in ("All", "Hue"):
                scaled_dh = dh * step_size
                perturb_dh = random.uniform(-6.0, 6.0) * step_size
                new_highlight = float((center["highlight_hue"] + scaled_dh + perturb_dh) % 360.0)
            elif axis == "Saturation":
                scaled_ds = (dh / 400.0) * step_size
                new_sat = float(np.clip(center["sat_shift"] + scaled_ds, -1.0, 1.0))
                zone_sat_offset = scaled_ds
            elif axis == "Luminance":
                scaled_dl = (dh / 400.0) * step_size
                new_light = float(np.clip(center["light_shift"] + scaled_dl, -1.0, 1.0))
                zone_light_offset = scaled_dl

            states.append({
                "harmony_mode": "Complementary",
                "hue_shift": center["hue_shift"],
                "sat_shift": new_sat,
                "light_shift": new_light,
                "step_size": center["step_size"],
                "blending": center["blending"],
                "balance": center["balance"],
                # Active complementary split injection with non-zero baseline saturation
                "highlight_hue": new_highlight,
                "highlight_sat": float(np.clip(center["highlight_sat"] + 0.20 + zone_sat_offset, 0.05, 1.0)),
                "highlight_light": float(np.clip(center["highlight_light"] + zone_light_offset, -1.0, 1.0)),
                
                "shadow_hue": float((new_highlight + 180.0) % 360.0),
                "shadow_sat": float(np.clip(center["shadow_sat"] + 0.20 + zone_sat_offset, 0.05, 1.0)),
                "shadow_light": float(np.clip(center["shadow_light"] + zone_light_offset, -1.0, 1.0)),
                
                # Keep midtone locked or slightly shifted
                "midtone_hue": center["midtone_hue"],
                "midtone_sat": center["midtone_sat"],
                "midtone_light": float(np.clip(center["midtone_light"] + zone_light_offset * 0.5, -1.0, 1.0)),
            })
    return states

def generate_mutations(
    center: StateNode, 
    mode: Literal["Monochromatic", "Analogous", "Complementary"],
    axis: MutationAxis = "All"
) -> List[StateNode]:
    """
    Generates 9 mutations from the center state based on the specified harmony mode and active mutation axis.
    """
    step_size = center.get("step_size", 0.2)
    if mode == "Monochromatic":
        return generate_monochromatic_mutations(center, axis, step_size)
    elif mode == "Analogous":
        return generate_analogous_mutations(center, axis, step_size)
    elif mode == "Complementary":
        return generate_complementary_mutations(center, axis, step_size)
    else:
        raise ValueError(f"Unknown harmony mode: {mode}")

# --- Random Shift: random slider shifts toward the colours sampled in Color Harmony ---
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
