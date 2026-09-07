import cv2
import numpy as np
from typing import TypedDict, Literal, List

MutationAxis = Literal["All", "Hue", "Saturation", "Luminance"]

class StateNode(TypedDict):
    harmony_mode: Literal["Monochromatic", "Analogous", "Complementary"]
    hue_shift: float        # overall hue shift (0 - 360)
    sat_shift: float        # overall saturation shift (-1.0 to 1.0)
    light_shift: float      # overall lightness shift (-1.0 to 1.0)
    highlight_hue: float    # complementary target highlight hue (0 - 360)
    shadow_hue: float       # complementary target shadow hue (0 - 360)

def create_default_state(mode: Literal["Monochromatic", "Analogous", "Complementary"] = "Monochromatic") -> StateNode:
    """
    Creates a baseline StateNode dictionary.
    """
    return {
        "harmony_mode": mode,
        "hue_shift": 0.0,
        "sat_shift": 0.0,
        "light_shift": 0.0,
        "highlight_hue": 0.0,
        "shadow_hue": 180.0,
    }

def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """
    Ensures that the input NumPy image is in 3-channel RGB float32 format.
    """
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            return img[:, :, :3]
        return img
    else:
        raise ValueError(f"Invalid image array with dimensions: {img.ndim}")

def apply_grading(img: np.ndarray, state: StateNode) -> np.ndarray:
    """
    Applies color grading parameters from the state to the input image (float32, [0.0, 1.0]).
    Returns a graded float32 image in [0.0, 1.0] RGB format.
    """
    # 1. Ensure input is RGB float32
    rgb = ensure_rgb(img).copy()
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)

    # 2. Convert to HLS (Hue [0, 360], Lightness [0, 1], Saturation [0, 1])
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    # 3. Create luminosity masks (using original L channel)
    # Shadows: L < 0.3
    # Highlights: L > 0.7
    shadow_mask = L < 0.3
    highlight_mask = L > 0.7

    # 4. Apply Harmony mode specifics
    harmony_mode = state.get("harmony_mode", "Monochromatic")

    if harmony_mode == "Complementary":
        # Force highlight hue to target highlight_hue
        target_highlight = state.get("highlight_hue", 0.0)
        # Force shadow hue to target highlight_hue + 180 (modulo 360)
        target_shadow = (target_highlight + 180.0) % 360.0

        H[highlight_mask] = target_highlight
        H[shadow_mask] = target_shadow

        # Shift midtones by overall hue_shift
        base_shift = state.get("hue_shift", 0.0)
        midtone_mask = ~(shadow_mask | highlight_mask)
        H[midtone_mask] = (H[midtone_mask] + base_shift) % 360.0
    else:
        # Monochromatic or Analogous: Shift overall hue
        base_shift = state.get("hue_shift", 0.0)
        H = (H + base_shift) % 360.0

    # 5. Apply Saturation shift
    sat_shift = state.get("sat_shift", 0.0)
    S = np.clip(S + sat_shift, 0.0, 1.0)

    # 6. Apply Lightness shift
    light_shift = state.get("light_shift", 0.0)
    L = np.clip(L + light_shift, 0.0, 1.0)

    # 7. Merge channels and convert back to RGB
    hls_graded = np.stack([H, L, S], axis=2)
    rgb_graded = cv2.cvtColor(hls_graded, cv2.COLOR_HLS2RGB)

    return np.clip(rgb_graded, 0.0, 1.0)

def generate_monochromatic_mutations(center: StateNode, axis: MutationAxis = "All") -> List[StateNode]:
    """
    Monochromatic: Lock hue; mutate only saturation and lightness.
    Returns 9 states (index 4 is exactly center).
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
            # Respect Mutation Axis constraint
            actual_ds = ds if axis in ("All", "Saturation") else 0.0
            actual_dl = dl if axis in ("All", "Luminance") else 0.0
            
            states.append({
                "harmony_mode": "Monochromatic",
                "hue_shift": center["hue_shift"],
                "sat_shift": float(np.clip(center["sat_shift"] + actual_ds, -1.0, 1.0)),
                "light_shift": float(np.clip(center["light_shift"] + actual_dl, -1.0, 1.0)),
                "highlight_hue": center["highlight_hue"],
                "shadow_hue": center["shadow_hue"],
            })
    return states

def generate_analogous_mutations(center: StateNode, axis: MutationAxis = "All") -> List[StateNode]:
    """
    Analogous: Mutate hue within a narrow adjacent band (e.g., ±30 degrees).
    Returns 9 states (index 4 is exactly center).
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
            # Respect Mutation Axis constraint
            actual_dh = dh if axis in ("All", "Hue") else 0.0
            actual_ds = ds if axis in ("All", "Saturation") else 0.0
            actual_dl = 0.0
            if axis == "Luminance":
                actual_dl = ds  # Map 3x3 variation onto Lightness shift
                
            states.append({
                "harmony_mode": "Analogous",
                "hue_shift": float((center["hue_shift"] + actual_dh) % 360.0),
                "sat_shift": float(np.clip(center["sat_shift"] + actual_ds, -1.0, 1.0)),
                "light_shift": float(np.clip(center["light_shift"] + actual_dl, -1.0, 1.0)),
                "highlight_hue": center["highlight_hue"],
                "shadow_hue": center["shadow_hue"],
            })
    return states

def generate_complementary_mutations(center: StateNode, axis: MutationAxis = "All") -> List[StateNode]:
    """
    Complementary: Force highlight hues to a target, and shadow hues to target + 180 degrees.
    Returns 9 states (index 4 is exactly center).
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

            if axis in ("All", "Hue"):
                new_highlight = float((center["highlight_hue"] + dh) % 360.0)
            elif axis == "Saturation":
                # Scale -120..120 range of offsets to -0.3..0.3 for Saturation shift
                new_sat = float(np.clip(center["sat_shift"] + (dh / 400.0), -1.0, 1.0))
            elif axis == "Luminance":
                # Scale -120..120 range of offsets to -0.3..0.3 for Lightness shift
                new_light = float(np.clip(center["light_shift"] + (dh / 400.0), -1.0, 1.0))

            states.append({
                "harmony_mode": "Complementary",
                "hue_shift": center["hue_shift"],
                "sat_shift": new_sat,
                "light_shift": new_light,
                "highlight_hue": new_highlight,
                "shadow_hue": float((new_highlight + 180.0) % 360.0),
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
    if mode == "Monochromatic":
        return generate_monochromatic_mutations(center, axis)
    elif mode == "Analogous":
        return generate_analogous_mutations(center, axis)
    elif mode == "Complementary":
        return generate_complementary_mutations(center, axis)
    else:
        raise ValueError(f"Unknown harmony mode: {mode}")
