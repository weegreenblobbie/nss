import cv2
import numpy as np
import random
from typing import TypedDict, Literal, List

# Type alias for mutation constraint axis
MutationAxis = Literal["All", "Hue", "Saturation", "Luminance"]

class StateNode(TypedDict):
    harmony_mode: Literal["Monochromatic", "Analogous", "Complementary"]
    # Global adjustments
    hue_shift: float        # overall hue shift (0 - 360)
    sat_shift: float        # overall saturation shift (-1.0 to 1.0)
    light_shift: float      # overall lightness shift (-1.0 to 1.0)
    step_size: float        # global mutation intensity / step size (0.1 to 2.0)
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

def create_default_state(mode: Literal["Monochromatic", "Analogous", "Complementary"] = "Monochromatic") -> StateNode:
    """
    Creates a baseline StateNode dictionary with full 3-way color grading support.
    """
    return {
        "harmony_mode": mode,
        "hue_shift": 0.0,
        "sat_shift": 0.0,
        "light_shift": 0.0,
        "step_size": 1.0,
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
    Applies 3-way zone-based color grading parameters from the state to the input image.
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
    shadow_mask = L < 0.3
    highlight_mask = L > 0.7
    midtone_mask = (L >= 0.3) & (L <= 0.7)

    # 4. Apply 3-Way Zone Grading (Color Injection)
    harmony_mode = state.get("harmony_mode", "Monochromatic")

    if harmony_mode == "Complementary":
        # Complementary mode directly forces highlights and shadows to complementary hues
        t_high = state.get("highlight_hue", 60.0)
        t_shadow = (t_high + 180.0) % 360.0

        H[highlight_mask] = t_high
        H[shadow_mask] = t_shadow

        # Inject default rich complementary saturation
        S[highlight_mask] = np.clip(S[highlight_mask] + 0.15, 0.0, 1.0)
        S[shadow_mask] = np.clip(S[shadow_mask] + 0.15, 0.0, 1.0)

        # Shift midtones by overall hue_shift
        base_shift = state.get("hue_shift", 0.0)
        H[midtone_mask] = (H[midtone_mask] + base_shift) % 360.0
    else:
        # Standard 3-Way Color Grading: Inject distinct colors into Shadows, Midtones, and Highlights
        # Shadows Tint
        sh_hue = state.get("shadow_hue", 240.0)
        sh_sat = state.get("shadow_sat", 0.0)
        if sh_sat > 0.0:
            H[shadow_mask] = (H[shadow_mask] * (1.0 - sh_sat) + sh_hue * sh_sat) % 360.0
            S[shadow_mask] = np.clip(S[shadow_mask] + sh_sat, 0.0, 1.0)

        # Midtones Tint
        mid_hue = state.get("midtone_hue", 120.0)
        mid_sat = state.get("midtone_sat", 0.0)
        if mid_sat > 0.0:
            H[midtone_mask] = (H[midtone_mask] * (1.0 - mid_sat) + mid_hue * mid_sat) % 360.0
            S[midtone_mask] = np.clip(S[midtone_mask] + mid_sat, 0.0, 1.0)

        # Highlights Tint
        hi_hue = state.get("highlight_hue", 60.0)
        hi_sat = state.get("highlight_sat", 0.0)
        if hi_sat > 0.0:
            H[highlight_mask] = (H[highlight_mask] * (1.0 - hi_sat) + hi_hue * hi_sat) % 360.0
            S[highlight_mask] = np.clip(S[highlight_mask] + hi_sat, 0.0, 1.0)

        # Global Hue Shift
        base_shift = state.get("hue_shift", 0.0)
        H = (H + base_shift) % 360.0

    # Apply Zone Lightness Adjustments
    L[shadow_mask] = np.clip(L[shadow_mask] + state.get("shadow_light", 0.0), 0.0, 1.0)
    L[midtone_mask] = np.clip(L[midtone_mask] + state.get("midtone_light", 0.0), 0.0, 1.0)
    L[highlight_mask] = np.clip(L[highlight_mask] + state.get("highlight_light", 0.0), 0.0, 1.0)

    # 5. Apply Global Saturation and Lightness shifts
    sat_shift = state.get("sat_shift", 0.0)
    S = np.clip(S + sat_shift, 0.0, 1.0)

    light_shift = state.get("light_shift", 0.0)
    L = np.clip(L + light_shift, 0.0, 1.0)

    # 6. Merge channels and convert back to RGB
    hls_graded = np.stack([H, L, S], axis=2)
    rgb_graded = cv2.cvtColor(hls_graded, cv2.COLOR_HLS2RGB)

    return np.clip(rgb_graded, 0.0, 1.0)

def generate_monochromatic_mutations(center: StateNode, axis: MutationAxis = "All", step_size: float = 1.0) -> List[StateNode]:
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

            states.append({
                "harmony_mode": "Monochromatic",
                "hue_shift": center["hue_shift"],
                "sat_shift": float(np.clip(center["sat_shift"] + actual_ds, -1.0, 1.0)),
                "light_shift": float(np.clip(center["light_shift"] + actual_dl, -1.0, 1.0)),
                "step_size": center["step_size"],
                # Mutate zone strengths slightly for rich monochromatic variety
                "shadow_hue": center["shadow_hue"],
                "shadow_sat": float(np.clip(center["shadow_sat"] + actual_ds * 0.3, 0.0, 1.0)),
                "shadow_light": float(np.clip(center["shadow_light"] + actual_dl * 0.5, -1.0, 1.0)),
                
                "midtone_hue": center["midtone_hue"],
                "midtone_sat": float(np.clip(center["midtone_sat"] + actual_ds * 0.3, 0.0, 1.0)),
                "midtone_light": float(np.clip(center["midtone_light"] + actual_dl * 0.5, -1.0, 1.0)),
                
                "highlight_hue": center["highlight_hue"],
                "highlight_sat": float(np.clip(center["highlight_sat"] + actual_ds * 0.3, 0.0, 1.0)),
                "highlight_light": float(np.clip(center["highlight_light"] + actual_dl * 0.5, -1.0, 1.0)),
            })
    return states

def generate_analogous_mutations(center: StateNode, axis: MutationAxis = "All", step_size: float = 1.0) -> List[StateNode]:
    """
    Analogous: Mutate hue within a narrow adjacent band (e.g., ±25 degrees).
    Distributes analogous hue offsets across Shadows, Midtones, and Highlights.
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

            states.append({
                "harmony_mode": "Analogous",
                "hue_shift": float((center["hue_shift"] + actual_dh) % 360.0),
                "sat_shift": float(np.clip(center["sat_shift"] + actual_ds, -1.0, 1.0)),
                "light_shift": float(np.clip(center["light_shift"] + actual_dl, -1.0, 1.0)),
                "step_size": center["step_size"],
                # Perturb zone hues analogously to create beautifully separated color steps
                "shadow_hue": float((center["shadow_hue"] + actual_dh) % 360.0),
                "shadow_sat": float(np.clip(center["shadow_sat"] + (0.1 if actual_ds != 0 else 0.0), 0.0, 1.0)),
                "shadow_light": float(np.clip(center["shadow_light"] + actual_dl * 0.3, -1.0, 1.0)),

                "midtone_hue": float((center["midtone_hue"] + actual_dh) % 360.0),
                "midtone_sat": float(np.clip(center["midtone_sat"] + (0.1 if actual_ds != 0 else 0.0), 0.0, 1.0)),
                "midtone_light": float(np.clip(center["midtone_light"] + actual_dl * 0.3, -1.0, 1.0)),

                "highlight_hue": float((center["highlight_hue"] + actual_dh) % 360.0),
                "highlight_sat": float(np.clip(center["highlight_sat"] + (0.1 if actual_ds != 0 else 0.0), 0.0, 1.0)),
                "highlight_light": float(np.clip(center["highlight_light"] + actual_dl * 0.3, -1.0, 1.0)),
            })
    return states

def generate_complementary_mutations(center: StateNode, axis: MutationAxis = "All", step_size: float = 1.0) -> List[StateNode]:
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
                
                # Complementary highlight hue drives shadow hue at exactly +180 deg
                "highlight_hue": new_highlight,
                "highlight_sat": float(np.clip(center["highlight_sat"] + zone_sat_offset, 0.0, 1.0)),
                "highlight_light": float(np.clip(center["highlight_light"] + zone_light_offset, -1.0, 1.0)),
                
                "shadow_hue": float((new_highlight + 180.0) % 360.0),
                "shadow_sat": float(np.clip(center["shadow_sat"] + zone_sat_offset, 0.0, 1.0)),
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
    step_size = center.get("step_size", 1.0)
    if mode == "Monochromatic":
        return generate_monochromatic_mutations(center, axis, step_size)
    elif mode == "Analogous":
        return generate_analogous_mutations(center, axis, step_size)
    elif mode == "Complementary":
        return generate_complementary_mutations(center, axis, step_size)
    else:
        raise ValueError(f"Unknown harmony mode: {mode}")
