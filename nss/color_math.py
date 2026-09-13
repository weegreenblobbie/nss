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
    Applies professional 3-way zone-based color grading parameters from the state to the input image.
    Uses master blending and balance values to softly transition colors between zones.
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

    # 3. Compute soft zone masks based on master balance and blending parameters
    balance = state.get("balance", 0.0)
    # Shift effective lightness used for mask boundaries based on balance
    L_shifted = np.clip(L - 0.3 * balance, 0.0, 1.0)
    
    blending = state.get("blending", 0.5)
    # Softness width scales from 0.01 (hard borders) up to 0.40 (highly feathered overlap)
    softness = max(0.01, 0.05 + 0.35 * blending)

    # Shadows Soft Mask (centered at L = 0.3)
    shadow_weight = np.clip((0.3 + softness/2.0 - L_shifted) / softness, 0.0, 1.0)

    # Highlights Soft Mask (centered at L = 0.7)
    highlight_weight = np.clip((L_shifted - (0.7 - softness/2.0)) / softness, 0.0, 1.0)

    # Midtones Soft Mask (fills the remaining space between highlights and shadows)
    midtone_weight = np.clip(1.0 - shadow_weight - highlight_weight, 0.0, 1.0)

    # 4. Compute target zone colors
    harmony_mode = state.get("harmony_mode", "Monochromatic")

    sh_hue = state.get("shadow_hue", 240.0)
    sh_sat = state.get("shadow_sat", 0.0)
    mid_hue = state.get("midtone_hue", 120.0)
    mid_sat = state.get("midtone_sat", 0.0)
    hi_hue = state.get("highlight_hue", 60.0)
    hi_sat = state.get("highlight_sat", 0.0)

    # Apply global rotation offset
    rotation_offset = state.get("rotation", 0.0)
    sh_hue = (sh_hue + rotation_offset) % 360.0
    mid_hue = (mid_hue + rotation_offset) % 360.0
    hi_hue = (hi_hue + rotation_offset) % 360.0

    # Complementary coupling: Force Highlights and Shadows to opposite hues
    if harmony_mode == "Complementary":
        sh_hue = (hi_hue + 180.0) % 360.0
        sh_sat = np.clip(sh_sat + 0.15, 0.0, 1.0)
        hi_sat = np.clip(hi_sat + 0.15, 0.0, 1.0)

    # 5. Apply soft-mask color injection into H and S channels (weighted average blending)
    total_tint_weight = (shadow_weight * sh_sat) + (midtone_weight * mid_sat) + (highlight_weight * hi_sat)
    
    tint_mask = total_tint_weight > 0.0
    if np.any(tint_mask):
        # Convert target zone hues to radians for robust circular vector interpolation
        sh_hue_rad = np.radians(sh_hue)
        mid_hue_rad = np.radians(mid_hue)
        hi_hue_rad = np.radians(hi_hue)

        # Compute weighted sum of 2D Cartesian vector components to avoid 0/360 boundary leaps
        x_comp = (
            (shadow_weight[tint_mask] * sh_sat * np.cos(sh_hue_rad)) +
            (midtone_weight[tint_mask] * mid_sat * np.cos(mid_hue_rad)) +
            (highlight_weight[tint_mask] * hi_sat * np.cos(hi_hue_rad))
        )
        y_comp = (
            (shadow_weight[tint_mask] * sh_sat * np.sin(sh_hue_rad)) +
            (midtone_weight[tint_mask] * mid_sat * np.sin(mid_hue_rad)) +
            (highlight_weight[tint_mask] * hi_sat * np.sin(hi_hue_rad))
        )

        # Reconstruct target Hue via arctan2 and convert back to [0, 360) degrees
        target_h_rad = np.arctan2(y_comp, x_comp)
        
        # Softly blend original Hue with target Hue using total_tint_weight as factor to ensure linear scaling
        w = np.clip(total_tint_weight[tint_mask], 0.0, 1.0)
        orig_h_rad = np.radians(H[tint_mask])
        
        x_blend = (1.0 - w) * np.cos(orig_h_rad) + w * np.cos(target_h_rad)
        y_blend = (1.0 - w) * np.sin(orig_h_rad) + w * np.sin(target_h_rad)
        
        H[tint_mask] = np.degrees(np.arctan2(y_blend, x_blend)) % 360.0
        
        # Softly scale/inject Saturation
        S[tint_mask] = np.clip(S[tint_mask] + total_tint_weight[tint_mask], 0.0, 1.0)

    # 6. Apply Zone Lightness Adjustments based on soft weights
    L = np.clip(
        L + 
        (shadow_weight * state.get("shadow_light", 0.0)) +
        (midtone_weight * state.get("midtone_light", 0.0)) +
        (highlight_weight * state.get("highlight_light", 0.0)),
        0.0, 1.0
    )

    # 7. Apply Master Global Hue, Saturation, and Lightness shifts
    base_shift = state.get("hue_shift", 0.0)
    H = (H + base_shift) % 360.0

    sat_shift = state.get("sat_shift", 0.0)
    S = np.clip(S + sat_shift, 0.0, 1.0)

    light_shift = state.get("light_shift", 0.0)
    L = np.clip(L + light_shift, 0.0, 1.0)

    # 8. Merge channels and convert back to RGB
    hls_graded = np.stack([H, L, S], axis=2)
    rgb_graded = cv2.cvtColor(hls_graded, cv2.COLOR_HLS2RGB)

    return np.clip(rgb_graded, 0.0, 1.0)

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

def generate_random_harmony_state(mode: str) -> StateNode:
    """
    Generates a brand new random StateNode based on Harmony Mode (Complementary, Analogous, Triadic, Monochromatic)
    following Step 4 formulas.
    """
    import random
    def j(variance: float) -> float:
        return random.uniform(-variance, variance)
        
    H = random.uniform(0.0, 360.0)
    
    # Step A: Generate Zone-Constrained Saturation & Luminance (Luminance neutral, Saturation clamped to [0.0, 0.50])
    sh_sat = random.uniform(0.0, 0.50)
    sh_light = 0.0
    
    mid_sat = random.uniform(0.0, 0.50)
    mid_light = 0.0
    
    hi_sat = random.uniform(0.0, 0.50)
    hi_light = 0.0
    
    # Step B: Map Hues to Zones based on Harmony Mode
    if mode == "Complementary":
        sh_hue = (H + 180.0 + j(15.0)) % 360.0
        mid_hue = H
        hi_hue = (H + j(10.0)) % 360.0
    elif mode == "Analogous":
        sh_hue = (H - 30.0 + j(10.0)) % 360.0
        mid_hue = H
        hi_hue = (H + 30.0 + j(10.0)) % 360.0
    elif mode == "Triadic":
        sh_hue = H
        mid_hue = (H + 120.0 + j(15.0)) % 360.0
        hi_hue = (H + 240.0 + j(15.0)) % 360.0
    else:  # Monochromatic
        sh_hue = H
        mid_hue = (H + j(5.0)) % 360.0
        hi_hue = (H + j(5.0)) % 360.0
        
    state = create_default_state()
    state["harmony_mode"] = mode  # type: ignore
    state["shadow_hue"] = float(sh_hue)
    state["shadow_sat"] = float(sh_sat)
    state["shadow_light"] = float(sh_light)
    state["midtone_hue"] = float(mid_hue)
    state["midtone_sat"] = float(mid_sat)
    state["midtone_light"] = float(mid_light)
    state["highlight_hue"] = float(hi_hue)
    state["highlight_sat"] = float(hi_sat)
    state["highlight_light"] = float(hi_light)
    return state

def generate_explore_mutations(center: StateNode, mode: str, variation_strength: float) -> List[StateNode]:
    """
    Generates 9 mutations where the center is index 4.
    The other 8 outer slots apply a random mutation scaled by variation_strength (Step 5).
    """
    import random
    states: List[StateNode] = []
    
    def j(variance: float) -> float:
        return random.uniform(-variance, variance)
        
    hue_var = 15.0 * variation_strength
    sat_light_var = 0.20 * variation_strength
    
    for i in range(9):
        if i == 4:
            states.append(center.copy())
        else:
            mutated = center.copy()
            mutated["harmony_mode"] = mode  # type: ignore
            
            # Mutate each zone (Shadows, Midtones, Highlights)
            # Shadows
            mutated["shadow_hue"] = float((center["shadow_hue"] + j(hue_var)) % 360.0)
            mutated["shadow_sat"] = float(np.clip(center["shadow_sat"] + j(sat_light_var), 0.0, 0.50))
            mutated["shadow_light"] = float(center["shadow_light"])
            
            # Midtones
            mutated["midtone_hue"] = float((center["midtone_hue"] + j(hue_var)) % 360.0)
            mutated["midtone_sat"] = float(np.clip(center["midtone_sat"] + j(sat_light_var), 0.0, 0.50))
            mutated["midtone_light"] = float(center["midtone_light"])
            
            # Highlights
            mutated["highlight_hue"] = float((center["highlight_hue"] + j(hue_var)) % 360.0)
            mutated["highlight_sat"] = float(np.clip(center["highlight_sat"] + j(sat_light_var), 0.0, 0.50))
            mutated["highlight_light"] = float(center["highlight_light"])
            
            states.append(mutated)
            
    return states
