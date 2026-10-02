import cv2
import numpy as np
import random
import matplotlib.colors as mcolors
from dataclasses import dataclass
from typing import TypedDict, Literal, List, Tuple, Sequence, Dict, Optional, Union

# Type alias for mutation constraint axis
MutationAxis = Literal["All", "Hue", "Saturation", "Luminance"]

# Perceptual Hue Mapping: Calibrated against Adobe Lightroom color wheel
WHEEL_ANGLES = np.array([0.0, 26.0, 60.0, 153.0, 180.0, 239.0, 309.0, 360.0], dtype=np.float32)
HSL_ANGLES   = np.array([0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0], dtype=np.float32)


def wheel_to_hsl(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Maps visual color wheel angle(s) [0.0, 360.0] to mathematical HSL hue(s) [0.0, 360.0].
    Supports both scalar floats and NumPy arrays.
    """
    arr = np.asarray(angle)
    res = np.interp(arr % 360.0, WHEEL_ANGLES, HSL_ANGLES)
    if np.isscalar(angle):
        return float(res)
    return res


def hsl_to_wheel(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Maps mathematical HSL hue(s) [0.0, 360.0] to visual color wheel angle(s) [0.0, 360.0].
    Supports both scalar floats and NumPy arrays.
    """
    arr = np.asarray(angle)
    res = np.interp(arr % 360.0, HSL_ANGLES, WHEEL_ANGLES)
    if np.isscalar(angle):
        return float(res)
    return res

class GradedArray(np.ndarray):
    def __new__(cls, input_array, S_mask=None):
        obj = np.asarray(input_array).view(cls)
        obj.S_mask = S_mask
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.S_mask = getattr(obj, "S_mask", None)

    def __iter__(self):
        yield self.view(np.ndarray)
        yield self.S_mask

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

@dataclass(frozen=True)
class CalibrationProfile:
    """
    Calibration profile encapsulating Gaussian luma mask sigmas and 2x3 chrominance matrix.
    """
    sigma_sh: float = 0.5
    sigma_mid: float = 0.5
    sigma_hi: float = 0.5
    m00: float = 0.1
    m01: float = 0.1
    m10: float = 0.1
    m11: float = 0.1
    m20: float = 0.1
    m21: float = 0.1

    def to_dict(self) -> dict:
        return {
            "sigma_sh": float(self.sigma_sh),
            "sigma_mid": float(self.sigma_mid),
            "sigma_hi": float(self.sigma_hi),
            "m00": float(self.m00),
            "m01": float(self.m01),
            "m10": float(self.m10),
            "m11": float(self.m11),
            "m20": float(self.m20),
            "m21": float(self.m21),
        }

    def format_snippet(self, name: str) -> str:
        return (
            f'    "{name}": CalibrationProfile(\n'
            f'        sigma_sh={self.sigma_sh:.6f},\n'
            f'        sigma_mid={self.sigma_mid:.6f},\n'
            f'        sigma_hi={self.sigma_hi:.6f},\n'
            f'        m00={self.m00:.6f},\n'
            f'        m01={self.m01:.6f},\n'
            f'        m10={self.m10:.6f},\n'
            f'        m11={self.m11:.6f},\n'
            f'        m20={self.m20:.6f},\n'
            f'        m21={self.m21:.6f},\n'
            f'    ),'
        )


CALIBRATION_PROFILES: Dict[str, CalibrationProfile] = {
    "universal_base": CalibrationProfile(
        sigma_sh=0.423092,
        sigma_mid=0.379175,
        sigma_hi=0.050012,
        m00=0.217850,
        m01=0.018322,
        m10=-0.114965,
        m11=0.140136,
        m20=-0.073528,
        m21=-0.097759,
    ),
    "baseline": CalibrationProfile(
        sigma_sh=0.423092,
        sigma_mid=0.379175,
        sigma_hi=0.050012,
        m00=0.217850,
        m01=0.018322,
        m10=-0.114965,
        m11=0.140136,
        m20=-0.073528,
        m21=-0.097759,
    ),
}


def apply_grading(img: np.ndarray, state: StateNode, **kwargs) -> GradedArray:
    """
    Applies 3-way color grading using Gaussian relative luminance masks and UV chrominance projection.
    """
    rgb = ensure_rgb(img).astype(np.float32)

    profile = kwargs.get("profile")
    if profile is None:
        profile_name = kwargs.get("profile_name", "universal_base")
        profile = CALIBRATION_PROFILES.get(profile_name, CALIBRATION_PROFILES["universal_base"])

    sigma_sh = float(kwargs.get("sigma_sh", getattr(profile, "sigma_sh", 0.5)))
    sigma_mid = float(kwargs.get("sigma_mid", getattr(profile, "sigma_mid", 0.5)))
    sigma_hi = float(kwargs.get("sigma_hi", getattr(profile, "sigma_hi", 0.5)))
    m00 = float(kwargs.get("m00", getattr(profile, "m00", 0.1)))
    m01 = float(kwargs.get("m01", getattr(profile, "m01", 0.1)))
    m10 = float(kwargs.get("m10", getattr(profile, "m10", 0.1)))
    m11 = float(kwargs.get("m11", getattr(profile, "m11", 0.1)))
    m20 = float(kwargs.get("m20", getattr(profile, "m20", 0.1)))
    m21 = float(kwargs.get("m21", getattr(profile, "m21", 0.1)))

    # Relative Luminance
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

    # 3 Luma Masks
    mask_sh = np.exp(-((Y - 0.0) ** 2) / (2.0 * sigma_sh ** 2))
    mask_mid = np.exp(-((Y - 0.5) ** 2) / (2.0 * sigma_mid ** 2))
    mask_hi = np.exp(-((Y - 1.0) ** 2) / (2.0 * sigma_hi ** 2))

    rotation_offset = float(state.get("rotation", 0.0))
    sh_hue = (float(state.get("shadow_hue", 240.0)) + rotation_offset) % 360.0
    sh_sat = float(state.get("shadow_sat", 0.0))
    mid_hue = (float(state.get("midtone_hue", 120.0)) + rotation_offset) % 360.0
    mid_sat = float(state.get("midtone_sat", 0.0))
    hi_hue = (float(state.get("highlight_hue", 60.0)) + rotation_offset) % 360.0
    hi_sat = float(state.get("highlight_sat", 0.0))

    if state.get("harmony_mode", "Monochromatic") == "Complementary":
        sh_hue = (hi_hue + 180.0) % 360.0
        sh_sat = np.clip(sh_sat + 0.15, 0.0, 1.0)
        hi_sat = np.clip(hi_sat + 0.15, 0.0, 1.0)

    # Convert Hue/Sat to U/V for each zone
    U_sh = sh_sat * np.cos(np.radians(sh_hue))
    V_sh = sh_sat * np.sin(np.radians(sh_hue))
    delta_r_sh = m00 * U_sh + m01 * V_sh
    delta_g_sh = m10 * U_sh + m11 * V_sh
    delta_b_sh = m20 * U_sh + m21 * V_sh
    delta_rgb_sh = np.stack([delta_r_sh, delta_g_sh, delta_b_sh], axis=-1).astype(np.float32)

    U_mid = mid_sat * np.cos(np.radians(mid_hue))
    V_mid = mid_sat * np.sin(np.radians(mid_hue))
    delta_r_mid = m00 * U_mid + m01 * V_mid
    delta_g_mid = m10 * U_mid + m11 * V_mid
    delta_b_mid = m20 * U_mid + m21 * V_mid
    delta_rgb_mid = np.stack([delta_r_mid, delta_g_mid, delta_b_mid], axis=-1).astype(np.float32)

    U_hi = hi_sat * np.cos(np.radians(hi_hue))
    V_hi = hi_sat * np.sin(np.radians(hi_hue))
    delta_r_hi = m00 * U_hi + m01 * V_hi
    delta_g_hi = m10 * U_hi + m11 * V_hi
    delta_b_hi = m20 * U_hi + m21 * V_hi
    delta_rgb_hi = np.stack([delta_r_hi, delta_g_hi, delta_b_hi], axis=-1).astype(np.float32)

    # Apply the mask to scale the delta
    final_delta_rgb = (mask_sh[..., None] * delta_rgb_sh) + \
                      (mask_mid[..., None] * delta_rgb_mid) + \
                      (mask_hi[..., None] * delta_rgb_hi)

    output_img = np.clip(rgb + final_delta_rgb, 0.0, 1.0).astype(np.float32)

    # Global adjustments if present
    base_shift = float(state.get("hue_shift", 0.0))
    sat_shift = float(state.get("sat_shift", 0.0))
    light_shift = float(state.get("light_shift", 0.0))
    if base_shift != 0.0 or sat_shift != 0.0 or light_shift != 0.0:
        hls_g = cv2.cvtColor(output_img, cv2.COLOR_RGB2HLS)
        H_g, L_g, S_g = hls_g[:, :, 0], hls_g[:, :, 1], hls_g[:, :, 2]
        if base_shift != 0.0: H_g = (H_g + base_shift) % 360.0
        if sat_shift != 0.0: S_g = np.clip(S_g + sat_shift, 0.0, 1.0)
        if light_shift != 0.0: L_g = np.clip(L_g + light_shift, 0.0, 1.0)
        output_img = cv2.cvtColor(np.stack([H_g, L_g, S_g], axis=2).astype(np.float32), cv2.COLOR_HLS2RGB).astype(np.float32)
        output_img = np.clip(output_img, 0.0, 1.0)

    return GradedArray(output_img, mask_sh)

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
        
    # Scale bounds linearly: 100% (1.0) maps to max hue delta of 45.0 * 4.0 and sat delta of 0.25 * 4.0 (4x stronger delta)
    hue_var = 45.0 * variation_strength * 4.0
    sat_light_var = 0.25 * variation_strength * 4.0
    
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
