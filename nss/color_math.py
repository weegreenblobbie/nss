import cv2
import numpy as np
import random
from typing import TypedDict, Literal, List

# Type alias for mutation constraint axis
MutationAxis = Literal["All", "Hue", "Saturation", "Luminance"]

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

def apply_grading(img: np.ndarray, state: StateNode,
                  shadows_end_val: float = None,
                  shadow_exponent: float = None,
                  shadow_gain: float = None,
                  shadows_end_val_r: float = None,
                  shadows_end_val_g: float = None,
                  shadows_end_val_b: float = None,
                  shadow_exponent_r: float = None,
                  shadow_exponent_g: float = None,
                  shadow_exponent_b: float = None,
                  shadow_gain_r: float = None,
                  shadow_gain_g: float = None,
                  shadow_gain_b: float = None,
                  shadow_c0: float = None,
                  shadow_c1: float = None,
                  shadow_c2: float = None,
                  shadow_c3: float = None,
                  shadow_c4: float = None,
                  shadow_c5: float = None,
                  shadow_c0_r: float = None,
                  shadow_c1_r: float = None,
                  shadow_c2_r: float = None,
                  shadow_c3_r: float = None,
                  shadow_c4_r: float = None,
                  shadow_c5_r: float = None,
                  shadow_hue_w_r: List[float] = None,
                  shadow_hue_w_g: List[float] = None,
                  shadow_hue_w_b: List[float] = None,
                  midtone_gain: float = None,
                  midtone_gain_r: float = None,
                  midtone_gain_g: float = None,
                  midtone_gain_b: float = None,
                  mid_center_r: float = None,
                  mid_center_g: float = None,
                  mid_center_b: float = None,
                  mid_width_r: float = None,
                  mid_width_g: float = None,
                  mid_width_b: float = None,
                  mid_hue_w_r: List[float] = None,
                  mid_hue_w_g: List[float] = None,
                  mid_hue_w_b: List[float] = None) -> np.ndarray:
    """
    Applies professional 3-way zone-based color grading parameters from the state to the input image.
    Uses master blending and balance values to softly transition colors between zones.
    Returns a graded float32 image in [0.0, 1.0] RGB format.
    """
    # 1. Ensure input is RGB float32
    rgb = ensure_rgb(img).copy()
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)

    # === SSOT AUTO-INTEGRATED DEFAULTS START ===
    if shadows_end_val_r is None: shadows_end_val_r = 0.7000
    if shadows_end_val_g is None: shadows_end_val_g = 0.7000
    if shadow_exponent_r is None: shadow_exponent_r = 3.2158
    if shadow_exponent_g is None: shadow_exponent_g = 2.4037
    if shadow_gain_r is None: shadow_gain_r = 0.5008
    if shadow_gain_g is None: shadow_gain_g = 0.2572
    if shadow_gain_b is None: shadow_gain_b = 0.8868
    if shadow_c0 is None: shadow_c0 = 1.309182
    if shadow_c1 is None: shadow_c1 = -2.906569
    if shadow_c2 is None: shadow_c2 = -2.326751
    if shadow_c3 is None: shadow_c3 = 4.856898
    if shadow_c4 is None: shadow_c4 = 9.353697
    if shadow_c5 is None: shadow_c5 = -11.700607
    if shadow_c0_r is None: shadow_c0_r = 4.432887
    if shadow_c1_r is None: shadow_c1_r = -22.814433
    if shadow_c2_r is None: shadow_c2_r = 40.281313
    if shadow_c3_r is None: shadow_c3_r = -19.857885
    if shadow_c4_r is None: shadow_c4_r = -12.806680
    if shadow_c5_r is None: shadow_c5_r = 10.917661
    if shadow_hue_w_r is None: shadow_hue_w_r = [-0.891725, -0.08555, -1.0, -1.0, -1.0, -0.39296]
    if shadow_hue_w_g is None: shadow_hue_w_g = [-1.0, -1.0, -1.618074, -1.0, -1.0, -1.0]
    if shadow_hue_w_b is None: shadow_hue_w_b = [-0.421471, 1.012153, 0.321249, -1.0, -1.0, -1.0]
    if midtone_gain is None: midtone_gain = 1.0000
    if midtone_gain_r is None: midtone_gain_r = 0.384397
    if midtone_gain_g is None: midtone_gain_g = 0.252290
    if midtone_gain_b is None: midtone_gain_b = 0.254922
    if mid_center_r is None: mid_center_r = 0.584927
    if mid_center_g is None: mid_center_g = 0.616240
    if mid_center_b is None: mid_center_b = 0.493935
    if mid_width_r is None: mid_width_r = 0.222199
    if mid_width_g is None: mid_width_g = 0.225556
    if mid_width_b is None: mid_width_b = 0.300469
    if mid_hue_w_r is None: mid_hue_w_r = [0.262391, -4.236068, 0.0, 0.0, 0.0, 0.031369]
    if mid_hue_w_g is None: mid_hue_w_g = [0.0, -4.236068, -4.236068, -6.703841, 0.0, 0.0]
    if mid_hue_w_b is None: mid_hue_w_b = [-0.278594, 2.618034, 2.618034, 0.0, 0.0, 0.0]
    # === SSOT AUTO-INTEGRATED DEFAULTS END ===

    # 2. Compute pixel luminance using standard Perceptual Luma coefficients (Rec. 709)
    Y = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    L = Y

    # Calculate pixel's Saturation (S) and Hue (H)
    hls: np.ndarray = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    H: np.ndarray = hls[:, :, 0]
    S: np.ndarray = hls[:, :, 2]

    # 6-point Hue weight interpolation
    xp: np.ndarray = np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0], dtype=np.float32)
    
    yp_r: np.ndarray = np.array([shadow_hue_w_r[0], shadow_hue_w_r[1], shadow_hue_w_r[2],
                                 shadow_hue_w_r[3], shadow_hue_w_r[4], shadow_hue_w_r[5],
                                 shadow_hue_w_r[0]], dtype=np.float32)
    hue_weight_r: np.ndarray = np.interp(H, xp, yp_r)

    yp_g: np.ndarray = np.array([shadow_hue_w_g[0], shadow_hue_w_g[1], shadow_hue_w_g[2],
                                 shadow_hue_w_g[3], shadow_hue_w_g[4], shadow_hue_w_g[5],
                                 shadow_hue_w_g[0]], dtype=np.float32)
    hue_weight_g: np.ndarray = np.interp(H, xp, yp_g)

    yp_b: np.ndarray = np.array([shadow_hue_w_b[0], shadow_hue_w_b[1], shadow_hue_w_b[2],
                                 shadow_hue_w_b[3], shadow_hue_w_b[4], shadow_hue_w_b[5],
                                 shadow_hue_w_b[0]], dtype=np.float32)
    hue_weight_b: np.ndarray = np.interp(H, xp, yp_b)

    # 6-point Hue weight interpolation for midtones
    mid_yp_r: np.ndarray = np.array([mid_hue_w_r[0], mid_hue_w_r[1], mid_hue_w_r[2],
                                     mid_hue_w_r[3], mid_hue_w_r[4], mid_hue_w_r[5],
                                     mid_hue_w_r[0]], dtype=np.float32)
    mid_hue_weight_r: np.ndarray = np.interp(H, xp, mid_yp_r)

    mid_yp_g: np.ndarray = np.array([mid_hue_w_g[0], mid_hue_w_g[1], mid_hue_w_g[2],
                                     mid_hue_w_g[3], mid_hue_w_g[4], mid_hue_w_g[5],
                                     mid_hue_w_g[0]], dtype=np.float32)
    mid_hue_weight_g: np.ndarray = np.interp(H, xp, mid_yp_g)

    mid_yp_b: np.ndarray = np.array([mid_hue_w_b[0], mid_hue_w_b[1], mid_hue_w_b[2],
                                     mid_hue_w_b[3], mid_hue_w_b[4], mid_hue_w_b[5],
                                     mid_hue_w_b[0]], dtype=np.float32)
    mid_hue_weight_b: np.ndarray = np.interp(H, xp, mid_yp_b)

    # 3. Compute soft zone masks based on master balance and blending parameters
    balance = state.get("balance", 0.0)
    # Shift effective lightness used for mask boundaries based on balance
    L_shifted = np.clip(L - 0.3 * balance, 0.0, 1.0)
    
    blending = state.get("blending", 0.5)
    # Softness width scales from 0.01 (hard borders) up to 0.40 (highly feathered overlap)
    softness = max(0.01, 0.05 + 0.35 * blending)

    # Shadows Soft Mask (Squared quarter-cosine curve for smooth, natural falloff concentrating color in deep shadows and dropping off faster in mid-tones)
    end_val = shadows_end_val if shadows_end_val is not None else (0.3 + 0.4 * blending)
    exp_val = shadow_exponent if shadow_exponent is not None else 1.0000

    # Red
    if shadow_c0_r is not None:
        c0_r = shadow_c0_r
        c1_r = shadow_c1_r if shadow_c1_r is not None else 0.0
        c2_r = shadow_c2_r if shadow_c2_r is not None else 0.0
        c3_r = shadow_c3_r if shadow_c3_r is not None else 0.0
        c4_r = shadow_c4_r if shadow_c4_r is not None else 0.0
        c5_r = shadow_c5_r if shadow_c5_r is not None else 0.0
        x = Y
        mask_r = c0_r + c1_r * x + c2_r * (x ** 2) + c3_r * (x ** 3) + c4_r * (x ** 4) + c5_r * (x ** 5)
        mask_r = mask_r + S * hue_weight_r
        shadow_weight_r = np.clip(mask_r, 0.0, 1.0)
    else:
        end_r = shadows_end_val_r if shadows_end_val_r is not None else end_val
        t_shadow_r = np.clip((L - 0.02) / end_r, 0.0, 1.0)
        exp_r = shadow_exponent_r if shadow_exponent_r is not None else exp_val
        cos_val_r = np.clip(np.cos(t_shadow_r * np.pi / 2.0), 0.0, 1.0)
        mask_r = cos_val_r ** exp_r + S * hue_weight_r
        shadow_weight_r = np.clip(mask_r, 0.0, 1.0)

    # Green
    end_g = shadows_end_val_g if shadows_end_val_g is not None else end_val
    t_shadow_g = np.clip((L - 0.02) / end_g, 0.0, 1.0)
    exp_g = shadow_exponent_g if shadow_exponent_g is not None else exp_val
    cos_val_g = np.clip(np.cos(t_shadow_g * np.pi / 2.0), 0.0, 1.0)
    mask_g = cos_val_g ** exp_g + S * hue_weight_g
    shadow_weight_g = np.clip(mask_g, 0.0, 1.0)

    # Blue
    if shadow_c0 is not None:
        c0 = shadow_c0
        c1 = shadow_c1 if shadow_c1 is not None else 0.0
        c2 = shadow_c2 if shadow_c2 is not None else 0.0
        c3 = shadow_c3 if shadow_c3 is not None else 0.0
        c4 = shadow_c4 if shadow_c4 is not None else 0.0
        c5 = shadow_c5 if shadow_c5 is not None else 0.0
        x = Y
        mask_b = c0 + c1 * x + c2 * (x ** 2) + c3 * (x ** 3) + c4 * (x ** 4) + c5 * (x ** 5)
        mask_b = mask_b + S * hue_weight_b
        shadow_weight_b = np.clip(mask_b, 0.0, 1.0)
    else:
        end_b = shadows_end_val_b if shadows_end_val_b is not None else end_val
        t_shadow_b = np.clip((L - 0.02) / end_b, 0.0, 1.0)
        exp_b = shadow_exponent_b if shadow_exponent_b is not None else exp_val
        cos_val_b = np.clip(np.cos(t_shadow_b * np.pi / 2.0), 0.0, 1.0)
        mask_b = cos_val_b ** exp_b + S * hue_weight_b
        shadow_weight_b = np.clip(mask_b, 0.0, 1.0)

    shadow_weight = np.stack([shadow_weight_r, shadow_weight_g, shadow_weight_b], axis=2)

    # Highlights Soft Mask (centered at L = 0.7)
    highlight_weight = np.clip((L_shifted - (0.7 - softness/2.0)) / softness, 0.0, 1.0)

    # Midtones Soft Mask (fills the remaining space between highlights and shadows or uses Gaussian per channel)
    # Red Midtone Mask
    if mid_center_r is not None and mid_width_r is not None:
        midtone_weight_r = np.exp(-((Y - mid_center_r)**2) / (2 * mid_width_r**2))
        midtone_weight_r = np.clip(midtone_weight_r + S * mid_hue_weight_r, 0.0, 1.0)
    else:
        midtone_weight_r = np.clip(1.0 - shadow_weight_r - highlight_weight + S * mid_hue_weight_r, 0.0, 1.0)

    # Green Midtone Mask
    if mid_center_g is not None and mid_width_g is not None:
        midtone_weight_g = np.exp(-((Y - mid_center_g)**2) / (2 * mid_width_g**2))
        midtone_weight_g = np.clip(midtone_weight_g + S * mid_hue_weight_g, 0.0, 1.0)
    else:
        midtone_weight_g = np.clip(1.0 - shadow_weight_g - highlight_weight + S * mid_hue_weight_g, 0.0, 1.0)

    # Blue Midtone Mask
    if mid_center_b is not None and mid_width_b is not None:
        midtone_weight_b = np.exp(-((Y - mid_center_b)**2) / (2 * mid_width_b**2))
        midtone_weight_b = np.clip(midtone_weight_b + S * mid_hue_weight_b, 0.0, 1.0)
    else:
        midtone_weight_b = np.clip(1.0 - shadow_weight_b - highlight_weight + S * mid_hue_weight_b, 0.0, 1.0)

    midtone_weight = np.stack([midtone_weight_r, midtone_weight_g, midtone_weight_b], axis=2)

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

    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)

    # 5. Compute independent RGB deltas for each zone
    if sh_sat == 0.0:
        delta_sh = np.zeros_like(rgb)
    else:
        hls_sh = hls.copy()
        hls_sh[:, :, 0] = sh_hue
        hls_sh[:, :, 2] = sh_sat
        # Determine per-channel gains
        default_gain = shadow_gain if shadow_gain is not None else 0.2744
        gain_r = shadow_gain_r if shadow_gain_r is not None else default_gain
        gain_g = shadow_gain_g if shadow_gain_g is not None else default_gain
        gain_b = shadow_gain_b if shadow_gain_b is not None else default_gain
        
        gain = np.array([gain_r, gain_g, gain_b], dtype=np.float32)
        delta_sh = gain * (cv2.cvtColor(hls_sh, cv2.COLOR_HLS2RGB) - rgb)

    if mid_sat == 0.0:
        delta_mid = np.zeros_like(rgb)
    else:
        hls_mid = hls.copy()
        hls_mid[:, :, 0] = mid_hue
        hls_mid[:, :, 2] = mid_sat
        
        default_mid_gain = midtone_gain if midtone_gain is not None else 1.0
        mid_gain_r = midtone_gain_r if midtone_gain_r is not None else default_mid_gain
        mid_gain_g = midtone_gain_g if midtone_gain_g is not None else default_mid_gain
        mid_gain_b = midtone_gain_b if midtone_gain_b is not None else default_mid_gain
        
        mid_gain = np.array([mid_gain_r, mid_gain_g, mid_gain_b], dtype=np.float32)
        delta_mid = mid_gain * (cv2.cvtColor(hls_mid, cv2.COLOR_HLS2RGB) - rgb)

    if hi_sat == 0.0:
        delta_hi = np.zeros_like(rgb)
    else:
        hls_hi = hls.copy()
        hls_hi[:, :, 0] = hi_hue
        hls_hi[:, :, 2] = hi_sat
        delta_hi = cv2.cvtColor(hls_hi, cv2.COLOR_HLS2RGB) - rgb

    # Reshape weights for broadcasting (sw and mw are already 3-channel arrays of shape (H, W, 3))
    sw = shadow_weight
    mw = midtone_weight
    hw = highlight_weight[:, :, np.newaxis]

    # Combined RGB offset
    combined_offset = (sw * delta_sh) + (mw * delta_mid) + (hw * delta_hi)

    # Apply combined offset to the original RGB image
    rgb_graded = rgb + combined_offset

    # Apply Zone Lightness Adjustments as a luma offset if present
    sh_light = state.get("shadow_light", 0.0)
    mid_light = state.get("midtone_light", 0.0)
    hi_light = state.get("highlight_light", 0.0)
    if sh_light != 0.0 or mid_light != 0.0 or hi_light != 0.0:
        luma_offset = (sw * sh_light) + (mw * mid_light) + (hw * hi_light)
        rgb_graded += luma_offset

    rgb_graded = np.clip(rgb_graded, 0.0, 1.0)

    # 6. Apply Master Global Hue, Saturation, and Lightness shifts if present
    base_shift = state.get("hue_shift", 0.0)
    sat_shift = state.get("sat_shift", 0.0)
    light_shift = state.get("light_shift", 0.0)

    if base_shift != 0.0 or sat_shift != 0.0 or light_shift != 0.0:
        hls_graded = cv2.cvtColor(rgb_graded, cv2.COLOR_RGB2HLS)
        H_g = hls_graded[:, :, 0]
        L_g = hls_graded[:, :, 1]
        S_g = hls_graded[:, :, 2]

        if base_shift != 0.0:
            H_g = (H_g + base_shift) % 360.0
        if sat_shift != 0.0:
            S_g = np.clip(S_g + sat_shift, 0.0, 1.0)
        if light_shift != 0.0:
            L_g = np.clip(L_g + light_shift, 0.0, 1.0)

        hls_graded = np.stack([H_g, L_g, S_g], axis=2)
        rgb_graded = cv2.cvtColor(hls_graded, cv2.COLOR_HLS2RGB)

    return GradedArray(np.clip(rgb_graded, 0.0, 1.0), shadow_weight)

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
        
    # Scale bounds linearly: 100% (1.0) maps to max hue delta of 45.0 and sat delta of 0.25
    hue_var = 45.0 * variation_strength
    sat_light_var = 0.25 * variation_strength
    
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
