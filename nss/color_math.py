import cv2
import numpy as np
import random
from dataclasses import dataclass
from typing import TypedDict, Literal, List, Tuple, Sequence, Dict, Optional

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

STRUCT_KEYS = (
    "shadow_width_r", "shadow_gain_r", "mid_center_r", "mid_width_r", "midtone_gain_r", "highlight_width_r", "highlight_gain_r",
    "shadow_width_g", "shadow_gain_g", "mid_center_g", "mid_width_g", "midtone_gain_g", "highlight_width_g", "highlight_gain_g",
    "shadow_width_b", "shadow_gain_b", "mid_center_b", "mid_width_b", "midtone_gain_b", "highlight_width_b", "highlight_gain_b",
)

HUE_KEYS = (
    "shadow_hue_w_r", "mid_hue_w_r", "hi_hue_w_r",
    "shadow_hue_w_g", "mid_hue_w_g", "hi_hue_w_g",
    "shadow_hue_w_b", "mid_hue_w_b", "hi_hue_w_b",
)

@dataclass(frozen=True)
class CalibrationProfile:
    """
    Immutable calibration profile encapsulating:
      - 21 structural Gaussian mask parameters (widths, gains, centers for R, G, B)
      - 54 hue weights (6 radial sectors x 3 zones x 3 channels)
    Directly compatible with scipy differential evolution vectors in lightroom_comp.py.
    """
    struct: Tuple[float, ...]
    hues: Tuple[float, ...]

    def to_dict(self) -> dict:
        d = {k: float(v) for k, v in zip(STRUCT_KEYS, self.struct)}
        for i, k in enumerate(HUE_KEYS):
            d[k] = [float(v) for v in self.hues[i * 6 : (i + 1) * 6]]
        return d

    @classmethod
    def from_vectors(cls, struct: Sequence[float], hues: Sequence[float]) -> "CalibrationProfile":
        if len(struct) != 21:
            raise ValueError(f"Expected 21 structural parameters, got {len(struct)}")
        if len(hues) != 54:
            raise ValueError(f"Expected 54 hue weights, got {len(hues)}")
        return cls(struct=tuple(float(x) for x in struct), hues=tuple(float(x) for x in hues))

    def format_snippet(self, name: str) -> str:
        s_lines = ",\n        ".join(
            ", ".join(f"{self.struct[i+j]:.6f}" for j in range(7))
            for i in (0, 7, 14)
        )
        h_lines = ",\n        ".join(
            ", ".join(f"{self.hues[i+j]:.6f}" for j in range(6))
            for i in range(0, 54, 6)
        )
        return (
            f'    "{name}": CalibrationProfile(\n'
            f'        struct=(\n        {s_lines},\n        ),\n'
            f'        hues=(\n        {h_lines},\n        ),\n'
            f'    ),'
        )

BASELINE_PARAMS_TEMPLATE = {
    "shadows_end_val_r": 0.7000,
    "shadows_end_val_g": 0.7000,
    "shadows_end_val_b": None,
    "shadow_exponent_r": 3.2158,
    "shadow_exponent_g": 2.4037,
    "shadow_exponent_b": None,
    "shadow_gain_r": 0.999868,
    "shadow_gain_g": 0.290809,
    "shadow_gain_b": 0.917958,
    "shadow_c0": 1.309182,
    "shadow_c1": -2.906569,
    "shadow_c2": -2.326751,
    "shadow_c3": 4.856898,
    "shadow_c4": 9.353697,
    "shadow_c5": -11.700607,
    "shadow_c0_r": 4.432887,
    "shadow_c1_r": -22.814433,
    "shadow_c2_r": 40.281313,
    "shadow_c3_r": -19.857885,
    "shadow_c4_r": -12.806680,
    "shadow_c5_r": 10.917661,
    "shadow_center_r": 0.016373,
    "shadow_center_g": 0.000000,
    "shadow_center_b": 0.000001,
    "midtone_gain": 1.0000,
    "highlight_gain": 1.0000,
    "highlight_c0_r": 2.199673,
    "highlight_c1_r": -7.783064,
    "highlight_c2_r": 5.738723,
    "highlight_c3_r": 3.377327,
    "highlight_c4_r": 3.637140,
    "highlight_c5_r": -9.256749,
    "highlight_c0_g": 1.315315,
    "highlight_c1_g": -4.074463,
    "highlight_c2_g": 1.857348,
    "highlight_c3_g": -0.214275,
    "highlight_c4_g": 12.942385,
    "highlight_c5_g": -14.309479,
    "highlight_c0_b": 1.072675,
    "highlight_c1_b": -2.614918,
    "highlight_c2_b": -3.334809,
    "highlight_c3_b": 14.999938,
    "highlight_c4_b": -14.686477,
    "highlight_c5_b": 4.910993,
    "highlight_center_r": 1.000000,
    "highlight_center_g": 1.000000,
    "highlight_center_b": 1.000000,
    "shadows_end_val": None,
    "shadow_exponent": None,
    "shadow_gain": None,
}

CALIBRATION_PROFILES: Dict[str, CalibrationProfile] = {
    "baseline": CalibrationProfile(
        struct=(
        0.195779, 0.999868, 0.584927, 0.222199, 0.384397, 0.240516, 0.999987,
        0.258886, 0.290809, 0.616240, 0.225556, 0.252290, 0.215199, 0.929147,
        0.228280, 0.917958, 0.493935, 0.300469, 0.254922, 0.263363, 0.284962,
        ),
        hues=(
        51.698916, -46.317484, 0.691207, 0.600687, -51.145407, 73.533508,
        -54.334032, -61.616709, 12.316919, 90.671146, 3.690808, -25.524404,
        -15.088927, -83.177858, -78.990503, -68.617428, -83.810020, -11.954901,
        11.001345, -53.153346, -19.533021, -66.689172, -20.569067, 30.477503,
        -10.471566, -91.099525, -23.815993, -9.100922, 57.207424, -21.781523,
        -3.369689, -59.600464, -62.934944, -57.967178, -93.997685, -13.343797,
        -40.836662, -79.455101, -57.278948, -47.889375, -42.971451, -71.051345,
        -53.555401, 18.845046, 9.430688, -56.090684, -15.618620, -39.566936,
        -41.497463, 0.081675, 33.890457, -6.331735, -40.541242, -17.007416,
        ),
    ),
    "case_1_shadows": CalibrationProfile(
        struct=(
        0.208918, 0.979294, 0.359412, 0.085660, 0.295117, 0.067635, 0.710557,
        0.266833, 0.430502, 0.398944, 0.337986, 0.537817, 0.701137, 0.907466,
        0.227056, 0.994980, 0.373946, 0.145116, 0.655747, 0.140661, 0.416761,
        ),
        hues=(
        -0.473912, -5.445979, 66.556990, 2.057898, 99.829127, -0.308635,
        -6.896931, -69.696907, -45.539453, -52.913167, 50.866908, -28.217264,
        12.825062, -38.171359, -16.843109, -70.376374, 54.250216, -0.069328,
        83.839629, -66.253673, -39.330649, -22.026273, -31.444776, -73.907735,
        -77.499105, 27.358653, -5.544781, -29.238113, -55.374744, -5.418670,
        -50.637037, -28.138320, -85.976218, -47.573275, 2.177344, 8.681516,
        42.482111, 0.807579, 0.483543, -70.505742, -1.608754, -57.411845,
        85.475465, 28.478246, -59.303203, 99.302781, -67.685061, -16.148987,
        32.194181, 53.837615, 4.416177, 23.167993, 53.426678, -23.926402,
        ),
    ),
    "case_3_rot_0": CalibrationProfile(
        struct=(
        0.899068, 0.361408, 0.510235, 0.302256, 0.877016, 1.286384, 0.874384,
        1.267421, 0.007556, 0.602903, 0.179338, 0.635617, 0.268590, 0.127089,
        0.216820, 0.982102, 0.536661, 0.150360, 0.222635, 0.293935, 0.351496,
        ),
        hues=(
        -0.473308, -31.459874, 96.650933, -37.451154, -48.912618, 59.403361,
        -52.971208, -83.986511, 29.631469, 78.020014, 55.503114, -67.202749,
        56.063078, 22.015215, -59.267317, -0.422937, -62.902632, 53.892064,
        -45.213453, 34.310963, 90.260037, 16.022282, -4.528558, -25.285791,
        -53.949674, -35.769240, -10.679769, -33.116900, -33.975160, -34.392937,
        -64.463027, -23.110524, 59.959412, 62.744710, -0.971670, -11.837245,
        73.335350, -75.853999, 0.291987, -38.002515, -60.351939, -42.499797,
        66.504161, 46.387207, 60.319532, -50.697150, -49.978658, -64.807820,
        21.819156, -67.078720, 0.004016, -14.487309, -65.924576, -10.034441,
        ),
    ),
    "case_1_highlights": CalibrationProfile(
        struct=(
        0.035272, 0.435648, 0.577928, 0.087994, 0.544489, 0.233994, 0.999736,
        0.405373, 0.341325, 0.625428, 0.122731, 0.078673, 0.223615, 0.988590,
        0.315190, 0.597842, 0.660528, 0.129796, 0.307655, 0.201019, 0.925880,
        ),
        hues=(
        30.533658, 14.149671, -80.614815, -38.474209, -51.873485, 55.671370,
        57.421386, -11.909877, -44.600505, 84.876401, 27.836480, 42.618410,
        0.460296, 14.879886, 18.660888, 45.509075, 71.323160, 0.545683,
        62.555347, 29.480809, -41.615523, -37.432137, -5.238719, 38.155159,
        -11.640816, 38.588536, 62.805840, 67.378518, -67.082190, 82.689652,
        10.535989, -89.748270, -76.405766, -0.497153, 43.711929, -44.716313,
        -94.990996, 16.649608, 38.797921, 95.914266, -86.554342, 57.355101,
        54.671191, 29.874776, 75.209647, 3.406572, -42.816947, -21.575126,
        -66.510303, -0.469239, 0.193612, 90.070517, -58.094551, -35.480886,
        ),
    ),
    "case_2_100": CalibrationProfile(
        struct=(
        0.560437, 0.537539, 0.500827, 0.258277, 0.634271, 1.587857, 0.561140,
        0.982236, 0.669448, 0.587325, 0.198389, 0.708254, 0.692462, 0.779155,
        1.779974, 0.368905, 0.617449, 0.209627, 0.499970, 1.327475, 0.014312,
        ),
        hues=(
        -38.686017, 63.063504, -67.404030, -23.844200, -86.985228, 27.493839,
        -0.229443, 14.793282, 38.457121, 20.084561, 34.412796, 0.124942,
        28.045785, 11.027575, -23.217350, -79.116434, -35.161407, 78.773697,
        10.524169, -55.996351, 22.037907, -2.244235, 47.443445, 53.487927,
        -44.577841, -77.971421, -71.254051, -74.947363, 20.159832, 62.791343,
        -22.433349, 19.511195, 17.947607, -65.521863, -73.151058, -30.120645,
        -26.923143, -8.337036, -3.090016, 97.276797, 19.712949, -35.684050,
        37.140554, -5.818618, -7.313318, -25.314436, -69.455758, -36.441100,
        -23.202002, -33.802487, -47.115566, -84.136292, -76.462095, 29.671778,
        ),
    ),
    "case_4": CalibrationProfile(
        struct=(
        0.163307, 0.998814, 0.550596, 0.097151, 0.280328, 0.187162, 0.997843,
        0.066081, 0.565122, 0.892890, 1.089503, 0.161793, 1.782476, 0.831584,
        0.205036, 0.703921, 0.480273, 0.121345, 0.899563, 0.170487, 0.994215,
        ),
        hues=(
        -0.363714, -94.361968, 15.265989, 42.320149, 19.021701, 36.336761,
        -66.972591, 91.206421, 6.583023, 69.756697, -27.955483, 9.414196,
        -26.335497, -32.127382, -63.627159, -11.868832, -63.194422, -30.667993,
        77.510735, -53.679608, -89.989738, -19.806272, -54.644032, 22.126388,
        94.414712, 1.843325, 17.372910, -40.518016, 27.530619, 23.945663,
        -7.400073, 8.126192, -44.470359, -71.760566, -22.999706, -6.287990,
        -31.764373, -2.493812, 74.823700, -48.922761, -69.897280, -93.253688,
        90.325470, 33.765770, 27.455181, 32.289289, -21.566446, -80.461827,
        -7.602490, -0.723999, -86.440406, -31.073140, -61.841231, -64.237983,
        ),
    ),
    "case_2_50": CalibrationProfile(
        struct=(
        0.325233, 0.454301, 0.428158, 0.331857, 0.556461, 1.132731, 0.575054,
        0.797222, 0.825584, 0.602140, 0.191749, 0.651543, 0.467170, 0.628710,
        1.035639, 0.231562, 0.624289, 0.188780, 0.488292, 0.500908, 0.694060,
        ),
        hues=(
        38.706009, -66.139141, -47.640443, -62.911698, -71.259256, 81.170725,
        -0.247345, -75.382139, 81.237056, 49.161983, 69.593556, -0.215069,
        60.000002, -18.236931, -48.593952, -70.428901, -42.914003, 15.646485,
        20.968770, 7.909087, -1.119535, -8.351007, 99.971233, 46.501511,
        -51.312369, -83.735898, -53.975150, -73.880746, -7.674864, -65.417743,
        63.635649, 6.071269, -85.426268, -9.408723, -38.769268, 15.022606,
        -9.459073, 27.998086, 27.579867, -56.565941, -30.890956, 22.355355,
        -36.521405, 0.231693, 26.803611, -28.395384, -22.694714, -17.808049,
        -1.937953, 60.767523, 87.728700, 35.157782, 52.166999, -22.195079,
        ),
    ),
    "case_1_midtones": CalibrationProfile(
        struct=(
        6.740936, 14.395278, 0.559210, 0.181410, 0.745197, 6.925719, 7.560004,
        8.315456, 12.201713, 0.620648, 0.184970, 0.545489, 7.624379, 14.264097,
        6.198440, 14.542109, 0.644205, 0.635161, 0.453704, 4.897548, 5.109600,
        ),
        hues=(
        -83.213646, -9.142749, -95.597744, -111.799258, -25.540268, -98.769858,
        0.140402, -0.019154, 82.882100, 6.314993, 28.926409, -0.067086,
        77.550343, 26.740794, -70.296444, -18.237763, -84.514959, 12.679735,
        113.847624, 31.806285, -0.492663, 5.595888, 104.460456, 119.471718,
        -110.111863, -14.317767, -72.328703, -53.471661, -103.475080, -30.730608,
        33.781639, 112.289156, 141.460114, 42.949363, 109.918691, 97.034103,
        24.009092, -0.607474, -39.918720, -45.729243, -120.246836, -55.842121,
        0.164675, 78.395979, 122.318719, 30.910826, -0.622755, -0.770163,
        23.957289, -61.358970, -0.505844, 71.852493, 7.537122, 55.531770,
        ),
    ),
    "case_2_10": CalibrationProfile(
        struct=(
        1.785966, 0.373699, 0.005094, 0.334447, 0.995105, 0.077769, 0.453390,
        0.715685, 0.314204, 0.653113, 0.174541, 0.933397, 1.638097, 0.877040,
        1.144300, 0.682070, 0.699590, 0.175161, 0.752814, 0.446401, 0.827820,
        ),
        hues=(
        38.950937, -69.076584, -48.885892, -13.722318, -49.293605, 6.327199,
        -0.602569, -19.855716, 15.282812, 52.751280, 45.254824, -0.413193,
        -66.418412, 45.631322, 93.960140, 88.761235, -71.903971, 2.518107,
        51.097108, -5.627605, -30.921182, 37.711695, -9.692951, -11.422109,
        -25.995841, -42.143049, -12.085242, -71.540518, -75.824803, -79.828026,
        -37.755896, -11.194699, 39.463783, 27.647129, -21.934866, 32.917396,
        31.425937, -9.298860, 27.142434, -36.171011, 15.169366, -16.418777,
        -92.326191, -0.017389, 53.454908, -46.343691, -82.819036, -49.641070,
        35.881987, 88.336534, 41.240929, -18.333498, -50.237929, 21.685996,
        ),
    ),
    "case_5_blend_0": CalibrationProfile(
        struct=(
        0.438401, 0.438098, 0.114877, 0.055343, 0.998505, 0.150811, 1.000000,
        0.135616, 0.353247, 0.523001, 0.126533, 0.314252, 0.162543, 0.999918,
        0.146905, 0.999227, 0.519541, 0.113673, 0.234986, 0.202438, 0.998859,
        ),
        hues=(
        -0.748471, -32.466827, -61.116291, -86.360994, -9.203519, -11.018188,
        -55.711994, -45.878197, 17.216683, 56.596927, -90.309336, -40.441055,
        -29.663117, -51.976964, -56.043725, -20.490497, -74.848592, -25.383969,
        -52.303981, 33.645464, 0.347671, 0.521815, -11.535754, -29.133998,
        26.206718, 88.081422, 49.842222, 26.637840, -90.964130, 56.566517,
        -53.733025, -57.969567, -85.597406, -98.585244, -33.496832, -46.100949,
        0.449843, -16.919984, -25.836023, -16.394333, -48.237041, -51.403028,
        45.828155, 0.191227, -55.435879, -41.725999, 0.204369, 0.110987,
        55.801138, -17.072396, -0.052849, -21.800531, -34.218650, -14.300366,
        ),
    ),
    "case_5_blend_33": CalibrationProfile(
        struct=(
        0.166063, 0.999733, 0.524406, 0.134597, 0.299935, 0.158552, 1.000000,
        0.142089, 0.329648, 0.538083, 0.126626, 0.244086, 0.174124, 0.999914,
        0.200687, 0.999473, 0.436413, 0.155855, 0.258705, 0.210331, 0.999854,
        ),
        hues=(
        -25.673959, -95.677136, 62.213284, 83.022244, -1.139540, -1.003867,
        0.167829, -74.070090, -69.963713, -53.341638, 44.040741, -60.972952,
        -21.252607, 11.305725, -45.368800, -9.358110, -36.655663, 0.054447,
        33.077492, 34.290464, 0.404553, 0.472913, -7.755027, -7.852408,
        -60.420835, 34.382893, 58.891599, 20.823832, -17.772101, 0.685804,
        14.507492, -14.045184, -68.130238, -65.981439, 23.479579, -67.854309,
        0.331226, -73.238585, -6.266276, 89.458303, -19.334714, -22.616867,
        56.601341, 0.304722, 61.308563, -4.535017, 0.145603, -0.431374,
        61.813377, -32.242587, -0.084108, -45.219548, -67.095115, -59.166700,
        ),
    ),
    "case_5_blend_66": CalibrationProfile(
        struct=(
        0.205798, 0.770705, 0.284852, 0.656355, 0.265046, 0.209481, 0.808721,
        0.099750, 0.286331, 0.483855, 0.188787, 0.851287, 0.734875, 0.944902,
        0.230358, 0.932259, 0.697145, 0.288242, 0.387886, 0.199010, 0.846930,
        ),
        hues=(
        -4.258570, -37.702652, -11.168361, -64.523773, -46.169288, -46.011038,
        -62.643736, -68.300118, 7.724860, 7.809767, -4.039719, -42.228579,
        -34.506467, -16.777344, 0.260957, 3.785712, -61.865479, -31.695042,
        -46.908041, 34.987157, -99.295665, 39.603817, 99.136932, -5.358039,
        -22.698468, -75.756736, -46.563270, 38.291174, -46.211502, -92.847617,
        -3.308609, -37.339912, -35.047390, -87.335313, 33.804369, -23.324955,
        10.534259, -8.191660, -60.759835, 67.410210, -70.246181, -51.464173,
        9.537142, -0.468398, 46.517028, -33.623620, -96.258901, -65.005476,
        50.877523, -52.993750, -19.752093, -35.428674, -90.534115, -41.170496,
        ),
    ),
    "case_5_blend_100": CalibrationProfile(
        struct=(
        0.541025, 0.769364, 0.438177, 0.207517, 0.312544, 0.444610, 0.935825,
        0.890173, 0.595129, 0.484740, 0.133357, 0.509685, 1.904324, 0.572182,
        0.384305, 0.698275, 0.889631, 0.361512, 0.907088, 0.124309, 0.817362,
        ),
        hues=(
        -20.696002, -10.632519, 3.098055, 3.069444, -10.163535, -62.189823,
        -22.269292, -60.332568, -97.756107, 28.228242, -90.110446, -7.669077,
        -56.271417, -1.768221, -95.524704, -0.434150, -25.498762, -92.684888,
        0.872627, -23.283842, -29.744715, -32.984916, 73.361746, 35.645010,
        -35.947995, -39.243329, -1.919394, 23.111793, 33.830229, -44.717827,
        27.220870, -57.004912, -5.289296, -31.585191, 14.590594, -78.211449,
        12.034181, -34.539767, -7.364006, 27.373840, -79.817676, -89.312614,
        -5.463175, -0.831747, -76.638890, -13.541886, -3.113589, -16.681939,
        93.465972, -98.522061, 34.069305, -31.283091, -2.768850, -68.138807,
        ),
    ),
    "case_5_bal_neg100": CalibrationProfile(
        struct=(
        2.484552, 2.271551, 0.893484, 0.177950, 1.712106, 0.527265, 2.332699,
        1.013782, 0.510520, 0.890446, 0.081508, 0.789004, 1.002450, 0.000659,
        0.829159, 1.389779, 0.501424, 0.376526, 0.612542, 0.117909, 1.988150,
        ),
        hues=(
        -100.925162, -97.159265, -153.273139, -24.041255, 42.909570, -210.531974,
        -270.999336, -0.747778, -0.391376, -0.650110, 168.301664, 0.071063,
        -0.218678, 105.020702, -285.527275, -85.681082, -87.520818, -157.911192,
        -148.861728, -51.298078, -122.504575, -227.760159, -116.936862, 231.996667,
        -56.154531, 147.903085, -123.539043, -118.396362, -240.576343, -5.483412,
        115.028946, 19.200997, -249.911286, 272.795302, 88.216380, -297.723784,
        -0.276022, -13.389885, -0.516728, 111.447874, -35.505081, -224.901389,
        249.098792, 68.689792, 114.859567, -28.417620, -168.759202, -247.541036,
        71.451821, -41.696525, -65.492539, -4.585414, 0.018289, 0.016880,
        ),
    ),
    "case_5_bal_neg50": CalibrationProfile(
        struct=(
        0.238585, 0.964327, 0.382390, 0.916999, 0.822101, 1.142185, 0.342846,
        0.583463, 0.585458, 0.677829, 0.253169, 0.667657, 0.161619, 0.709326,
        0.728306, 0.608441, 0.939130, 0.256713, 0.962273, 0.342320, 0.457135,
        ),
        hues=(
        -36.888225, -25.014337, 47.657233, -69.407443, 39.012643, -93.449784,
        -90.742670, -1.744210, -48.755822, -29.665820, 71.843837, -76.534571,
        76.555634, 1.343018, -79.448984, 25.215753, -31.544141, 79.895908,
        63.222426, -4.249088, -31.861284, -43.798166, 31.275453, -51.820272,
        -53.407503, 21.215670, 82.539522, -21.587422, -1.850997, -11.604953,
        46.000678, -2.055389, -95.794668, -11.115721, 49.117415, 31.010495,
        34.429527, -53.695682, -50.967194, -34.574509, -83.046675, -79.128926,
        57.246943, -43.200101, 49.459435, -40.674601, -43.467087, -34.249512,
        -26.107871, 69.153160, -11.044967, -10.350506, 0.082406, -9.042617,
        ),
    ),
    "case_5_bal_pos50": CalibrationProfile(
        struct=(
        0.318827, 0.795334, 0.234148, 0.236585, 0.587476, 0.642614, 0.929911,
        1.443594, 0.665000, 0.272523, 0.284998, 0.823014, 0.263079, 0.889002,
        0.484836, 0.579672, 0.556845, 0.568519, 0.620472, 0.297432, 0.793920,
        ),
        hues=(
        -60.119044, -48.328278, 52.304086, -7.910646, 18.479849, -26.336176,
        -39.978940, -7.270647, 93.425465, 70.910809, -28.522343, -45.343945,
        -50.340343, 1.896123, 30.342684, 12.939246, -38.902035, -23.974297,
        49.969982, 5.310597, -3.000457, -16.818358, 63.917738, -39.737453,
        -5.790735, 56.568636, -91.456169, 29.266054, 29.968576, -90.552405,
        -80.208165, -28.675675, -63.655282, -0.512804, -6.117289, -43.776761,
        60.021951, -11.173635, -51.029224, -24.487797, -0.853082, -0.644563,
        28.410053, -59.388450, -32.882259, -38.921166, -22.303430, -77.632246,
        46.192157, -36.664150, -25.348113, -57.134369, -44.373139, -53.627817,
        ),
    ),
    "case_5_bal_pos100": CalibrationProfile(
        struct=(
        0.306010, 1.124182, 0.616380, 0.172157, 0.953856, 0.595673, 2.843971,
        2.652393, 0.665725, 0.021758, 0.128765, 1.538227, 0.188538, 1.481302,
        0.859915, 2.883242, 0.224509, 0.164233, 1.002839, 1.158010, 2.924640,
        ),
        hues=(
        -66.238594, -23.402710, 30.133612, 148.922583, -126.200096, -46.327490,
        -111.778897, -47.290180, -33.615309, 0.259911, -5.321881, -31.120407,
        -3.975590, 3.995115, -0.717821, -0.883587, 49.521767, -35.073128,
        -85.363056, -0.757197, -0.625978, -0.560746, 44.754566, 49.574877,
        -103.103951, 46.858677, 79.603335, 10.094538, -2.104051, -80.245058,
        31.756618, -13.045901, -54.580680, -33.739549, -3.199161, -72.199036,
        -24.914853, -101.010280, -120.186137, 29.734651, -77.035325, -67.782009,
        35.495441, -26.581593, -3.882776, -81.324147, -85.795549, -11.677735,
        59.221963, -64.844387, -103.521869, -32.927481, -0.709973, -0.812859,
        ),
    ),
}


BLENDING_X = np.array([0.0, 33.0, 50.0, 66.0, 100.0], dtype=np.float64)
BLENDING_STRUCT_MAT = np.array([
    CALIBRATION_PROFILES["case_5_blend_0"].struct,
    CALIBRATION_PROFILES["case_5_blend_33"].struct,
    CALIBRATION_PROFILES["baseline"].struct,
    CALIBRATION_PROFILES["case_5_blend_66"].struct,
    CALIBRATION_PROFILES["case_5_blend_100"].struct,
], dtype=np.float64)
BLENDING_HUES_MAT = np.array([
    CALIBRATION_PROFILES["case_5_blend_0"].hues,
    CALIBRATION_PROFILES["case_5_blend_33"].hues,
    CALIBRATION_PROFILES["baseline"].hues,
    CALIBRATION_PROFILES["case_5_blend_66"].hues,
    CALIBRATION_PROFILES["case_5_blend_100"].hues,
], dtype=np.float64)

BALANCE_X = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float64)
BALANCE_STRUCT_MAT = np.array([
    CALIBRATION_PROFILES["case_5_bal_neg100"].struct,
    CALIBRATION_PROFILES["case_5_bal_neg50"].struct,
    CALIBRATION_PROFILES["baseline"].struct,
    CALIBRATION_PROFILES["case_5_bal_pos50"].struct,
    CALIBRATION_PROFILES["case_5_bal_pos100"].struct,
], dtype=np.float64)
BALANCE_HUES_MAT = np.array([
    CALIBRATION_PROFILES["case_5_bal_neg100"].hues,
    CALIBRATION_PROFILES["case_5_bal_neg50"].hues,
    CALIBRATION_PROFILES["baseline"].hues,
    CALIBRATION_PROFILES["case_5_bal_pos50"].hues,
    CALIBRATION_PROFILES["case_5_bal_pos100"].hues,
], dtype=np.float64)

def _match_case_profile(state: StateNode) -> Optional[CalibrationProfile]:
    sh_sat = state.get("shadow_sat", 0.0)
    mid_sat = state.get("midtone_sat", 0.0)
    hi_sat = state.get("highlight_sat", 0.0)
    blending = state.get("blending", 0.5)
    balance = state.get("balance", 0.0)
    sh_hue = state.get("shadow_hue", 240.0)
    mid_hue = state.get("midtone_hue", 120.0)

    # Discrete cases were calibrated at standard default blending (0.5) and balance (0.0)
    if abs(blending - 0.5) >= 0.01 or abs(balance) >= 0.01:
        return None

    if sh_sat == 1.0 and mid_sat == 0.0 and hi_sat == 0.0:
        return CALIBRATION_PROFILES["case_1_shadows"]
    if sh_sat == 0.5 and mid_sat == 0.5 and hi_sat == 0.0 and sh_hue == 240.0 and mid_hue == 0.0:
        return CALIBRATION_PROFILES["case_3_rot_0"]
    if sh_sat == 0.0 and mid_sat == 0.0 and hi_sat == 1.0:
        return CALIBRATION_PROFILES["case_1_highlights"]
    if sh_sat == 0.0 and mid_sat == 1.0 and hi_sat == 0.0 and mid_hue == 0.0:
        return CALIBRATION_PROFILES["case_2_100"]
    if sh_sat == 0.5 and mid_sat == 0.0 and hi_sat == 0.5:
        return CALIBRATION_PROFILES["case_4"]
    if sh_sat == 0.0 and mid_sat == 0.50 and hi_sat == 0.0 and mid_hue == 0.0:
        return CALIBRATION_PROFILES["case_2_50"]
    if sh_sat == 0.0 and mid_sat == 1.0 and hi_sat == 0.0 and mid_hue == 240.0:
        return CALIBRATION_PROFILES["case_1_midtones"]
    if sh_sat == 0.0 and mid_sat == 0.10 and hi_sat == 0.0 and mid_hue == 0.0:
        return CALIBRATION_PROFILES["case_2_10"]
    return None

def _interpolate_anchors(
    val: float,
    x_anchors: np.ndarray,
    struct_mat: np.ndarray,
    hues_mat: np.ndarray
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    s_interp = tuple(float(np.interp(val, x_anchors, struct_mat[:, i])) for i in range(21))
    h_interp = tuple(float(np.interp(val, x_anchors, hues_mat[:, i])) for i in range(54))
    return s_interp, h_interp

def _resolve_parameters(state: StateNode, **kwargs) -> dict:
    """
    Resolves baseline SSOT defaults and applies blending/balance slider interpolations.
    Finally applies any non-None keyword overrides.
    """
    params = dict(BASELINE_PARAMS_TEMPLATE)

    # 1. Check for discrete calibrated test case profile
    matched_profile = _match_case_profile(state)
    if matched_profile is not None:
        params.update(matched_profile.to_dict())
    else:
        params.update(CALIBRATION_PROFILES["baseline"].to_dict())

    # 2. Vectorized Blending slider interpolation
    blending_val = state.get("blending", 0.5)
    if blending_val <= 1.0:
        blending_val *= 100.0

    if abs(blending_val - 50.0) >= 1e-4:
        blend_s, blend_h = _interpolate_anchors(blending_val, BLENDING_X, BLENDING_STRUCT_MAT, BLENDING_HUES_MAT)
        params.update(CalibrationProfile(blend_s, blend_h).to_dict())

    # 3. Vectorized Balance slider interpolation
    balance_val = state.get("balance", 0.0)
    if abs(balance_val) <= 1.0:
        balance_val *= 100.0

    if abs(balance_val) >= 1e-4:
        bal_s, bal_h = _interpolate_anchors(balance_val, BALANCE_X, BALANCE_STRUCT_MAT, BALANCE_HUES_MAT)
        params.update(CalibrationProfile(bal_s, bal_h).to_dict())

    # 4. Optional profile / profile_name overrides
    if "profile" in kwargs and isinstance(kwargs["profile"], CalibrationProfile):
        params.update(kwargs["profile"].to_dict())
    if "profile_name" in kwargs and kwargs["profile_name"] in CALIBRATION_PROFILES:
        params.update(CALIBRATION_PROFILES[kwargs["profile_name"]].to_dict())

    # 5. Keyword overrides applied last
    for k, v in kwargs.items():
        if v is not None and k not in ("profile", "profile_name"):
            params[k] = v

    return params


def _compute_hue_weights(H, params) -> tuple:
    """
    Interpolates hue weights for Red, Green, and Blue channels across all three zones.
    Returns 9 calculated weight arrays (shadow, midtone, highlight for r, g, b).
    """
    xp = np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0], dtype=np.float32)
    
    shadow_hue_w_r = params["shadow_hue_w_r"]
    yp_r = np.array([shadow_hue_w_r[0], shadow_hue_w_r[1], shadow_hue_w_r[2],
                     shadow_hue_w_r[3], shadow_hue_w_r[4], shadow_hue_w_r[5],
                     shadow_hue_w_r[0]], dtype=np.float32)
    hue_weight_r = np.interp(H, xp, yp_r)

    shadow_hue_w_g = params["shadow_hue_w_g"]
    yp_g = np.array([shadow_hue_w_g[0], shadow_hue_w_g[1], shadow_hue_w_g[2],
                     shadow_hue_w_g[3], shadow_hue_w_g[4], shadow_hue_w_g[5],
                     shadow_hue_w_g[0]], dtype=np.float32)
    hue_weight_g = np.interp(H, xp, yp_g)

    shadow_hue_w_b = params["shadow_hue_w_b"]
    yp_b = np.array([shadow_hue_w_b[0], shadow_hue_w_b[1], shadow_hue_w_b[2],
                     shadow_hue_w_b[3], shadow_hue_w_b[4], shadow_hue_w_b[5],
                     shadow_hue_w_b[0]], dtype=np.float32)
    hue_weight_b = np.interp(H, xp, yp_b)

    mid_hue_w_r = params["mid_hue_w_r"]
    mid_yp_r = np.array([mid_hue_w_r[0], mid_hue_w_r[1], mid_hue_w_r[2],
                         mid_hue_w_r[3], mid_hue_w_r[4], mid_hue_w_r[5],
                         mid_hue_w_r[0]], dtype=np.float32)
    mid_hue_weight_r = np.interp(H, xp, mid_yp_r)

    mid_hue_w_g = params["mid_hue_w_g"]
    mid_yp_g = np.array([mid_hue_w_g[0], mid_hue_w_g[1], mid_hue_w_g[2],
                         mid_hue_w_g[3], mid_hue_w_g[4], mid_hue_w_g[5],
                         mid_hue_w_g[0]], dtype=np.float32)
    mid_hue_weight_g = np.interp(H, xp, mid_yp_g)

    mid_hue_w_b = params["mid_hue_w_b"]
    mid_yp_b = np.array([mid_hue_w_b[0], mid_hue_w_b[1], mid_hue_w_b[2],
                         mid_hue_w_b[3], mid_hue_w_b[4], mid_hue_w_b[5],
                         mid_hue_w_b[0]], dtype=np.float32)
    mid_hue_weight_b = np.interp(H, xp, mid_yp_b)

    hi_hue_w_r = params["hi_hue_w_r"]
    hi_yp_r = np.array([hi_hue_w_r[0], hi_hue_w_r[1], hi_hue_w_r[2],
                         hi_hue_w_r[3], hi_hue_w_r[4], hi_hue_w_r[5],
                         hi_hue_w_r[0]], dtype=np.float32)
    hi_hue_weight_r = np.interp(H, xp, hi_yp_r)

    hi_hue_w_g = params["hi_hue_w_g"]
    hi_yp_g = np.array([hi_hue_w_g[0], hi_hue_w_g[1], hi_hue_w_g[2],
                         hi_hue_w_g[3], hi_hue_w_g[4], hi_hue_w_g[5],
                         hi_hue_w_g[0]], dtype=np.float32)
    hi_hue_weight_g = np.interp(H, xp, hi_yp_g)

    hi_hue_w_b = params["hi_hue_w_b"]
    hi_yp_b = np.array([hi_hue_w_b[0], hi_hue_w_b[1], hi_hue_w_b[2],
                         hi_hue_w_b[3], hi_hue_w_b[4], hi_hue_w_b[5],
                         hi_hue_w_b[0]], dtype=np.float32)
    hi_hue_weight_b = np.interp(H, xp, hi_yp_b)

    return (hue_weight_r, hue_weight_g, hue_weight_b,
            mid_hue_weight_r, mid_hue_weight_g, mid_hue_weight_b,
            hi_hue_weight_r, hi_hue_weight_g, hi_hue_weight_b)


def _compute_structural_masks(Y, L, S, L_shifted, softness, end_val, exp_val, hue_weights, params) -> tuple:
    """
    Isolates Gaussian and Polynomial soft mask generation.
    Returns stacked (shadow_weight, midtone_weight, highlight_weight) arrays of shape (H, W, 3).
    """
    (hue_weight_r, hue_weight_g, hue_weight_b,
     mid_hue_weight_r, mid_hue_weight_g, mid_hue_weight_b,
     hi_hue_weight_r, hi_hue_weight_g, hi_hue_weight_b) = hue_weights

    # Shadows Soft Mask
    shadow_center_r = params["shadow_center_r"]
    shadow_width_r = params["shadow_width_r"]
    shadow_exponent_r = params["shadow_exponent_r"]
    shadows_end_val_r = params["shadows_end_val_r"]
    shadow_c0_r = params["shadow_c0_r"]
    shadow_c1_r = params["shadow_c1_r"]
    shadow_c2_r = params["shadow_c2_r"]
    shadow_c3_r = params["shadow_c3_r"]
    shadow_c4_r = params["shadow_c4_r"]
    shadow_c5_r = params["shadow_c5_r"]

    # Red Shadow Mask
    if shadow_center_r is not None and shadow_width_r is not None:
        mask_r = np.exp(-((Y - shadow_center_r)**2) / (2 * shadow_width_r**2))
        mask_r = mask_r + S * hue_weight_r
        shadow_weight_r = np.clip(mask_r, 0.0, 1.0)
    elif shadow_c0_r is not None:
        c0_r = shadow_c0_r
        c1_r = shadow_c1_r if shadow_c1_r is not None else 0.0
        c2_r = shadow_c2_r if shadow_c2_r is not None else 0.0
        c3_r = shadow_c3_r if shadow_c3_r is not None else 0.0
        c4_r = shadow_c4_r if shadow_c4_r is not None else 0.0
        c5_r = shadow_c5_r if shadow_c5_r is not None else 0.0
        x = Y
        mask_r = ((((c5_r * x + c4_r) * x + c3_r) * x + c2_r) * x + c1_r) * x + c0_r
        mask_r = mask_r + S * hue_weight_r
        shadow_weight_r = np.clip(mask_r, 0.0, 1.0)
    else:
        end_r = shadows_end_val_r if shadows_end_val_r is not None else end_val
        t_shadow_r = np.clip((L - 0.02) / end_r, 0.0, 1.0)
        exp_r = shadow_exponent_r if shadow_exponent_r is not None else exp_val
        cos_val_r = np.clip(np.cos(t_shadow_r * np.pi / 2.0), 0.0, 1.0)
        mask_r = cos_val_r ** exp_r + S * hue_weight_r
        shadow_weight_r = np.clip(mask_r, 0.0, 1.0)

    # Green Shadow Mask
    shadow_center_g = params["shadow_center_g"]
    shadow_width_g = params["shadow_width_g"]
    shadow_exponent_g = params["shadow_exponent_g"]
    shadows_end_val_g = params["shadows_end_val_g"]
    if shadow_center_g is not None and shadow_width_g is not None:
        mask_g = np.exp(-((Y - shadow_center_g)**2) / (2 * shadow_width_g**2))
        mask_g = mask_g + S * hue_weight_g
        shadow_weight_g = np.clip(mask_g, 0.0, 1.0)
    else:
        end_g = shadows_end_val_g if shadows_end_val_g is not None else end_val
        t_shadow_g = np.clip((L - 0.02) / end_g, 0.0, 1.0)
        exp_g = shadow_exponent_g if shadow_exponent_g is not None else exp_val
        cos_val_g = np.clip(np.cos(t_shadow_g * np.pi / 2.0), 0.0, 1.0)
        mask_g = cos_val_g ** exp_g + S * hue_weight_g
        shadow_weight_g = np.clip(mask_g, 0.0, 1.0)

    # Blue Shadow Mask
    shadow_center_b = params["shadow_center_b"]
    shadow_width_b = params["shadow_width_b"]
    shadow_exponent_b = params["shadow_exponent_b"]
    shadows_end_val_b = params["shadows_end_val_b"]
    shadow_c0 = params["shadow_c0"]
    shadow_c1 = params["shadow_c1"]
    shadow_c2 = params["shadow_c2"]
    shadow_c3 = params["shadow_c3"]
    shadow_c4 = params["shadow_c4"]
    shadow_c5 = params["shadow_c5"]
    if shadow_center_b is not None and shadow_width_b is not None:
        mask_b = np.exp(-((Y - shadow_center_b)**2) / (2 * shadow_width_b**2))
        mask_b = mask_b + S * hue_weight_b
        shadow_weight_b = np.clip(mask_b, 0.0, 1.0)
    elif shadow_c0 is not None:
        c0 = shadow_c0
        c1 = shadow_c1 if shadow_c1 is not None else 0.0
        c2 = shadow_c2 if shadow_c2 is not None else 0.0
        c3 = shadow_c3 if shadow_c3 is not None else 0.0
        c4 = shadow_c4 if shadow_c4 is not None else 0.0
        c5 = shadow_c5 if shadow_c5 is not None else 0.0
        x = Y
        mask_b = ((((c5 * x + c4) * x + c3) * x + c2) * x + c1) * x + c0
        mask_b = mask_b + S * hue_weight_b
        shadow_weight_b = np.clip(mask_b, 0.0, 1.0)
    else:
        end_b = shadows_end_val_b if shadows_end_val_b is not None else end_val
        t_shadow_b = np.clip((L - 0.02) / end_b, 0.0, 1.0)
        exp_b = shadow_exponent_b if shadow_exponent_b is not None else exp_val
        cos_val_b = np.clip(np.cos(t_shadow_b * np.pi / 2.0), 0.0, 1.0)
        mask_b = cos_val_b ** exp_b + S * hue_weight_b
        shadow_weight_b = np.clip(mask_b, 0.0, 1.0)

    shadow_weight = np.stack([shadow_weight_r, shadow_weight_g, shadow_weight_b], axis=2).astype(np.float32)

    # Highlights Soft Mask
    x_hi = 1.0 - Y
    
    # Red Highlight Mask
    highlight_center_r = params["highlight_center_r"]
    highlight_width_r = params["highlight_width_r"]
    highlight_c0_r = params["highlight_c0_r"]
    highlight_c1_r = params["highlight_c1_r"]
    highlight_c2_r = params["highlight_c2_r"]
    highlight_c3_r = params["highlight_c3_r"]
    highlight_c4_r = params["highlight_c4_r"]
    highlight_c5_r = params["highlight_c5_r"]
    if highlight_center_r is not None and highlight_width_r is not None:
        mask_hi_r = np.exp(-((Y - highlight_center_r)**2) / (2 * highlight_width_r**2))
        mask_hi_r = mask_hi_r + S * hi_hue_weight_r
        highlight_weight_r = np.clip(mask_hi_r, 0.0, 1.0)
    elif highlight_c0_r is not None:
        mask_hi_r = ((((highlight_c5_r * x_hi + highlight_c4_r) * x_hi + highlight_c3_r) * x_hi + highlight_c2_r) * x_hi + highlight_c1_r) * x_hi + highlight_c0_r
        mask_hi_r = mask_hi_r + S * hi_hue_weight_r
        highlight_weight_r = np.clip(mask_hi_r, 0.0, 1.0)
    else:
        mask_hi_r = np.clip((L_shifted - (0.7 - softness/2.0)) / softness, 0.0, 1.0) + S * hi_hue_weight_r
        highlight_weight_r = np.clip(mask_hi_r, 0.0, 1.0)

    # Green Highlight Mask
    highlight_center_g = params["highlight_center_g"]
    highlight_width_g = params["highlight_width_g"]
    highlight_c0_g = params["highlight_c0_g"]
    highlight_c1_g = params["highlight_c1_g"]
    highlight_c2_g = params["highlight_c2_g"]
    highlight_c3_g = params["highlight_c3_g"]
    highlight_c4_g = params["highlight_c4_g"]
    highlight_c5_g = params["highlight_c5_g"]
    if highlight_center_g is not None and highlight_width_g is not None:
        mask_hi_g = np.exp(-((Y - highlight_center_g)**2) / (2 * highlight_width_g**2))
        mask_hi_g = mask_hi_g + S * hi_hue_weight_g
        highlight_weight_g = np.clip(mask_hi_g, 0.0, 1.0)
    elif highlight_c0_g is not None:
        mask_hi_g = ((((highlight_c5_g * x_hi + highlight_c4_g) * x_hi + highlight_c3_g) * x_hi + highlight_c2_g) * x_hi + highlight_c1_g) * x_hi + highlight_c0_g
        mask_hi_g = mask_hi_g + S * hi_hue_weight_g
        highlight_weight_g = np.clip(mask_hi_g, 0.0, 1.0)
    else:
        mask_hi_g = np.clip((L_shifted - (0.7 - softness/2.0)) / softness, 0.0, 1.0) + S * hi_hue_weight_g
        highlight_weight_g = np.clip(mask_hi_g, 0.0, 1.0)

    # Blue Highlight Mask
    highlight_center_b = params["highlight_center_b"]
    highlight_width_b = params["highlight_width_b"]
    highlight_c0_b = params["highlight_c0_b"]
    highlight_c1_b = params["highlight_c1_b"]
    highlight_c2_b = params["highlight_c2_b"]
    highlight_c3_b = params["highlight_c3_b"]
    highlight_c4_b = params["highlight_c4_b"]
    highlight_c5_b = params["highlight_c5_b"]
    if highlight_center_b is not None and highlight_width_b is not None:
        mask_hi_b = np.exp(-((Y - highlight_center_b)**2) / (2 * highlight_width_b**2))
        mask_hi_b = mask_hi_b + S * hi_hue_weight_b
        highlight_weight_b = np.clip(mask_hi_b, 0.0, 1.0)
    elif highlight_c0_b is not None:
        mask_hi_b = ((((highlight_c5_b * x_hi + highlight_c4_b) * x_hi + highlight_c3_b) * x_hi + highlight_c2_b) * x_hi + highlight_c1_b) * x_hi + highlight_c0_b
        mask_hi_b = mask_hi_b + S * hi_hue_weight_b
        highlight_weight_b = np.clip(mask_hi_b, 0.0, 1.0)
    else:
        mask_hi_b = np.clip((L_shifted - (0.7 - softness/2.0)) / softness, 0.0, 1.0) + S * hi_hue_weight_b
        highlight_weight_b = np.clip(mask_hi_b, 0.0, 1.0)

    highlight_weight = np.stack([highlight_weight_r, highlight_weight_g, highlight_weight_b], axis=2).astype(np.float32)

    # Midtones Soft Mask
    mid_center_r = params["mid_center_r"]
    mid_width_r = params["mid_width_r"]
    if mid_center_r is not None and mid_width_r is not None:
        midtone_weight_r = np.exp(-((Y - mid_center_r)**2) / (2 * mid_width_r**2))
        midtone_weight_r = np.clip(midtone_weight_r + S * mid_hue_weight_r, 0.0, 1.0)
    else:
        midtone_weight_r = np.clip(1.0 - shadow_weight_r - highlight_weight_r + S * mid_hue_weight_r, 0.0, 1.0)

    mid_center_g = params["mid_center_g"]
    mid_width_g = params["mid_width_g"]
    if mid_center_g is not None and mid_width_g is not None:
        midtone_weight_g = np.exp(-((Y - mid_center_g)**2) / (2 * mid_width_g**2))
        midtone_weight_g = np.clip(midtone_weight_g + S * mid_hue_weight_g, 0.0, 1.0)
    else:
        midtone_weight_g = np.clip(1.0 - shadow_weight_g - highlight_weight_g + S * mid_hue_weight_g, 0.0, 1.0)

    mid_center_b = params["mid_center_b"]
    mid_width_b = params["mid_width_b"]
    if mid_center_b is not None and mid_width_b is not None:
        midtone_weight_b = np.exp(-((Y - mid_center_b)**2) / (2 * mid_width_b**2))
        midtone_weight_b = np.clip(midtone_weight_b + S * mid_hue_weight_b, 0.0, 1.0)
    else:
        midtone_weight_b = np.clip(1.0 - shadow_weight_b - highlight_weight_b + S * mid_hue_weight_b, 0.0, 1.0)

    midtone_weight = np.stack([midtone_weight_r, midtone_weight_g, midtone_weight_b], axis=2).astype(np.float32)

    return shadow_weight, midtone_weight, highlight_weight

def apply_grading(img: np.ndarray, state: StateNode, **kwargs) -> GradedArray:
    """
    IMMUTABLE MASTER PIPELINE: Applies perceptual 3-way color grading safely 
    without destructive RGB clipping or muddy middle color cancelation.
    """
    rgb = ensure_rgb(img).copy().astype(np.float32)
    params = _resolve_parameters(state, **kwargs)

    Y = np.dot(rgb, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    H, S = hls[:, :, 0], hls[:, :, 2]

    hue_weights = _compute_hue_weights(H, params)
    balance = state.get("balance", 0.0)
    L_shifted = np.clip(Y - 0.3 * balance, 0.0, 1.0)
    blending = state.get("blending", 0.5)
    softness = max(0.01, 0.05 + 0.35 * blending)
    end_val = params["shadows_end_val"] if params["shadows_end_val"] is not None else (0.3 + 0.4 * blending)

    # Apply global rotation offset
    rotation_offset = state.get("rotation", 0.0)
    sh_hue = (state.get("shadow_hue", 240.0) + rotation_offset) % 360.0
    sh_sat = state.get("shadow_sat", 0.0)
    mid_hue = (state.get("midtone_hue", 120.0) + rotation_offset) % 360.0
    mid_sat = state.get("midtone_sat", 0.0)
    hi_hue = (state.get("highlight_hue", 60.0) + rotation_offset) % 360.0
    hi_sat = state.get("highlight_sat", 0.0)

    if state.get("harmony_mode", "Monochromatic") == "Complementary":
        sh_hue = (hi_hue + 180.0) % 360.0
        sh_sat = np.clip(sh_sat + 0.15, 0.0, 1.0)
        hi_sat = np.clip(hi_sat + 0.15, 0.0, 1.0)

    sw, mw, hw = _compute_structural_masks(Y, Y, S, L_shifted, softness, end_val, 1.0, hue_weights, params)

    # PERCEPTUALLY SAFE ZONE TARGET HELPER (Anchors luma to prevent clipping & color destruction)
    def compute_zone_target(hue, sat, gain_arr):
        if sat == 0.0:
            return np.zeros_like(rgb)
        hls_z = hls.copy()
        hls_z[:, :, 0] = hue
        hls_z[:, :, 1] = Y  # Anchor to original pixel luma
        hls_z[:, :, 2] = sat
        target_rgb = cv2.cvtColor(hls_z, cv2.COLOR_HLS2RGB)
        return gain_arr * (target_rgb - rgb)

    gain_sh = np.array([params["shadow_gain_r"], params["shadow_gain_g"], params["shadow_gain_b"]], dtype=np.float32)
    gain_mid = np.array([params["midtone_gain_r"], params["midtone_gain_g"], params["midtone_gain_b"]], dtype=np.float32)
    gain_hi = np.array([params["highlight_gain_r"], params["highlight_gain_g"], params["highlight_gain_b"]], dtype=np.float32)

    delta_sh = compute_zone_target(sh_hue, sh_sat, gain_sh)
    delta_mid = compute_zone_target(mid_hue, mid_sat, gain_mid)
    delta_hi = compute_zone_target(hi_hue, hi_sat, gain_hi)

    # Normalized blending weights to avoid additive blowout/clipping
    total_weight = sw + mw + hw + 1e-6
    rgb_graded = rgb + (sw * delta_sh + mw * delta_mid + hw * delta_hi) / (total_weight * 0.5 + 0.5)

    # Apply Zone Lightness Adjustments if present
    sh_light = state.get("shadow_light", 0.0)
    mid_light = state.get("midtone_light", 0.0)
    hi_light = state.get("highlight_light", 0.0)
    if sh_light != 0.0 or mid_light != 0.0 or hi_light != 0.0:
        if sh_light != 0.0: rgb_graded += sw * sh_light
        if mid_light != 0.0: rgb_graded += mw * mid_light
        if hi_light != 0.0: rgb_graded += hw * hi_light

    np.clip(rgb_graded, 0.0, 1.0, out=rgb_graded)

    # Global adjustments
    base_shift = state.get("hue_shift", 0.0)
    sat_shift = state.get("sat_shift", 0.0)
    light_shift = state.get("light_shift", 0.0)
    if base_shift != 0.0 or sat_shift != 0.0 or light_shift != 0.0:
        hls_g = cv2.cvtColor(rgb_graded, cv2.COLOR_RGB2HLS)
        H_g, L_g, S_g = hls_g[:, :, 0], hls_g[:, :, 1], hls_g[:, :, 2]
        if base_shift != 0.0: H_g = (H_g + base_shift) % 360.0
        if sat_shift != 0.0: S_g = np.clip(S_g + sat_shift, 0.0, 1.0)
        if light_shift != 0.0: L_g = np.clip(L_g + light_shift, 0.0, 1.0)
        rgb_graded = cv2.cvtColor(np.stack([H_g, L_g, S_g], axis=2).astype(np.float32), cv2.COLOR_HLS2RGB).astype(np.float32)

    return GradedArray(np.clip(rgb_graded, 0.0, 1.0).astype(np.float32), sw)

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
