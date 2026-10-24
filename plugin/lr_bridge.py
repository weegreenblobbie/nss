"""
Hand-over files between the Lightroom Classic plug-in (lightroom/nss.lrdevplugin) and the nss color GUI.

The plug-in writes a request file of `key=value` lines: `tiff` (the exported picture, everything but the mixer and the color wheels baked in), `result` (where to write the answer)
and the current Lightroom develop settings of the 24 mixer sliders and the color grading wheels under Lightroom's own key names.  The GUI starts its sliders from them and, when
you press Apply, writes the same keys to the result file; the plug-in applies them to the photo.  No result file (or a non-zero exit) means: leave the photo alone.

    req = read_request(path)            # req.tiff, req.result, req.mixer {"hue","sat","lum"}, req.wheels {"shadows","midtones","highlights","global"}, req.blending, req.balance
    write_result(req.result, mixer, wheels, blending, balance)

Units are Lightroom's: mixer -100..100; wheels (hue 0..359, sat 0..100, lum -100..100); blending 0..100; balance -100..100.  The mixer / wheel arguments are the ones of
nss.color_mixer.ColorMixer.apply (lists of eight, red .. magenta) and tests.test_data.grade_image.grade.
"""
from dataclasses import dataclass

from plugin.handover import read_values

BANDS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
MIXER = (("hue", "HueAdjustment"), ("sat", "SaturationAdjustment"), ("lum", "LuminanceAdjustment"))
# Lightroom still stores the shadow and highlight hue / saturation under the old split-toning names (checked on a Lightroom Classic 15.6 photo)
WHEEL_KEYS = {
    "shadows": ("SplitToningShadowHue", "SplitToningShadowSaturation", "ColorGradeShadowLum"),
    "midtones": ("ColorGradeMidtoneHue", "ColorGradeMidtoneSat", "ColorGradeMidtoneLum"),
    "highlights": ("SplitToningHighlightHue", "SplitToningHighlightSaturation", "ColorGradeHighlightLum"),
    "global": ("ColorGradeGlobalHue", "ColorGradeGlobalSat", "ColorGradeGlobalLum"),
}
BLENDING_KEY, BALANCE_KEY = "ColorGradeBlending", "SplitToningBalance"
DEFAULT_BLENDING, DEFAULT_BALANCE = 50, 0


@dataclass
class Request:
    tiff: str
    result: str
    mixer: dict
    wheels: dict
    blending: float
    balance: float


def _number(values, key, default):
    try:
        return float(values[key])
    except (KeyError, ValueError):
        return float(default)


def read_request(path):
    v = read_values(path)
    if "tiff" not in v or "result" not in v:
        raise ValueError(f"{path}: request needs tiff= and result= lines")
    mixer = {name: [_number(v, prefix + band, 0.0) for band in BANDS] for name, prefix in MIXER}
    wheels = {name: tuple(_number(v, key, 0.0) for key in keys) for name, keys in WHEEL_KEYS.items()}
    return Request(v["tiff"], v["result"], mixer, wheels, _number(v, BLENDING_KEY, DEFAULT_BLENDING), _number(v, BALANCE_KEY, DEFAULT_BALANCE))


def _clamp(x, lo, hi):
    return int(round(min(max(x, lo), hi)))


def result_lines(mixer, wheels, blending, balance):
    """key=value lines in Lightroom's key names, rounded to the integers Lightroom stores."""
    out = []
    for name, prefix in MIXER:
        values = mixer[name]
        if len(values) != len(BANDS):
            raise ValueError(f"mixer[{name!r}]: expected {len(BANDS)} values")
        out += [f"{prefix}{band}={_clamp(x, -100, 100)}" for band, x in zip(BANDS, values)]
    for name, (hue_key, sat_key, lum_key) in WHEEL_KEYS.items():
        hue, sat, lum = wheels[name]
        out += [f"{hue_key}={int(round(hue)) % 360}", f"{sat_key}={_clamp(sat, 0, 100)}", f"{lum_key}={_clamp(lum, -100, 100)}"]
    out += [f"{BLENDING_KEY}={_clamp(blending, 0, 100)}", f"{BALANCE_KEY}={_clamp(balance, -100, 100)}"]
    return out


def write_result(path, mixer, wheels, blending, balance):
    lines = result_lines(mixer, wheels, blending, balance)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")