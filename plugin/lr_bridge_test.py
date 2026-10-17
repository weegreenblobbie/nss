import pytest
from nss.plugin import lr_bridge as lb


def sample():
    mixer = {"hue": [0, -8, -12, -18, 10, 6, 0, 0], "sat": [0, 12, 18, 22, 15, 30, 0, 0], "lum": [0, 8, 5, -12, 14, -22, 0, 0]}
    wheels = {"shadows": (200.0, 30.0, 0.0), "midtones": (0.0, 0.0, 0.0), "highlights": (40.0, 30.0, -5.0)}
    return mixer, wheels


def test_result_round_trips_through_a_request(tmp_path):
    mixer, wheels = sample()
    path = str(tmp_path / "r.txt")
    lb.write_result(path, mixer, wheels, 60, -10)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    with open(path, "a", encoding="utf-8") as f:
        f.write("tiff=C:\\tmp\\a=b.tif\nresult=C:\\tmp\\res.txt\n")
    req = lb.read_request(path)
    assert req.mixer == {k: [float(x) for x in v] for k, v in mixer.items()}
    assert req.wheels == wheels and req.blending == 60 and req.balance == -10
    assert req.tiff == "C:\\tmp\\a=b.tif" and req.result == "C:\\tmp\\res.txt"
    assert "HueAdjustmentOrange=-8" in text and "ColorGradeHighlightLum=-5" in text and "SplitToningBalance=-10" in text


def test_missing_keys_take_neutral_values(tmp_path):
    path = tmp_path / "r.txt"
    path.write_text("tiff=a.tif\nresult=b.txt\n", encoding="utf-8")
    req = lb.read_request(str(path))
    assert req.mixer["hue"] == [0.0] * 8 and req.wheels["shadows"] == (0.0, 0.0, 0.0) and req.blending == 50 and req.balance == 0


def test_request_needs_paths(tmp_path):
    path = tmp_path / "r.txt"
    path.write_text("HueAdjustmentRed=3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        lb.read_request(str(path))


def test_values_are_rounded_clamped_and_hue_wraps():
    mixer = {"hue": [120] * 8, "sat": [-120.4] * 8, "lum": [2.5] * 8}
    wheels = {"shadows": (-10.0, 130.0, 0.0), "midtones": (360.0, 0.0, 0.0), "highlights": (359.6, 0.0, 0.0)}
    lines = dict(line.split("=") for line in lb.result_lines(mixer, wheels, 130, -130))
    assert lines["HueAdjustmentRed"] == "100" and lines["SaturationAdjustmentBlue"] == "-100" and lines["LuminanceAdjustmentAqua"] in ("2", "3")
    assert lines["SplitToningShadowHue"] == "350" and lines["SplitToningShadowSaturation"] == "100"
    assert lines["ColorGradeMidtoneHue"] == "0" and lines["SplitToningHighlightHue"] == "0"
    assert lines["ColorGradeBlending"] == "100" and lines["SplitToningBalance"] == "-100"


def test_all_keys_are_written_once():
    mixer, wheels = sample()
    keys = [line.split("=")[0] for line in lb.result_lines(mixer, wheels, 50, 0)]
    assert len(keys) == 24 + 9 + 2 and len(set(keys)) == len(keys)


def test_wrong_number_of_bands_is_refused():
    mixer, wheels = sample()
    mixer["hue"] = [0] * 7
    with pytest.raises(ValueError):
        lb.result_lines(mixer, wheels, 50, 0)