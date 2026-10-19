"""Tests of moon.py: exactly centred crops, the batch common size, the orientation mapping, detect_moon's (row, col) conversion,
the confidence threshold and the notfound path."""
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

import moon
from nss.tiff_file import TiffFile
from nss.utils import Circle
from plugin import moon_bridge



def _centre(crop: tuple, width: int, height: int) -> tuple[float, float]:
    """Pixel centre of a crop given in fractions."""
    return (crop[0] + crop[2]) / 2 * width, (crop[1] + crop[3]) / 2 * height


@pytest.mark.parametrize("cx, cy", [(300.0, 200.0), (123.4, 310.7), (1.5, 2.25), (598.5, 399.0), (450.0, 30.0)])
@pytest.mark.parametrize("aspect", [1.5, 1.0, 0.02, 40.0])
def test_crop_is_centred_exactly_and_is_the_largest_that_fits(cx: float, cy: float, aspect: float) -> None:
    """On a 600 x 400 image the crop's centre is the moon to 1e-9 px, it stays inside, has the aspect and touches a limit."""
    crop = moon.crop_for(cx, cy, 600, 400, aspect)
    assert np.allclose(_centre(crop, 600, 400), (cx, cy), atol=1e-9, rtol=0)
    assert min(crop) >= -1e-12 and crop[2] <= 1 + 1e-12 and crop[3] <= 1 + 1e-12
    w, h = (crop[2] - crop[0]) * 600, (crop[3] - crop[1]) * 400
    assert w / h == pytest.approx(aspect)
    assert w == pytest.approx(2 * min(cx, 600 - cx)) or h == pytest.approx(2 * min(cy, 400 - cy))


def test_aspect_zero_is_the_image_aspect_and_a_fitting_size_is_kept() -> None:
    """Aspect 0 takes 600 / 400; a requested 100 x 80 px that fits is kept, a 500 x 400 px one that does not falls back to the largest (450 x 360)."""
    crop = moon.crop_for(250.0, 180.0, 600, 400, 0.0)
    assert (crop[2] - crop[0]) * 600 / ((crop[3] - crop[1]) * 400) == pytest.approx(1.5)
    kept = moon.crop_for(250.0, 180.0, 600, 400, 1.25, (100.0, 80.0))
    assert np.allclose(kept, (200 / 600, 140 / 400, 300 / 600, 220 / 400))
    fallback = moon.crop_for(250.0, 180.0, 600, 400, 1.25, (500.0, 400.0))
    assert np.allclose(fallback, (25 / 600, 0.0, 475 / 600, 360 / 400))


def test_batch_uses_one_pixel_size_for_photos_of_different_sizes() -> None:
    """The common size is the smallest max_crop at the first photo's aspect, the same in pixels on a 600 x 400 and a 300 x 500 photo."""
    results = [{"status": "ok", "cx": 300.0, "cy": 200.0, "width": 600, "height": 400, "crop_width": 300.0, "crop_height": 200.0, "orientation": "AB",
                "angle": 0.0},
               {"status": "notfound"},
               {"status": "ok", "cx": 100.0, "cy": 260.0, "width": 300, "height": 500, "crop_width": 50.0, "crop_height": 100.0, "orientation": "AB",
                "angle": 0.0}]
    moon.add_crops(results, "batch")
    assert "left" not in results[1]
    for r in (results[0], results[2]):
        crop = (r["left"], r["top"], r["right"], r["bottom"])
        assert np.allclose(((crop[2] - crop[0]) * r["width"], (crop[3] - crop[1]) * r["height"]), (200.0, 200.0 / 1.5))
        assert np.allclose(_centre(crop, r["width"], r["height"]), (r["cx"], r["cy"]))


def test_single_keeps_each_photos_own_crop_size() -> None:
    """Single mode: each photo keeps its current 120 x 90 px crop, moved onto the moon."""
    results = [{"status": "ok", "cx": 250.0, "cy": 180.0, "width": 600, "height": 400, "crop_width": 120.0, "crop_height": 90.0, "orientation": "AB",
                "angle": 0.0}]
    moon.add_crops(results, "single")
    assert np.allclose([results[0][k] for k in ("left", "top", "right", "bottom")], (190 / 600, 135 / 400, 310 / 600, 225 / 400))


def _turned_corners(crop: tuple, width: int, height: int, angle: float) -> np.ndarray:
    """The four corners (4, 2) in pixels of a crop given in fractions, turned by angle degrees about its centre (an independent check)."""
    cx, cy = _centre(crop, width, height)
    a, b = (crop[2] - crop[0]) * width / 2, (crop[3] - crop[1]) * height / 2
    t = np.radians(angle)
    offsets = np.array([[-a, -b], [a, -b], [a, b], [-a, b]]) @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    return offsets + (cx, cy)


@pytest.mark.parametrize("cx, cy", [(300.0, 200.0), (123.4, 310.7), (450.0, 60.0)])
@pytest.mark.parametrize("angle", [3.5, -12.0, 30.0, -45.0])
@pytest.mark.parametrize("aspect", [1.5, 1.0, 0.25])
def test_turned_crop_is_centred_and_its_corners_stay_inside(cx: float, cy: float, angle: float, aspect: float) -> None:
    """On a 600 x 400 image a crop turned by angle is centred on the moon, keeps the aspect, has every corner inside and one on an edge."""
    crop = moon.crop_for(cx, cy, 600, 400, aspect, angle=angle)
    assert np.allclose(_centre(crop, 600, 400), (cx, cy), atol=1e-9, rtol=0)
    assert (crop[2] - crop[0]) * 600 / ((crop[3] - crop[1]) * 400) == pytest.approx(aspect)
    corners = _turned_corners(crop, 600, 400, angle)
    assert corners.min() >= -1e-6 and (corners[:, 0] <= 600 + 1e-6).all() and (corners[:, 1] <= 400 + 1e-6).all()
    gaps = np.concatenate([corners[:, 0], 600 - corners[:, 0], corners[:, 1], 400 - corners[:, 1]])
    assert gaps.min() == pytest.approx(0.0, abs=1e-6)


def test_turned_crop_keeps_a_fitting_size_and_angle_zero_is_unchanged() -> None:
    """Turned 30 degrees about (300, 200): 200 x 100 px fits and is kept, 500 x 300 px does not (its corners would leave the image);
    angle 0 gives the untilted crop."""
    kept = moon.crop_for(300.0, 200.0, 600, 400, 2.0, (200.0, 100.0), angle=30.0)
    assert np.allclose(kept, (200 / 600, 150 / 400, 400 / 600, 250 / 400))
    fallback = moon.crop_for(300.0, 200.0, 600, 400, 5 / 3, (500.0, 300.0), angle=30.0)
    assert (fallback[2] - fallback[0]) * 600 < 500.0
    assert moon.crop_for(250.0, 180.0, 600, 400, 1.25, angle=0.0) == moon.crop_for(250.0, 180.0, 600, 400, 1.25)


def test_angle_reaches_the_crop_of_a_request() -> None:
    """A result with angle 20 gets a smaller crop (read back through crop_size) than the same moon at angle 0, still centred and 3:2."""
    results = [{"status": "ok", "cx": 300.0, "cy": 200.0, "width": 600, "height": 400, "crop_width": 600.0, "crop_height": 400.0, "orientation": "AB",
                "angle": a} for a in (0.0, 20.0)]
    for r in results:
        moon.add_crops([r], "single")
    sizes = [moon.crop_size((r["left"], r["top"], r["right"], r["bottom"]), 600, 400, "AB", r["angle"]) for r in results]
    assert np.allclose(sizes[0], (600.0, 400.0)) and sizes[1][0] < 500.0 and sizes[1][0] / sizes[1][1] == pytest.approx(1.5)
    assert np.allclose(_centre((results[1]["left"], results[1]["top"], results[1]["right"], results[1]["bottom"]), 600, 400), (300.0, 200.0))


# Lightroom Classic, an 8256 x 5504 photo at CropAngle -2.21: the stored crop and the size of its export (verify runs)
LIGHTROOM_TURNED_CROPS = [((0.034218, 0.069086, 0.989082, 0.946113), (7691, 5127)), ((0.033967, 0.068844, 0.989079, 0.946099), (7693, 5129))]


@pytest.mark.parametrize("crop, exported", LIGHTROOM_TURNED_CROPS)
def test_crop_size_reads_lightroom_corners(crop: tuple, exported: tuple) -> None:
    """The stored numbers of a turned crop are its corners: crop_size gives the exported size to 1 px (as a rectangle they would be 7883 x 4827)."""
    assert np.allclose(moon.crop_size(crop, 8256, 5504, "AB", -2.21), exported, atol=1.0)


@pytest.mark.parametrize("orientation", list(moon.ORIENTATIONS))
@pytest.mark.parametrize("angle", [0.0, -2.21, 7.5, -30.0])
def test_lightroom_crop_round_trips_through_crop_size(orientation: str, angle: float) -> None:
    """A 250 x 140 px displayed crop centred at (310, 190) on a 600 x 400 display: lightroom_crop keeps the centre (the midpoint of the corners)
    and crop_size reads back 250 x 140; angle 0 is to_stored."""
    crop = (185 / 600, 120 / 400, 435 / 600, 260 / 400)
    stored = moon.lightroom_crop(crop, 600, 400, orientation, angle)
    assert np.allclose(moon.crop_size(stored, 600, 400, orientation, angle), (250.0, 140.0))
    assert np.allclose(_centre(moon.to_display(stored, orientation), 600, 400), (310.0, 190.0))
    if angle == 0.0:
        assert np.allclose(stored, moon.to_stored(crop, orientation))


@pytest.mark.parametrize("orientation", list(moon.ORIENTATIONS))
def test_orientation_round_trip(orientation: str) -> None:
    """to_stored undoes to_display for every orientation."""
    crop = (0.1, 0.25, 0.7, 0.4)
    assert np.allclose(moon.to_stored(moon.to_display(crop, orientation), orientation), crop)


def test_orientation_maps_a_strip() -> None:
    """Rotate Left (counter-clockwise, Lightroom reports DA) and Flip Horizontal + Rotate Right (AD) show the stored left strip at the bottom,
    BC (clockwise) and CB (transpose) at the top, CD (180) and BA (flip horizontal) at the right; an unknown code is a ValueError."""
    strip = (0.0, 0.0, 0.25, 1.0)
    assert np.allclose(moon.to_display(strip, "AB"), strip)
    assert np.allclose(moon.to_display(strip, "DA"), (0.0, 0.75, 1.0, 1.0))
    assert np.allclose(moon.to_display(strip, "AD"), (0.0, 0.75, 1.0, 1.0))
    assert np.allclose(moon.to_display(strip, "BC"), (0.0, 0.0, 1.0, 0.25))
    assert np.allclose(moon.to_display(strip, "CB"), (0.0, 0.0, 1.0, 0.25))
    assert np.allclose(moon.to_display(strip, "CD"), (0.75, 0.0, 1.0, 1.0))
    assert np.allclose(moon.to_display(strip, "BA"), (0.75, 0.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        moon.to_display(strip, "AC")


# Lightroom Classic, one photo: the moon's offset (px) from the crop centre as center moon reported it for each orientation code
LIGHTROOM_OFFSETS = {"AB": (96.2, 142.5), "BC": (-142.5, 96.2), "CD": (-96.2, -142.4), "DA": (142.5, -96.2), "BA": (-96.2, 142.5),
                     "DC": (96.2, -142.5), "AD": (-142.4, -96.2), "CB": (142.4, 96.2)}


@pytest.mark.parametrize("orientation", list(LIGHTROOM_OFFSETS))
def test_orientation_table_reproduces_the_lightroom_offsets(orientation: str) -> None:
    """The unrotated offset (96.2, 142.5) turned by the table's u, v axes is the offset Lightroom's render showed for that code."""
    origin, u, v = moon.ORIENTATIONS[orientation]
    shown = 96.2 * np.array(u) + 142.5 * np.array(v)
    assert np.allclose(shown, LIGHTROOM_OFFSETS[orientation], atol=0.15)


def test_find_moon_converts_row_col_to_pixel_centres_and_rejects_implausible_or_doubtful_circles() -> None:
    """detect_moon's (row 10, col 20) is x 20.5, y 10.5; radius 2, a centre outside the image or a confidence below min_confidence is
    not a moon, each with its reason."""
    image = np.zeros((40, 60, 3), np.float32)
    with patch("moon.utils.detect_moon", return_value=(Circle(center=(10.0, 20.0), radius=7.0), 0.95)):
        assert moon.find_moon(image) == (20.5, 10.5, 7.0, 0.95, "")
        assert moon.find_moon(image, min_confidence=0.97)[3:] == (0.95, "confidence 0.950 below 0.970")
    with patch("moon.utils.detect_moon", return_value=(Circle(center=(10.0, 20.0), radius=2.0), 1.0)):
        assert moon.find_moon(image)[4].startswith("implausible circle")
    with patch("moon.utils.detect_moon", return_value=(Circle(center=(10.0, 75.0), radius=7.0), 1.0)):
        assert moon.find_moon(image)[4].startswith("implausible circle")


def test_black_image_is_notfound_and_the_result_is_still_written(tmp_path) -> None:
    """The real detect_moon finds nothing on a black 40 x 60 TIFF: status notfound, result file written, exit code 0."""
    tiff = str(tmp_path / "black.tif")
    TiffFile.from_array(np.zeros((40, 60, 3), np.float32)).saveas(tiff)
    request, result = tmp_path / "request.txt", tmp_path / "result.txt"
    request.write_text(f"result={result}\nmode=measure\nsize=21\ncount=1\nphoto.1.id=7\nphoto.1.image={tiff}\n", encoding="utf-8")
    with patch.object(sys, "argv", ["moon.py", "--request", str(request)]):
        assert moon.main() == 0
    lines = result.read_text(encoding="utf-8").splitlines()
    assert "photo.1.status=notfound" in lines and "photo.1.width=60" in lines and not any(line.startswith("photo.1.left") for line in lines)


def test_request_run_with_a_rotated_photo_and_verify_offsets(tmp_path) -> None:
    """A DA photo whose stored crop is its left half: the new crop, shown in the TIFF frame, is centred on the moon; the offset is from the
    current crop's centre.  Measuring a full-frame crop (verify) gives the offset from the TIFF centre and no crop.  An unreadable TIFF is
    an error for that photo only."""
    tiff = str(tmp_path / "a.tif")
    TiffFile.from_array(np.zeros((40, 60, 3), np.float32)).saveas(tiff)          # displayed 60 wide, 40 high
    photos = [moon_bridge.Photo("1", tiff, (0.0, 0.0, 0.5, 1.0), "DA"), moon_bridge.Photo("2", str(tmp_path / "missing.tif"), (0, 0, 1, 1), "AB")]
    with patch("moon.utils.detect_moon", return_value=(Circle(center=(14.5, 24.5), radius=6.0), 0.97)):       # x 25, y 15
        first, missing = moon.run(moon_bridge.Request("", "measure", 21, photos))
        moon.add_crops([first, missing], "single")
        verify, = moon.run(moon_bridge.Request("", "measure", 21, [moon_bridge.Photo("3", tiff, (0, 0, 1, 1), "AB")]))
    assert missing["status"] == "error" and "missing.tif" in missing["message"] and first["confidence"] == 0.97
    assert (first["cx"], first["cy"], first["dx"], first["dy"]) == (25.0, 15.0, -5.0, -15.0)     # current crop: the bottom half, centre (30, 30)
    shown = moon.to_display((first["left"], first["top"], first["right"], first["bottom"]), "DA")
    assert np.allclose(_centre(shown, 60, 40), (25.0, 15.0))
    assert np.allclose(((shown[2] - shown[0]) * 60, (shown[3] - shown[1]) * 40), (50.0, 50.0 / 3))     # 60 x 20 does not fit; 3:1 limited by x 25 of 60
    assert (verify["dx"], verify["dy"]) == (-5.0, -5.0) and "left" not in verify


def test_measure_writes_progress_and_stops_at_the_cancel_file(tmp_path) -> None:
    """After each photo the progress file holds '<done> <count>'; once the cancel file exists the remaining photos are canceled."""
    tiff = str(tmp_path / "a.tif")
    TiffFile.from_array(np.zeros((40, 60, 3), np.float32)).saveas(tiff)
    progress, cancel = str(tmp_path / "progress.txt"), str(tmp_path / "cancel.txt")
    photos = [moon_bridge.Photo(str(i), tiff, (0, 0, 1, 1), "AB") for i in range(3)]
    seen = []

    def detect(image: np.ndarray, size: int) -> tuple:
        """A moon at x 25.5, y 15.5; asks to cancel while the second photo is measured, and notes the progress file each time."""
        seen.append(open(progress, encoding="utf-8").read() if os.path.exists(progress) else "")
        if len(seen) == 2:
            open(cancel, "w").close()
        return Circle(center=(15.0, 25.0), radius=6.0), 1.0

    with patch("moon.utils.detect_moon", side_effect=detect):
        results = moon.run(moon_bridge.Request("", "measure", 21, photos, progress=progress, cancel=cancel))
    assert [r["status"] for r in results] == ["ok", "ok", "canceled"]
    assert seen == ["", "1 3\n"] and open(progress, encoding="utf-8").read() == "2 3\n"


def test_batch_request_of_measurements_gives_crops_without_tiffs(tmp_path) -> None:
    """A batch request carries the measurements of a 600 x 400 and a 300 x 500 photo (no TIFFs): both get the same 200 x 133.3 px crop,
    centred on their moons."""
    request, result = tmp_path / "request.txt", tmp_path / "result.txt"
    lines = [f"result={result}", "mode=batch", "count=2"]
    for i, (cx, cy, width, height, crop_w, crop_h) in enumerate([(300, 200, 600, 400, 300, 200), (100, 260, 300, 500, 50, 100)], start=1):
        lines += [f"photo.{i}.{key}={value}" for key, value in (("id", 10 + i), ("cx", cx), ("cy", cy), ("width", width), ("height", height),
                  ("crop_width", crop_w), ("crop_height", crop_h), ("orientation", "AB"), ("angle", 0))]
    request.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with patch.object(sys, "argv", ["moon.py", "--request", str(request)]):
        assert moon.main() == 0
    values = dict(line.split("=", 1) for line in result.read_text(encoding="utf-8").splitlines())
    for i, (width, height, cx, cy) in enumerate([(600, 400, 300, 200), (300, 500, 100, 260)], start=1):
        crop = [float(values[f"photo.{i}.{side}"]) for side in ("left", "top", "right", "bottom")]
        assert np.allclose(((crop[2] - crop[0]) * width, (crop[3] - crop[1]) * height), (200.0, 200.0 / 1.5))
        assert np.allclose(_centre(crop, width, height), (cx, cy))


def _disc_tiff(path: str, row: float, col: float) -> str:
    """A 300 x 420 TIFF with an anti-aliased disc of radius 80 centred at (row, col); returns path."""
    rows, cols = np.mgrid[0:300, 0:420]
    disc = 0.5 * np.clip(80.0 - np.hypot(rows - row, cols - col) + 0.5, 0.0, 1.0)
    TiffFile.from_array(np.repeat(disc[..., None], 3, axis=2)).saveas(path)
    return path


def test_two_worker_processes_give_the_same_results_in_order(tmp_path) -> None:
    """The real detect_moon in 2 worker processes: the same moons, in request order (the black photo stays notfound in its place),
    as 1 worker, and the progress file ends at '3 3'."""
    tiffs = [_disc_tiff(str(tmp_path / "a.tif"), 140.3, 180.7), str(tmp_path / "black.tif"), _disc_tiff(str(tmp_path / "c.tif"), 160.0, 230.5)]
    TiffFile.from_array(np.zeros((300, 420, 3), np.float32)).saveas(tiffs[1])
    photos = [moon_bridge.Photo(str(i), tiff, (0, 0, 1, 1), "AB") for i, tiff in enumerate(tiffs)]
    progress = str(tmp_path / "progress.txt")
    one = moon.run(moon_bridge.Request("", "measure", 31, photos, workers=1))
    two = moon.run(moon_bridge.Request("", "measure", 31, photos, workers=2, progress=progress))
    assert [r["status"] for r in two] == ["ok", "notfound", "ok"]
    assert two == one
    assert np.allclose((two[0]["cx"], two[0]["cy"]), (181.2, 140.8), atol=0.1) and np.allclose((two[2]["cx"], two[2]["cy"]), (231.0, 160.5), atol=0.1)
    assert open(progress, encoding="utf-8").read() == "3 3\n"


def test_a_locked_progress_file_does_not_stop_measuring(tmp_path) -> None:
    """Windows refuses to replace a progress file the plug-in has open: that update is skipped, the photos are still measured."""
    tiff = _disc_tiff(str(tmp_path / "a.tif"), 140.3, 180.7)
    photos = [moon_bridge.Photo(str(i), tiff, (0, 0, 1, 1), "AB") for i in range(2)]
    with patch("moon.os.replace", side_effect=PermissionError(13, "The process cannot access the file")):
        results = moon.run(moon_bridge.Request("", "measure", 31, photos, progress=str(tmp_path / "progress.txt")))
    assert [r["status"] for r in results] == ["ok", "ok"]
