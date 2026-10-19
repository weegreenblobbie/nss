"""Tests of plugin/moon_bridge.py: request parsing, its checks and the result lines."""
import pytest

from plugin import moon_bridge as mb


def test_request_is_parsed_with_defaults(tmp_path) -> None:
    """Photos in order with their crop, orientation and angle; a missing crop is the full frame, orientation AB, angle 0; values may contain '='."""
    path = tmp_path / "r.txt"
    path.write_text("result=C:\\tmp\\res=1.txt\nmode=measure\nsize=81\nmin_confidence=0.75\nprogress=p.txt\ncancel=c.txt\nworkers=3\ncount=2\n"
                    "photo.1.id=11\nphoto.1.image=C:\\a.tif\nphoto.1.left=0.1\nphoto.1.top=0.2\nphoto.1.right=0.7\nphoto.1.bottom=0.9\n"
                    "photo.1.orientation=DA\nphoto.1.angle=-3.25\n"
                    "photo.2.id=12\nphoto.2.image=C:\\b.tif\n", encoding="utf-8")
    req = mb.read_request(str(path))
    assert (req.result, req.mode, req.size, req.min_confidence) == ("C:\\tmp\\res=1.txt", "measure", 81, 0.75)
    assert (req.progress, req.cancel, req.workers) == ("p.txt", "c.txt", 3)
    assert req.photos == [mb.Photo("11", "C:\\a.tif", (0.1, 0.2, 0.7, 0.9), "DA", -3.25),
                          mb.Photo("12", "C:\\b.tif", (0.0, 0.0, 1.0, 1.0), "AB", 0.0)]


@pytest.mark.parametrize("text", ["mode=single\n", "result=r.txt\nmode=fast\n", "result=r.txt\nmode=single\ncount=1\nphoto.1.id=3\n"])
def test_incomplete_requests_are_refused(tmp_path, text: str) -> None:
    """No result=, an unknown mode or a photo without image= is a ValueError."""
    path = tmp_path / "r.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        mb.read_request(str(path))


def test_result_lines_and_file(tmp_path) -> None:
    """Floats get 6 decimals, ints stay, messages become one line, missing keys and unknown keys are left out; count= leads the file."""
    results = [{"status": "ok", "confidence": 0.9875, "cx": 25.0, "cy": 15.25, "width": 60, "left": 0.125, "message": "moon\nfound", "crop_px": (1, 2)},
               {"status": "error", "message": "OSError: no such file"}]
    lines = mb.result_lines(results)
    assert lines == ["photo.1.status=ok", "photo.1.confidence=0.987500", "photo.1.cx=25.000000", "photo.1.cy=15.250000", "photo.1.width=60",
                     "photo.1.left=0.125000", "photo.1.message=moon found", "photo.2.status=error", "photo.2.message=OSError: no such file"]
    path = tmp_path / "res.txt"
    mb.write_result(str(path), results)
    assert path.read_text(encoding="utf-8").splitlines() == ["count=2"] + lines


def test_min_confidence_defaults(tmp_path) -> None:
    """Without min_confidence= the request takes DEFAULT_MIN_CONFIDENCE."""
    path = tmp_path / "r.txt"
    path.write_text("result=r.txt\nmode=single\ncount=0\n", encoding="utf-8")
    assert mb.read_request(str(path)).min_confidence == mb.DEFAULT_MIN_CONFIDENCE == 0.9


def test_single_and_batch_requests_carry_measurements(tmp_path) -> None:
    """A batch request's photos are measurement dicts (numbers as floats, status ok); a photo missing one of them is a ValueError."""
    path = tmp_path / "r.txt"
    keys = "photo.1.id=4\nphoto.1.cx=25.5\nphoto.1.cy=15.25\nphoto.1.width=60\nphoto.1.height=40\nphoto.1.crop_width=50\nphoto.1.crop_height=20\n"
    path.write_text("result=r.txt\nmode=batch\ncount=1\n" + keys + "photo.1.orientation=DA\nphoto.1.angle=-2.21\n", encoding="utf-8")
    req = mb.read_request(str(path))
    assert req.photos == [] and req.measured == [{"id": "4", "cx": 25.5, "cy": 15.25, "width": 60.0, "height": 40.0, "crop_width": 50.0,
                                                  "crop_height": 20.0, "angle": -2.21, "orientation": "DA", "status": "ok"}]
    path.write_text("result=r.txt\nmode=single\ncount=1\n" + keys, encoding="utf-8")         # no angle
    with pytest.raises(ValueError):
        mb.read_request(str(path))


def test_workers_default_to_one(tmp_path) -> None:
    """Without workers= a request measures in one worker."""
    path = tmp_path / "r.txt"
    path.write_text("result=r.txt\nmode=measure\ncount=0\n", encoding="utf-8")
    assert mb.read_request(str(path)).workers == 1
