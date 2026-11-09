"""Tests of plugin.stack_bridge: reading the request, writing the result, the exit codes."""
from pathlib import Path

import pytest

from plugin import stack_bridge as sb
from plugin.handover import read_values


def write_request(path: Path, **overrides: str) -> str:
    """Writes a request for a clean stack of a target and three frames; `overrides` replace or (with None) drop lines."""
    values = {
        "mode": "clean", "target": "C:\\tmp\\nss_stack_0.tif", "count": "3", "frame.1": "C:\\tmp\\nss_stack_1.tif",
        "frame.2": "C:\\tmp\\nss_stack_2.tif", "frame.3": "C:\\tmp\\nss_stack_3.tif",
        "output": "D:\\photos\\a_clean.tif", "output.1": "D:\\photos\\b_clean.tif", "output.2": "D:\\photos\\c_clean.tif",
        "output.3": "D:\\photos\\d_clean.tif", "result": "C:\\tmp\\result.txt",
    }
    values.update(overrides)
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items() if value is not None), encoding="utf-8")
    return str(path)


def test_request_is_read_in_frame_order(tmp_path: Path) -> None:
    """The mode, the target, the frames in order, the output and the result come through; work_dim defaults."""
    request = sb.read_request(write_request(tmp_path / "request.txt"))
    assert request == sb.Request(
        "clean", "C:\\tmp\\nss_stack_0.tif",
        ["C:\\tmp\\nss_stack_1.tif", "C:\\tmp\\nss_stack_2.tif", "C:\\tmp\\nss_stack_3.tif"],
        "D:\\photos\\a_clean.tif", "C:\\tmp\\result.txt", sb.DEFAULT_WORK_DIM,
        ["D:\\photos\\a_clean.tif", "D:\\photos\\b_clean.tif", "D:\\photos\\c_clean.tif", "D:\\photos\\d_clean.tif"],
    )


def test_focus_mode_and_work_dim_and_paths_with_equals_signs(tmp_path: Path) -> None:
    """focus is accepted, work_dim is read, and a path keeps an '=' in it."""
    request = sb.read_request(write_request(tmp_path / "r.txt", mode="focus", work_dim="2000", count="1", **{"frame.1": "C:\\a=b\\f.tif", "output.2": None, "output.3": None}))
    assert request.mode == "focus" and request.work_dim == 2000 and request.frames[0] == "C:\\a=b\\f.tif"


@pytest.mark.parametrize("dropped", ["mode", "target", "count", "output", "result", "frame.2", "output.1", "output.3"])
def test_request_needs_every_line(tmp_path: Path, dropped: str) -> None:
    """A request without one of its lines (or with a frame missing from the count) is refused, naming it."""
    with pytest.raises(ValueError, match=dropped.replace(".", r"\.")):
        sb.read_request(write_request(tmp_path / "request.txt", **{dropped: None}))


@pytest.mark.parametrize("field, value", [("mode", "stretch"), ("count", "0"), ("count", "many"), ("work_dim", "tiny"), ("work_dim", "0")])
def test_request_values_are_checked(tmp_path: Path, field: str, value: str) -> None:
    """An unknown mode, a frame count that is not a positive whole number, or a bad work_dim is refused."""
    with pytest.raises(ValueError, match=field):
        sb.read_request(write_request(tmp_path / "request.txt", **{field: value}))


def test_saved_result_round_trips(tmp_path: Path) -> None:
    """A saved result names the output and which photo was the target (0 the one the plug-in sent first), and exits 0."""
    path = tmp_path / "result.txt"
    sb.write_result(str(path), sb.SAVED, "D:\\photos\\a_clean_2.tif")
    assert read_values(str(path)) == {"status": "saved", "output": "D:\\photos\\a_clean_2.tif", "target": "0"}
    sb.write_result(str(path), sb.SAVED, "D:\\photos\\c_clean.tif", target=2)
    assert read_values(str(path)) == {"status": "saved", "output": "D:\\photos\\c_clean.tif", "target": "2"}
    assert sb.EXIT_CODES[sb.SAVED] == 0


def test_cancelled_and_error_results() -> None:
    """cancelled is the status alone (exit 1); error carries its message on one line (exit 2)."""
    assert sb.result_lines(sb.CANCELLED, "ignored.tif") == ["status=cancelled"]
    assert sb.result_lines(sb.ERROR, message="registration failed:\n  too few matches") == ["status=error", "message=registration failed: too few matches"]
    assert (sb.EXIT_CODES[sb.CANCELLED], sb.EXIT_CODES[sb.ERROR]) == (1, 2)


def test_unknown_status_is_refused() -> None:
    """Only saved, cancelled and error are statuses."""
    with pytest.raises(ValueError, match="status"):
        sb.result_lines("done")
