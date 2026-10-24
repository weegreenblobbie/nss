"""Tests of plugin.icm_bridge: reading the request, writing the result, the exit codes."""
from pathlib import Path
import pytest
from plugin import icm_bridge as ib
from plugin.handover import read_values


def test_request_paths_are_read(tmp_path: Path) -> None:
    """tiff, output and result come through, a value keeping its '='."""
    path = tmp_path / "request.txt"
    path.write_text("tiff=C:\\tmp\\in=1.tif\noutput=D:\\photos\\a_icm.tif\nresult=C:\\tmp\\result.txt\n", encoding="utf-8")
    assert ib.read_request(str(path)) == ib.Request("C:\\tmp\\in=1.tif", "D:\\photos\\a_icm.tif", "C:\\tmp\\result.txt")


@pytest.mark.parametrize("dropped", ["tiff", "output", "result"])
def test_request_needs_every_path(tmp_path: Path, dropped: str) -> None:
    """A request without one of its three paths is refused."""
    lines = [f"{key}=x.{key}" for key in ("tiff", "output", "result") if key != dropped]
    path = tmp_path / "request.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=dropped):
        ib.read_request(str(path))


def test_saved_result_round_trips(tmp_path: Path) -> None:
    """A saved result names the output and exits 0."""
    path = tmp_path / "result.txt"
    ib.write_result(str(path), ib.SAVED, "D:\\photos\\a_icm_2.tif")
    assert read_values(str(path)) == {"status": "saved", "output": "D:\\photos\\a_icm_2.tif"}
    assert ib.EXIT_CODES[ib.SAVED] == 0


def test_cancelled_and_error_results() -> None:
    """cancelled is the status alone (exit 1); error carries its message on one line (exit 2)."""
    assert ib.result_lines(ib.CANCELLED, "ignored.tif") == ["status=cancelled"]
    assert ib.result_lines(ib.ERROR, message="disk full\n  while writing") == ["status=error", "message=disk full while writing"]
    assert (ib.EXIT_CODES[ib.CANCELLED], ib.EXIT_CODES[ib.ERROR]) == (1, 2)


def test_unknown_status_is_refused() -> None:
    """Only saved, cancelled and error are written."""
    with pytest.raises(ValueError):
        ib.result_lines("done")
