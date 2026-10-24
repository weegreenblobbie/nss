"""Tests of plugin.handover.read_values."""
from pathlib import Path
from plugin.handover import read_values


def test_values_keep_equals_signs_and_skip_other_lines(tmp_path: Path) -> None:
    """Keys and values are stripped, a value keeps its '=' and lines without '=' are left out."""
    path = tmp_path / "r.txt"
    path.write_text(" tiff = C:\\tmp\\a=b.tif \n\nno equals here\nstatus=saved\n", encoding="utf-8")
    assert read_values(str(path)) == {"tiff": "C:\\tmp\\a=b.tif", "status": "saved"}
