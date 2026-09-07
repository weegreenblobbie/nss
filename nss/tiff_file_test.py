"""Tests of nss.tiff_file: stored values kept (no min/max stretch), exact round trip, ICC profile kept, clipping on save."""
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from nss.tiff_file import TiffFile
from nss.icc import gamma_icc_bytes

ICC = gamma_icc_bytes("Adobe RGB (1998)", [(0.60974, 0.31111, 0.01947), (0.20528, 0.62567, 0.06087), (0.14919, 0.06322, 0.74457)], 563 / 256)


def test_read_keeps_stored_values(tmp_path: Path) -> None:
    """No min/max stretch: a uint16 image spanning 1000..30000 reads back as value / 65535."""
    raw = np.linspace(1000, 30000, 7 * 5 * 3).astype(np.uint16).reshape(7, 5, 3)
    path = str(tmp_path / "a.tif")
    TiffFile.from_array(raw / 65535.0, icc=ICC).saveas(path)
    t = TiffFile().read(path)
    assert t.array.dtype == np.float32
    assert_allclose(t.array, raw / 65535.0, atol=1e-7)
    assert t.array.min() > 0.01 and t.array.max() < 0.5


def test_round_trip_is_exact_and_keeps_icc(tmp_path: Path) -> None:
    """read then saveas reproduces the uint16 data exactly and keeps the ICC profile."""
    raw = np.random.default_rng(2).integers(0, 65536, (6, 9, 3), dtype=np.uint16)
    a, b = str(tmp_path / "a.tif"), str(tmp_path / "b.tif")
    TiffFile.from_array(raw / 65535.0, icc=ICC).saveas(a)
    t = TiffFile().read(a, dtype=np.float64)
    assert t.iccprofile == ICC
    t.saveas(b)
    u = TiffFile().read(b, dtype=np.float64)
    assert np.array_equal(np.round(u.array * 65535.0).astype(np.uint16), raw)
    assert u.iccprofile == ICC


def test_save_clips_to_range(tmp_path: Path) -> None:
    """Values outside 0..1 are clipped when saved as uint16."""
    path = str(tmp_path / "c.tif")
    TiffFile.from_array(np.array([[[-0.2, 0.5, 1.3]]] * 3), icc=ICC).saveas(path)
    assert_allclose(TiffFile().read(path, dtype=np.float64).array[0, 0], [0.0, 32768 / 65535.0, 1.0])
