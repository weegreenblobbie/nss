"""Sanity tests of the generated Midtones tables (midtones.py)."""
import numpy as np

from nss.color_grading.tonal import TonalWheel
from nss.color_grading.midtones import Midtones


def test_is_tonal_wheel() -> None:
    """Midtones is a TonalWheel with the right tint factor."""
    assert issubclass(Midtones, TonalWheel)
    assert not Midtones.CHROMA_ONLY


def test_table_shapes() -> None:
    """The tables have the shapes of the level axis (801 points, 5 Blending rows)."""
    assert np.shape(Midtones.MASK) == (5, 801)
    assert np.shape(Midtones.LUM) == (801,)


def test_tables_finite() -> None:
    """Every table value is finite."""
    for table in (Midtones.MASK, Midtones.LUM):
        assert np.all(np.isfinite(table))


def test_mask_non_negative() -> None:
    """The mask is never negative."""
    assert np.min(Midtones.MASK) >= 0.0


def test_no_cap_tables() -> None:
    """Midtones has no window, cap or black lift."""
    assert Midtones.CAP_HI is None and Midtones.CAP_LO is None
    assert Midtones.LIFT_L is None and Midtones.LIFT_Y is None
