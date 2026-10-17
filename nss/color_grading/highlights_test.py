"""Sanity tests of the generated Highlights tables (highlights.py)."""
import numpy as np

from nss.color_grading.tonal import TonalWheel
from nss.color_grading.highlights import Highlights


def test_is_tonal_wheel() -> None:
    """Highlights is a TonalWheel with the right tint factor."""
    assert issubclass(Highlights, TonalWheel)
    assert Highlights.CHROMA_ONLY


def test_table_shapes() -> None:
    """The tables have the shapes of the level axis (801 points, 5 Blending rows)."""
    assert np.shape(Highlights.MASK) == (5, 801)
    for table in (Highlights.LUM, Highlights.CAP_HI, Highlights.CAP_LO):
        assert np.shape(table) == (801,)


def test_tables_finite() -> None:
    """Every table value is finite."""
    for table in (Highlights.MASK, Highlights.LUM, Highlights.CAP_HI, Highlights.CAP_LO, Highlights.CAP_A):
        assert np.all(np.isfinite(table))


def test_mask_non_negative() -> None:
    """The mask is never negative."""
    assert np.min(Highlights.MASK) >= 0.0


def test_cap_window_ordered() -> None:
    """The window's upper edge is never below its lower edge."""
    assert np.all(Highlights.CAP_HI >= Highlights.CAP_LO)


def test_no_black_lift() -> None:
    """Highlights has no black lift."""
    assert Highlights.LIFT_L is None and Highlights.LIFT_Y is None
