"""Sanity tests of the generated Shadows tables (shadows.py)."""
import numpy as np

from nss.color_grading.tonal import TonalWheel
from nss.color_grading.shadows import Shadows


def test_is_tonal_wheel() -> None:
    """Shadows is a TonalWheel with the right tint factor."""
    assert issubclass(Shadows, TonalWheel)
    assert Shadows.CHROMA_ONLY


def test_table_shapes() -> None:
    """The tables have the shapes of the level axis (801 points, 5 Blending rows)."""
    assert np.shape(Shadows.MASK) == (5, 801)
    for table in (Shadows.LUM, Shadows.CAP_HI, Shadows.CAP_LO):
        assert np.shape(table) == (801,)
    assert np.shape(Shadows.LIFT_L) == np.shape(Shadows.LIFT_Y)


def test_tables_finite() -> None:
    """Every table value is finite."""
    for table in (Shadows.MASK, Shadows.LUM, Shadows.CAP_HI, Shadows.CAP_LO, Shadows.LIFT_L, Shadows.LIFT_Y, Shadows.CAP_A):
        assert np.all(np.isfinite(table))


def test_mask_non_negative() -> None:
    """The mask is never negative."""
    assert np.min(Shadows.MASK) >= 0.0


def test_cap_window_ordered() -> None:
    """The window's upper edge is never below its lower edge."""
    assert np.all(Shadows.CAP_HI >= Shadows.CAP_LO)


def test_lift_levels_increasing() -> None:
    """The black-lift level axis increases, as np.interp requires."""
    assert np.all(np.diff(Shadows.LIFT_L) > 0)
