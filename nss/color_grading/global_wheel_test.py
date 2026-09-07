"""Sanity tests of the generated Global wheel tables (global_wheel.py)."""
import numpy as np
from numpy.testing import assert_allclose

from nss.color_grading.global_wheel import SATMULT_U, SATMULT_POS, SATMULT_NEG


def test_table_shapes() -> None:
    """All three tables have 41 points."""
    for table in (SATMULT_U, SATMULT_POS, SATMULT_NEG):
        assert np.shape(table) == (41,)


def test_level_axis() -> None:
    """The level axis is u = -13 .. -3 in steps of 0.25."""
    assert_allclose(SATMULT_U, np.linspace(-13.0, -3.0, 41))


def test_tables_finite_and_positive() -> None:
    """The multipliers are finite and positive."""
    for table in (SATMULT_POS, SATMULT_NEG):
        assert np.all(np.isfinite(table))
        assert np.all(table > 0.0)


def test_one_at_the_top() -> None:
    """Both tables are 1 from u = -4.25 up."""
    top = SATMULT_U >= -4.25
    assert_allclose(SATMULT_POS[top], 1.0)
    assert_allclose(SATMULT_NEG[top], 1.0)


def test_raised_channels_boosted_near_black() -> None:
    """Raised channels are boosted by more than 1.5 below u = -8."""
    assert np.all(SATMULT_POS[SATMULT_U <= -8.0] > 1.5)
