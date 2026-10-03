import numpy as np
import pytest
from numpy.testing import assert_allclose

from nss.color_grading.math import rgb_to_lab_pure


def test_rgb_to_lab_pure_black():
    """Verify that pure black (0, 0, 0) transforms to (L*=0, a*=0, b*=0)."""
    # Non-square dimensions per Rule 20.1
    black_rgb = np.zeros((7, 5, 3), dtype=np.uint16)
    lab = rgb_to_lab_pure(black_rgb)

    assert lab.shape == (7, 5, 3)
    assert lab.dtype == np.float64
    assert_allclose(lab, 0.0, atol=1e-5)


def test_rgb_to_lab_pure_white():
    """Verify that pure white (65535, 65535, 65535) transforms to (L*=100, a*=0, b*=0)."""
    # Non-square dimensions per Rule 20.1
    white_rgb = np.full((7, 5, 3), 65535, dtype=np.uint16)
    lab = rgb_to_lab_pure(white_rgb)

    assert lab.shape == (7, 5, 3)
    assert lab.dtype == np.float64
    assert_allclose(lab[:, :, 0], 100.0, atol=0.01)
    assert_allclose(lab[:, :, 1], 0.0, atol=0.01)
    assert_allclose(lab[:, :, 2], 0.0, atol=0.01)


def test_rgb_to_lab_pure_shape_and_dtype():
    """Verify that output array preserves input spatial shape and returns np.float64."""
    # Non-square dimensions per Rule 20.1
    input_rgb = np.linspace(0, 65535, 13 * 11 * 3, dtype=np.float32).reshape(13, 11, 3)
    lab = rgb_to_lab_pure(input_rgb)

    assert lab.shape == (13, 11, 3)
    assert lab.dtype == np.float64


def test_rgb_to_lab_pure_value_flow():
    """Verify value flow through non-zero color patches (Rule 20.2)."""
    # Pure red 16-bit patch: high L*, positive a*, positive b*
    red_rgb = np.zeros((9, 6, 3), dtype=np.uint16)
    red_rgb[:, :, 0] = 65535
    lab = rgb_to_lab_pure(red_rgb)

    assert lab.shape == (9, 6, 3)
    assert lab.dtype == np.float64
    # Standard sRGB pure red in D65 Lab: L* ~ 53.24, a* ~ 80.09, b* ~ 67.20
    assert np.all(lab[:, :, 0] > 50.0)
    assert np.all(lab[:, :, 1] > 70.0)
    assert np.all(lab[:, :, 2] > 60.0)
