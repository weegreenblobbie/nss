import os
import cv2
import numpy as np
import pytest

from nss.color_grading.math import rgb_to_lab_pure
from nss.color_grading.highlights import Highlights


def test_assertion_1d_array_raises():
    """Test that a 1D array raises an AssertionError requiring a 2D array."""
    engine = Highlights()
    arr_1d = np.zeros(10, dtype=np.float32)
    with pytest.raises(AssertionError, match="must be a 2D array"):
        engine.get_mask(arr_1d)


def test_assertion_3d_array_raises():
    """Test that a 3D array raises an AssertionError requiring a 2D array."""
    engine = Highlights()
    arr_3d = np.zeros((13, 11, 3), dtype=np.float32)
    with pytest.raises(AssertionError, match="must be a 2D array"):
        engine.get_mask(arr_3d)


def test_assertion_float64_dtype_raises():
    """Test that a float64 array raises an AssertionError requiring np.float32."""
    engine = Highlights()
    arr_f64 = np.zeros((13, 11), dtype=np.float64)
    with pytest.raises(AssertionError, match="must be np.float32"):
        engine.get_mask(arr_f64)


def test_assertion_uint8_dtype_raises():
    """Test that a uint8 array raises an AssertionError requiring np.float32."""
    engine = Highlights()
    arr_u8 = np.zeros((13, 11), dtype=np.uint8)
    with pytest.raises(AssertionError, match="must be np.float32"):
        engine.get_mask(arr_u8)


def test_assertion_non_array_raises():
    """Test that non-numpy array input raises an AssertionError."""
    engine = Highlights()
    with pytest.raises(AssertionError, match="must be a numpy array"):
        engine.get_mask([[0.1, 0.2], [0.3, 0.4]])


def test_valid_2d_array_execution():
    """Test that a valid 2D float32 array produces a matching 2D mask."""
    engine = Highlights()
    # Non-square dimensions with non-zero distinct values (Rules 20.1 & 20.2)
    lum = np.linspace(0.05, 0.95, 13 * 11, dtype=np.float32).reshape(13, 11)
    mask = engine.get_mask(lum, balance=0, blend=50)

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (13, 11)
    assert mask.dtype == np.float32 or mask.dtype == np.float64
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
    assert np.max(mask) > 0.0


def test_baseline_highlights():
    filepath = os.path.join("tests", "test_data", "set_c_hi_bal_0.tif")
    if not os.path.exists(filepath):
        filepath = os.path.join("/workspace", "tests", "test_data", "set_c_hi_bal_0.tif")
    if not os.path.exists(filepath):
        return

    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    assert img is not None, f"Failed to load {filepath}"
    row = img[100:101, :]
    lab = rgb_to_lab_pure(row)

    a_channel = lab[..., 1][0]
    b_channel = lab[..., 2][0]
    chroma = np.sqrt(a_channel**2 + b_channel**2)
    ground_truth = chroma / np.max(chroma)

    engine = Highlights()
    luminance_array = np.linspace(0.0, 1.0, 2048, dtype=np.float32).reshape(1, -1)
    mask = engine.get_mask(luminance_array, balance=0, blend=50)[0]

    rmse = np.sqrt(np.mean((ground_truth - mask)**2)) * 65535.0
    print(f"\nBaseline (Bal 0, Blend 50) -> RMSE: {rmse:.1f} DN")

    assert rmse < 500  # Should be well below this threshold


def test_highlights_non_square_2d_shape_and_value_flow():
    """Verify non-square 2D array execution and value flow (Rules 20.1 & 20.2)."""
    engine = Highlights()
    lum_2d = np.linspace(0.1, 0.9, 13 * 11, dtype=np.float32).reshape(13, 11)
    mask = engine.get_mask(lum_2d, balance=25, blend=60)

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (13, 11)
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
    assert np.max(mask) > 0.1
    assert not np.all(mask == mask[0, 0])
