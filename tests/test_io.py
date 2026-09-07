import numpy as np
import pytest
from nss.image_utils import load_tiff_to_float32, to_uint8_display

def test_to_uint8_display() -> None:
    # Test float32 to uint8 mapping and bounds clipping
    input_arr = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    expected = np.array([0, 0, 128, 255, 255], dtype=np.uint8)
    output = to_uint8_display(input_arr)
    np.testing.assert_array_equal(output, expected)

def test_load_tiff_to_float32() -> None:
    # The load_tiff_to_float32 function uses TiffFile, which in tests is mocked
    # by tests/conftest.py to return [[0, 32768, 65535]].
    arr, tiff_obj = load_tiff_to_float32("dummy_path.tif")
    
    assert arr.dtype == np.float32
    # Verify values are normalized between 0.0 and 1.0 based on MockTiffFile asarray(): [[0, 32768, 65535]]
    assert np.isclose(arr[0, 0], 0.0)
    assert np.isclose(arr[0, 2], 1.0)
    assert np.isclose(arr[0, 1], 0.5, atol=1e-3)
