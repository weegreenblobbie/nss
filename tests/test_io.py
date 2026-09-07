import numpy as np
import pytest
from nss.image_utils import to_uint8_display

def test_to_uint8_display() -> None:
    # Test float32 to uint8 mapping and bounds clipping
    input_arr = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    expected = np.array([0, 0, 128, 255, 255], dtype=np.uint8)
    output = to_uint8_display(input_arr)
    np.testing.assert_array_equal(output, expected)
