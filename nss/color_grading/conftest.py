"""Shared fixtures of the color grading tests: a synthetic Adobe RGB profile and a ramp builder."""
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
import pytest

from nss.icc import gamma_icc_bytes

ADOBE_RGB_PRIMARIES = [(0.60974, 0.31111, 0.01947), (0.20528, 0.62567, 0.06087), (0.14919, 0.06322, 0.74457)]      # D50-adapted XYZ
ADOBE_RGB_GAMMA = 563 / 256


@pytest.fixture(scope="session")
def adobe_icc() -> bytes:
    """Synthetic Adobe RGB (1998) ICC profile bytes, the working space of the Lightroom test exports."""
    return gamma_icc_bytes("Adobe RGB (1998)", ADOBE_RGB_PRIMARIES, ADOBE_RGB_GAMMA)


@pytest.fixture(scope="session")
def ramp() -> Callable[..., np.ndarray]:
    """ramp(end, width) -> (width, 3) encoded levels going linearly from black to the colour `end`."""
    def build(end: ArrayLike, width: int = 2048) -> np.ndarray:
        """(width, 3) ramp from black to `end`."""
        return np.linspace(0.0, 1.0, width)[:, None] * np.asarray(end, float)
    return build
