"""Luminosity Mask Engine for Shadows in Adobe Lightroom Color Grading.

Implements empirical 21-knot quadratic Akima spline interpolation across
Balance (-100 to +100) and Blending (0 to 100) parameter states.
"""

from typing import Union
import numpy as np
from scipy.interpolate import Akima1DInterpolator


class LumaMaskShadows:
    """Shadows luminosity mask generator based on reverse-engineered Lightroom math."""

    # 21-knot quadratic lattice: x = linspace(0.0, 1.0, 21)**2
    fixed_x = np.linspace(0.0, 1.0, 21) ** 2

    # Empirical normalized Y-values at fixed_x coordinates
    SHADOWS_BALANCE_KNOTS = {
        -100: [0.0, 0.02728, 0.09806, 0.18073, 0.26325, 0.34323, 0.42642, 0.51267, 0.6006, 0.69237, 0.78502, 0.87371, 0.9515, 0.99426, 0.99344, 0.93309, 0.79248, 0.56346, 0.27986, 0.07615, 0.0],
        -50: [0.0, 0.03551, 0.12763, 0.23524, 0.34253, 0.44553, 0.55145, 0.65991, 0.76608, 0.87001, 0.96134, 0.99761, 0.98309, 0.89366, 0.72451, 0.50568, 0.30874, 0.20085, 0.11492, 0.03651, 0.0],
        0: [0.0, 0.04805, 0.17269, 0.31827, 0.46187, 0.5969, 0.73011, 0.86, 0.96686, 0.99899, 0.97119, 0.85142, 0.65784, 0.46045, 0.34036, 0.2804, 0.20917, 0.13466, 0.06741, 0.01867, 0.0],
        50: [0.0, 0.07198, 0.2576, 0.47259, 0.67588, 0.85075, 0.98069, 0.98367, 0.88748, 0.6922, 0.4911, 0.39551, 0.35281, 0.2969, 0.23503, 0.17374, 0.11745, 0.06942, 0.03244, 0.00832, 0.0],
        100: [0.0, 0.17557, 0.61475, 0.95879, 0.96183, 0.73153, 0.52341, 0.47727, 0.41754, 0.35053, 0.28394, 0.22292, 0.17154, 0.12882, 0.09352, 0.06438, 0.04126, 0.02335, 0.0101, 0.0021, 0.0],
    }

    SHADOWS_BLEND_KNOTS = {
        0: [0.0, 0.05061, 0.18189, 0.33523, 0.48536, 0.62822, 0.76582, 0.89795, 0.99276, 0.99262, 0.90675, 0.69484, 0.39439, 0.12199, 0.00167, 0.00011, 0.00017, 0.00022, 0.00021, 0.00021, 0.0],
        25: [0.0, 0.04965, 0.17845, 0.32887, 0.47616, 0.61631, 0.75301, 0.88435, 0.98714, 0.99787, 0.93691, 0.7642, 0.50901, 0.268, 0.14694, 0.11999, 0.08941, 0.0576, 0.0288, 0.00792, 0.0],
        50: [0.0, 0.04805, 0.17269, 0.31827, 0.46187, 0.5969, 0.73011, 0.86, 0.96686, 0.99899, 0.97119, 0.85142, 0.65784, 0.46045, 0.34036, 0.2804, 0.20917, 0.13466, 0.06741, 0.01867, 0.0],
        75: [0.0, 0.04233, 0.15426, 0.28554, 0.41831, 0.54688, 0.68131, 0.79626, 0.89991, 0.97887, 0.98349, 0.92848, 0.81978, 0.6886, 0.57673, 0.47681, 0.35555, 0.22856, 0.11428, 0.03178, 0.0],
        100: [0.0, 0.0336, 0.12403, 0.23287, 0.34378, 0.45464, 0.57575, 0.68821, 0.77366, 0.85961, 0.9345, 0.99033, 0.98595, 0.94005, 0.84317, 0.69768, 0.5191, 0.3325, 0.16576, 0.04597, 0.0],
    }

    def _get_slider_knots(self, value: float, knot_dict: dict[Union[int, float], list[float]]) -> np.ndarray:
        """Interpolate the 21 knot heights for an arbitrary slider position."""
        keys = sorted(knot_dict.keys())
        matrix = np.array([knot_dict[k] for k in keys], dtype=np.float64)
        interpolated = np.array([
            np.interp(value, keys, matrix[:, j])
            for j in range(len(self.fixed_x))
        ], dtype=np.float64)
        return interpolated

    def get_mask(self, luminance_array: np.ndarray, balance: float = 0, blend: float = 50) -> np.ndarray:
        """Calculate the normalized Shadows mask weights for input luminance.

        Parameters
        ----------
        luminance_array : np.ndarray
            A 2D NumPy array representing normalized luminance in the range [0.0, 1.0].
            Must satisfy:
            - Type: np.ndarray
            - Dimensions: Exactly 2D (ndim == 2, shape: (height, width))
            - Dtype: np.float32 for performance and precision
        balance : float, optional
            Lightroom Balance slider in range [-100, 100]. Default is 0.
        blend : float, optional
            Lightroom Blending slider in range [0, 100]. Default is 50.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of shape (height, width) with normalized shadow mask weights
            clipped to [0.0, 1.0].
        """
        assert isinstance(luminance_array, np.ndarray), "luminance_array must be a numpy array"
        assert luminance_array.ndim == 2, f"luminance_array must be a 2D array, got ndim={luminance_array.ndim}"
        assert luminance_array.dtype == np.float32, (
            f"luminance_array must be np.float32 for performance, got {luminance_array.dtype}"
        )

        # 1. Get interpolated 21-knot arrays for the requested UI states
        base_y = np.array(self.SHADOWS_BALANCE_KNOTS[0])
        bal_y = self._get_slider_knots(balance, self.SHADOWS_BALANCE_KNOTS)
        blend_y = self._get_slider_knots(blend, self.SHADOWS_BLEND_KNOTS)

        # 2. Additive Delta Combination (Clamp 0.0 to 1.0)
        final_y = np.clip(base_y + (bal_y - base_y) + (blend_y - base_y), 0.0, 1.0)

        # 3. Generate Akima Spline and evaluate pixels
        spline = Akima1DInterpolator(self.fixed_x, final_y)

        # 4. Map the normalized luminance image to mask weights
        return spline(luminance_array)
