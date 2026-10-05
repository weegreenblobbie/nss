import numpy as np
from scipy.interpolate import Akima1DInterpolator, CloughTocher2DInterpolator


class Shadows:
    """
    High-performance runtime engine for generating Lightroom-compatible Shadows luma masks.
    
    Uses a dual-interpolator architecture:
    1. Z-Axis 2D Surface (CloughTocher2DInterpolator): A 2D Delaunay triangulation mesh with smooth 
       cubic C1 continuous Bézier patches that interpolates the 21 geometric knots across the coupled 
       (Balance, Blend) slider parameter space. The interpolator is constructed once during initialization 
       and cached for high-performance 2D surface evaluation across frames.
    2. X-Axis Akima (Akima1DInterpolator): Maps the resulting 21 knots to the continuous pixel luminance range.
    """

    FIXED_X = np.array([
        0.0, 0.0025, 0.01, 0.0225, 0.04, 0.0625, 0.09, 0.1225, 0.16, 0.2025, 
        0.25, 0.3025, 0.36, 0.4225, 0.49, 0.5625, 0.64, 0.7225, 0.81, 0.9025, 1.0
    ])

    SHADOWS_KNOTS = {
        (-100, 50): [0.0, 0.02728, 0.09806, 0.18073, 0.26325, 0.34323, 0.42642, 0.51267, 0.6006, 0.69237, 0.78502, 0.87371, 0.9515, 0.99426, 0.99344, 0.93309, 0.79248, 0.56346, 0.27986, 0.07615, 0.0],
        (-50, 50):  [0.0, 0.03551, 0.12763, 0.23524, 0.34253, 0.44553, 0.55145, 0.65991, 0.76608, 0.87001, 0.96134, 0.99761, 0.98309, 0.89366, 0.72451, 0.50568, 0.30874, 0.20085, 0.11492, 0.03651, 0.0],
        (0, 50):    [0.0, 0.04805, 0.17269, 0.31827, 0.46187, 0.5969,  0.73011, 0.86,    0.96686, 0.99899, 0.97119, 0.85142, 0.65784, 0.46045, 0.34036, 0.2804,  0.20917, 0.13466, 0.06741, 0.01867, 0.0],
        (25, 50):   [0.0, 0.05771, 0.20652, 0.38182, 0.55061, 0.70715, 0.85135, 0.97764, 0.9969,  0.94772, 0.7979,  0.59034, 0.42247, 0.35489, 0.29894, 0.23307, 0.16461, 0.10093, 0.04849, 0.01293, 0.0],
        (50, 50):   [0.0, 0.07198, 0.2576,  0.47259, 0.67588, 0.85075, 0.98069, 0.98367, 0.88748, 0.6922,  0.4911,  0.39551, 0.35281, 0.2969,  0.23503, 0.17374, 0.11745, 0.06942, 0.03244, 0.00832, 0.0],
        (60, 50):   [0.0, 0.08021, 0.28602, 0.52447, 0.74339, 0.91769, 0.99795, 0.93368, 0.76434, 0.54645, 0.41458, 0.37493, 0.32389, 0.26511, 0.20506, 0.14893, 0.09914, 0.05791, 0.02678, 0.00678, 0.0],
        (70, 50):   [0.0, 0.0919,  0.32771, 0.59467, 0.82929, 0.97516, 0.9823,  0.83273, 0.60641, 0.4409,  0.39645, 0.34814, 0.29092, 0.23199, 0.17575, 0.12557, 0.08262, 0.04786, 0.02193, 0.00538, 0.0],
        (80, 50):   [0.0, 0.10826, 0.38297, 0.6886,  0.92484, 0.99638, 0.89049, 0.66242, 0.4675,  0.41596, 0.36918, 0.31136, 0.25208, 0.19622, 0.14611, 0.10314, 0.06713, 0.03853, 0.01744, 0.00415, 0.0],
        (90, 50):   [0.0, 0.12966, 0.46755, 0.81216, 0.98945, 0.92749, 0.70481, 0.49237, 0.43634, 0.38799, 0.32734, 0.26663, 0.20964, 0.16005, 0.11726, 0.08209, 0.05282, 0.0303,  0.01346, 0.00309, 0.0],
        (100, 50):  [0.0, 0.17557, 0.61475, 0.95879, 0.96183, 0.73153, 0.52341, 0.47727, 0.41754, 0.35053, 0.28394, 0.22292, 0.17154, 0.12882, 0.09352, 0.06438, 0.04126, 0.02335, 0.0101,  0.0021,  0.0],
        (0, 0):     [0.0, 0.05061, 0.18189, 0.33523, 0.48536, 0.62822, 0.76582, 0.89795, 0.99276, 0.99262, 0.90675, 0.69484, 0.39439, 0.12199, 0.00167, 0.00011, 0.00017, 0.00022, 0.00021, 0.00021, 0.0],
        (0, 5):     [0.0, 0.05043, 0.18126, 0.33406, 0.48366, 0.62603, 0.76322, 0.89561, 0.99164, 0.99339, 0.91192, 0.70725, 0.41496, 0.14809, 0.02762, 0.02135, 0.01581, 0.01013, 0.00501, 0.00126, 0.0],
        (0, 10):    [0.0, 0.05027, 0.18067, 0.33297, 0.48209, 0.62398, 0.7618,  0.89311, 0.99076, 0.99481, 0.91803, 0.72044, 0.43681, 0.17582, 0.05514, 0.04395, 0.03276, 0.02107, 0.01052, 0.00282, 0.0],
        (0, 15):    [0.0, 0.05008, 0.17998, 0.3317,  0.48025, 0.6216,  0.75889, 0.88995, 0.98964, 0.99584, 0.92394, 0.73434, 0.45966, 0.20492, 0.08402, 0.06791, 0.05052, 0.03254, 0.01627, 0.0044,  0.0],
        (0, 25):    [0.0, 0.04965, 0.17845, 0.32887, 0.47616, 0.61631, 0.75301, 0.88435, 0.98714, 0.99787, 0.93691, 0.7642,  0.50901, 0.268,   0.14694, 0.11999, 0.08941, 0.0576,  0.0288,  0.00792, 0.0],
        (0, 75):    [0.0, 0.04233, 0.15426, 0.28554, 0.41831, 0.54688, 0.68131, 0.79626, 0.89991, 0.97887, 0.98349, 0.92848, 0.81978, 0.6886,  0.57673, 0.47681, 0.35555, 0.22856, 0.11428, 0.03178, 0.0],
        (0, 100):   [0.0, 0.0336,  0.12403, 0.23287, 0.34378, 0.45464, 0.57575, 0.68821, 0.77366, 0.85961, 0.9345,  0.99033, 0.98595, 0.94005, 0.84317, 0.69768, 0.5191,  0.3325,  0.16576, 0.04597, 0.0],
        (-100, 0):  [0.0, 0.02764, 0.09935, 0.18311, 0.26671, 0.34775, 0.43202, 0.51941, 0.60823, 0.7012,  0.79474, 0.88367, 0.96059, 0.9974,  0.9872,  0.90965, 0.74195, 0.47763, 0.16417, 0.00011, 0.0],
        (-100, 100):[0.0, 0.02213, 0.08229, 0.15358, 0.22796, 0.30422, 0.39288, 0.48602, 0.57692, 0.65598, 0.73335, 0.81632, 0.89239, 0.95438, 0.9931,  0.99581, 0.92202, 0.75615, 0.51018, 0.21293, 0.0],
        (100, 0):   [0.0, 0.19961, 0.68451, 0.99994, 0.80701, 0.37512, 0.02478, 0.0,     0.0,     0.0,     0.0,     0.00078, 0.00043, 0.00049, 0.00072, 0.00047, 0.00068, 0.00089, 0.00087, 0.00085, 0.0],
        (100, 100): [0.0, 0.09288, 0.33568, 0.60018, 0.82592, 0.96652, 0.99463, 0.92219, 0.80238, 0.66053, 0.52631, 0.40962, 0.31293, 0.2335,  0.16902, 0.11682, 0.07508, 0.04261, 0.01921, 0.00469, 0.0],
    }

    def __init__(self):
        # Build the 2D surface interpolator once during initialization
        points = list(self.SHADOWS_KNOTS.keys())
        values = list(self.SHADOWS_KNOTS.values())
        # Upgrade from flat linear triangles to smooth cubic C1 continuous patches
        self.surface_interpolator = CloughTocher2DInterpolator(points, values)

    def get_mask(self, luminance_array, balance=0, blend=50):
        """
        Generates the Shadows luma mask for a given image.
        
        Args:
            luminance_array (np.ndarray): 2D array of the image's grayscale pixel values.
                Must be strictly 2D and dtype np.float32, normalized to [0.0, 1.0].
            balance (float): UI Balance slider value within [-100, 100].
            blend (float): UI Blending slider value within [0, 100].

        Returns:
            np.ndarray: Evaluated 2D float32 luma mask with values clamped to [0.0, 1.0].
        """
        assert isinstance(luminance_array, np.ndarray), "luminance_array must be a numpy array"
        assert luminance_array.ndim == 2, "luminance_array must be a 2D array"
        assert luminance_array.dtype == np.float32, "luminance_array must be np.float32 for performance"

        # 1. Ask the 2D surface for the exact 21 knots at this (Bal, Blend) coordinate
        raw_y = self.surface_interpolator(balance, blend)
        final_y = raw_y[0] if raw_y.ndim > 1 else raw_y
        final_y = np.clip(final_y, 0.0, 1.0)
        
        # 2. Fit the Akima spline to generate the continuous function
        spline = Akima1DInterpolator(self.FIXED_X, final_y)
        
        # 3. Map the normalized luminance image array through the spline
        luma_clamped = np.clip(luminance_array, 0.0, 1.0)
        mask = spline(luma_clamped)
        
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

