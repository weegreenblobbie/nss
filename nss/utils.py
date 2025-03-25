import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def _gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return np.exp(-((x - mx)**2.0 / (2.0 * sx**2.0) + (y - my)**2.0 / (2.0 * sy**2.0)))


def _generate_non_overlapping_creators(outer_radius, num_circles, min_radius, max_radius):
    circles = []
    
    while len(circles) < num_circles:
        # Generate random center and radius.
        radius = random.uniform(min_radius, max_radius)
        rr = outer_radius - radius - 1
        x = random.uniform(-rr, rr)
        y = random.uniform(-rr, rr)
        if outer_radius - radius < np.hypot(x, y):
            continue
        # Check for overlaps
        overlap = False
        for cx, cy, cr in circles:
            distance = np.hypot(cx - x, cy - y)
            if distance <= radius + cr:
                overlap = True
                break
        
        # Add circle if no overlap
        if not overlap:
            circles.append((x, y, radius))
            
    return circles


def _write_circle(array, center, radius, value, assign=False):
    M, N = array.shape
    m_center, n_center = center
    for m in range(m_center - radius, min(m_center + radius, M)):
            for n in range(n_center - radius, min(n_center + radius, N)):
                dist = np.hypot(m - m_center, n - n_center)
                if dist <= radius:
                    if assign:
                        array[m,n] = value
                    else:
                        array[m,n] += value


def generate_test_image(image_shape=(1600,2400), num_stars_min=16, num_stars_max=32, radius_moon=None, num_moon_creators=None):
    """
    Generate 16-bit array representing a star field for testing the algorithms
    in this package.
    """
    assert len(image_shape) == 2
    assert image_shape[0] * image_shape[1] > 0
    assert num_stars_min >= 0
    assert num_stars_max >= num_stars_min

    M, N = image_shape

    print(image_shape)

    out = np.zeros(image_shape, dtype=np.float32)
    num_stars = np.random.randint(num_stars_min, num_stars_max)

    for _ in range(num_stars):
        m = np.random.randint(0, M)
        n = np.random.randint(0, N)
        out[m,n] = 1.0

    # Create gaussian kernel.
    kernel_size = np.random.randint(1, 3)
    x = np.linspace(-kernel_size, kernel_size, 2 * kernel_size + 1)
    x, y = np.meshgrid(x, x)
    g = _gaus2d(x, y)

    out = convolve2d(out, g, mode='full')

    if radius_moon is not None:
        assert radius_moon >= 100

        # Moon center.
        m_center = np.random.randint(radius_moon, M - radius_moon)
        n_center = np.random.randint(radius_moon, N - radius_moon)
        _write_circle(out, (m_center, n_center), radius_moon, 0.500, assign=True)

        # Add random creators.
        for creator in _generate_non_overlapping_creators(radius_moon, 100, 5, radius_moon//4):
            mc, nc, r = creator
            mc = int(mc)
            nc = int(nc)
            r = int(r)
            center = (m_center + mc, n_center + nc)
            _write_circle(out, center, r, np.random.normal(0.500, 0.300))

    return out


def imshow(array):
    """
    Plots a 2d array to the current maptloblib axes.
    """

    peak = np.nanmax(array)

    if peak < 1.0:
        peak = 1.0

    # Grayscale.
    if array.ndim == 2:
        plt.imshow(array / peak, cmap = 'gray', interpolation='nearest')

    # Color image.
    elif array.ndim == 3:
        plt.imshow(array / peak, interpolation='nearest')

    else:
        raise RuntimeError("don't know how to handle array with shape %s" % repr(array.shape))
