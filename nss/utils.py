import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def _gaus2d(x=0.0, y=0.0, mx=0.0, my=0.0, sx=1.0, sy=1.0):
    return np.exp(-((x - mx)**2.0 / (2.0 * sx**2.0) + (y - my)**2.0 / (2.0 * sy**2.0)))


def _make_2d_gaussian(size):
    x = np.linspace(-size, size, 2 * size + 1)
    x, y = np.meshgrid(x, x)
    g = _gaus2d(x, y)
    g /= np.sum(g.flatten())
    return g


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


def _write_circle(array, center, radius, value):
    M, N = array.shape
    m_center, n_center = center
    for m in range(max(0, m_center - radius), min(m_center + radius, M)):
        for n in range(max(0, n_center - radius), min(n_center + radius, N)):
            dist = np.hypot(m - m_center, n - n_center)
            if dist <= radius:
                array[m,n] += value


def generate_moon(radius, num_creators=100):
    """
    Generates an array radius x radius with the specified number of creators.
    """
    out = np.zeros((2*radius,2*radius), dtype=np.float32)
    _write_circle(out, (radius, radius), radius, 0.250)
    # Add random creators.
    for creator in _generate_non_overlapping_creators(radius, num_creators, 5, radius//3):
        mc, nc, r = creator
        mc = int(mc + radius)
        nc = int(nc + radius)
        r = int(r)
        _write_circle(out, (mc, nc), r, np.random.normal(0.250, 0.100))
    out /= np.max(out.flatten())
    print(f"max(moon) = {np.max(out.flatten())}")
    return out


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

    out = np.zeros(image_shape, dtype=np.float32)
    num_stars = np.random.randint(num_stars_min, num_stars_max)

    if radius_moon is not None:
        assert radius_moon >= 100
        m0 = np.random.randint(radius_moon, M - 2 * radius_moon)
        n0 = np.random.randint(radius_moon, N - 2 * radius_moon)
        moon = generate_moon(radius_moon, 100)
        m1 = m0 + moon.shape[0]
        n1 = n0 + moon.shape[1]
        if m1 > M:
            m1 = M
        if n1 > N:
            n1 = N
        out[m0:m1, n0:n1] = moon[0:m1-m0, 0:n1-n0]

    # Blur with a gaussian.
    size = np.random.randint(1, 3)
    g = _make_2d_gaussian(size)
    
    for _ in range(num_stars):
        m = np.random.randint(0, M)
        n = np.random.randint(0, N)
        out[m,n] = 4.0

    out = convolve2d(out, g, mode="same", boundary="symm")

    # Add some noise.
    out += np.random.uniform(0.0, 0.15, (M,N))
    out /= np.max(out.flatten())
    out[out <= 0.0] = 0.0
    return (50_000 * out).astype(np.uint16)


def imshow(array):
    """
    Plots a 2d array to the current maptloblib axes.
    """

    # Grayscale.
    if array.ndim == 2:
        plt.imshow(array, cmap = 'gray', interpolation='nearest')

    # Color image.
    elif array.ndim == 3:
        plt.imshow(array, interpolation='nearest')

    else:
        raise RuntimeError("don't know how to handle array with shape %s" % repr(array.shape))
