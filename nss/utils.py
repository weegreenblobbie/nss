from dataclasses import dataclass
import datetime
import sys
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import convolve2d


@dataclass(kw_only=True)
class Circle:
    center: tuple[int, int]
    radius: float
    def to_int(self):
        return int(self.center[0] + 0.5), int(self.center[1] + 0.5), int(self.radius + 0.5)
    def __str__(self):
        """
        Custom __str__ method to format the output.
        """
        cx, cy = self.center
        return f"Circle(center=({cx:.2f}, {cy:.2f}), radius={self.radius:.2f}))"
    
    def __sub__(self, other):
        dm = self.center[0] - other.center[0]
        dn = self.center[1] - other.center[1]
        dr = self.radius - other.radius
        return Circle(center=(dm, dn), radius=dr)


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
        radius = np.random.uniform(min_radius, max_radius)
        rr = outer_radius - radius - 1
        x = np.random.uniform(-rr, rr)
        y = np.random.uniform(-rr, rr)
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

    vmax = np.nanmax(array)
    vmin = np.nanmin(array)
    array = np.array(array) / vmax
    array[np.isnan(array)] = 0.0

    # Grayscale.
    if array.ndim == 2:
        plt.imshow(array, interpolation='nearest', cmap="gray")

    # Color image.
    elif array.ndim == 3:
        plt.imshow(array, interpolation='nearest')

    else:
        raise RuntimeError("don't know how to handle array with shape %s" % repr(array.shape))

def auto_threshold_first_valley(data, bins=32):
    hist, t_axis = np.histogram(data.flatten(), bins=bins)
    
    abs_peak_idx = np.argmax(hist)

    last = hist[abs_peak_idx] + 1.0

    print(f"{abs_peak_idx}:last: {last}")

    i = abs_peak_idx

    while last > hist[i]:
        last = hist[i]
        i += 1
        if i >= hist.size:
            break

    return np.mean(t_axis[i:i + 1])


def auto_threshold_otsu(data, bins=32):
    """
    Calculates the Otsu threshold for a histogram computed over the data.

    Args:
        data (numpy.ndarray): The data to compute the histogram.
        bins (int): The number of bins to use for the histogram.

    Returns:
        float: The Otsu threshold value.
    """
    hist, t_axis = np.histogram(data.flatten(), bins=bins)
    total = np.sum(hist.flatten())
    if total <= 0.0:
        return 0.0
    max_variance = 0.0
    best_index = 0.0
    sum_b = 0.0
    sum_1 = np.sum(np.arange(hist.size) * hist)
    w_b = 0.0 # weight for the background class
    w_f = 0.0 # weight for the foreground class
    for i in np.arange(hist.size):
        w_b += hist[i]
        if w_b <= 0.0:
            continue
        w_f = total - w_b
        if w_f <= 0.0:
            break
        sum_b += i * hist[i]
        mean_b = sum_b / w_b
        mean_f = (sum_1 - sum_b) / w_f

        variance_between = w_b * w_f * (mean_b - mean_f) ** 2.0

        if variance_between > max_variance:
            max_variance = variance_between
            best_index = i

    return np.mean(t_axis[best_index:best_index + 1])


def detect_circle(image_array, min_radius=400, max_radius=700, param1=150, param2=30, min_dist=400):
    """
    Detects circles in an image using the Hough Circle Transform.

    Args:
        image_array (numpy.ndarray): The input grayscale image array.
        min_radius (int, optional): Minimum circle radius. Defaults to 10.
        max_radius (int, optional): Maximum circle radius. Defaults to 100.
        param1 (int, optional): Upper threshold for Canny edge detection. Defaults to 50.
        param2 (int, optional): Accumulator threshold for circle center detection. Defaults to 30.
        min_dist (int, optional): Minimum distance between detected circle centers. Defaults to 20.

    Returns:
        list: A list of detected circles, where each circle is represented as (x, y, radius),
              or None if no circles are found.
    """

    # Apply Gaussian blur to reduce noise and improve circle detection.
    #blurred_image = cv2.GaussianBlur(image_array, (5, 5), 0)
    blurred_image = image_array

    # Detect circles using Hough Circle Transform.
    circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        1,  # Inverse resolution ratio (1 = same resolution)
        min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None:
        # Convert circle coordinates and radii to integers.
        circles = np.round(circles[0, :]).astype("int")
        return circles.tolist()  # Convert to list of lists for easier handling
    else:
        return None


def sobel_edge_detection(image_array):
    """
    Applies Sobel edge detection to an image.

    Args:
        image_array (numpy.ndarray): The input grayscale image array.

    Returns:
        numpy.ndarray: The edge-detected image array.
    """
    assert image_array.ndim == 2

    # Apply Sobel filters in the x and y directions.
    sobelx = cv2.Sobel(image_array, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image_array, cv2.CV_32F, 0, 1, ksize=5)

    # Calculate the magnitude of the gradient.
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    return gradient_magnitude

def sobel_edge_detection_2(image_array):
    """
    Computes the Sobel transform with 45-degree rotated kernels.

    Args:
        image_array (numpy.ndarray): The input grayscale image array.

    Returns:
        numpy.ndarray: The gradient magnitude image.
    """
    """
    Applies Sobel edge detection to an image.

    Args:
        image_array (numpy.ndarray): The input grayscale image array.

    Returns:
        numpy.ndarray: The edge-detected image array.
    """
    assert image_array.ndim == 2

    # Apply Sobel filters in the x and y directions.
    sobelx = cv2.Sobel(image_array, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image_array, cv2.CV_32F, 0, 1, ksize=5)

    # rotate array 45 degrees.
    image_45 = scipy.ndimage.rotate(image_array, angle=45, reshape=False)
    sobelx_45 = cv2.Sobel(image_45, cv2.CV_32F, 1, 0, ksize=5)
    sobely_45 = cv2.Sobel(image_45, cv2.CV_32F, 0, 1, ksize=5)
    sobelx_45 = scipy.ndimage.rotate(sobelx_45, angle=-45, reshape=False)
    sobely_45 = scipy.ndimage.rotate(sobely_45, angle=-45, reshape=False)

    gradient = np.sqrt(sobelx**2 + sobely**2 + sobelx_45**2 + sobely_45**2)

    return gradient

def to_gray(image):

    match image.ndim:
        case 2:
            return image
        case 3:
            # Luminance in the Adobe Color colorspace
            #     https://community.adobe.com/t5/photoshop-ecosystem-discussions/how-to-measure-true-luminosity-of-a-color-sample-by-script/td-p/12094302
            return 0.29999 * image[..., 0] + 0.587 * image[..., 1] + 0.1131 * image[..., 2]

    raise ValueError(f"Expecting image.ndim in {{2, 3}}, got {image.ndim}")


def plot_patches(data, windows, show=True, title="windows"):

    black = "black"
    white = "white"
    red = "red"

    plt.figure(facecolor=black, figsize=(16,8))
    plt.title(title, color=white)
    ax = plt.gca()
    imshow(data)

    for win in windows:
        m0, n0 = win["pos"]
        size = win["size"]
        rect = plt.Rectangle((n0, m0), size, size, ls='-', color=white, fill=False)
        ax.add_patch(rect)
        ax.text(n0, m0 + size/2.0, f"%.3f" % win["score"], color=red)

    if show:
        plt.show()


def fit_circle(points):
    """
    Fits a circle to a set of (x, y) points using the least-squares method.

    Args:
        points (list or numpy.ndarray): A list or array of (x, y) points.

    Returns:
        tuple: (center_x, center_y, radius) of the best-fit circle, or None if fitting fails.
    """

    points = np.array(points)
    if len(points) < 3:  # Need at least 3 points to fit a circle.
        return None

    x = points[:, 0]
    y = points[:, 1]

    # Least-squares fitting.
    A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    b = x**2 + y**2

    try:
        center_x, center_y, c = np.linalg.lstsq(A, b, rcond=None)[0]
        radius = np.sqrt(center_x**2 + center_y**2 + c)
        return center_x, center_y, radius
    except np.linalg.LinAlgError:
        # Handle cases where the least-squares fit fails.
        return None


def detect_moon(image, grid_size=200, plot=False):
    """
    Detect the center and radius of the moon located in the image.

    Args:
        image (numpy.ndarray): The input image array.

        grid_size (int): The size of the grid patches used to detect the moon's
                         edge.

    Returns:
        m, n, r (float): The detected moon's center and radius.
    """
    assert image.dtype == np.float32
    sobel = sobel_edge_detection(to_gray(image))

    sobel /= np.nanmax(sobel)
    M, N = sobel.shape
    grid_step = grid_size // 2

    candidates = []
    for m in range(0, M - grid_step, grid_step):
        for n in range(0, N - grid_step, grid_step):
            m0 = m
            m1 = m0 + grid_size + 1
            n0 = n
            n1 = n0 + grid_size + 1
            p = np.array(sobel[m0 : m1, n0 : n1])
            pmax = np.nanmax(p)
            pmin = np.nanmin(p)
            psum = np.nansum(p)
            score = (pmax - pmin) * psum
            candidates.append(dict(score=score, pos=(m0,n0), size=grid_size + 1, pmax=pmax, p=p))

    candidates = sorted(candidates, key = lambda x: x["score"], reverse=True)

    # Normalize the scores.
    max_score = max(candidates[0]["score"], 1.0)

    for can in candidates:
        can["score"] /= max_score

    candidates = [can for can in candidates if can["score"] >= 0.6]

    if plot:
        plot_patches(image, candidates, show=False, title="")

    raw_points = set()
    for can in candidates:
        m0, n0 = can["pos"]
        p = can["p"]
        pmax = can["pmax"]
        mask = p > 0.90 * pmax
        mm, nn = np.where(mask)
        for m, n in [(m0 + m, n0 + n) for m, n in zip(mm, nn)]:
            raw_points.add((m,n))

    raw_points = list(raw_points)

    np.random.shuffle(raw_points)
    raw_points = raw_points[:100]
    m, n, r = fit_circle(raw_points)

    if plot:
        ax = plt.gca()
        for y, x in raw_points:
            plt.scatter(x, y, color="red")
    
        rect = plt.Circle((n, m), r, ls='-', color="red", fill=False)
        ax.add_patch(rect)

    return Circle(center=(m,n), radius=r)
    

def align_moon_images(target, source, target_circle, source_circle):
    """
    Align the source to the target using the detected circles as the starting
    point.

    Returns the heading and rotation to apply on source to be aligned with
    target.
    """
    assert target.dtype == np.float32
    assert source.dtype == np.float32
    assert target.ndim == 3
    assert source.ndim == 3
    assert target.shape == source.shape

    radius = int(max(target_circle.radius, source_circle.radius) + 0.5)

    # Perform a single alignment using a patch 2x the radius + a little 
    # margin.
    delta = target_circle - source_circle

    # m0, n0 is the upper left corner of the patch we'll slice out of the 
    # source image.
    src_m0, src_n0, src_r = source_circle.to_int()
    tgt_m0, tgt_n0, tgt_r = target_circle.to_int()
    off_m, off_n, _ = delta.to_int()

    m0 = src_m0 - src_r
    n0 = src_n0 - src_r
    m1 = m0 + 2 * src_r + 1
    n1 = n0 + 2 * src_r + 1

    src = source[m0 : m1, n0 : n1, :]

    tgt_pos = (tgt_m0 - tgt_r, tgt_n0 - tgt_r)
    offset = (off_m, off_n)

    # DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
    if False:
        plt.figure()
        imshow(src)
        plt.title("src")
        plt.figure()
        imshow(target[tgt_pos[0]:tgt_pos[0] + 2*radius, tgt_pos[1]:tgt_pos[1] + 2*radius])
        plt.title("tgt")
        plt.show()
    # DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG

    print(f"offset: {offset}")
    print(f"Target: pos: {tgt_pos[0]},{tgt_pos[1]}, radius: {tgt_r}")
    print(f"Source: pos: {m0},{n0}, radius: {src_r}")

    return brute_force_align(
        src,
        target,
        tgt_pos,
        offset,
    )

def brute_force_align(
        src,
        tgt,
        tgt_pos,
        offset,
    ):
    tgt = to_gray(tgt)

    da_pool = np.arange(-3.0, 3.1, 0.1)

    m0 = tgt_pos[0]
    n0 = tgt_pos[1]

    patch = Patch(src, (m0 - offset[0], n0 - offset[1]))

    jobs = []

    for dm in range(-3, 4, 1):
        for dn in range(-3, 4, 1):
            for da in da_pool:
                jobs.append( (dm, dn, da) )

    angle = 0.0
    best = patch.score(tgt, (m0, n0), angle)
    print(f"Computing root-mean-squared error...")
    scores = [best]
    erase = "\b" * 128
    for i, job in enumerate(jobs):
        log(f"{erase}    {i+1:4d}/{len(jobs)}: translate: {best.heading[0]:3d},{best.heading[1]:3d} angle:{best.angle:7.4f} score: {best.score:.5g}       ")
        dm, dn, a = job
        m = m0 + dm
        n = n0 + dn
        #scores.append(patch.score(tgt, (m, n), a))
        #scores = sorted(scores, key = lambda x: x.score, reverse=True)
        #best = scores[0]
        new = patch.score(tgt, (m, n), a)
        if new.score > best.score:
            best = new

    log("\n")
    return best


def pad_array(array, radius):
    assert array.dtype == np.float32, f"Expected np.float32, got {array.dtype}"
    radius = int(radius + 0.5)

    if array.ndim == 2:
        return _pad_array_2d(array, radius)

    else:
        return _pad_array_3d(array, radius)


def _pad_array_2d(array, radius):

    M, N = array.shape

    out = np.ones((M + 2 * radius, N + 2 * radius), array.dtype)

    out[radius : radius + M, radius : radius + N] = array

    return out


def _pad_array_3d(array, radius):

    M, N, O = array.shape

    out = np.ones((M + 2 * radius, N + 2 * radius, O), array.dtype)

    out[radius : radius + M, radius : radius + N, :] = array

    return out

def unpad_array(array, radius):

    if array.ndim == 2:
        return _unpad_array_2d(array, radius)

    else:
        return _unpad_array_3d(array, radius)


def _unpad_array_2d(array, radius):

    M, N = array.shape

    return array[
        radius : radius + M - 2 * radius,
        radius : radius + N - 2 * radius
    ]


def _unpad_array_3d(array, radius):

    M, N, _ = array.shape

    return array[
        radius : radius + M - 2 * radius,
        radius : radius + N - 2 * radius,
        :
    ]


def log(msg):

    sys.stdout.write(msg)
    sys.stdout.flush()


def zscore(array, valid):

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')
        warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')

        array = np.array(array)

        array[~valid] = np.nan

        mu = np.nanmean(array)
        sig = np.nanstd(array)

        if np.isnan(mu) or np.isnan(sig):
            return np.zeros_like(array)

        out = (array - mu) / sig

        return out



class timeit(object):

    def __init__(self, msg=None):
        if msg:
            log(msg)

    def __enter__(self):
        self.d0 = datetime.datetime.now()

    def __exit__(self, *args):
        delta = datetime.datetime.now() - self.d0
        total_seconds = delta.total_seconds()
        if total_seconds > 60.0:
            delta = str(delta)
        else:
            delta = "%.2f seconds" % total_seconds
        log(f"took {delta}\n")


class Aligned(object):
    '''
    This class stores an empty RGB image on construction.  The algorithms take
    pactches and align them to the reference images, then the patches are given
    to this class and the rgb values are copied out of the patch and stored.

    This class can be pickled to disk and the resulting tiff can also be saved.
    '''


    def __init__(self, shape):
        assert len(shape) >= 2
        self._rgb = np.nan * np.ones(shape, np.float32)
        self._count = np.ones(shape[0:2], np.float32)


    def add_patch(self, patch):

        array = patch.color_array()

        M, N = array.shape[0:2]

        m, n = patch.tgt_pos() # upper left corner of patch

        target = self._rgb[m : m + M, n : n + N, :]

        # Only write out pixels that are not NaN, rotating patches introduce
        # NaNs.

        valid = ~np.isnan(array)

        target[valid] = array[valid]

        target = self._count[m : m + M, n : n + N]

        target[valid[:,:,0]] += 1.0


    def get_rgb(self):

        rgb = np.array(self._rgb)

        return rgb


    def pickle(self, filename):

        d = dict(
            rgb = self._rgb,
            count = self._count,
        )

        with open(filename, 'wb') as fd:
            pickle.dump(d, fd)

        print("wrote %s" % filename)


    def savetiff(self, filename, resize = 1.0):

        rgb = self.get_rgb()

        rgb -= np.nanmin(rgb)
        rgb /= np.nanmax(rgb)

#~        invalid = np.isnan(rgb)

#~        rgb[invalid] = 0.0

#~        rgb = imresize(rgb, resize)

        # Convert float32 (0.0 to 1.0) to uint16.

        rgb *= (1 << 16) - 1

        rgb = rgb.astype(np.uint16)

        rgb = unpad_array(rgb, RADIUS)

        imwrite(filename, rgb, photometric = 'rgb')


    @classmethod
    def from_pickle(cls, filename):

        with open(filename, 'rb') as fd:
            d = pickle.load(fd)

        obj = cls(d['rgb'].shape)

        obj._rgb = d['rgb']
        obj._count = d['count']

        print("read %s" % filename)

        return obj

class Patch(object):
    '''
    A patch is a square sub image taking out of the source image.  It is scored
    using a cross correlation of the overlapping pixels.

    This class copies the source image in RGB, after the final translation
    and rotation is found, the RGB pixles are given to the Stacker class to be
    written to the final aligned image.
    '''

    def __init__(self, rgb, start_pos):

        assert \
            rgb.ndim == 3, \
            "Error, expecting patch to be a color image, got shape = %s" % \
                repr(rgb.shape)

        self._rgb = np.array(rgb)
        self._gray = to_gray(rgb)
        self._gray[np.isnan(self._gray)] = 0.0
        self._cache = dict()
        self._start_pos = start_pos
        self._angle = None
        self._heading = None

        self._rotate_kwargs = dict(
            reshape = False,
            mode = 'constant',
            cval = np.nan
        )
        self.shape = self._gray.shape

    def color_array(self):
        return self._rgb

    def rotate(self, angle):
        key = int(angle * 1000 + 0.5)
        if key == 0:
            return self._gray

        if key in self._cache:
            return self._cache[key]

        # Compute the rotation on cache miss.
        p = scipy.ndimage.rotate(self._gray, angle, **self._rotate_kwargs)
        self._cache[key] = p

        return p


    def score(self, tgt, tgt_pos, angle, plot=False):

        assert \
            tgt.ndim == 2, \
            "Error, expecting target array to be gray scale, (shape = %s)" % \
                repr(tgt.shape)

        M, N = tgt.shape  # height and width of target image aligning to

        size = self._gray.shape[0]  # size of the patch

        m, n = tgt_pos # upper left corner on target image

        assert m >= 0, f"invalid m: {m}"
        assert n >= 0, f"invalid n: {n}"

        # compute indices into target image and limit at image boundary

        m0 = m
        m1 = m0 + size

        n0 = n
        n1 = n0 + size

        if m0 < 0:
            m0 = 0
            m1 = size

        if m1 > M:
            m1 = M
            m0 = m1 - size

        if n0 < 0:
            n0 = 0
            n1 = size

        if n1 > N:
            n1 = N
            n0 = n1 - size

        # extract patch from target

        t = tgt[m0 : m1, n0 : n1]

        # rotate the patch to angle

        p = self.rotate(angle)

        assert p.shape == t.shape, f"shape mismatch: {p.shape} != {t.shape}, {m0}:{m1},{n0}:{n1}, (tgt.shape: {tgt.shape})"

        # Root mean squared error.
        score = rms_score(t, p)

        rs = RegScore(score, self._start_pos, tgt_pos, angle)

        if plot:
            plt.figure()
            imshow(self._gray)
            plt.title("src")

            plt.figure()
            imshow(t)
            plt.title("tgt")

            plt.figure()
            imshow(t - p)
            plt.title(f"score={score:.4g} tgt_pos:{rs.heading}, angle:{angle:.4f}")

            #plt.show()

        return rs

    def plot(self, title):
        plt.figure()
        imshow(self._gray)
        plt.title(title)
        plt.show()



class RegScore(object):
    '''
    A class to hold the registration score along with registration parameters
    so they can be tracked together.
    '''

    def __init__(self, score, src_pos, tgt_pos, angle):
        self.score = score
        self.src_pos = tuple(src_pos)
        self.tgt_pos = tuple(tgt_pos)
        self.heading = (tgt_pos[0] - src_pos[0], tgt_pos[1] - src_pos[1])
        self.angle = angle

    def __lt__(self, rhs):
        return self.score < rhs.score

    def __gt__(self, rhs):
        return self.score > rhs.score

    def __ge__(self, rhs):
        return self.score >= rhs.score

    def __str__(self):
        return "score=%.3f, src_pos=%s, tgt_pos=%s, heading=%s, angle=%s" % (
            self.score,
            self.src_pos,
            self.tgt_pos,
            self.heading,
            self.angle,
        )


def zscore(array, valid):

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')
        warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')

        array = np.array(array)

        array[~valid] = np.nan

        mu = np.nanmean(array)
        sig = np.nanstd(array)

        if np.isnan(mu) or np.isnan(sig):
            return np.zeros_like(array)

        out = (array - mu) / sig

        return out

def rms_score(p, t):
    _max = np.sqrt(np.nanmean(t ** 2.0))
    return float(_max - np.sqrt(np.nanmean((t - p) ** 2.0)))

def xcorr_score(p, t):
    '''
    xcorr that deals with nans.

    The patch and target arrays are z-scored which scales each image in units of
    standard deviation.  This way, we are comparing the shapes in the images and
    relative scaling doesn't matter.

    score = average(patch * target)

    When shapes match well, the score approaches 1.0, when they mismatch they
    approach 0, and if they match exactly opposite, approach -1.0.
    '''

    assert p.shape == t.shape, f"shape mismatch: {p.shape} != {t.shape}"

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        warnings.filterwarnings('ignore', r'invalid value encountered in less')

        valid = ~np.isnan(p) * ~np.isnan(t)
        valid_count = np.sum(valid.astype(np.int32))

        if valid_count == 0:
            return 0.0

        p = zscore(p, valid)
        t = zscore(t, valid)

        s = np.nanmean(p * t)

        if np.isnan(s) or np.isinf(s):
            return 0.0

        # deweight scores as the number of valid pixels drops off

        weight = float(valid_count) / p.size

        #if weight < 0.50:
        #    weight *= 2.0

        s *= weight

        if s > 1.0001 or s < -1.0001:

            print("Score {} out of bounds".format(s))
            print("valid_count = %s" % (repr(valid_count)))
            print("weight = %s" % (repr(weight)))
            print("s = %s" % (repr(s)))

            plt.figure()
            imshow(p)
            plt.title('p (%s, %s)' % (np.nanmin(p), np.nanmax(p)))

            plt.figure()
            imshow(t)
            plt.title('t (%s, %s)' % (np.nanmin(t), np.nanmax(t)))

            tmp = p * t

            plt.figure()
            imshow(tmp)
            plt.title('p*t (%s, %s)' % (np.nanmin(tmp), np.nanmax(tmp)))

            plt.show()

            raise RuntimeError(
                "score %s is outside expected range (-1 : 1)" % s
            )

#        s = np.sign(s) * (s**2)

    return float(s)


def apply_alignment(image_array, delta_m, delta_n, angle, rotation_pos):
    """
    Applies translation and rotation to an image array.

    Args:
        image_array (numpy.ndarray): The input image array.
        delta_m (int): Translation along the vertical axis (rows).
        delta_n (int): Translation along the horizontal axis (columns).
        angle (float): Rotation angle in degrees (counter-clockwise).
        rotation_pos (tuple of int): Rotatate about this position.
    Returns:
        numpy.ndarray: The translated and rotated image array.
    """

    # 1. Translation.
    M, N = image_array.shape[:2]  # Handle grayscale or color images.
    translation_matrix = np.float32([[1, 0, delta_n], [0, 1, delta_m]])
    translated_image = cv2.warpAffine(image_array, translation_matrix, (M, N))

    # 2. Rotation.
    rotation_matrix = cv2.getRotationMatrix2D(rotation_pos, angle, 1)  # Rotate around the center.
    rotated_image = cv2.warpAffine(translated_image, rotation_matrix, (M, N))

    return rotated_image