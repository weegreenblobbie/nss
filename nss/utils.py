import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
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

    # Grayscale.
    if array.ndim == 2:
        plt.imshow(array, cmap = 'gray', interpolation='nearest')

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


def detect_moon(image, grid_size=200):
    """
    Detect the center and radius of the moon located in the image.

    Args:
        image (numpy.ndarray): The input image array.

        grid_size (int): The size of the grid patches used to detect the moon's
                         edge.

    Returns:
        m, n, r (float): The detected moon's center and radius.
    """
    sobel = sobel_edge_detection(to_gray(image))

    sobel /= np.nanmax(sobel)
    #sobel = (256 * sobel).astype(np.uint8)
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

    candidates = [can for can in candidates if can["score"] > 0.5]

    plot_patches(image, [], show=False, title="")

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

    print(f"len(raw_points): {len(raw_points)}" )

    np.random.shuffle(raw_points)

    raw_points = raw_points[:100]

    import matplotlib.pyplot as plt

    ax = plt.gca()

    #for y, x in raw_points:
    #    plt.scatter(x, y, color="red")

    m, n, r = fit_circle(raw_points)
    
    rect = plt.Circle((n, m), r, ls='-', color="red", fill=False)
    ax.add_patch(rect)

    plt.show()
        






    