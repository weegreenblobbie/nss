import numpy as np
import numpy.random
import secrets

from matplotlib import pyplot as plt
import tifffile.tifffile
from tifffile.tifffile import imread

from nss import utils
from nss.align import to_gray

#numpy.random.seed(19881108)

def test_generate_test_image():
    #backup = utils.generate_test_image(radius_moon=300).astype(np.float64)
    #backup = to_gray(imread("lunar-eclipse/20250313_232149.99.tif")).astype(np.float32)
    #backup = to_gray(imread("lunar-eclipse/20250313_232746.99.tif")).astype(np.float32)
    #backup = to_gray(imread("lunar-eclipse/20250313_232842.99.tif")).astype(np.float32)
    #backup = to_gray(imread("lunar-eclipse/20250313_223345.99.tif")).astype(np.float32) # "cresent" 50%
    #backup = to_gray(imread("lunar-eclipse/20250313_230824.99.tif")).astype(np.float32) # "cresent" 25%
    backup = to_gray(imread("lunar-eclipse/20250313_231429.00.tif")).astype(np.float32) # "cresent" 10%
    
    utils.detect_moon(backup)

    xxxxxxxx

    print(f"shape: {backup.shape}")

    array = np.array(backup)
    #array -= np.min(array.flatten())
    #array[array < 0] = 0.0
    #array **= 0.25

    #threshold = utils.auto_threshold_first_valley(array, bins=64)
    threshold = utils.auto_threshold_otsu(array, bins=64)
    #threshold = 6.92
    print(f"threshold: {threshold}")
    # Threshold.
    plt.figure()
    plt.hist(array.flatten(), bins=64)
    ax = plt.gca()
    ax.axvline(threshold, color="red")
    plt.grid(True)

    #array[array < threshold] = 0
    sobel = array

    #sobel = utils.sobel_edge_detection(array)
    sobel = utils.sobel_edge_detection_2(array)
    sobel /= np.max(sobel.flatten())
    sobel = (256 * sobel).astype(np.uint8)

    #center, radius = utils.contours(sobel)
    #circles = [[center[0], center[1], radius]]

    plt.figure()
    utils.imshow(sobel)
    plt.title("Sobel edges")

    circles = utils.detect_circle(sobel)
    #circles = []
    
    for x, y, r in circles:
        print(f"{y:4d},{x:4d} radius: {r}")
        rect = plt.Circle((y, x), r, ls='-', color="red", fill=False)
        plt.gca().add_patch(rect)

    plt.show()

    xxxxxxx


    M, N = array.shape

    # Compute centroid.
    m_axis = np.arange(0, M)
    n_axis = np.arange(0, N)

    mvec = np.sum(array, axis=1).astype(np.float64)
    nvec = np.sum(array, axis=0).astype(np.float64)

    print(f"array.shape: {array.shape}")
    print(f"mvec.shape: {mvec.shape}")
    print(f"nvec.shape: {nvec.shape}")

    print(f"m_axis.shape: {m_axis.shape}")
    print(f"n_axis.shape: {n_axis.shape}")

    m = np.sum(mvec * m_axis) / np.sum(mvec)
    n = np.sum(nvec * n_axis) / np.sum(nvec)

    print(f"m: {m}")
    print(f"n: {n}")
    
    plt.figure()
    plt.plot(mvec, "r-", label="mvec")
    plt.plot(nvec, "b-", label="bvec")
    plt.legend()
    ax = plt.gca()
    ax.axvline(m, color="red")
    ax.axvline(n, color="blue")
    plt.grid(True)

    # Compute moon diameter in vertical and horizontal axes.
    mthresh = 0.05 * np.max(mvec)
    mvec = np.array(mvec)
    mvec[mvec < mthresh] = 0
    mvec = mvec > 0.0

    nthresh = 0.05 * np.max(nvec)
    nvec = np.array(nvec)
    nvec[nvec < nthresh] = 0
    nvec = nvec > 0.0
    
    plt.figure()
    plt.plot(mvec, "r-", label="mvec")
    plt.plot(nvec, "b-", label="nvec")
    plt.legend()

    m_diameter = np.sum(mvec)
    n_diameter = np.sum(nvec)
    diameter = max(m_diameter, n_diameter)
    print(f"diameter: {diameter}")

    plt.figure()
    utils.imshow(backup)
    plt.plot(n, m, "r+", linewidth=10)

    radius = diameter / 2.0
    m0 = m - radius
    n0 = n - radius

    rect = plt.Rectangle((n0, m0), diameter, diameter, ls='-', color="red", fill=False)
    plt.gca().add_patch(rect)


    plt.show()


