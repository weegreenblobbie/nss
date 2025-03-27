import numpy as np
import numpy.random
import secrets

from matplotlib import pyplot as plt

from nss import utils

numpy.random.seed(19881108)

def test_generate_test_image():
    array = utils.generate_test_image(radius_moon=400).astype(np.float64)

    threshold = utils.auto_threshold_otsu(array)

    # Threshold.
    plt.figure()
    plt.hist(array.flatten(), bins=64)
    ax = plt.gca()
    ax.axvline(threshold, color="red")
    plt.grid(True)

    array[array < threshold] = 0

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
    mthresh = utils.auto_threshold_otsu(mvec ** 0.5)
    mvec = np.array(mvec)
    mvec[mvec < mthresh] = 0
    mvec = mvec > 0.0

    nthresh = utils.auto_threshold_otsu(nvec ** 0.5)
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
    utils.imshow(array)
    plt.plot(n, m, "r+", linewidth=10)

    radius = diameter / 2.0
    m0 = m - radius
    n0 = n - radius

    rect = plt.Rectangle((n0, m0), diameter, diameter, ls='-', color="red", fill=False)
    plt.gca().add_patch(rect)


    plt.show()


