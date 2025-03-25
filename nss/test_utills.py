import numpy as np
import numpy.random

from matplotlib import pyplot as plt

from nss import utils

def test_generate_test_image():
    array = utils.generate_test_image(radius_moon=400)
    plt.figure()
    utils.imshow(array)
    plt.show()
