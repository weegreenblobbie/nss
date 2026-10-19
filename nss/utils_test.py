"""Tests of nss.utils.detect_moon on a synthetic 300 x 420 moon (disc radius 80) with stars that pass moon_edge_boxes' selection."""
import warnings

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.patches import Circle as CirclePatch, Rectangle

from nss import utils

matplotlib.use("Agg")       # test_plot_draws_kept_and_dropped_boxes draws off screen

CENTRE, RADIUS, SIZE = (140.3, 180.7), 80.0, 31        # (row, col) of the disc, its radius and the grid size


def _scene(stars: list[tuple[float, float, float]]) -> np.ndarray:
    """A (300, 420, 3) float32 image: an anti-aliased disc of 0.5 plus Gaussian stars (row, col, peak)."""
    rows, cols = np.mgrid[0:300, 0:420]
    img = 0.5 * np.clip(RADIUS - np.hypot(rows - CENTRE[0], cols - CENTRE[1]) + 0.5, 0.0, 1.0)
    for m, n, peak in stars:
        img += peak * np.exp(-((rows - m) ** 2 + (cols - n) ** 2) / 2.0)
    return np.repeat(img[..., None], 3, axis=2).astype(np.float32)


def _off_limb(image: np.ndarray) -> int:
    """Number of edge boxes whose centre is more than 15 px off the disc's edge."""
    return sum(abs(np.hypot(*(box.mean(axis=0) - CENTRE)) - RADIUS) > 15 for origin, box in utils.moon_edge_boxes(image, SIZE))


def _assert_on_the_disc(circle: utils.Circle) -> None:
    """The circle is the disc to 0.1 px."""
    assert np.allclose(circle.center, CENTRE, atol=0.1) and circle.radius == pytest.approx(RADIUS, abs=0.1)


def test_clean_disc() -> None:
    """Without stars every edge point is on the circle: confidence 1."""
    circle, confidence = utils.detect_moon(_scene([]), SIZE)
    _assert_on_the_disc(circle)
    assert confidence == 1.0


def test_far_star_in_four_overlapping_boxes_is_dropped() -> None:
    """A star at (262, 395) passes in 4 overlapping boxes and pulls a plain fit_circle over all edge points by over 10 px; detect_moon finds
    the disc and the star's points lower the confidence."""
    image = _scene([(262.0, 395.0, 1.5)])
    assert _off_limb(image) == 4
    plain = utils.fit_circle(np.unique(np.concatenate([box for origin, box in utils.moon_edge_boxes(image, SIZE)]), axis=0))
    assert np.hypot(plain[0] - CENTRE[0], plain[1] - CENTRE[1]) > 10.0
    circle, confidence = utils.detect_moon(image, SIZE)
    _assert_on_the_disc(circle)
    assert 0.98 < confidence < 1.0


def test_star_near_the_edge_is_dropped_by_its_residual() -> None:
    """A star 27 px outside the edge, too close for the distance filter, still goes: its boxes are the furthest from the fitted circle."""
    image = _scene([(140.0, 287.0, 1.2)])
    assert _off_limb(image) == 6
    _assert_on_the_disc(utils.detect_moon(image, SIZE)[0])


def test_far_and_near_stars_together() -> None:
    """Both stars at once: the far one by distance, the near one by residual."""
    image = _scene([(262.0, 395.0, 1.8), (140.0, 287.0, 1.2)])
    assert _off_limb(image) == 10
    _assert_on_the_disc(utils.detect_moon(image, SIZE)[0])


def test_too_few_consistent_boxes_raise() -> None:
    """A star in the corner bright enough that only its own box passes: ValueError, not a wrong circle; a black image has no boxes at all."""
    with pytest.raises(ValueError):
        utils.detect_moon(_scene([(4.0, 4.0, 20.0)]), SIZE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError):
            utils.detect_moon(np.zeros((40, 60, 3), np.float32), 21)


def test_plot_draws_kept_and_dropped_boxes() -> None:
    """plot=True gives the same circle and draws one rectangle per edge box: the 4 star boxes red, the others white, plus the circle."""
    image = _scene([(262.0, 395.0, 1.5)])
    circle, confidence = utils.detect_moon(image, SIZE, plot=True)
    assert (circle, confidence) == utils.detect_moon(image, SIZE)
    patches = plt.gca().patches
    rectangles = [p for p in patches if isinstance(p, Rectangle)]
    red = [p for p in rectangles if p.get_edgecolor()[:3] == (1.0, 0.0, 0.0)]
    assert len(rectangles) == len(utils.moon_edge_boxes(image, SIZE)) and len(red) == 4
    assert sum(isinstance(p, CirclePatch) for p in patches) == 1
    plt.close("all")
