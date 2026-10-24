"""Tests of nss.icm_render: shutter labels, frame counts, the fitted path, the sampled motion and the render."""
import numpy as np
from scipy.interpolate import splev
from nss.icm_render import Motion, fit_path, frame_count, render, sample_motion, shutter_seconds


def _image() -> np.ndarray:
    """A 13 x 21 RGB float32 picture with distinct values per pixel and channel."""
    return np.linspace(0.05, 0.95, 13 * 21 * 3, dtype=np.float32).reshape(13, 21, 3)


def test_shutter_labels_and_frame_counts() -> None:
    """Fractions and seconds are read; a multi-exposure counts samples, otherwise seconds times samples, at least 2."""
    assert shutter_seconds("1/60") == 1 / 60 and shutter_seconds("2.5s") == 2.5 and shutter_seconds("1s") == 1.0
    assert frame_count(1.6, 60, False) == 96 and frame_count(1 / 100, 60, False) == 2 and frame_count(1.6, 61, True) == 61


def test_fit_path_maps_the_stroke_into_image_fractions() -> None:
    """A stroke across the picture shown at (10, 20) size 300 x 200 starts and ends at its fractions of the picture."""
    stroke = np.array([[40.0 + 30 * i, 70.0 + 10 * i] for i in range(9)])      # (40, 70) .. (280, 150)
    tck = fit_path(stroke, (10.0, 20.0, 300.0, 200.0))
    x, y = splev([0.0, 1.0], tck)
    assert np.allclose([x[0], y[0], x[1], y[1]], [0.1, 0.25, 0.9, 0.65], atol=1e-6)


def test_fit_path_needs_four_distinct_points() -> None:
    """Repeated points do not count; fewer than four distinct ones give no path."""
    stroke = np.array([[5.0, 5.0], [5.0, 5.0], [9.0, 6.0], [9.0, 6.0], [14.0, 8.0]])
    assert fit_path(stroke, (0.0, 0.0, 30.0, 20.0)) is None


def test_sample_motion_maps_the_curves() -> None:
    """Even speed walks the path evenly; rotation 0.75 is 22.5 degrees and zoom 0.25 a scale of 0.75."""
    stroke = np.array([[10.0 * i, 5.0] for i in range(10)])
    tck = fit_path(stroke, (0.0, 0.0, 90.0, 10.0))
    motion = sample_motion(tck, np.full(5, 0.4), np.full(5, 0.75), np.full(5, 0.25))
    assert np.allclose(motion.path[:, 0], [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-6) and np.allclose(motion.path[:, 1], 0.5, atol=1e-6)
    assert np.allclose(motion.angle, 22.5) and np.allclose(motion.scale, 0.75)


def test_render_averages_the_shifted_copies() -> None:
    """Two frames one pixel apart average each pixel with its left neighbour; progress counts the frames."""
    image = _image()
    motion = Motion(np.array([[0.5, 0.5], [0.5 + 1 / 21, 0.5]]), np.zeros(2), np.ones(2))
    calls = []
    out = render(image, motion, lambda done, total: calls.append((done, total)))
    assert out.shape == (13, 21, 3) and out.dtype == np.float32
    assert np.allclose(out[:, 1:], (image[:, 1:] + image[:, :-1]) / 2, atol=1e-6)
    assert calls == [(1, 2), (2, 2)]


def test_render_stops_when_cancelled() -> None:
    """A cancelled render returns None."""
    motion = Motion(np.array([[0.2, 0.3], [0.6, 0.7], [0.4, 0.5]]), np.array([5.0, 10.0, 15.0]), np.array([1.0, 1.1, 1.2]))
    assert render(_image(), motion, cancelled=lambda: True) is None
