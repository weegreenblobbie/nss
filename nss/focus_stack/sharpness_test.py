"""Tests for nss.focus_stack.sharpness."""
import cv2
import numpy as np

from nss.focus_stack.register_test import make_texture
from nss.focus_stack.sharpness import sharpness_map

SHAPE = (480, 640)  # rows, cols: non-square


def test_sharpness_map_is_2d_float32_and_non_negative() -> None:
    """A non-square colour image gives a (rows, cols) float32 map with no negative values."""
    img = make_texture((13, 12), seed=1)
    out = sharpness_map(img)
    assert out.shape == (13, 12)
    assert out.dtype == np.float32
    assert out.min() >= 0.0


def test_sharpness_map_of_flat_image_is_zero_and_of_texture_is_not() -> None:
    """A constant image has no detail, a textured one does."""
    flat = np.full((13, 12, 3), 0.4, dtype=np.float32)
    assert np.allclose(sharpness_map(flat), 0.0)
    assert sharpness_map(make_texture(SHAPE, seed=2)).mean() > 0.0


def test_sharpness_map_scales_with_contrast() -> None:
    """Doubling the contrast about mid-grey doubles the sharpness."""
    img = make_texture(SHAPE, seed=3)
    base = sharpness_map(img)
    doubled = sharpness_map(0.5 + 2.0 * (img - 0.5))
    assert np.allclose(doubled, 2.0 * base, rtol=1e-3, atol=1e-6)


def test_sharpness_map_ranks_sharp_half_above_blurred_half() -> None:
    """Left half sharp, right half blurred: well away from the seam the left scores far higher."""
    sharp = make_texture(SHAPE, seed=4)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3.0)
    img = sharp.copy()
    img[:, SHAPE[1] // 2:] = blurred[:, SHAPE[1] // 2:]
    out = sharpness_map(img)
    left = out[40:-40, 40:SHAPE[1] // 2 - 40].mean()
    right = out[40:-40, SHAPE[1] // 2 + 40:-40].mean()
    assert left > 3.0 * right


def test_sharpness_map_scores_real_edges_above_sensor_noise() -> None:
    """A textured picture beats a flat one carrying noise of 0.02 standard deviation."""
    rng = np.random.default_rng(5)
    noisy_flat = (0.5 + 0.02 * rng.standard_normal((*SHAPE, 3))).astype(np.float32)
    textured = make_texture(SHAPE, seed=5)
    assert sharpness_map(textured).mean() > 3.0 * sharpness_map(noisy_flat).mean()
