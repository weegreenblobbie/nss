"""Tests for nss.focus_stack.register."""
import cv2
import numpy as np
import pytest

from nss.focus_stack.register import (
    RegistrationError,
    estimate_homography,
    luma,
    warp_to_target,
)

SHAPE = (480, 640)  # rows, cols: non-square


def make_texture(shape: tuple[int, int], seed: int) -> np.ndarray:
    """A smooth random float32 RGB texture in 0..1 with plenty of corners for SIFT."""
    rng = np.random.default_rng(seed)
    noise = rng.random((*shape, 3)).astype(np.float32)
    fine = cv2.GaussianBlur(noise, (0, 0), 2.0)
    coarse = cv2.GaussianBlur(noise, (0, 0), 8.0)
    tex = fine - coarse
    tex -= tex.min()
    return (tex / tex.max()).astype(np.float32)


def make_homography() -> np.ndarray:
    """Scale 0.8, rotation 5 degrees, shift (30, -20) and a small perspective term."""
    angle = np.deg2rad(5.0)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array(
        [[0.8 * cos, -0.8 * sin, 30.0],
         [0.8 * sin, 0.8 * cos, -20.0],
         [2e-5, 1e-5, 1.0]]
    )


def corner_error(est: np.ndarray, truth: np.ndarray, shape: tuple[int, int]) -> float:
    """Largest distance in pixels between where two homographies send the image corners."""
    rows, cols = shape
    corners = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]], dtype=np.float64).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(corners, est)
    b = cv2.perspectiveTransform(corners, truth)
    return float(np.max(np.linalg.norm(a - b, axis=-1)))


def test_luma_is_2d_and_weights_channels() -> None:
    """luma collapses the channel axis and keeps non-square shape."""
    img = np.zeros((4, 2, 3), dtype=np.float32)
    img[..., 0] = 0.2
    img[..., 1] = 0.4
    img[..., 2] = 0.9
    out = luma(img)
    assert out.shape == (4, 2)
    assert np.allclose(out, 0.5)


def test_estimate_homography_recovers_known_transform() -> None:
    """The estimated matrix maps source corners to within half a pixel of the truth."""
    source = make_texture(SHAPE, seed=1)
    truth = make_homography()
    target = cv2.warpPerspective(source, truth, (SHAPE[1], SHAPE[0]), flags=cv2.INTER_LINEAR)
    result = estimate_homography(target, source, max_dim=2000)
    assert result.inliers >= 30
    assert corner_error(result.matrix, truth, SHAPE) < 0.5


def test_estimate_homography_at_reduced_size_scales_back_to_full_size() -> None:
    """Matching on downsampled copies still yields a full-resolution matrix."""
    source = make_texture((960, 1280), seed=2)
    truth = make_homography()
    truth[2, :2] = [1e-5, 5e-6]
    target = cv2.warpPerspective(source, truth, (1280, 960), flags=cv2.INTER_LINEAR)
    result = estimate_homography(target, source, max_dim=640)
    assert corner_error(result.matrix, truth, (960, 1280)) < 2.0


def test_estimate_homography_raises_for_unrelated_images() -> None:
    """Two unrelated textures cannot be registered."""
    a = make_texture(SHAPE, seed=3)
    b = make_texture(SHAPE, seed=4)
    with pytest.raises(RegistrationError):
        estimate_homography(a, b, max_dim=2000)


def test_warp_to_target_puts_source_pixels_at_matrix_positions() -> None:
    """A pure (+5 col, +3 row) shift moves a bright pixel by that amount in a target-shaped output."""
    source = np.zeros((13, 12, 3), dtype=np.float32)
    source[4, 2] = (0.25, 0.5, 0.75)
    matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
    out = warp_to_target(source, matrix, (9, 20))
    assert out.shape == (9, 20, 3)
    assert np.allclose(out[7, 7], (0.25, 0.5, 0.75))
    assert np.count_nonzero(out) == 3
