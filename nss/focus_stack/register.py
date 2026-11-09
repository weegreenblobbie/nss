"""Coarse registration: a homography that maps a source frame's pixels onto a target picture's pixels."""
from dataclasses import dataclass

import cv2
import numpy as np

from nss.image_utils import downsample_image

RATIO = 0.75  # Lowe's ratio test threshold
RANSAC_PIXELS = 3.0  # reprojection tolerance in downsampled pixels
MIN_INLIERS = 30


class RegistrationError(Exception):
    """Raised when two pictures cannot be matched well enough to trust a homography."""


@dataclass
class Homography:
    """A full-resolution 3x3 source-to-target matrix and the number of matches that support it."""

    matrix: np.ndarray
    inliers: int


def luma(image: np.ndarray) -> np.ndarray:
    """The mean of the colour channels of a float image, or the image itself when it is already 2D."""
    return image if image.ndim == 2 else image.mean(axis=-1, dtype=np.float32)


def small_luma(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """An 8-bit luma copy no larger than `max_dim`, and the 3x3 matrix taking full-size pixels to its pixels."""
    gray = downsample_image(luma(image), max_dim)
    scale_x = gray.shape[1] / image.shape[1]
    scale_y = gray.shape[0] / image.shape[0]
    to_small = np.array(
        [[scale_x, 0.0, (scale_x - 1.0) / 2.0],
         [0.0, scale_y, (scale_y - 1.0) / 2.0],
         [0.0, 0.0, 1.0]]
    )
    return (np.clip(gray, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), to_small


def estimate_homography(target: np.ndarray, source: np.ndarray, max_dim: int = 1600) -> Homography:
    """SIFT matches on downsampled luma, filtered by the ratio test and RANSAC, as a full-resolution matrix."""
    target_gray, target_scale = small_luma(target, max_dim)
    source_gray, source_scale = small_luma(source, max_dim)

    sift = cv2.SIFT_create()
    target_points, target_desc = sift.detectAndCompute(target_gray, None)
    source_points, source_desc = sift.detectAndCompute(source_gray, None)
    if target_desc is None or source_desc is None or len(target_points) < 4 or len(source_points) < 4:
        raise RegistrationError("too few features to register")

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(source_desc, target_desc, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance]
    if len(good) < MIN_INLIERS:
        raise RegistrationError(f"only {len(good)} matches passed the ratio test, need {MIN_INLIERS}")

    source_xy = np.float32([source_points[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    target_xy = np.float32([target_points[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    small_matrix, inlier_mask = cv2.findHomography(source_xy, target_xy, cv2.RANSAC, RANSAC_PIXELS)
    inliers = 0 if inlier_mask is None else int(inlier_mask.sum())
    if small_matrix is None or inliers < MIN_INLIERS:
        raise RegistrationError(f"only {inliers} RANSAC inliers, need {MIN_INLIERS}")

    matrix = np.linalg.inv(target_scale) @ small_matrix @ source_scale
    return Homography(matrix / matrix[2, 2], inliers)


def warp_to_target(source: np.ndarray, matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """`source` resampled with Lanczos onto a (rows, cols) grid, zero where the source has no pixels."""
    rows, cols = shape
    return cv2.warpPerspective(source, matrix, (cols, rows), flags=cv2.INTER_LANCZOS4)
