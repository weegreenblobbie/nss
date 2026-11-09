"""Dense residual alignment: the per-pixel shift left over after a homography, and a single-resample warp using it."""
import cv2
import numpy as np

from nss.focus_stack.register import luma
from nss.image_utils import downsample_image

NORM_SIGMA = 10.0  # Gaussian sigma in downsampled pixels for the local mean and contrast
NORM_FLOOR = 1e-3  # added to the local variance so flat areas are not amplified into noise
NORM_RANGE = 3.0  # local standard deviations mapped to each side of mid-grey


def normalize_local(gray: np.ndarray) -> np.ndarray:
    """An 8-bit copy of a float luma image with local mean removed and local contrast equalised."""
    mean = cv2.GaussianBlur(gray, (0, 0), NORM_SIGMA)
    var = cv2.GaussianBlur(gray * gray, (0, 0), NORM_SIGMA) - mean * mean
    unit = (gray - mean) / np.sqrt(np.maximum(var, 0.0) + NORM_FLOOR)
    return (np.clip(unit / NORM_RANGE, -1.0, 1.0) * 127.5 + 127.5).astype(np.uint8)


def estimate_flow(target: np.ndarray, source: np.ndarray, max_dim: int = 1600) -> np.ndarray:
    """A (rows, cols, 2) float32 (dx, dy) in full-size pixels with target(x) ~ source(x + flow(x)).

    `source` must already be on the target's pixel grid.
    """
    target_gray = normalize_local(downsample_image(luma(target), max_dim))
    source_gray = normalize_local(downsample_image(luma(source), max_dim))
    small = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(target_gray, source_gray, None)
    rows, cols = target.shape[:2]
    flow = cv2.resize(small, (cols, rows), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= cols / small.shape[1]
    flow[..., 1] *= rows / small.shape[0]
    return flow


def warp_with_flow(source: np.ndarray, matrix: np.ndarray, flow: np.ndarray, *, row_offset: int = 0) -> np.ndarray:
    """`source` on the grid of `flow`: each target pixel x samples source at matrix^-1 (x + flow(x)), in one Lanczos resample.

    `flow` may be a strip of the target; `row_offset` is the target row of its first row.
    """
    rows, cols = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(cols, dtype=np.float64), np.arange(row_offset, row_offset + rows, dtype=np.float64))
    x = grid_x + flow[..., 0]
    y = grid_y + flow[..., 1]
    inv = np.linalg.inv(matrix)
    denom = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2]
    map_x = ((inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]) / denom).astype(np.float32)
    map_y = ((inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]) / denom).astype(np.float32)
    return cv2.remap(source, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
