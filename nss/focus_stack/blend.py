"""Merging aligned frames into the target: coverage, gain matching, the agreement check and the weighted composite."""
import warnings

import cv2
import numpy as np

from nss.focus_stack.flow import warp_with_flow
from nss.focus_stack.register import luma

GAIN_FLOOR = 1e-6  # keeps the gain ratio finite where the source is black
VARIANCE_FLOOR = 1e-12  # product of the two local variances below which a neighbourhood counts as featureless


def source_coverage(source_shape: tuple[int, int], matrix: np.ndarray, flow: np.ndarray, margin: int) -> np.ndarray:
    """A bool (rows, cols) map of target pixels fully inside the warped source, `margin` pixels from its edge."""
    covered = warp_with_flow(np.ones(source_shape, dtype=np.float32), matrix, flow) > 0.999
    kernel = np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8)
    return cv2.erode(covered.astype(np.uint8), kernel).astype(bool)


def gain_map(target: np.ndarray, source: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """A smooth per-channel gain: the ratio of the Gaussian-smoothed target to source over `valid` pixels."""
    weight = valid.astype(np.float32)[..., None]
    target_level = cv2.GaussianBlur(target * weight, (0, 0), sigma)
    source_level = cv2.GaussianBlur(source * weight, (0, 0), sigma)
    return target_level / (source_level + GAIN_FLOOR)


def match_gain(target: np.ndarray, source: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """`source` times its `gain_map`, so it has the target's brightness."""
    return source * gain_map(target, source, valid, sigma)


def agreement_map(target: np.ndarray, source: np.ndarray, lowpass_sigma: float, window_sigma: float) -> np.ndarray:
    """A float32 map in -1..1: the local normalised correlation of the low-passed luma of `target` and `source`.

    Near 1 where both show the same structure, whatever their gain or sharpness; near 0 where one frame has
    structure the other lacks. `lowpass_sigma` removes the fine detail that differs with focus, and
    `window_sigma` sets the neighbourhood the correlation is taken over. Both are in pixels.
    """
    a = cv2.GaussianBlur(luma(target).astype(np.float32), (0, 0), lowpass_sigma)
    b = cv2.GaussianBlur(luma(source).astype(np.float32), (0, 0), lowpass_sigma)
    a -= a.mean()
    b -= b.mean()
    mean_a = cv2.GaussianBlur(a, (0, 0), window_sigma)
    mean_b = cv2.GaussianBlur(b, (0, 0), window_sigma)
    cov = cv2.GaussianBlur(a * b, (0, 0), window_sigma) - mean_a * mean_b
    var_a = cv2.GaussianBlur(a * a, (0, 0), window_sigma) - mean_a * mean_a
    var_b = cv2.GaussianBlur(b * b, (0, 0), window_sigma) - mean_b * mean_b
    return np.clip(cov / np.sqrt(np.maximum(var_a, 0.0) * np.maximum(var_b, 0.0) + VARIANCE_FLOOR), -1.0, 1.0)


def masked_median(layers: list[np.ndarray], covered: list[np.ndarray]) -> np.ndarray:
    """The per-pixel, per-channel median of `layers` over the layers that cover each pixel; where none does, the first layer."""
    stack = np.stack(layers)
    stack[~np.stack(covered)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
    return np.where(np.isnan(median), layers[0], median).astype(np.float32)


def composite(target: np.ndarray, sources: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """The weighted sum of `target` (weights[0]) and each of `sources` (weights[1:]); the weights should add up to 1 at every pixel."""
    out = target * weights[0][..., None]
    for source, weight in zip(sources, weights[1:]):
        out += source * weight[..., None]
    return out
