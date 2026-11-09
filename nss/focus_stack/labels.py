"""Which frame each pixel comes from: the label map, and the blending weights made from it."""
import cv2
import numpy as np


def choose_labels(
    target_sharp: np.ndarray,
    frame_sharps: list[np.ndarray],
    eligible: list[np.ndarray],
    bias_stops: float,
    tidy_radius: int,
) -> np.ndarray:
    """An int16 label per pixel: 0 for the target, k for frame k (counting from 1).

    A pixel takes the sharpest frame among those `eligible` there, but only if that frame is sharper than the
    target by more than `bias_stops` (powers of two). Each label is then tidied: holes of its own or of the
    target narrower than `2 * tidy_radius + 1` pixels are filled and strips narrower than that are dropped.
    Tidying never takes a pixel from another frame.
    """
    scores = np.stack([np.where(ok, sharp, -np.inf) for sharp, ok in zip(frame_sharps, eligible)])
    best = scores.argmax(axis=0)
    chosen = scores.max(axis=0) > target_sharp * 2.0 ** bias_stops
    labels = np.where(chosen, best + 1, 0).astype(np.int16)

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tidy_radius + 1, 2 * tidy_radius + 1))
    tidy = np.zeros_like(labels)
    for label in range(1, len(frame_sharps) + 1):
        closed = cv2.morphologyEx((labels == label).astype(np.uint8), cv2.MORPH_CLOSE, disk)
        claim = cv2.morphologyEx(closed, cv2.MORPH_OPEN, disk).astype(bool)
        tidy[claim & (tidy == 0) & ((labels == 0) | (labels == label))] = label
    return tidy


GROW_SIGMAS = 2.0  # how many feather sigmas a region grows into the target's pixels before it is blurred


def label_weights(labels: np.ndarray, eligible: list[np.ndarray], feather_sigma: float) -> np.ndarray:
    """A float32 (frames + 1, rows, cols) array of blending weights that add up to 1 at every pixel.

    Index 0 is the target and index k is frame k. A frame's weight is 0 wherever the frame is not `eligible`; the target takes
    what is left. A region is fully opaque inside: it is first grown by 2 `feather_sigma` pixels into the pixels that show
    the target, and only then blurred with a Gaussian of `feather_sigma` pixels, so the soft edge lies outside the region and
    even a patch smaller than the blur shows its frame at full strength. Growth never enters another frame's region. Where
    the growths of two frames overlap they are scaled down to share the pixel.
    """
    weights = np.zeros((len(eligible) + 1, *labels.shape), dtype=np.float32)
    reach = GROW_SIGMAS * feather_sigma
    free = labels == 0
    for k, ok in enumerate(eligible, start=1):
        own = labels == k
        if not own.any():
            continue
        distance = cv2.distanceTransform((~own).astype(np.uint8), cv2.DIST_L2, 3)
        grown = own | ((distance <= reach) & free)
        weights[k] = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), feather_sigma) * ok
    weights[1:] /= np.maximum(weights[1:].sum(axis=0), 1.0)
    weights[0] = np.clip(1.0 - weights[1:].sum(axis=0), 0.0, 1.0)
    return weights
