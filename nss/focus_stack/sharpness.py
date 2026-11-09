"""Local sharpness: how much fine detail each pixel's neighbourhood holds."""
import cv2
import numpy as np

from nss.focus_stack.register import luma


def sharpness_map(image: np.ndarray, blur_sigma: float = 1.0, window_sigma: float = 4.0) -> np.ndarray:
    """A (rows, cols) float32 map of the Gaussian-weighted mean absolute Laplacian of the luma.

    `blur_sigma` smooths first so sensor noise scores low, `window_sigma` sets the neighbourhood size.
    Both are in pixels, so scale them with the resolution of `image`.
    """
    gray = luma(image).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    detail = np.abs(cv2.Laplacian(smooth, cv2.CV_32F))
    return cv2.GaussianBlur(detail, (0, 0), window_sigma)
