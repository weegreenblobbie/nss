"""
Intentional camera movement (ICM) image math: the motion path fitted to a drawn stroke, the camera motion per frame from the path and the
curves, and the average of the image moved by it.  The path is kept in fractions of the image's width and height, so the same motion renders a
screen-size preview and the full-size picture alike.
"""
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from scipy.interpolate import splev, splprep

PATH_SMOOTHING = 5000.0      # splprep's s for a stroke in screen pixels: the sum of squared distances the smoothed path may stray from it
MAX_ANGLE = 45.0             # degrees of rotation at the rotation curve's ends (0 and 1); 0.5 is none
MIN_SCALE = 0.5              # scale at the zoom curve's bottom; 0.5 is 1.0 and the top is MIN_SCALE + 1


@dataclass(frozen=True)
class Motion:
    """The camera motion at each frame: path (n, 2) in fractions of the image width / height, angle (n,) in degrees, scale (n,)."""
    path: np.ndarray
    angle: np.ndarray
    scale: np.ndarray


def shutter_seconds(text: str) -> float:
    """The exposure time of a shutter label: '1/60' is 1/60, '2.5s' is 2.5."""
    text = text.replace("s", "")
    if "/" in text:
        num, den = text.split("/")
        return float(num) / float(den)
    return float(text)


def frame_count(shutter: float, samples: int, multi_exposure: bool) -> int:
    """Frames averaged: samples for a multi-exposure, else shutter seconds times samples per second (at least 2)."""
    return samples if multi_exposure else max(2, int(shutter * samples))


def fit_path(stroke: np.ndarray, image_rect: tuple[float, float, float, float]) -> tuple | None:
    """
    The smoothed spline (splprep's tck) of a stroke (m, 2) drawn in screen pixels over the image shown at image_rect (x, y, width, height),
    in fractions of the image; None when the stroke has fewer than 4 distinct points.
    """
    moved = np.any(np.diff(stroke, axis=0) != 0, axis=1)
    points = np.concatenate([stroke[:1], stroke[1:][moved]]).astype(float)
    if len(points) < 4:
        return None
    (knots, (cx, cy), degree), _ = splprep(points.T, s=PATH_SMOOTHING)
    x, y, width, height = image_rect
    # a spline is linear in its coefficients: mapping them maps the curve exactly
    return knots, [(cx - x) / width, (cy - y) / height], degree


def sample_motion(tck: tuple, speed: np.ndarray, rotation: np.ndarray, zoom: np.ndarray) -> Motion:
    """
    The motion along the path tck for the curves' values (0..1) at each frame: the running sum of speed sets how far along the path a frame is,
    rotation 0..1 turns -MAX_ANGLE..MAX_ANGLE degrees and zoom 0..1 scales MIN_SCALE..MIN_SCALE + 1.
    """
    progress = np.cumsum(speed)
    progress -= progress[0]
    progress = progress / progress[-1] if progress[-1] > 0 else np.linspace(0.0, 1.0, len(speed))
    x, y = splev(progress, tck)
    return Motion(np.column_stack([x, y]), (np.asarray(rotation) - 0.5) * 2.0 * MAX_ANGLE, MIN_SCALE + np.asarray(zoom))


def render(image: np.ndarray, motion: Motion, progress: Callable[[int, int], None] = lambda done, total: None,
           cancelled: Callable[[], bool] = lambda: False) -> np.ndarray | None:
    """
    The average (float32) of the image turned and scaled about the path's mean point and shifted along the path from its start, one copy per
    frame, edges reflected.  progress(done, total) is called after each frame; None once cancelled() is true (checked before each frame).
    """
    height, width = image.shape[:2]
    path = motion.path * (width, height)
    center = tuple(path.mean(axis=0))
    total = len(path)
    accumulator = np.zeros(image.shape, dtype=np.float32)
    for i in range(total):
        if cancelled():
            return None
        matrix = cv2.getRotationMatrix2D(center, float(motion.angle[i]), float(motion.scale[i]))
        matrix[:, 2] += path[i] - path[0]
        accumulator += cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
        progress(i + 1, total)
    return accumulator / total
