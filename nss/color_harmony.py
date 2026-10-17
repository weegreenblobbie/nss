"""
Colour harmony targets for sampled image colours: the harmony modes (wedges of target hue), the best-fit rotation and the wedges drawn in mixer-input hue.

Hues are degrees on any hue axis with a mapping f in the table format of nss.color_mixer.hue_remap (the Color Harmony panel uses the Color Grading
wheel's angle).  A wedge is (centre, width) in OUTPUT hue; it is drawn at the input hues f maps into it (`wedge_arcs`), so a sample at input hue h
is in harmony when its output hue f(h) lies in the wedge.
"""
import numpy as np
from numpy.typing import ArrayLike

from nss.color_mixer.hue_remap import hue_remap_forward, hue_remap_inverse

# mode -> wedge centre offsets (degrees, relative to the rotation R) and wedge width; tuned by eye
HARMONY_MODES = {
    "None": {"offsets": (), "width": 0.0},
    "Monochromatic": {"offsets": (0.0,), "width": 15.0},
    "Analogous": {"offsets": (-30.0, 0.0, 30.0), "width": 30.0},
    "Complementary": {"offsets": (0.0, 180.0), "width": 30.0},
    "Split-Complementary": {"offsets": (0.0, 150.0, 210.0), "width": 30.0},
    "Triadic": {"offsets": (0.0, 120.0, 240.0), "width": 30.0},
    "Double Complementary": {"offsets": (0.0, 60.0, 180.0, 240.0), "width": 30.0},
}
ROTATION_STEP = 0.5         # degrees between the rotations tried by best_rotation
SAMPLE_SIZE = 5             # side in pixels of the neighbourhood whose per-channel median is a sample's colour


def median_rgb(img: np.ndarray, x: int, y: int, size: int = SAMPLE_SIZE) -> np.ndarray:
    """Per-channel median (3,) of the size x size pixels of img (h, w, 3) centred on column x, row y (cut at the image border)."""
    half = size // 2
    patch = img[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1]
    return np.median(patch.reshape(-1, 3), axis=0)


def angdiff(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Signed difference a - b in degrees, wrapped to [-180, 180)."""
    return (np.asarray(a, np.float64) - np.asarray(b, np.float64) + 180.0) % 360.0 - 180.0


def nearest_wedge(hues: ArrayLike, rotation: float, offsets: tuple) -> np.ndarray:
    """Index into offsets of the wedge centre (rotation + offset) nearest to each hue."""
    centres = rotation + np.asarray(offsets, np.float64)
    return np.argmin(np.abs(angdiff(np.asarray(hues, np.float64)[:, None], centres[None, :])), axis=1)


def best_rotation(hues: ArrayLike, offsets: tuple) -> float:
    """Rotation R in 0..360 (ROTATION_STEP grid, lowest R on ties) minimising sum_i min_k |angdiff(hue_i, R + offset_k)|; 0 without hues or offsets."""
    hues = np.asarray(hues, np.float64)
    if hues.size == 0 or len(offsets) == 0:
        return 0.0
    rotations = np.arange(0.0, 360.0, ROTATION_STEP)
    centres = rotations[:, None] + np.asarray(offsets, np.float64)[None, :]                   # (R, k)
    cost = np.abs(angdiff(hues[None, :, None], centres[:, None, :])).min(axis=2).sum(axis=1)
    return float(rotations[np.argmin(cost)])


def in_wedge(hues_out: ArrayLike, centre: float, width: float) -> np.ndarray:
    """True where an output hue lies inside the wedge (centre, width)."""
    return np.abs(angdiff(hues_out, centre)) <= 0.5 * width


def wedge_arcs(lut: np.ndarray, centre: float, width: float) -> list[tuple[float, float]]:
    """The input-hue arcs (start, end), start in 0..360 and end > start, that f (the hue_remap table lut) maps into the output wedge (centre, width).
    One arc when f is monotonic there, several where f folds over."""
    edges = np.sort(np.concatenate([hue_remap_inverse(lut, centre - 0.5 * width), hue_remap_inverse(lut, centre + 0.5 * width)]))
    ends = np.append(edges[1:], edges[0] + 360.0)
    mid = 0.5 * (edges + ends)
    keep = in_wedge(hue_remap_forward(lut, mid), centre, width)
    return [(float(s), float(e)) for s, e in zip(edges[keep], ends[keep])]
