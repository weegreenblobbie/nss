"""Stacks extra frames onto a target picture: align each, find where each is sharper, and blend those regions in."""
from dataclasses import dataclass

import numpy as np

from nss.focus_stack.blend import agreement_map, composite, gain_map, masked_median, source_coverage
from nss.focus_stack.flow import estimate_flow, warp_with_flow
from nss.focus_stack.labels import choose_labels, label_weights
from nss.focus_stack.register import estimate_homography, warp_to_target
from nss.focus_stack.sharpness import sharpness_map
from nss.image_utils import downsample_image

# Pixel sizes below are for a working picture 3000 px on its long side and are scaled to the actual size.
REFERENCE_DIM = 3000
BLUR_SIGMA = 1.0  # sharpness: smoothing before the Laplacian
SHARPNESS_SIGMA = 7.5  # sharpness: neighbourhood the Laplacian is averaged over
GAIN_SIGMA = 75.0  # smoothing of the brightness match between the frames
COVERAGE_MARGIN = 18  # pixels kept clear of a frame's edge
TIDY_RADIUS = 9  # holes and strips narrower than this are tidied from the selection
FEATHER_SIGMA = 11.25  # softness of the region edges
AGREE_LOWPASS = 3.0  # agreement: smoothing that hides sharpness differences
AGREE_WINDOW = 8.0  # agreement: neighbourhood of the correlation
BIAS_STOPS = 0.2  # a frame must be sharper than the target by more than this many stops
MIN_AGREEMENT = 0.9  # correlation below this means the frames show different content


@dataclass
class AlignedFrame:
    """One extra frame aligned to the target, at the working resolution."""

    aligned: np.ndarray  # the frame warped onto the target and brightness matched
    coverage: np.ndarray  # bool: target pixels well inside the frame
    agreement: np.ndarray | None  # -1..1 correlation of the aligned frame with the target; None when aligned without flow
    sharpness: np.ndarray | None  # sharpness map of the aligned frame; None when aligned without flow
    inliers: int  # matches supporting the coarse alignment
    matrix: np.ndarray  # 3x3 taking the original frame's pixels to the original target's pixels
    flow: np.ndarray | None  # (dx, dy) residual shift on the working grid, in working pixels; None when aligned by the matrix alone
    gain: np.ndarray  # per-channel brightness gain of the warped frame on the working grid

    @property
    def eligible(self) -> np.ndarray:
        """Bool map of the pixels where this frame may be used: covered and, if it has an agreement map, in agreement with the target."""
        return self.coverage if self.agreement is None else self.coverage & (self.agreement >= MIN_AGREEMENT)

    def eligible_in(self, rows: slice, cols: slice, manual: np.ndarray) -> np.ndarray:
        """`eligible` for a box, where the pixels in `manual` (chosen by hand) skip the agreement test but never the coverage."""
        covered = self.coverage[rows, cols]
        return covered if self.agreement is None else covered & ((self.agreement[rows, cols] >= MIN_AGREEMENT) | manual)


@dataclass
class StackResult:
    """The stacking of all frames, at the working resolution."""

    target: np.ndarray  # the target downsampled to the working size
    target_sharpness: np.ndarray | None  # sharpness map of the target; None when stacked for a clean plate
    frames: list[AlignedFrame]
    labels: np.ndarray  # int16: 0 where the target is used, k where frame k is, frames + 1 where the median is
    weights: np.ndarray  # (layers, rows, cols) blending weights, index 0 the target, then the frames, then the median if any; adding up to 1
    result: np.ndarray  # the composite
    replaced_fraction: float  # share of the pixels covered by any frame that do not take the target
    feather_sigma: float  # softness of the region edges in working pixels, as used for `weights`
    median: np.ndarray | None = None  # the clean plate, the last layer: the per-pixel median of the target and the frames, or `clean2`'s plate; None unless stacked for one
    plate_is_median: bool = True  # False when the plate is not the per-pixel median of the warped frames (`clean2`), which `plate_mix` then describes
    plate_mix: np.ndarray | None = None  # float32 (rows, cols, layers + 1): how much of the target, each frame and the median make the plate at each pixel; None when `plate_is_median`


def resize_matrix(full_shape: tuple[int, int], small_shape: tuple[int, int]) -> np.ndarray:
    """The 3x3 taking pixel positions of a (rows, cols) picture to those of its area-downsampled copy."""
    scale_y = small_shape[0] / full_shape[0]
    scale_x = small_shape[1] / full_shape[1]
    return np.array(
        [[scale_x, 0.0, (scale_x - 1.0) / 2.0],
         [0.0, scale_y, (scale_y - 1.0) / 2.0],
         [0.0, 0.0, 1.0]]
    )


def align_frame(target: np.ndarray, full_target_shape: tuple[int, int], source: np.ndarray, work_dim: int) -> AlignedFrame:
    """Aligns `source` to `target`, a copy of the full-size target downsampled to the working size.

    Raises RegistrationError when the two pictures cannot be matched.
    """
    full_source_shape = source.shape[:2]
    source = downsample_image(source, work_dim)
    scale = max(target.shape[:2]) / REFERENCE_DIM

    homography = estimate_homography(target, source)
    flow = estimate_flow(target, warp_to_target(source, homography.matrix, target.shape[:2]), max_dim=REFERENCE_DIM)
    coverage = source_coverage(source.shape[:2], homography.matrix, flow, round(COVERAGE_MARGIN * scale))
    warped = warp_with_flow(source, homography.matrix, flow)
    gain = gain_map(target, warped, coverage, GAIN_SIGMA * scale)
    aligned = warped * gain
    return AlignedFrame(
        aligned=aligned,
        coverage=coverage,
        agreement=agreement_map(target, aligned, AGREE_LOWPASS * scale, AGREE_WINDOW * scale),
        sharpness=sharpness_map(aligned, BLUR_SIGMA * scale, SHARPNESS_SIGMA * scale),
        inliers=homography.inliers,
        matrix=np.linalg.inv(resize_matrix(full_target_shape, target.shape[:2]))
        @ homography.matrix
        @ resize_matrix(full_source_shape, source.shape[:2]),
        flow=flow,
        gain=gain,
    )


def stack_frames(target: np.ndarray, sources: list[np.ndarray], work_dim: int) -> StackResult:
    """Aligns every frame in `sources` to `target` and blends in the regions where a frame is sharper, at most `work_dim` px on the long side.

    Raises RegistrationError as soon as one frame cannot be matched.
    """
    full_target_shape = target.shape[:2]
    target = downsample_image(target, work_dim)
    scale = max(target.shape[:2]) / REFERENCE_DIM
    frames = [align_frame(target, full_target_shape, source, work_dim) for source in sources]

    target_sharpness = sharpness_map(target, BLUR_SIGMA * scale, SHARPNESS_SIGMA * scale)
    eligible = [frame.eligible for frame in frames]
    labels = choose_labels(target_sharpness, [frame.sharpness for frame in frames], eligible, BIAS_STOPS, round(TIDY_RADIUS * scale))
    feather_sigma = FEATHER_SIGMA * scale
    weights = label_weights(labels, eligible, feather_sigma)
    covered = np.any([frame.coverage for frame in frames], axis=0)
    return StackResult(
        target=target,
        target_sharpness=target_sharpness,
        frames=frames,
        labels=labels,
        weights=weights,
        result=composite(target, [frame.aligned for frame in frames], weights),
        replaced_fraction=float(1.0 - weights[0][covered].mean()),
        feather_sigma=feather_sigma,
    )


def resample_rows(small: np.ndarray, full_shape: tuple[int, int], first: int, last: int) -> np.ndarray:
    """Rows `first` to `last` of `small` bilinearly resized to `full_shape` (rows, cols), without resizing the rest.

    Gives what cv2.resize would give for those rows, with the border pixels repeated.
    """
    rows, cols = full_shape

    def taps(full_count: int, small_count: int, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """The lower source index and the weight of the next index, for full-size positions start..stop."""
        position = np.clip((np.arange(start, stop) + 0.5) * small_count / full_count - 0.5, 0.0, small_count - 1.0)
        lower = np.minimum(position.astype(int), small_count - 2)
        return lower, (position - lower).astype(np.float32)

    row_lower, row_weight = taps(rows, small.shape[0], first, last)
    col_lower, col_weight = taps(cols, small.shape[1], 0, cols)
    row_weight = row_weight.reshape((-1,) + (1,) * (small.ndim - 1))
    col_weight = col_weight.reshape((1, -1) + (1,) * (small.ndim - 2))
    across = small[row_lower] * (1.0 - row_weight) + small[row_lower + 1] * row_weight
    return across[:, col_lower] * (1.0 - col_weight) + across[:, col_lower + 1] * col_weight


def render_strip(
    target: np.ndarray,
    sources: list[np.ndarray],
    frames: list[AlignedFrame],
    weights: np.ndarray,
    first: int,
    last: int,
    plate_mix: np.ndarray | None = None,
) -> np.ndarray:
    """Rows `first` to `last` of the full-size composite of `target` and `sources`.

    `frames` and `weights` (index 0 the target, then one per frame, then the median if there is one) are at the working size,
    as `stack_frames`, `stack_clean` or an `EditState` give them. A layer with no weight in the strip is not read at all. The
    last layer, the plate, is computed from the strip's own warped frames: the per-pixel median, or with a working-size `plate_mix`
    (`StackResult.plate_mix` of `clean2`) the mix of the target, the frames and the median it describes, where a layer the mix
    does not use in the strip is not read either.
    """
    rows, cols = target.shape[:2]
    work_rows, work_cols = weights.shape[1:]
    full = (rows, cols)
    strip_weights = [resample_rows(weight, full, first, last) for weight in weights]
    strip = target[first:last] * strip_weights[0][..., None]
    need_plate = len(weights) == len(frames) + 2 and bool(strip_weights[-1].any())
    # share of the strip's output that each layer, then the median, contributes through the plate
    plate_share = [None] * (len(frames) + 2)
    if need_plate and plate_mix is None:
        plate_share[-1] = strip_weights[-1]
    elif need_plate:
        mix = resample_rows(plate_mix, full, first, last) * strip_weights[-1][..., None]
        plate_share = [mix[..., k] if mix[..., k].any() else None for k in range(len(frames) + 2)]
    need_median = plate_share[-1] is not None
    if plate_share[0] is not None:
        strip += target[first:last] * plate_share[0][..., None]
    median_layers = [target[first:last]]
    median_covered = [np.ones((last - first, cols), dtype=bool)]
    for k, (frame, source, weight) in enumerate(zip(frames, sources, strip_weights[1:]), start=1):
        used = bool(weight.any())
        if not (used or need_median or plate_share[k] is not None):
            continue
        if frame.flow is None:
            flow = np.zeros((last - first, cols, 2), dtype=np.float32)
        else:
            flow = resample_rows(frame.flow, full, first, last)
            flow[..., 0] *= cols / work_cols
            flow[..., 1] *= rows / work_rows
        aligned = warp_with_flow(source, frame.matrix, flow, row_offset=first)
        aligned *= resample_rows(frame.gain, full, first, last)
        if need_median:
            median_layers.append(aligned)
            median_covered.append(resample_rows(frame.coverage.astype(np.float32), full, first, last) > 0.5)
        if used:
            strip += aligned * weight[..., None]
        if plate_share[k] is not None:
            strip += aligned * plate_share[k][..., None]
    if need_median:
        strip += masked_median(median_layers, median_covered) * plate_share[-1][..., None]
    return strip


def render_full(
    target: np.ndarray,
    sources: list[np.ndarray],
    frames: list[AlignedFrame],
    weights: np.ndarray,
    tile_rows: int,
    plate_mix: np.ndarray | None = None,
) -> np.ndarray:
    """The composite at the full size of `target`, built `tile_rows` rows at a time.

    Each pixel is computed on its own, so the strip height changes the memory used and not the picture. `plate_mix` is as for `render_strip`.
    """
    rows = target.shape[0]
    out = np.empty_like(target)
    for first in range(0, rows, tile_rows):
        last = min(first + tile_rows, rows)
        out[first:last] = render_strip(target, sources, frames, weights, first, last, plate_mix)
    return out
