"""Clean-plate stacking: remove people, or anything that moved, from the target using the other frames and their per-pixel median.

The frames must be shot from the same spot. They are aligned by a homography alone: a dense flow is dragged around by whoever is
in one frame and not another, and bends the background into their outline. The median of the aligned frames shows the empty scene
wherever a majority of them do. Each foreground blob of the target then takes the one frame that is clean across it, with the
median (layer frames + 1) filling in where that frame has a person of its own.
"""
import cv2
import numpy as np
from scipy import ndimage

from nss.focus_stack.blend import composite, gain_map, masked_median
from nss.focus_stack.labels import label_weights
from nss.focus_stack.register import estimate_homography, warp_to_target
from nss.focus_stack.stack import (
    COVERAGE_MARGIN,
    FEATHER_SIGMA,
    GAIN_SIGMA,
    REFERENCE_DIM,
    AlignedFrame,
    StackResult,
    resize_matrix,
)
from nss.image_utils import downsample_image

# Sizes in pixels are for a working picture 3000 px on its long side and are scaled to the actual size; deviations are 0..1 levels.
DEVIATION_SIGMA = 2.0  # smoothing of the difference from the median
STRONG = 0.08  # a target deviation this big is certainly foreground
WEAK = 0.03  # a deviation this big joins a foreground blob if it touches a strong one
AGREE_BELOW = 0.03  # a frame agrees with the median where its deviation is below this
CLEAN_MEAN = 0.04  # a frame is clean over a blob if its mean deviation there is below this
BAD_DEVIATION = 0.05  # a chosen frame has a person of its own where its deviation is above this
COVER_FRACTION = 0.95  # a frame must cover this much of a blob to be used for it
CLOSE_RADIUS = 15  # gaps in a blob narrower than this are closed
GROW_RADIUS = 10  # how far a blob grows to take in its soft edge
BAD_GROW = 8  # how far a chosen frame's own person is grown before it is replaced
BAD_LIMIT = 0.3  # a frame with a person at more than this share of the target's foreground pixels shows the same people
TRUST_FRACTION = 0.7  # a blob is replaced only if the median is trusted over this share of the pixels where the target has a person
MIN_AREA = 1500  # blobs smaller than this many pixels are ignored
PLATE_DIFF = 0.06  # the pairwise plate and the median differ where they are further apart than this
HOLE_SIGMAS = 3  # a frame's own person is grown by this many feather sigmas before the plate replaces it, so the feathering cannot bring it back, in `clean2`
RING_RADIUS = 20  # how far around a region the calm pixels that say what is behind it are taken
PRIOR_REACH = 0.1  # the ring continues into a region only as far as this share of a Gaussian window is ring
GAIN_MIN_LEVEL = 0.05  # pixels darker than this in either picture are left out of the global brightness ratio
GAIN_STRIDE = 3  # the global brightness ratio looks at every this-many-th pixel in each direction


def disk(radius: int) -> np.ndarray:
    """An elliptical structuring element of the given radius."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def global_gain(target: np.ndarray, warped: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The brightness ratio target / warped for each channel: the median over the `valid` pixels that are not very dark in either picture.

    Unlike a locally smoothed ratio, it cannot be pulled around by a big object in either picture, as long as the object
    covers under half of the pixels. Every `GAIN_STRIDE`th pixel is used. Returns ones when no pixel qualifies.
    """
    step = (slice(None, None, GAIN_STRIDE), slice(None, None, GAIN_STRIDE))
    target, warped = target[step], warped[step]
    usable = valid[step] & (target.mean(axis=-1) > GAIN_MIN_LEVEL) & (warped.mean(axis=-1) > GAIN_MIN_LEVEL)
    if not usable.any():
        return np.ones(3, dtype=np.float32)
    return np.median(target[usable] / np.maximum(warped[usable], 1e-4), axis=0).astype(np.float32)


def align_frame_homography(target: np.ndarray, full_target_shape: tuple[int, int], source: np.ndarray, work_dim: int) -> AlignedFrame:
    """Aligns `source` to `target` (a copy of the full-size target at the working size) by a homography alone.

    Its brightness is matched with one robust gain per channel (`global_gain`); `correct_gains` refines that locally once the
    foreground is known. The result has no flow, agreement or sharpness map. Raises RegistrationError when the pictures cannot be matched.
    """
    full_source_shape = source.shape[:2]
    source = downsample_image(source, work_dim)
    scale = max(target.shape[:2]) / REFERENCE_DIM

    homography = estimate_homography(target, source)
    warped = warp_to_target(source, homography.matrix, target.shape[:2])
    inside = warp_to_target(np.ones(source.shape[:2], dtype=np.float32), homography.matrix, target.shape[:2]) > 0.999
    margin = round(COVERAGE_MARGIN * scale)
    coverage = cv2.erode(inside.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8)).astype(bool)
    gain = np.ones((*warped.shape[:2], 3), dtype=np.float32) * global_gain(target, warped, coverage)
    return AlignedFrame(
        aligned=warped * gain,
        coverage=coverage,
        agreement=None,
        sharpness=None,
        inliers=homography.inliers,
        matrix=np.linalg.inv(resize_matrix(full_target_shape, target.shape[:2]))
        @ homography.matrix
        @ resize_matrix(full_source_shape, source.shape[:2]),
        flow=None,
        gain=gain,
    )


def deviation_maps(layers: list[np.ndarray], covered: list[np.ndarray], median: np.ndarray, sigma: float) -> np.ndarray:
    """A (layers, rows, cols) float32 array: how far each layer is from `median`, the largest channel difference smoothed by `sigma`.

    The deviation is 0 where a layer does not cover, apart from the smoothing's reach.
    """
    maps = [
        cv2.GaussianBlur((np.abs(layer - median).max(axis=-1) * ok).astype(np.float32), (0, 0), sigma)
        for layer, ok in zip(layers, covered)
    ]
    return np.stack(maps)


def trusted_median(deviation: np.ndarray, covered: list[np.ndarray], agree_below: float) -> np.ndarray:
    """A bool map of the pixels where a strict majority of the layers that cover them agree with the median."""
    covering = np.stack(covered)
    agreeing = (covering & (deviation < agree_below)).sum(axis=0)
    return agreeing * 2 > covering.sum(axis=0)


def foreground_blobs(
    deviation: np.ndarray, strong: float, weak: float, close_radius: int, grow_radius: int, min_area: int
) -> np.ndarray:
    """An int32 map numbering the foreground blobs of a deviation map 1, 2, ... (0 elsewhere).

    A blob is a connected area above `weak` that holds a pixel above `strong`; gaps narrower than `close_radius` are closed,
    holes are filled, the blob grows by `grow_radius`, and blobs under `min_area` pixels are dropped.
    """
    count, components = cv2.connectedComponents((deviation > weak).astype(np.uint8))
    has_core = np.zeros(count, dtype=bool)
    has_core[np.unique(components[deviation > strong])] = True
    has_core[0] = False
    mask = has_core[components].astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(close_radius))
    mask = cv2.dilate(ndimage.binary_fill_holes(mask).astype(np.uint8), disk(grow_radius))
    count, blobs, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= min_area]
    renumber = np.zeros(count, dtype=np.int32)
    renumber[kept] = np.arange(1, len(kept) + 1)
    return renumber[blobs]


def auto_labels(
    deviation: np.ndarray,
    covered: list[np.ndarray],
    trusted: np.ndarray,
    blobs: np.ndarray,
    clean_mean: float,
    bad_deviation: float,
    bad_grow: int,
    bad_limit: float,
    trust_fraction: float,
) -> np.ndarray:
    """An int16 label per pixel: 0 the target, k frame k, `len(deviation)` the median (index 0 of `deviation` is the target).

    A blob is only touched where the median is trusted over at least `trust_fraction` of the pixels at which the target has a
    person (deviation over `bad_deviation`): if most frames disagree about what is behind the target's people, they are people
    who stay, and the blob is left as the target. A blob goes whole to the covering frame with the lowest mean deviation,
    provided that is under `clean_mean` and the frame has a person of its own at no more than `bad_limit` of the pixels where
    the target has one. Where the chosen frame has such a person (grown by `bad_grow`) the pixel takes the median if it is
    trusted and the target if not. A blob with no usable frame takes the median where it is trusted.
    """
    median_label = len(deviation)
    labels = np.zeros(blobs.shape, dtype=np.int16)
    grow = disk(bad_grow)
    for index, box in enumerate(ndimage.find_objects(blobs), start=1):
        if box is None:
            continue
        inside = blobs[box] == index
        crop, trusted_crop = labels[box], trusted[box]
        target_person = inside & (deviation[0][box] > bad_deviation)
        if target_person.any() and trusted_crop[target_person].mean() < trust_fraction:
            continue
        usable = []
        for frame in range(1, len(deviation)):
            own_person = inside & (deviation[frame][box] > bad_deviation)
            mean = float(deviation[frame][box][inside].mean())
            shared = (own_person & target_person).sum()
            if mean < clean_mean and covered[frame][box][inside].mean() >= COVER_FRACTION and shared <= bad_limit * target_person.sum():
                usable.append((mean, frame, own_person))
        if not usable:
            crop[inside & trusted_crop] = median_label
            continue
        _, frame, own_person = min(usable, key=lambda item: item[0])
        crop[inside] = frame
        bad = cv2.dilate(own_person.astype(np.uint8), grow).astype(bool) & inside
        crop[bad & trusted_crop] = median_label
        crop[bad & ~trusted_crop] = 0
    return labels


def suggest_layer(frames: list[AlignedFrame], median: np.ndarray, region: tuple[slice, slice], inside: np.ndarray, clean_mean: float) -> int:
    """The layer a new patch over `inside` (a bool crop of `region`) should start on: the covering frame closest to the median, or the median.

    A frame qualifies if it covers at least `COVER_FRACTION` of the patch and its mean deviation from the median there is under
    `clean_mean`; the closest qualifying frame is returned (frame k is layer k). With none, the result is the median layer.
    """
    best = None
    for layer, frame in enumerate(frames, start=1):
        if frame.coverage[region][inside].mean() < COVER_FRACTION:
            continue
        mean = float(np.abs(frame.aligned[region] - median[region]).max(axis=-1)[inside].mean())
        if mean < clean_mean and (best is None or mean < best[0]):
            best = (mean, layer)
    return best[1] if best else len(frames) + 1


def compare_to_median(target: np.ndarray, frames: list[AlignedFrame], scale: float) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """The layers (target first), their coverage maps, their median and each layer's deviation from it."""
    covered = [np.ones(target.shape[:2], dtype=bool)] + [frame.coverage for frame in frames]
    layers = [target] + [frame.aligned for frame in frames]
    median = masked_median(layers, covered)
    return layers, covered, median, deviation_maps(layers, covered, median, DEVIATION_SIGMA * scale)


def correct_gains(target: np.ndarray, frames: list[AlignedFrame], deviation: np.ndarray, scale: float) -> None:
    """Matches each frame's brightness to the target again, using only pixels where neither the target nor the frame is foreground.

    The first match is pulled around by whoever is in the target, so the gain is estimated afresh from the un-matched warp.
    Where fewer than 20% of the pixels within reach are reliable the gain fades to the frame's overall brightness ratio, so a
    large foreground blob cannot darken or brighten its middle.
    """
    sigma = GAIN_SIGMA * scale
    foreground = cv2.dilate((deviation[0] > WEAK).astype(np.uint8), disk(round(GROW_RADIUS * scale))).astype(bool)
    for frame, frame_deviation in zip(frames, deviation[1:]):
        reliable = frame.coverage & ~foreground & (frame_deviation < WEAK)
        warped = frame.aligned / np.maximum(frame.gain, 1e-3)
        gain = gain_map(target, warped, reliable, sigma)
        overall = np.ones(3, dtype=np.float32)
        if reliable.any():
            overall = target[reliable].mean(axis=0) / np.maximum(warped[reliable].mean(axis=0), 1e-6)
        reach = cv2.GaussianBlur(reliable.astype(np.float32), (0, 0), sigma)
        gain = overall + (gain - overall) * np.clip(reach / 0.2, 0.0, 1.0)[..., None]
        frame.aligned = warped * gain
        frame.gain = gain


def stack_clean(target: np.ndarray, sources: list[np.ndarray], work_dim: int) -> StackResult:
    """Aligns every frame in `sources` to `target` and replaces the target's foreground with what the other frames show behind it.

    Works on pictures at most `work_dim` px on the long side. Raises RegistrationError as soon as one frame cannot be matched.
    """
    full_target_shape = target.shape[:2]
    target = downsample_image(target, work_dim)
    scale = max(target.shape[:2]) / REFERENCE_DIM
    frames = [align_frame_homography(target, full_target_shape, source, work_dim) for source in sources]

    correct_gains(target, frames, compare_to_median(target, frames, scale)[3], scale)
    layers, covered, median, deviation = compare_to_median(target, frames, scale)
    everywhere = covered[0]
    trusted = trusted_median(deviation, covered, AGREE_BELOW)
    # the target is only called foreground where a majority of the covering layers agree on what is behind it
    blobs = foreground_blobs(deviation[0] * trusted, STRONG, WEAK, round(CLOSE_RADIUS * scale), round(GROW_RADIUS * scale), round(MIN_AREA * scale**2))
    labels = auto_labels(deviation, covered, trusted, blobs, CLEAN_MEAN, BAD_DEVIATION, round(BAD_GROW * scale), BAD_LIMIT, TRUST_FRACTION)

    feather_sigma = FEATHER_SIGMA * scale
    weights = label_weights(labels, [frame.coverage for frame in frames] + [everywhere], feather_sigma)
    return StackResult(
        target=target,
        target_sharpness=None,
        frames=frames,
        labels=labels,
        weights=weights,
        result=composite(target, [frame.aligned for frame in frames] + [median], weights),
        replaced_fraction=float(1.0 - weights[0][np.any(covered[1:], axis=0)].mean()),
        feather_sigma=feather_sigma,
        median=median,
    )


def ring_prior(image: np.ndarray, ring: np.ndarray, sigma: float) -> np.ndarray:
    """What `image` would look like inside a region, extended smoothly from the `ring` pixels around it.

    The ring pixels are averaged under a Gaussian of `sigma`; deeper into the region than the ring's reach the result fades to the
    ring's median colour. Returns a float32 (rows, cols, 3) array.
    """
    weight = ring.astype(np.float32)
    reach = cv2.GaussianBlur(weight, (0, 0), sigma)
    smooth = cv2.GaussianBlur(image * weight[..., None], (0, 0), sigma) / np.maximum(reach, 1e-6)[..., None]
    fallback = np.median(image[ring], axis=0)
    fade = np.clip(reach / PRIOR_REACH, 0.0, 1.0)[..., None]
    return (fallback + (smooth - fallback) * fade).astype(np.float32)


def pairwise_plate(layers: list[np.ndarray], covered: list[np.ndarray], sigma: float) -> np.ndarray:
    """A float32 plate: at each pixel the mean of the two covering layers that are closest to each other, the distance smoothed by `sigma`.

    Two layers that match are most likely both showing the background. It is wrong where a person is at the same place in two
    layers; where fewer than two layers cover, the plate is layer 0.
    """
    plate = layers[0].copy()
    best = np.full(layers[0].shape[:2], np.inf, dtype=np.float32)
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            distance = cv2.GaussianBlur(np.abs(layers[i] - layers[j]).max(axis=-1).astype(np.float32), (0, 0), sigma)
            closer = covered[i] & covered[j] & (distance < best)
            best[closer] = distance[closer]
            plate[closer] = (layers[i][closer] + layers[j][closer]) / 2
    return plate


def choose_plate(pairwise: np.ndarray, median: np.ndarray, trusted: np.ndarray, sigma: float, ring_radius: int) -> np.ndarray:
    """The `pairwise` plate, except in each area where it is over `PLATE_DIFF` from the `median`, the median is `trusted` and fits the surroundings better.

    The two plates fail in different places (the pairwise one where a person is at the same spot in two layers, the median where a
    person is in most layers), so they mostly agree. Each connected area where they differ is judged on its own: both are compared with
    the `ring_prior` of the calm pixels (where the plates agree) within `ring_radius`, and the one with the smaller median
    largest-channel distance inside the area wins; ties go to the pairwise plate. The median is only a candidate where it is
    `trusted` over at least `TRUST_FRACTION` of the area: where no majority of layers agree it is an arbitrary mix, not a background.
    """
    gap = cv2.GaussianBlur(np.abs(pairwise - median).max(axis=-1).astype(np.float32), (0, 0), sigma)
    differing = gap > PLATE_DIFF
    _, areas = cv2.connectedComponents(differing.astype(np.uint8))
    plate = pairwise.copy()
    grow = disk(ring_radius)
    for index, box in enumerate(ndimage.find_objects(areas), start=1):
        crop = tuple(slice(max(s.start - ring_radius, 0), s.stop + ring_radius) for s in box)
        inside = areas[crop] == index
        if trusted[crop][inside].mean() < TRUST_FRACTION:
            continue
        ring = cv2.dilate(inside.astype(np.uint8), grow).astype(bool) & ~differing[crop]
        if not ring.any():
            plate[crop][inside] = median[crop][inside]
            continue
        prior = ring_prior(median[crop], ring, sigma=max(inside.shape) / 4.0)
        pairwise_error, median_error = (float(np.median(np.abs(image[crop] - prior).max(axis=-1)[inside])) for image in (pairwise, median))
        if median_error < pairwise_error:
            plate[crop][inside] = median[crop][inside]
    return plate


def plate_labels(deviation: np.ndarray, covered: list[np.ndarray], blobs: np.ndarray, own_grow: int) -> np.ndarray:
    """An int16 label per pixel: 0 the target, k frame k, `len(deviation)` the plate (index 0 of `deviation` is the target).

    `deviation` is each layer's distance from the plate (`deviation_maps`). A blob where the target's mean deviation is already under
    `CLEAN_MEAN` is left alone: it is people who stay, which the plate is made of, not people to remove. Otherwise the blob goes
    whole to the covering frame with the lowest mean deviation there, if that is under `CLEAN_MEAN`; where that frame has a person of
    its own (deviation over `BAD_DEVIATION`, grown by `own_grow`, which must exceed the feather so it cannot bring the person back) the pixel takes the plate. A blob with no such frame takes the plate whole.
    """
    plate_label = len(deviation)
    labels = np.zeros(blobs.shape, dtype=np.int16)
    grow = disk(own_grow)
    for index, box in enumerate(ndimage.find_objects(blobs), start=1):
        if box is None:
            continue
        inside = blobs[box] == index
        if deviation[0][box][inside].mean() < CLEAN_MEAN:
            continue
        crop = labels[box]
        usable = [
            (float(deviation[frame][box][inside].mean()), frame)
            for frame in range(1, len(deviation))
            if covered[frame][box][inside].mean() >= COVER_FRACTION
        ]
        usable = [item for item in usable if item[0] < CLEAN_MEAN]
        if not usable:
            crop[inside] = plate_label
            continue
        _, frame = min(usable)
        crop[inside] = frame
        own_person = cv2.dilate((inside & (deviation[frame][box] > BAD_DEVIATION)).astype(np.uint8), grow).astype(bool)
        crop[inside & own_person] = plate_label
    return labels


def stack_clean2(target: np.ndarray, sources: list[np.ndarray], work_dim: int) -> StackResult:
    """Like `stack_clean`, but the clean plate comes from the closest pair of layers at each pixel (`pairwise_plate`) checked against the median (`choose_plate`).

    The blobs are where the target is far from the plate (`deviation_maps`). Each takes one frame (`plate_labels`), the plate filling in
    where that frame has a person of its own. The plate is in `StackResult.median`.
    """
    full_target_shape = target.shape[:2]
    target = downsample_image(target, work_dim)
    scale = max(target.shape[:2]) / REFERENCE_DIM
    frames = [align_frame_homography(target, full_target_shape, source, work_dim) for source in sources]

    correct_gains(target, frames, compare_to_median(target, frames, scale)[3], scale)
    layers, covered, median, median_deviation = compare_to_median(target, frames, scale)
    sigma = DEVIATION_SIGMA * scale
    trusted = trusted_median(median_deviation, covered, AGREE_BELOW)
    plate = choose_plate(pairwise_plate(layers, covered, sigma), median, trusted, sigma, round(RING_RADIUS * scale))
    deviation = deviation_maps(layers, covered, plate, sigma)
    blobs = foreground_blobs(deviation[0], STRONG, WEAK, round(CLOSE_RADIUS * scale), round(GROW_RADIUS * scale), round(MIN_AREA * scale**2))
    labels = plate_labels(deviation, covered, blobs, round(HOLE_SIGMAS * FEATHER_SIGMA * scale))

    feather_sigma = FEATHER_SIGMA * scale
    weights = label_weights(labels, [frame.coverage for frame in frames] + [covered[0]], feather_sigma)
    return StackResult(
        target=target,
        target_sharpness=None,
        frames=frames,
        labels=labels,
        weights=weights,
        result=composite(target, [frame.aligned for frame in frames] + [plate], weights),
        replaced_fraction=float(1.0 - weights[0][np.any(covered[1:], axis=0)].mean()),
        feather_sigma=feather_sigma,
        median=plate,
        plate_is_median=False,
    )
