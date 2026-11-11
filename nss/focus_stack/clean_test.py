"""Tests for nss.focus_stack.clean: the median layer, its trust test, the foreground blobs, the automatic labels and the clean2 plate."""
import cv2
import numpy as np

from nss.focus_stack.clean import (
    align_frame_homography,
    suggest_layer,
    auto_labels,
    choose_plate,
    deviation_maps,
    foreground_blobs,
    global_gain,
    pairwise_plate,
    plate_mix,
    plate_labels,
    ring_prior,
    stack_clean,
    stack_clean2,
    trusted_median,
)
from nss.focus_stack.register_test import corner_error, make_homography, make_texture
from nss.focus_stack.stack import AlignedFrame

SHAPE = (480, 640)  # rows, cols: non-square
PERSON_SPOTS = [(200, 300), (80, 60), (330, 180), (150, 470)]  # (row, col) of the "person" in the target and in each frame


BIG = (600, 800)  # a larger scene than the 480 x 640 view the target shows, so shifted frames still cover the whole view
VIEW = (slice(60, 540), slice(80, 720))


def make_big_background() -> np.ndarray:
    """A textured float32 RGB scene with mid-range brightness, larger than the view."""
    return make_texture(BIG, seed=21) * 0.6 + 0.3


def make_background() -> np.ndarray:
    """The empty view the target and the frames show."""
    return np.ascontiguousarray(make_big_background()[VIEW])


def with_person(scene: np.ndarray, index: int) -> np.ndarray:
    """The big scene with a bright 70 x 40 block, the 'person', at spot `index` of the view."""
    image = scene.copy()
    row, col = PERSON_SPOTS[index]
    image[VIEW[0].start + row:VIEW[0].start + row + 70, VIEW[1].start + col:VIEW[1].start + col + 40] = 0.97
    return image


def make_people_pair() -> tuple[np.ndarray, list[np.ndarray]]:
    """A target with a person at the first spot and three handheld-shifted frames with their people elsewhere."""
    scene = make_big_background()
    frames = []
    for index, (shift_x, shift_y, degrees) in enumerate([(6.0, -4.0, 0.8), (-5.0, 7.0, -0.6), (3.0, 3.0, 1.1)], start=1):
        angle = np.deg2rad(degrees)
        matrix = np.array([[np.cos(angle), -np.sin(angle), shift_x], [np.sin(angle), np.cos(angle), shift_y], [0.0, 0.0, 1.0]])
        warped = cv2.warpPerspective(with_person(scene, index), matrix, (BIG[1], BIG[0]), flags=cv2.INTER_LINEAR)
        frames.append(np.ascontiguousarray(warped[VIEW]))
    return np.ascontiguousarray(with_person(scene, 0)[VIEW]), frames


def test_align_frame_homography_has_no_flow_agreement_or_sharpness() -> None:
    """A frame seen through a known homography is aligned by the matrix alone, matching the target where it covers."""
    background = make_background()
    truth = make_homography()
    target = cv2.warpPerspective(background, truth, (SHAPE[1], SHAPE[0]), flags=cv2.INTER_LINEAR)
    frame = align_frame_homography(target, SHAPE, 0.8 * background, work_dim=640)
    assert frame.flow is None and frame.agreement is None and frame.sharpness is None
    assert corner_error(frame.matrix, truth, SHAPE) < 0.5
    assert 0.3 < frame.coverage.mean() < 0.95
    assert np.abs(frame.aligned - target)[frame.coverage].mean() < 0.03
    assert frame.gain.shape == (*SHAPE, 3) and frame.inliers >= 30


def test_deviation_maps_are_the_blurred_largest_channel_difference_where_covered() -> None:
    """A 0.3 change in one channel shows as a deviation of about 0.3 inside the block, 0 outside, and 0 where uncovered."""
    median = np.full((40, 60, 3), 0.5, dtype=np.float32)
    layer = median.copy()
    layer[10:20, 15:30, 1] += 0.3
    layer[30:38, 40:55, 2] += 0.3
    covered = np.ones((40, 60), dtype=bool)
    covered[30:38, 40:55] = False
    dev = deviation_maps([layer, median], [covered, np.ones((40, 60), dtype=bool)], median, sigma=0.5)
    assert dev.shape == (2, 40, 60)
    assert np.allclose(dev[0, 12:18, 18:27], 0.3, atol=0.02)
    assert dev[0, 0:5, 0:5].max() < 1e-6 and dev[0, 31:37, 41:54].max() == 0.0
    assert dev[1].max() < 1e-6


def test_trusted_median_needs_a_strict_majority_of_the_covering_layers() -> None:
    """3 of 5 agreeing is trusted, 2 of 5 is not; with only 3 covering, 2 agreeing is trusted, and 2 of 4 is not."""
    deviation = np.full((5, 4, 6), 0.5, dtype=np.float32)
    covered = np.ones((5, 4, 6), dtype=bool)
    deviation[:3, 0, 0] = 0.0                  # 3 of 5 agree
    deviation[:2, 0, 1] = 0.0                  # 2 of 5 agree
    covered[3:, 0, 2] = False
    deviation[:2, 0, 2] = 0.0                  # 2 of the 3 covering layers agree
    covered[4, 0, 3] = False
    deviation[:2, 0, 3] = 0.0                  # 2 of the 4 covering layers agree
    trusted = trusted_median(deviation, list(covered), agree_below=0.03)
    assert trusted.shape == (4, 6)
    assert trusted[0, 0] and not trusted[0, 1] and trusted[0, 2] and not trusted[0, 3]


def test_foreground_blobs_grow_from_strong_cores_close_holes_and_drop_specks() -> None:
    """A strong core grows into its weak halo; a ring has its hole filled; weak-only areas and tiny cores are dropped."""
    deviation = np.zeros((60, 80), dtype=np.float32)
    deviation[15:35, 25:45] = 0.05
    deviation[20:30, 30:40] = 0.2
    deviation[45:55, 60:70] = 0.05                 # weak only: no strong core
    deviation[2:3, 2:3] = 0.3                      # a speck
    deviation[42:56, 5:19] = 0.2
    deviation[46:52, 9:15] = 0.0                   # hole inside a ring of strong deviation
    blobs = foreground_blobs(deviation, strong=0.08, weak=0.03, close_radius=1, grow_radius=1, min_area=30)
    assert blobs.shape == (60, 80) and blobs.dtype == np.int32
    assert blobs[16, 26] > 0 and blobs[16, 26] == blobs[28, 38]
    assert blobs[48, 12] > 0 and blobs[48, 12] != blobs[28, 38]
    assert blobs[50, 65] == 0 and blobs[2, 2] == 0
    assert len(np.unique(blobs)) == 3


def make_label_case() -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Deviation maps (target + 2 frames), coverage, trust and four blobs, each exercising one rule of auto_labels."""
    shape = (40, 70)
    deviation = np.zeros((3, *shape), dtype=np.float32)
    deviation[0, 10:20, 10:30] = 0.3
    deviation[2, 10:20, 10:30] = 0.2                       # blob 1: frame 1 clean, frame 2 has a person
    deviation[0, 25:35, 10:30] = 0.3
    deviation[1, 28:30, 15:18] = 0.3                       # blob 2: frame 1 clean except a small person of its own
    deviation[2, 25:35, 10:30] = 0.1
    deviation[:, 5:9, 40:60] = 0.2                         # blob 3: no clean frame at all
    deviation[0, 30:38, 40:55] = 0.3
    deviation[2, 30:38, 40:55] = 0.0                       # blob 4: frame 1 clean but not covering enough, frame 2 clean
    covered = [np.ones(shape, dtype=bool) for _ in range(3)]
    covered[1][30:38, 48:55] = False
    trusted = np.ones(shape, dtype=bool)
    trusted[25:35, 28:30] = False                          # a tenth of blob 2 where the median cannot be trusted
    trusted[5:9, 50:60] = False                            # half of blob 3
    blobs = np.zeros(shape, dtype=np.int32)
    blobs[10:20, 10:30] = 1
    blobs[25:35, 10:30] = 2
    blobs[5:9, 40:60] = 3
    blobs[30:38, 40:55] = 4
    return deviation, covered, trusted, blobs


def labels_for(deviation: np.ndarray, covered: list[np.ndarray], trusted: np.ndarray, blobs: np.ndarray, **changes: float) -> np.ndarray:
    """auto_labels with the production rules for the limits, except those given in `changes`, and no growing of a chosen frame's people."""
    rules = dict(clean_mean=0.04, bad_deviation=0.05, bad_grow=0, bad_limit=0.3, trust_fraction=0.8)
    rules.update(changes)
    return auto_labels(deviation, covered, trusted, blobs, **rules)


def test_auto_labels_give_each_blob_to_its_cleanest_frame() -> None:
    """Blob 1 goes to frame 1, the clean one; blob 4 skips the frame that does not cover it and goes to frame 2."""
    deviation, covered, trusted, blobs = make_label_case()
    labels = labels_for(deviation, covered, trusted, blobs)
    assert labels.shape == blobs.shape and labels.dtype == np.int16
    assert (labels[10:20, 10:30] == 1).all()
    assert (labels[30:38, 40:55] == 2).all()
    assert (labels[blobs == 0] == 0).all()


def test_auto_labels_replace_a_chosen_frames_own_person_with_the_median_or_the_target() -> None:
    """Where the chosen frame deviates strongly, the pixel takes the median (label 3) if it is trusted, otherwise the target."""
    deviation, covered, trusted, blobs = make_label_case()
    labels = labels_for(deviation, covered, trusted, blobs)
    assert (labels[28:30, 15:18] == 3).all()
    assert (labels[25:28, 10:28] == 1).all() and (labels[30:35, 10:28] == 1).all()
    grown = labels_for(deviation, covered, trusted, blobs, bad_grow=1)
    assert grown[27, 16] == 3 and grown[30, 16] == 3 and grown[28, 14] == 3 and grown[28, 18] == 3
    assert grown[26, 16] == 1
    trusted[28:30, 15:18] = False
    labels = labels_for(deviation, covered, trusted, blobs)
    assert (labels[28:30, 15:18] == 0).all()


def test_auto_labels_use_the_median_only_where_trusted_when_no_frame_is_clean() -> None:
    """With no usable frame a blob takes the median where it is trusted, if that is enough of it, and keeps the target elsewhere."""
    deviation, covered, trusted, blobs = make_label_case()
    labels = labels_for(deviation, covered, trusted, blobs, trust_fraction=0.4)
    assert (labels[5:9, 40:50] == 3).all()
    assert (labels[5:9, 50:60] == 0).all()


def test_auto_labels_leave_a_blob_alone_when_the_median_is_trusted_over_too_little_of_its_people() -> None:
    """Trusted over half of blob 3 is not enough at the default 80%: the whole blob stays the target, so no person is half overwritten."""
    deviation, covered, trusted, blobs = make_label_case()
    labels = labels_for(deviation, covered, trusted, blobs)
    assert (labels[5:9, 40:60] == 0).all()


def test_auto_labels_measure_trust_over_the_targets_people_not_over_the_whole_blob() -> None:
    """A median trusted everywhere in a big blob except under the person is not enough: the blob stays the target."""
    shape = (40, 40)
    deviation = np.zeros((2, *shape), dtype=np.float32)
    deviation[0, 10:16, 10:16] = 0.3
    covered = [np.ones(shape, dtype=bool), np.ones(shape, dtype=bool)]
    blobs = np.zeros(shape, dtype=np.int32)
    blobs[10:30, 10:30] = 1
    trusted = np.ones(shape, dtype=bool)
    trusted[10:16, 10:16] = False
    assert (labels_for(deviation, covered, trusted, blobs) == 0).all()
    trusted[10:16, 10:16] = True
    assert (labels_for(deviation, covered, trusted, blobs)[10:30, 10:30] == 1).all()


def test_auto_labels_reject_a_frame_that_shows_the_same_people_as_the_target() -> None:
    """A frame with a person of its own at 55% of the target's person pixels is not usable at a 30% limit, and is at a 60% limit."""
    deviation, covered, trusted, blobs = make_label_case()
    deviation[1, 25:35, 10:21] = 0.06                       # mean 0.033 passes the clean test, yet over 55% of the blob is a person
    deviation[1, 25:35, 21:30] = 0.0
    deviation[2, 25:35, 10:30] = 0.2
    trusted[:] = True
    strict = labels_for(deviation, covered, trusted, blobs)
    assert (strict[25:35, 10:30] == 3).all()                # no usable frame: the median, which is trusted here
    lenient = labels_for(deviation, covered, trusted, blobs, bad_limit=0.6)
    assert (lenient[25:35, 21:30] == 1).all() and (lenient[25:35, 10:21] == 3).all()


def test_stack_clean_removes_a_person_that_is_elsewhere_in_the_other_frames() -> None:
    """The target's person is replaced by background from the frames; everything away from it is the target unchanged."""
    target, frames = make_people_pair()
    plan = stack_clean(target, frames, work_dim=640)
    assert plan.weights.shape == (5, *SHAPE) and plan.median.shape == (*SHAPE, 3)
    assert plan.target_sharpness is None and all(frame.flow is None for frame in plan.frames)
    assert np.allclose(plan.weights.sum(axis=0), 1.0, atol=1e-5)
    row, col = PERSON_SPOTS[0]
    background = make_background()
    person = (slice(row + 5, row + 65), slice(col + 5, col + 35))
    assert np.abs(target - background)[person].mean() > 0.2
    assert np.abs(plan.result - background)[person].mean() < 0.06
    assert plan.labels[row + 35, col + 20] != 0
    far = np.ones(SHAPE, dtype=bool)
    far[max(0, row - 40):row + 110, max(0, col - 40):col + 80] = False
    assert np.abs(plan.result - target)[far].max() < 1e-4
    assert 0.0 < plan.replaced_fraction < 0.1


def test_auto_labels_compare_a_frames_people_to_the_targets_not_to_the_whole_blob() -> None:
    """A mostly-background blob whose small person core is also a person in the frame is rejected, though the frame's mean is low."""
    shape = (40, 40)
    deviation = np.zeros((3, *shape), dtype=np.float32)
    deviation[0, 10:16, 10:16] = 0.3                      # the target's person: a 6 x 6 core in a 20 x 20 blob
    deviation[1, 10:16, 10:16] = 0.06                     # the frame shows a person at the same place (mean over the blob 0.005)
    deviation[2, 10:30, 10:30] = 0.2
    covered = [np.ones(shape, dtype=bool) for _ in range(3)]
    blobs = np.zeros(shape, dtype=np.int32)
    blobs[10:30, 10:30] = 1
    trusted = np.ones(shape, dtype=bool)
    assert (labels_for(deviation, covered, trusted, blobs)[10:30, 10:30] == 3).all()    # no usable frame: the median
    deviation[1, 10:16, 10:16] = 0.0                      # the frame is empty where the target has its person
    assert (labels_for(deviation, covered, trusted, blobs)[10:30, 10:30] == 1).all()


def flat_frame(level: float, covered: np.ndarray | None = None) -> AlignedFrame:
    """A flat 30 x 40 frame at `level` that covers everything unless `covered` says otherwise."""
    shape = (30, 40)
    return AlignedFrame(
        aligned=np.full((*shape, 3), level, dtype=np.float32),
        coverage=np.ones(shape, dtype=bool) if covered is None else covered,
        agreement=None, sharpness=None, inliers=50, matrix=np.eye(3), flow=None, gain=np.ones((*shape, 3), dtype=np.float32),
    )


def test_suggest_layer_prefers_the_cleanest_covering_frame_and_falls_back_to_the_median() -> None:
    """A new patch starts on the covering frame closest to the median if it is clean there; otherwise on the median layer."""
    median = np.full((30, 40, 3), 0.5, dtype=np.float32)
    region = (slice(5, 15), slice(5, 25))
    inside = np.ones((10, 20), dtype=bool)
    frames = [flat_frame(0.7), flat_frame(0.51), flat_frame(0.505)]
    assert suggest_layer(frames, median, region, inside, clean_mean=0.04) == 3
    assert suggest_layer(frames[:2], median, region, inside, clean_mean=0.04) == 2
    assert suggest_layer([frames[0]], median, region, inside, clean_mean=0.04) == 2       # none clean: the median, layer 2
    half_covered = np.ones((30, 40), dtype=bool)
    half_covered[:, 10:] = False
    assert suggest_layer([flat_frame(0.5, half_covered), frames[0]], median, region, inside, clean_mean=0.04) == 3


def make_car_pair() -> tuple[np.ndarray, list[np.ndarray]]:
    """A target with a big dark 'car' (200 x 300 px, much larger than the brightness smoothing) and the three frames of make_people_pair."""
    target = make_background()
    target[140:340, 160:460] = 0.04
    return target, make_people_pair()[1]


def test_a_large_dark_object_in_the_target_does_not_darken_the_frames() -> None:
    """Matching brightness to a target with a huge dark object must not crush the frames there to a black oval."""
    target, frames = make_car_pair()
    plan = stack_clean(target, frames, work_dim=640)
    background = make_background()
    inside = (slice(180, 300), slice(200, 420))
    for frame in plan.frames:
        assert np.abs(frame.aligned - background)[inside].mean() < 0.05
        assert frame.gain[inside].mean() > 0.9


def test_stack_clean_replaces_a_large_dark_object_with_the_background_not_a_black_oval() -> None:
    """The car is gone from the result: its area shows the background texture, not the dark target and not a darkened frame."""
    target, frames = make_car_pair()
    plan = stack_clean(target, frames, work_dim=640)
    inside = (slice(180, 300), slice(200, 420))
    assert np.abs(target - make_background())[inside].mean() > 0.3
    assert np.abs(plan.result - make_background())[inside].mean() < 0.08


def test_global_gain_is_the_median_ratio_and_ignores_a_big_object() -> None:
    """A frame 25% darker gives a gain of 1.25 per channel even when a block covering 30% of the picture differs wildly."""
    rng = np.random.default_rng(2)
    frame = (0.2 + 0.6 * rng.random((40, 60, 3))).astype(np.float32) * np.array([1.0, 0.8, 0.6], dtype=np.float32)
    target = frame * 1.25
    target[:, :18] = 0.05                                  # a dark object over 30% of the picture
    gain = global_gain(target, frame, np.ones((40, 60), dtype=bool))
    assert gain.shape == (3,) and np.allclose(gain, 1.25, atol=0.01)
    assert np.allclose(global_gain(target, frame, np.zeros((40, 60), dtype=bool)), 1.0)



def test_stack_clean2_removes_a_person_that_is_elsewhere_in_the_other_frames() -> None:
    """Same scene as `stack_clean`: the target's person is replaced by background and everything away from it is untouched."""
    target, frames = make_people_pair()
    plan = stack_clean2(target, frames, work_dim=640)
    assert plan.weights.shape == (5, *SHAPE) and plan.median.shape == (*SHAPE, 3)
    assert not plan.plate_is_median
    assert plan.plate_mix.shape == (*SHAPE, 5) and np.allclose(plan.plate_mix.sum(axis=-1), 1.0)
    assert np.allclose(plan.weights.sum(axis=0), 1.0, atol=1e-5)
    row, col = PERSON_SPOTS[0]
    person = (slice(row + 5, row + 65), slice(col + 5, col + 35))
    assert np.abs(plan.result - make_background())[person].mean() < 0.06
    far = np.ones(SHAPE, dtype=bool)
    far[max(0, row - 40):row + 110, max(0, col - 40):col + 80] = False
    assert np.abs(plan.result - target)[far].max() < 1e-4



def flat_layers(levels: list[float], shape: tuple[int, int] = (30, 40)) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """One flat RGB layer per level and a full-coverage map for each."""
    return [np.full((*shape, 3), level, dtype=np.float32) for level in levels], [np.ones(shape, dtype=bool) for _ in levels]


def test_ring_prior_continues_the_ring_into_the_region_and_fades_to_its_median_colour() -> None:
    """A ring of a smooth ramp is extended inward; far from the ring the colour is the ring's median."""
    shape = (30, 44)
    image = np.zeros((*shape, 3), dtype=np.float32)
    image[...] = np.linspace(0.2, 0.6, shape[1], dtype=np.float32)[None, :, None]
    ring = np.zeros(shape, dtype=bool)
    ring[4:26, 10:34] = True
    ring[8:22, 14:30] = False
    prior = ring_prior(image, ring, sigma=3.0)
    assert prior.shape == (*shape, 3) and prior.dtype == np.float32
    assert abs(prior[15, 15, 0] - image[15, 15, 0]) < 0.06
    assert abs(prior[15, 22, 0] - np.median(image[ring][:, 0])) < 0.06


def test_pairwise_plate_is_the_mean_of_the_two_closest_covering_layers() -> None:
    """Layers 2 and 3 agree where layers 0 and 1 show the same person; layer 1 does not cover there, so it cannot match layer 0."""
    layers, covered = flat_layers([0.30, 0.30, 0.30, 0.45])
    layers[0][5:15, 8:20] = 0.9                              # a person in layer 0 ...
    layers[1][5:15, 8:20] = 0.9                              # ... and at the same place in layer 1, which does not cover there
    covered[1][5:15, 8:20] = False
    layers[2][5:15, 8:20] = 0.30
    layers[3][5:15, 8:20] = 0.45
    plate, pair = pairwise_plate(layers, covered, sigma=0.5)
    assert plate.shape == (30, 40, 3) and plate.dtype == np.float32
    assert np.allclose(plate[7:13, 10:18], 0.375, atol=1e-5)  # layers 2 and 3
    assert np.allclose(plate[20:, 25:], 0.30, atol=1e-5)
    assert pair.shape == (30, 40, 2) and pair.dtype == np.int16
    assert (pair[7:13, 10:18] == (2, 3)).all()                # the indices say which layers the plate is the mean of
    assert (pair[20:, 25:] == (0, 1)).all()


def test_choose_plate_takes_the_median_or_the_pairwise_plate_by_which_fits_the_surroundings() -> None:
    """Where the plates differ, the one that continues the calm 0.5 around the area wins; where they agree nothing changes."""
    pairwise = np.full((30, 40, 3), 0.5, dtype=np.float32)
    median = pairwise.copy()
    pairwise[5:10, 5:12] = 0.1                               # the pairwise plate has a dark spot, the median is close to its surroundings
    median[5:10, 5:12] = 0.52
    median[18:25, 25:34] = 0.9                               # the median has a person, the pairwise plate is close to its surroundings
    pairwise[18:25, 25:34] = 0.48
    trusted = np.ones((30, 40), dtype=bool)
    plate, use_median = choose_plate(pairwise, median, trusted, sigma=0.5, ring_radius=5)
    assert plate.shape == (30, 40, 3) and plate.dtype == np.float32
    assert np.allclose(plate[6:9, 6:11], 0.52, atol=1e-5)
    assert np.allclose(plate[19:24, 26:33], 0.48, atol=1e-5)
    assert np.allclose(plate[:4], 0.5)
    assert use_median[6:9, 6:11].all() and not use_median[19:24, 26:33].any() and not use_median[:4].any()
    trusted[5:10, 5:12] = False                              # no majority of layers backs the median there: the pairwise plate stays
    plate, use_median = choose_plate(pairwise, median, trusted, sigma=0.5, ring_radius=5)
    assert np.allclose(plate[6:9, 6:11], 0.1, atol=1e-5)


def test_plate_labels_give_each_blob_to_its_cleanest_frame_and_the_plate_where_there_is_none() -> None:
    """Frame 1 is the cleanest over blob 1 apart from a small person of its own, which takes the plate; blob 2 has no clean frame."""
    deviation = np.zeros((4, 30, 40), dtype=np.float32)
    deviation[0, 10:20, 3:15] = 0.3                          # the target's people
    deviation[0, 5:12, 25:38] = 0.3
    deviation[1] = 0.01
    deviation[1, 12:14, 6:10] = 0.2                          # frame 1's own person
    deviation[2] = 0.10
    deviation[3] = 0.03
    deviation[1:, 5:12, 25:38] = 0.2                         # blob 2: every frame is far from the plate
    blobs = np.zeros((30, 40), dtype=np.int32)
    blobs[10:20, 3:15], blobs[5:12, 25:38] = 1, 2
    covered = [np.ones((30, 40), dtype=bool) for _ in range(4)]
    labels = plate_labels(deviation, covered, blobs, own_grow=1)
    assert labels.shape == (30, 40) and labels.dtype == np.int16
    assert (labels[12:14, 6:10] == 4).all() and labels[18, 13] == 1 and labels[10, 3] == 1
    assert (labels[5:12, 25:38] == 4).all() and (labels[blobs == 0] == 0).all()
    covered[1][10:20, 3:15] = False                          # frame 1 does not cover blob 1: frame 3 is the cleanest left
    assert (plate_labels(deviation, covered, blobs, own_grow=1)[10:20, 3:15] == 3).all()


def test_plate_labels_leave_a_blob_alone_where_the_target_is_already_close_to_the_plate() -> None:
    """People who stay are what the plate is made of: the target deviates a little over a blob, so no frame replaces it."""
    deviation = np.zeros((3, 30, 40), dtype=np.float32)
    deviation[0, 10:20, 3:15] = 0.03                         # under CLEAN_MEAN on average, yet some pixels are over BAD_DEVIATION
    deviation[0, 12:14, 5:9] = 0.09
    deviation[1:] = 0.01
    blobs = np.zeros((30, 40), dtype=np.int32)
    blobs[10:20, 3:15] = 1
    covered = [np.ones((30, 40), dtype=bool) for _ in range(3)]
    assert (plate_labels(deviation, covered, blobs, own_grow=1) == 0).all()
    deviation[0, 10:20, 3:15] = 0.3                          # a real person: the cleanest frame takes the blob
    assert (plate_labels(deviation, covered, blobs, own_grow=1)[10:20, 3:15] == 1).all()


def test_plate_mix_says_how_the_layers_and_median_make_the_plate() -> None:
    """Half on each of a pixel's two layers, or all on the median; applying it to the layers gives back the plate it came from."""
    layers, covered = flat_layers([0.30, 0.30, 0.30, 0.45])
    layers[0][5:15, 8:20] = 0.9
    layers[2][5:15, 8:20] = 0.35
    covered[1][5:15, 8:20] = False
    median = np.full_like(layers[0], 0.6)
    pairwise, pair = pairwise_plate(layers, covered, sigma=0.5)
    use_median = np.zeros(pairwise.shape[:2], dtype=bool)
    use_median[20:, 25:] = True
    plate = np.where(use_median[..., None], median, pairwise)
    mix = plate_mix(pair, use_median, len(layers))
    assert mix.shape == (30, 40, 5) and mix.dtype == np.float32
    assert np.allclose(mix.sum(axis=-1), 1.0)
    assert np.allclose(mix[7:13, 10:18, [2, 3]], 0.5) and not mix[7:13, 10:18, [0, 1, 4]].any()
    assert np.allclose(mix[20:, 25:, 4], 1.0) and not mix[20:, 25:, :4].any()
    assert np.allclose(sum(mix[..., k, None] * layer for k, layer in enumerate(layers + [median])), plate, atol=1e-5)


def test_full_size_render_of_clean2_with_its_plate_mix_gives_its_own_result() -> None:
    """When the working size is the full size, rendering from the frames with `plate_mix` reproduces the plate-based working result."""
    from nss.focus_stack.stack import render_full
    target, frames = make_people_pair()
    plan = stack_clean2(target, frames, work_dim=640)
    assert plan.result.shape[:2] == target.shape[:2]
    full = render_full(target, frames, plan.frames, plan.weights, tile_rows=100, plate_mix=plan.plate_mix)
    assert np.abs(full - plan.result).max() < 0.02
