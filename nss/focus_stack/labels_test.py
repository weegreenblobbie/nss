"""Tests for nss.focus_stack.labels."""
import cv2
import numpy as np

from nss.focus_stack.labels import choose_labels, label_weights

SHAPE = (30, 50)  # rows, cols: non-square


def flat(value: float) -> np.ndarray:
    """A float32 map of SHAPE holding `value`."""
    return np.full(SHAPE, value, dtype=np.float32)


def all_true() -> np.ndarray:
    """A bool map of SHAPE that is True everywhere."""
    return np.ones(SHAPE, dtype=bool)


def test_choose_labels_picks_a_frame_only_where_it_beats_the_target_by_the_bias() -> None:
    """Frame 1 four times sharper on the left half gets label 1 there; the equally sharp right half stays 0."""
    sharp = flat(1.0)
    sharp[:, :25] = 4.0
    labels = choose_labels(flat(1.0), [sharp], [all_true()], bias_stops=0.5, tidy_radius=0)
    assert labels.shape == SHAPE
    assert (labels[:, :25] == 1).all()
    assert (labels[:, 25:] == 0).all()


def test_choose_labels_needs_more_than_the_bias() -> None:
    """1.2 times sharper is chosen with no bias and not with half a stop."""
    assert (choose_labels(flat(1.0), [flat(1.2)], [all_true()], 0.0, 0) == 1).all()
    assert (choose_labels(flat(1.0), [flat(1.2)], [all_true()], 0.5, 0) == 0).all()


def test_choose_labels_takes_the_sharpest_of_several_frames() -> None:
    """Left third only frame 1 is sharper, middle third both are and frame 2 is sharper still, right third neither."""
    one, two = flat(1.0), flat(1.0)
    one[:, :34] = 4.0
    two[:, 17:34] = 6.0
    labels = choose_labels(flat(1.0), [one, two], [all_true(), all_true()], 0.5, 0)
    assert (labels[:, :17] == 1).all()
    assert (labels[:, 17:34] == 2).all()
    assert (labels[:, 34:] == 0).all()


def test_choose_labels_never_picks_a_frame_where_it_is_not_eligible() -> None:
    """Frame 2 is sharpest everywhere but ineligible on the left half, so frame 1 wins there; frame 1 is ineligible on the top rows."""
    one, two = flat(4.0), flat(8.0)
    eligible_one, eligible_two = all_true(), all_true()
    eligible_two[:, :25] = False
    eligible_one[:10, :25] = False
    labels = choose_labels(flat(1.0), [one, two], [eligible_one, eligible_two], 0.5, 0)
    assert (labels[10:, :25] == 1).all()
    assert (labels[:10, :25] == 0).all()
    assert (labels[:, 25:] == 2).all()


def test_choose_labels_tidy_fills_holes_and_drops_thin_strips_without_stealing() -> None:
    """With radius 3 a 3x3 hole in frame 1's block is filled and a 3-wide strip is dropped; frame 2's 10x10 block inside keeps its label."""
    shape = (40, 60)
    target = np.ones(shape, dtype=np.float32)
    one = np.ones(shape, dtype=np.float32)
    two = np.ones(shape, dtype=np.float32)
    one[5:35, 5:35] = 4.0
    one[8:11, 8:11] = 1.0
    two[20:30, 20:30] = 9.0
    one[5:35, 50:53] = 4.0
    on = np.ones(shape, dtype=bool)
    untidy = choose_labels(target, [one, two], [on, on], 0.5, 0)
    tidy = choose_labels(target, [one, two], [on, on], 0.5, 3)
    assert untidy[9, 9] == 0
    assert untidy[20, 51] == 1
    assert tidy[9, 9] == 1
    assert (tidy[:, 48:] == 0).all()
    assert (tidy[22:28, 22:28] == 2).all()
    assert (tidy[12:18, 12:18] == 1).all()


def test_choose_labels_tidy_keeps_a_region_touching_the_image_border() -> None:
    """A block against the image edge is neither eroded nor shifted by tidying."""
    shape = (40, 60)
    sharp = np.ones(shape, dtype=np.float32)
    sharp[:, :20] = 4.0
    labels = choose_labels(np.ones(shape, dtype=np.float32), [sharp], [np.ones(shape, dtype=bool)], 0.5, 3)
    assert (labels[:, :20] == 1).all()
    assert (labels[:, 20:] == 0).all()


def test_label_weights_sum_to_one_and_the_soft_edge_lies_outside_the_region() -> None:
    """Weights add to 1 everywhere; a region is fully opaque up to its edge and fades out over the target's pixels beyond it."""
    labels = np.zeros(SHAPE, dtype=np.int16)
    labels[:, :20] = 1
    labels[:, 30:] = 2
    weights = label_weights(labels, [all_true(), all_true()], feather_sigma=1.5)
    assert weights.shape == (3, *SHAPE)
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-5)
    assert weights[1, 15, 3] > 0.99 and weights[2, 15, 45] > 0.99
    assert weights[1, 15, 19] > 0.97 and weights[1, 15, 20] > 0.9          # opaque to the edge, still strong just outside it
    assert 0.1 < weights[1, 15, 23] < 0.9 and weights[1, 15, 28] < 0.05    # the fade is outside
    assert weights[0, 15, 25] > 0.75                                        # the middle of the 10-pixel gap is mostly the target


def test_label_weights_give_the_target_what_an_ineligible_frame_cannot() -> None:
    """Where a frame is not eligible its weight is 0 and the target takes it, so weights still sum to 1."""
    labels = np.ones(SHAPE, dtype=np.int16)
    eligible = all_true()
    eligible[:, :10] = False
    weights = label_weights(labels, [eligible], feather_sigma=1.5)
    assert (weights[1, :, :10] == 0.0).all()
    assert np.allclose(weights[0, :, :10], 1.0)
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-5)


def test_a_small_region_is_fully_opaque_inside() -> None:
    """A 6 x 6 patch, smaller than the blur, shows its layer at (nearly) full strength over its whole interior, not a blend with the target.

    Before the soft edge was moved outside the region, its centre was only about 60% opaque; its corners are the weakest pixels now.
    """
    labels = np.zeros(SHAPE, dtype=np.int16)
    labels[12:18, 20:26] = 1
    weights = label_weights(labels, [all_true()], feather_sigma=3.0)
    assert weights[1, 14:16, 22:24].min() > 0.98
    assert weights[1, 12:18, 20:26].min() > 0.93
    assert weights[0, 12:18, 20:26].max() < 0.07
    assert weights[1, 15, 40] < 0.01


def test_a_regions_growth_never_enters_another_regions_pixels() -> None:
    """Next to a region of another frame, growth stops at the border: deep inside it the neighbour's weight is negligible."""
    labels = np.zeros(SHAPE, dtype=np.int16)
    labels[:, :20] = 1
    labels[:, 20:] = 2
    weights = label_weights(labels, [all_true(), all_true()], feather_sigma=1.5)
    assert weights[1, 15, 26] < 0.01 and weights[2, 15, 13] < 0.01
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-5)


def test_growths_that_overlap_in_the_targets_pixels_are_shared_and_weights_stay_valid() -> None:
    """Two regions 4 pixels apart both grow into the gap; the weights still add to 1, none is negative, and each stays opaque inside."""
    labels = np.zeros(SHAPE, dtype=np.int16)
    labels[10:20, 10:20] = 1
    labels[10:20, 24:34] = 2
    weights = label_weights(labels, [all_true(), all_true()], feather_sigma=2.0)
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-5) and weights.min() >= 0.0
    assert weights[1, 12:18, 10:16].min() > 0.9 and weights[2, 12:18, 28:34].min() > 0.9


