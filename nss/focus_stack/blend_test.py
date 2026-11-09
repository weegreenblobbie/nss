"""Tests for nss.focus_stack.blend."""
import cv2
import numpy as np

from nss.focus_stack.blend import agreement_map, composite, gain_map, masked_median, match_gain, source_coverage
from nss.focus_stack.labels import choose_labels, label_weights
from nss.focus_stack.register_test import make_texture
from nss.focus_stack.sharpness import sharpness_map

SHAPE = (480, 640)  # rows, cols: non-square


def test_source_coverage_excludes_uncovered_columns_and_a_margin() -> None:
    """A +5 column shift leaves columns 0..4 empty, and a margin of 1 also drops column 5."""
    matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    flow = np.zeros((13, 12, 2), dtype=np.float32)
    covered = source_coverage((13, 12), matrix, flow, margin=1)
    assert covered.shape == (13, 12)
    assert covered.dtype == bool
    assert not covered[:, :6].any()
    assert covered[:, 7:].all()


def test_match_gain_removes_a_smooth_multiplicative_difference() -> None:
    """A source dimmed by 0.5..0.8 across the columns is brought back to the target's level."""
    target = make_texture(SHAPE, seed=1) * 0.6 + 0.3
    dim = np.linspace(0.5, 0.8, SHAPE[1], dtype=np.float32)[None, :, None]
    source = target * dim
    valid = np.ones(SHAPE, dtype=bool)
    matched = match_gain(target, source, valid, sigma=20.0)
    inner = (slice(40, -40), slice(40, -40))
    before = np.abs(source - target)[inner].mean()
    after = np.abs(matched - target)[inner].mean()
    assert matched.shape == source.shape
    assert after < 0.1 * before


def test_match_gain_ignores_pixels_outside_the_valid_region() -> None:
    """Black pixels the source does not cover do not pull the gain down near the valid edge."""
    target = make_texture(SHAPE, seed=2) * 0.6 + 0.3
    source = target * 0.5
    valid = np.ones(SHAPE, dtype=bool)
    valid[:, :200] = False
    source[:, :200] = 0.0
    matched = match_gain(target, source, valid, sigma=20.0)
    edge = (slice(40, -40), slice(205, 230))
    assert np.abs(matched - target)[edge].mean() < 0.02


def test_composite_is_the_weighted_sum_per_channel() -> None:
    """Weights 0.25 / 0.5 / 0.25 give a quarter of the target, half of frame 1 and a quarter of frame 2."""
    target = np.full((13, 12, 3), 0.8, dtype=np.float32)
    one = np.zeros((13, 12, 3), dtype=np.float32)
    one[..., 1] = 0.4
    two = np.full((13, 12, 3), 0.2, dtype=np.float32)
    weights = np.empty((3, 13, 12), dtype=np.float32)
    weights[0], weights[1], weights[2] = 0.25, 0.5, 0.25
    out = composite(target, [one, two], weights)
    assert out.shape == (13, 12, 3)
    assert np.allclose(out[..., 0], 0.25)
    assert np.allclose(out[..., 1], 0.45)


def test_blurred_half_of_target_is_repaired_from_a_dimmer_sharp_source() -> None:
    """Target blurred on the right, source sharp but 20% darker: the composite is sharp there and unchanged on the left."""
    sharp = make_texture(SHAPE, seed=3) * 0.6 + 0.3
    half = SHAPE[1] // 2
    target = sharp.copy()
    target[:, half:] = cv2.GaussianBlur(sharp, (0, 0), 3.0)[:, half:]
    source = 0.8 * sharp
    valid = np.ones(SHAPE, dtype=bool)

    matched = match_gain(target, source, valid, sigma=30.0)
    labels = choose_labels(sharpness_map(target), [sharpness_map(matched)], [valid], bias_stops=0.5, tidy_radius=0)
    result = composite(target, [matched], label_weights(labels, [valid], feather_sigma=4.0))

    left = (slice(40, -40), slice(40, half - 60))
    right = (slice(40, -40), slice(half + 60, -40))
    assert sharpness_map(result)[right].mean() > 2.0 * sharpness_map(target)[right].mean()
    assert np.abs(result - target)[left].max() < 0.02


INNER = (slice(40, -40), slice(40, -40))


def test_agreement_map_is_one_for_identical_images() -> None:
    """The same textured picture agrees with itself everywhere away from the border."""
    img = make_texture(SHAPE, seed=10)
    out = agreement_map(img, img, lowpass_sigma=2.0, window_sigma=6.0)
    assert out.shape == SHAPE
    assert out.dtype == np.float32
    assert out[INNER].min() > 0.99


def test_agreement_map_ignores_gain_and_offset() -> None:
    """A brighter, lower-contrast copy still agrees."""
    img = make_texture(SHAPE, seed=11)
    out = agreement_map(img, 0.5 * img + 0.2, lowpass_sigma=2.0, window_sigma=6.0)
    assert out[INNER].min() > 0.99


def test_agreement_map_still_agrees_when_one_side_is_blurred() -> None:
    """A blurred copy of the target matches it, so a sharper source is never rejected for being sharper."""
    img = make_texture(SHAPE, seed=12)
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    out = agreement_map(img, blurred, lowpass_sigma=3.0, window_sigma=6.0)
    assert out[INNER].mean() > 0.9


def test_agreement_map_drops_for_shifted_and_unrelated_content() -> None:
    """A copy shifted by 8 columns and an unrelated texture both score low."""
    img = make_texture(SHAPE, seed=13)
    shifted = np.roll(img, 8, axis=1)
    unrelated = make_texture(SHAPE, seed=14)
    assert agreement_map(img, shifted, 2.0, 6.0)[INNER].mean() < 0.5
    assert abs(agreement_map(img, unrelated, 2.0, 6.0)[INNER].mean()) < 0.3


def test_agreement_map_is_near_zero_where_the_target_is_flat() -> None:
    """A flat target against a textured source (content only one frame sees) does not agree."""
    flat = np.full((*SHAPE, 3), 0.3, dtype=np.float32)
    textured = make_texture(SHAPE, seed=15)
    out = agreement_map(flat, textured, lowpass_sigma=2.0, window_sigma=6.0)
    assert np.abs(out[INNER]).max() < 0.05


def test_gain_map_is_the_ratio_that_match_gain_applies() -> None:
    """A source at half the target's level gets a gain near 2 per channel, and match_gain is source times that gain."""
    target = make_texture(SHAPE, seed=20) * 0.6 + 0.3
    source = target * np.array([0.5, 0.4, 0.8], dtype=np.float32)
    valid = np.ones(SHAPE, dtype=bool)
    gain = gain_map(target, source, valid, sigma=20.0)
    assert gain.shape == (*SHAPE, 3)
    assert np.allclose(gain[INNER].mean(axis=(0, 1)), [2.0, 2.5, 1.25], rtol=0.02)
    assert np.allclose(match_gain(target, source, valid, sigma=20.0), source * gain)


def test_masked_median_uses_only_the_layers_that_cover_a_pixel() -> None:
    """The median is per pixel and per channel over covering layers; two covering layers give their mean."""
    scales = np.array([1.0, 0.5, 0.25], dtype=np.float32)
    layers = [np.full((4, 2, 3), value, dtype=np.float32) * scales for value in (0.2, 0.8, 0.5, 0.9)]
    covered = [np.ones((4, 2), dtype=bool) for _ in layers]
    covered[3][:, 0] = False
    out = masked_median(layers, covered)
    assert out.shape == (4, 2, 3)
    assert np.allclose(out[:, 0], 0.5 * scales)
    assert np.allclose(out[:, 1], 0.65 * scales)


def test_masked_median_falls_back_to_the_first_layer_where_nothing_covers() -> None:
    """A pixel no layer covers takes the first layer's value instead of NaN."""
    layers = [np.full((4, 2, 3), 0.3, dtype=np.float32), np.full((4, 2, 3), 0.7, dtype=np.float32)]
    none = [np.zeros((4, 2), dtype=bool), np.zeros((4, 2), dtype=bool)]
    out = masked_median(layers, none)
    assert np.allclose(out, 0.3) and not np.isnan(out).any()
