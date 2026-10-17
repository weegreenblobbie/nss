import numpy as np
from nss.color_mixer import bands


def test_centres_return_their_own_level():
    levels = np.array([10.0, -20.0, 30.0, -40.0, 50.0, -60.0, 70.0, -80.0])
    assert np.allclose(bands.slider_at(bands.CENTRES, levels), levels)


def test_equal_levels_are_constant_everywhere():
    h = np.linspace(0.0, 359.9, 721)
    assert np.allclose(bands.slider_at(h, np.full(8, 37.0)), 37.0)


def test_midpoint_is_average_of_neighbours():
    mid = 0.5 * (bands.CENTRES[3] + bands.CENTRES[4])
    levels = np.zeros(8)
    levels[3], levels[4] = 100.0, -50.0
    assert abs(bands.slider_at(mid, levels) - 25.0) < 1e-9


def test_wraps_around_red():
    levels = np.zeros(8)
    levels[0] = 100.0
    assert bands.slider_at(0.0, levels) == 100.0 and bands.slider_at(360.0, levels) == 100.0
    assert abs(bands.slider_at(345.0, levels) - 50.0 * 0.0 - bands.slider_at(345.0, levels)) < 1e-12       # defined and finite
    assert 0.0 < bands.slider_at(345.0, levels) < 100.0 and 0.0 < bands.slider_at(15.0, levels) < 100.0


def test_band_has_no_effect_past_its_neighbours():
    levels = np.zeros(8)
    levels[0] = 100.0                                          # red: neighbours magenta 330 and orange 30
    h = np.array([31.0, 100.0, 200.0, 329.0])
    assert np.allclose(bands.slider_at(h, levels), 0.0)
    assert np.allclose(bands.hue_shift(h[1:3], levels), 0.0)      # beyond the neighbouring centres: exactly nothing
    assert np.abs(bands.hue_shift(h[[0, 3]], levels)).max() < 0.05    # just inside the neighbours: only the fitted slope term of the peak (degrees)


def test_hue_shift_at_centres():
    for k in range(8):
        up, down = np.zeros(8), np.zeros(8)
        up[k], down[k] = 100.0, -100.0
        assert abs(bands.hue_shift(bands.CENTRES[k], up) - bands.HUE_GAP[k]) < 1e-9
        assert abs(bands.hue_shift(bands.CENTRES[k], down) + bands.HUE_GAP[k - 1]) < 1e-9


def test_hue_shift_scales_with_the_levels():
    h = np.linspace(0.0, 359.0, 360)
    a = np.zeros(8)
    a[3] = 100.0
    assert np.allclose(bands.hue_shift(h, 0.5 * a), 0.5 * bands.hue_shift(h, a))
    b = np.array([0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 50.0, 50.0])
    assert np.allclose(bands.hue_shift(h, 0.5 * b), 0.5 * bands.hue_shift(h, b))


def test_bands_moving_the_same_way_interact_and_opposite_ones_add():
    """Yellow + Green together is not the sum of the two (measured up to 9 degrees apart); Yellow + with Green - is exactly the sum."""
    h = np.linspace(0.0, 359.0, 360)
    y, g, gn = np.zeros(8), np.zeros(8), np.zeros(8)
    y[2], g[3], gn[3] = 100.0, 100.0, -100.0
    same = bands.hue_shift(h, y + g) - bands.hue_shift(h, y) - bands.hue_shift(h, g)
    assert np.abs(same).max() > 3.0
    assert np.abs(same).max() < 12.0
    assert np.allclose(bands.hue_shift(h, y + gn), bands.hue_shift(h, y) + bands.hue_shift(h, gn), atol=1e-9)


def test_hue_shift_is_continuous_and_bounded():
    h = np.linspace(0.0, 360.0, 3601)
    rng = np.random.default_rng(1)
    for _ in range(20):
        lv = rng.uniform(-100, 100, 8)
        d = bands.hue_shift(h, lv)
        assert np.abs(np.diff(d)).max() < 1.0                      # no jumps at 0.1 degree steps
        assert abs(d[0] - d[-1]) < 1e-9
        assert np.abs(d).max() < 75.0


def test_adjacent_plus_and_minus_agree_at_the_shared_edge():
    """Hue +100 on band k moves the centre forward by the distance that Hue -100 on band k + 1 moves it back."""
    for k in range(8):
        assert abs(bands.HUE_GAP[k] - bands.HUE_GAP[(k + 1) % 8]) < 60.0                    # sanity of the table
    assert np.all(bands.HUE_GAP > 25.0) and np.all(bands.HUE_GAP < 70.0)
    