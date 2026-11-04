"""Tests of the Random Shift functions in nss.color_explore.random_shift: mixer band weights and damping, the wheel targets, the enabled
panels, clamping and the explore grid."""
import numpy as np

from nss.color_grading.grading import Grading
from nss.color_explore.random_shift import ShiftColors, explore_states, mixer_weights, random_shift, shift_colors, tonal_zone
from nss.color_explore.state import StateNode, create_default_state
from nss.color_mixer.hue_remap import LINEAR_PROPHOTO


class FixedDraws:
    """Stands in for np.random.Generator: every uniform draw is r and every integer draw is 0."""

    def __init__(self, r: float) -> None:
        """Every draw returns r."""
        self.r = r

    def uniform(self, low: float, high: float, size: int) -> np.ndarray:
        """size draws, all r."""
        return np.full(size, self.r)

    def integers(self, high: int) -> int:
        """Always the first choice."""
        return 0


def _state() -> StateNode:
    """A state with distinct non-zero mixer and wheel values."""
    state = create_default_state()
    state["mixer_hue"] = [10.0, 20.0, -30.0, 40.0, 0.0, 5.0, -6.0, 7.0]
    state["mixer_sat"] = [-1.0, 2.0, 3.0, -4.0, 5.0, 6.0, 7.0, 8.0]
    state["mixer_lum"] = [11.0, -12.0, 13.0, 14.0, 15.0, -16.0, 17.0, 18.0]
    state.update(shadow_hue=200.0, shadow_sat=0.3, shadow_light=-0.4, midtone_hue=100.0, midtone_sat=0.2, midtone_light=0.2,
                 highlight_hue=50.0, highlight_sat=0.1, highlight_light=0.3)
    return state


def _colors(mixer_hue: list, wheel_hue: list, sat: list, value: list, zone: list) -> ShiftColors:
    """ShiftColors from lists."""
    return ShiftColors(*(np.array(v, np.float64) for v in (mixer_hue, wheel_hue, sat, value)), np.array(zone))


ORANGE = _colors([30.0], [40.0], [0.5], [1.0], [1])        # on the orange band's centre, in the midtones


def test_weights_follow_the_band_and_damp_grey_and_dark_colours() -> None:
    """A hue on a band centre weighs 1 there; halfway between red (0) and orange (30) each weighs 0.5; sat 0.05 halves it, value 0.02 scales by 0.2."""
    assert np.allclose(mixer_weights(_colors([30.0], [0.0], [1.0], [1.0], [0])), np.eye(8)[1])
    assert np.allclose(mixer_weights(_colors([15.0], [0.0], [1.0], [1.0], [0])), [0.5, 0.5, 0, 0, 0, 0, 0, 0])
    assert np.allclose(mixer_weights(_colors([15.0], [0.0], [0.05], [1.0], [0])), [0.25, 0.25, 0, 0, 0, 0, 0, 0])
    assert np.allclose(mixer_weights(_colors([30.0], [0.0], [1.0], [0.02], [0])), 0.2 * np.eye(8)[1])


def test_colours_on_one_band_take_the_largest_weight_not_the_sum() -> None:
    """Orange gets 1 from the colour on its centre and 0.5 from the one halfway to red: 1, not 1.5."""
    assert np.allclose(mixer_weights(_colors([30.0, 15.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0, 0])), [0.5, 1.0, 0, 0, 0, 0, 0, 0])


def test_random_shift_values() -> None:
    """r = 1, strength 0.2: orange's Hue / Sat / Lum move by 40 (2x), other bands stay; the midtone wheel goes to the orange's wheel hue 40 + 72 and
    moves from sat 0.2 and light 0.2 by 0.2 each; Shadows and Highlights have no colours and stay."""
    state = _state()
    out = random_shift(state, ORANGE, 0.2, True, True, FixedDraws(1.0))
    assert out["mixer_hue"] == [10.0, 60.0, -30.0, 40.0, 0.0, 5.0, -6.0, 7.0]
    assert out["mixer_sat"] == [-1.0, 42.0, 3.0, -4.0, 5.0, 6.0, 7.0, 8.0]
    assert out["mixer_lum"] == [11.0, 28.0, 13.0, 14.0, 15.0, -16.0, 17.0, 18.0]
    assert (out["midtone_hue"], out["midtone_sat"], out["midtone_light"]) == (112.0, 0.4, 0.4)
    for key in ("shadow_hue", "shadow_sat", "shadow_light", "highlight_hue", "highlight_sat", "highlight_light"):
        assert out[key] == state[key]
    assert state == _state()                     # the input is not changed


def test_unchecked_panels_are_untouched() -> None:
    """Mixer off keeps the 24 mixer values; Grading off keeps every wheel."""
    state = _state()
    rng = np.random.default_rng(3)
    no_mixer = random_shift(state, ORANGE, 0.5, False, True, rng)
    assert all(no_mixer[f"mixer_{kind}"] == state[f"mixer_{kind}"] for kind in ("hue", "sat", "lum"))
    assert no_mixer["midtone_hue"] != state["midtone_hue"]
    no_grading = random_shift(state, ORANGE, 0.5, True, False, rng)
    assert {k: v for k, v in no_grading.items() if not k.startswith("mixer_")} == {k: v for k, v in state.items() if not k.startswith("mixer_")}
    assert no_grading["mixer_hue"] != state["mixer_hue"]


def test_values_are_clamped_and_hue_wraps() -> None:
    """Strength 1, r = 1: mixer 95 + 200, wheel sat 0.2 + 1 and light 0.9 + 1 clamp to 100, 1 and 1; r = -1: Lum -12 - 200 clamps to -100,
    wheel sat 0.2 - 1 clamps to 0 and light -0.9 - 1 to -1.  Strength 0.2: hue 350 + 72 wraps to 62."""
    state = _state()
    state["mixer_hue"][1] = 95.0
    state["midtone_light"] = 0.9
    up = random_shift(state, ORANGE, 1.0, True, True, FixedDraws(1.0))
    assert up["mixer_hue"][1] == 100.0 and up["midtone_sat"] == 1.0 and up["midtone_light"] == 1.0
    state["midtone_light"] = -0.9
    down = random_shift(state, ORANGE, 1.0, True, True, FixedDraws(-1.0))
    assert down["mixer_sat"][1] == -100.0 and down["mixer_lum"][1] == -100.0 and down["midtone_sat"] == 0.0 and down["midtone_light"] == -1.0
    wrapped = random_shift(state, _colors([30.0], [350.0], [0.5], [1.0], [1]), 0.2, False, True, FixedDraws(1.0))
    assert wrapped["midtone_hue"] == 62.0


def test_zero_strength_moves_only_the_wheel_hues_to_a_colour_of_their_zone() -> None:
    """Strength 0: the mixer and every sat and light stay; each wheel with colours takes the wheel hue of one of its own zone's colours (both
    midtone colours turn up over the seeds) and the wheel without colours keeps its hue."""
    state = _state()
    colors = _colors([30.0, 100.0, 200.0], [20.0, 250.0, 90.0], [0.1, 0.6, 0.4], [1.0, 1.0, 1.0], [1, 1, 0])
    midtone_hues, shadow_hues = set(), set()
    for seed in range(20):
        out = random_shift(state, colors, 0.0, True, True, np.random.default_rng(seed))
        assert all(out[f"mixer_{kind}"] == state[f"mixer_{kind}"] for kind in ("hue", "sat", "lum"))
        assert all(out[f"{p}_{k}"] == state[f"{p}_{k}"] for p in ("shadow", "midtone", "highlight") for k in ("sat", "light"))
        assert out["highlight_hue"] == state["highlight_hue"]
        midtone_hues.add(out["midtone_hue"])
        shadow_hues.add(out["shadow_hue"])
    assert midtone_hues == {20.0, 250.0} and shadow_hues == {90.0}


def test_seeded_generator_is_reproducible() -> None:
    """The same seed gives the same shift; another seed a different one."""
    state = _state()
    first = random_shift(state, ORANGE, 0.3, True, True, np.random.default_rng(11))
    assert random_shift(state, ORANGE, 0.3, True, True, np.random.default_rng(11)) == first
    assert random_shift(state, ORANGE, 0.3, True, True, np.random.default_rng(12)) != first


def test_tonal_zone_and_hsv_of_colours() -> None:
    """Dark, middle and bright greys fall in Shadows, Midtones and Highlights; sat and value are the HSV ones of the encoded colour."""
    rgb = np.array([[0.01, 0.01, 0.01], [0.18, 0.18, 0.18], [0.9, 0.9, 0.9], [0.8, 0.4, 0.2]])
    assert list(tonal_zone(Grading(), rgb[:3], LINEAR_PROPHOTO, 0.0, 0.5)) == [0, 1, 2]
    colors = shift_colors(np.array([1.0, 2.0, 3.0, 25.0]), np.array([4.0, 5.0, 6.0, 30.0]), rgb, LINEAR_PROPHOTO, Grading(), 0.0, 0.5)
    assert np.allclose(colors.sat, [0.0, 0.0, 0.0, 0.75]) and np.allclose(colors.value, [0.01, 0.18, 0.9, 0.8])
    assert list(colors.mixer_hue) == [1.0, 2.0, 3.0, 25.0] and list(colors.wheel_hue) == [4.0, 5.0, 6.0, 30.0]


def test_explore_states_leave_the_middle_alone() -> None:
    """Index 4 is the centre unchanged; the 8 others are independent shifts of it."""
    state = _state()
    grid = explore_states(state, ORANGE, 0.3, True, True, np.random.default_rng(2))
    assert len(grid) == 9 and grid[4] == state and grid[4] is not state
    outer = [grid[i] for i in range(9) if i != 4]
    assert all(s != state for s in outer)
    assert len({s["midtone_hue"] for s in outer}) > 1
