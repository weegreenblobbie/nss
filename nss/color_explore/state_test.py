"""Tests of nss.color_explore.state: the neutral default state."""
from nss.color_explore.state import StateNode, create_default_state


def test_default_state_is_neutral_with_the_default_hues() -> None:
    """Every StateNode key is set; all wheels have Sat 0 and Light 0, Blending 0.5, Balance and Rotation 0, hues 240 / 120 / 60 / 0."""
    state = create_default_state()
    assert set(state) == set(StateNode.__annotations__)
    assert all(state[f"{wheel}_{key}"] == 0.0 for wheel in ("shadow", "midtone", "highlight", "global") for key in ("sat", "light"))
    assert (state["blending"], state["balance"], state["rotation"]) == (0.5, 0.0, 0.0)
    assert (state["shadow_hue"], state["midtone_hue"], state["highlight_hue"], state["global_hue"]) == (240.0, 120.0, 60.0, 0.0)


def test_each_call_returns_a_fresh_state() -> None:
    """Changing one default state does not change the next."""
    first = create_default_state()
    first["shadow_sat"] = 0.4
    assert create_default_state()["shadow_sat"] == 0.0
