"""Tests of nss.color_gui.presets: JSON preset save / load, including presets from before the Global wheel."""
import json
from pathlib import Path

from nss.color_explore.state import StateNode, create_default_state
from nss.color_gui.presets import save_preset, load_preset, state_from_preset, preset_from_state, VERSION


def _graded_state() -> StateNode:
    """A state with distinct, non-default values in every group, Global included."""
    state = create_default_state()
    state.update(shadow_hue=210.0, shadow_sat=0.6, shadow_light=-0.15, midtone_sat=0.4, highlight_hue=45.0, highlight_sat=0.25,
                 blending=0.7, balance=-0.5, rotation=15.0, global_hue=30.0, global_sat=0.4, global_light=0.1)
    return state


def test_round_trip(tmp_path: Path) -> None:
    """save_preset then load_preset gives the same state."""
    path = str(tmp_path / "look.json")
    state = _graded_state()
    save_preset(path, state)
    assert load_preset(path) == state


def test_file_format(tmp_path: Path) -> None:
    """The file is {"version": 1, "state": {...}} and carries the Global values."""
    path = str(tmp_path / "look.json")
    save_preset(path, _graded_state())
    data = json.load(open(path))
    assert data["version"] == VERSION
    assert data["state"]["global_hue"] == 30.0 and data["state"]["global_sat"] == 0.4 and data["state"]["global_light"] == 0.1


def test_legacy_file_without_global_defaults_to_zero(tmp_path: Path) -> None:
    """A preset without global_* keys loads with Global 0 and keeps its other values."""
    legacy = preset_from_state(_graded_state())
    for k in ("global_hue", "global_sat", "global_light"):
        del legacy["state"][k]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(legacy))
    state = load_preset(str(path))
    assert (state["global_hue"], state["global_sat"], state["global_light"]) == (0.0, 0.0, 0.0)
    assert state["shadow_hue"] == 210.0 and state["balance"] == -0.5


def test_unknown_keys_ignored_and_values_are_floats() -> None:
    """Unknown keys are dropped (also the old harmony_mode and step_size of earlier presets), numbers become floats."""
    state = state_from_preset({"version": 1, "state": {"shadow_sat": "0.3", "not_a_key": 5, "harmony_mode": "Triadic", "step_size": 0.2}})
    assert state["shadow_sat"] == 0.3
    assert "not_a_key" not in state
    assert "harmony_mode" not in state and "step_size" not in state
