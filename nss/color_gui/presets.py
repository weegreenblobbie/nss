"""
Grading presets as JSON:  {"version": 1, "state": {<StateNode keys>}}.
Loading starts from create_default_state(), so keys missing from older files (e.g. the Global wheel's global_*) take their defaults and unknown
keys are ignored.
"""
import json

from nss.color_math import StateNode, create_default_state

VERSION = 1


def preset_from_state(state: StateNode) -> dict:
    """The JSON document of a state: the version and every State Node key."""
    return {"version": VERSION, "state": {k: state[k] for k in create_default_state() if k in state}}


def state_from_preset(preset: dict) -> StateNode:
    """A full State Node from a preset document; missing keys take their defaults, unknown keys are ignored."""
    state = create_default_state()
    for k, v in preset.get("state", {}).items():
        if k in state:
            state[k] = v if k == "harmony_mode" else float(v)
    return state


def save_preset(path: str, state: StateNode) -> None:
    """Writes the state to path as a JSON preset."""
    with open(path, "w") as f:
        json.dump(preset_from_state(state), f, indent=2)


def load_preset(path: str) -> StateNode:
    """Reads a JSON preset from path."""
    with open(path) as f:
        return state_from_preset(json.load(f))
