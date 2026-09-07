"""Tests of nss.color_gui.grading_state: State Node -> Grading.apply arguments."""
import numpy as np
import pytest

from nss.color_math import create_default_state
from nss.color_gui.grading_state import grading_kwargs, grade
from nss.color_grading.grading import Grading
from nss.icc import gamma_icc_bytes

ICC = gamma_icc_bytes("Adobe RGB (1998)", [(0.60974, 0.31111, 0.01947), (0.20528, 0.62567, 0.06087), (0.14919, 0.06322, 0.74457)], 563 / 256)


def test_default_state_is_neutral() -> None:
    """The default state maps to neutral sliders (Blending 50, every Sat and Lum 0)."""
    kw = grading_kwargs(create_default_state())
    assert kw["balance"] == 0.0 and kw["blend"] == 50.0
    for p in ("sh", "mi", "hi", "gl"):
        assert kw[f"{p}_s"] == 0.0 and kw[f"{p}_l"] == 0.0
    assert kw["gl_h"] == 0.0


def test_units_and_rotation() -> None:
    """GUI units become Lightroom units and Rotation is added to every hue, Global included."""
    state = create_default_state()
    state.update(shadow_hue=210.0, shadow_sat=0.6, shadow_light=-0.15, blending=0.7, balance=-0.5, rotation=20.0,
                 global_hue=350.0, global_sat=0.4, global_light=0.1)
    kw = grading_kwargs(state)
    assert kw["sh_h"] == pytest.approx(230.0) and kw["sh_s"] == pytest.approx(60.0) and kw["sh_l"] == pytest.approx(-15.0)
    assert kw["blend"] == pytest.approx(70.0) and kw["balance"] == pytest.approx(-50.0)
    assert kw["gl_h"] == pytest.approx(10.0) and kw["gl_s"] == pytest.approx(40.0) and kw["gl_l"] == pytest.approx(10.0)


def test_legacy_state_without_global() -> None:
    """A state saved before the Global wheel maps to a neutral Global."""
    state = create_default_state()
    for k in ("global_hue", "global_sat", "global_light"):
        del state[k]
    kw = grading_kwargs(state)
    assert (kw["gl_h"], kw["gl_s"], kw["gl_l"]) == (0.0, 0.0, 0.0)


def test_grade_passes_global_to_grading(monkeypatch: pytest.MonkeyPatch) -> None:
    """grade() hands the profile and the gl_* values to Grading.apply."""
    seen = {}
    monkeypatch.setattr(Grading, "apply", lambda self, img, icc, **kw: seen.update(kw, icc=icc) or img)
    state = create_default_state()
    state.update(global_hue=120.0, global_sat=0.25, global_light=-0.3)
    grade(Grading(), np.zeros((4, 3, 3)), ICC, state)
    assert seen["icc"] is ICC
    assert (seen["gl_h"], seen["gl_s"], seen["gl_l"]) == pytest.approx((120.0, 25.0, -30.0))


def test_grade_global_changes_image() -> None:
    """A neutral state returns the input; a Global Sat changes it and keeps dtype and shape."""
    img = np.random.default_rng(4).random((7, 5, 3)).astype(np.float32)
    state = create_default_state()
    assert grade(Grading(), img, ICC, state) is img
    state["global_sat"] = 0.5
    out = grade(Grading(), img, ICC, state)
    assert out.dtype == np.float32 and out.shape == (7, 5, 3)
    assert np.abs(out - img).max() > 1e-3
