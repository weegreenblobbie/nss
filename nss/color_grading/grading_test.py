"""Tests of the public API nss.color_grading.grading.Grading."""
from collections.abc import Callable

import numpy as np
import pytest

from nss.color_grading.grading import Grading

PURE_COLOURS = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1), "cyan": (0, 1, 1), "magenta": (1, 0, 1), "yellow": (1, 1, 0)}
TEAL_ORANGE = dict(sh_h=200, sh_s=60, mi_h=30, mi_s=20, hi_h=30, hi_s=60)


@pytest.fixture(scope="module")
def grading() -> Grading:
    """One Grading shared by the module's tests."""
    return Grading()


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    """A random non-square image, encoded 0..1."""
    return np.random.default_rng(7).random((13, 12, 3))


def test_identity_returns_input(grading: Grading, adobe_icc: bytes, image: np.ndarray) -> None:
    """With every Sat and Lum at 0 the input object itself is returned."""
    assert grading.apply(image, adobe_icc) is image
    assert grading.apply(image, adobe_icc, balance=80, blend=10, sh_h=200, mi_h=30, hi_h=120, gl_h=300) is image


def test_requires_icc(grading: Grading, image: np.ndarray) -> None:
    """No ICC profile raises ValueError."""
    with pytest.raises(ValueError, match="ICC profile"):
        grading.apply(image, None, sh_s=30)


def test_rejects_bad_last_dimension(grading: Grading, adobe_icc: bytes) -> None:
    """A last dimension other than 3 raises ValueError."""
    with pytest.raises(ValueError, match="expected"):
        grading.apply(np.zeros((5, 4, 4)), adobe_icc, sh_s=30)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(13, 12, 3), (29, 3)])
def test_dtype_and_shape_preserved(grading: Grading, adobe_icc: bytes, dtype: type, shape: tuple[int, ...]) -> None:
    """float32 / float64 and (H, W, 3) / (N, 3) come back unchanged."""
    img = np.random.default_rng(1).random(shape).astype(dtype)
    out = grading.apply(img, adobe_icc, **TEAL_ORANGE)
    assert out.shape == shape
    assert out.dtype == dtype
    assert np.abs(out - img).max() > 1e-3


EXTREMES = [
    dict(sh_h=200, sh_s=100, mi_h=30, mi_s=100, hi_h=120, hi_s=100),
    dict(sh_s=0, sh_l=100, mi_l=100, hi_l=100),
    dict(sh_s=0, sh_l=-100, mi_l=-100, hi_l=-100),
    dict(sh_h=0, sh_s=100, sh_l=100, hi_h=240, hi_s=100, hi_l=-100, gl_h=60, gl_s=100, gl_l=100),
]


@pytest.mark.parametrize("settings", EXTREMES)
@pytest.mark.parametrize("balance, blend", [(-100, 0), (100, 100), (-100, 100), (100, 0), (0, 50)])
def test_extreme_settings_finite_and_in_range(grading: Grading, adobe_icc: bytes, image: np.ndarray, settings: dict, balance: float, blend: float) -> None:
    """Extreme sliders keep the output finite and in 0..1."""
    out = grading.apply(image, adobe_icc, balance=balance, blend=blend, **settings)
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_teal_shadows_push_blue_over_red(grading: Grading, adobe_icc: bytes) -> None:
    """Teal shadows make a dark gray bluer than red."""
    dark = np.full((3, 2, 3), 0.15)
    out = grading.apply(dark, adobe_icc, sh_h=200, sh_s=60)
    assert np.all(out[..., 2] > out[..., 0])


def test_orange_highlights_push_red_over_blue(grading: Grading, adobe_icc: bytes) -> None:
    """Orange highlights make a light gray redder than blue."""
    bright = np.full((3, 2, 3), 0.85)
    out = grading.apply(bright, adobe_icc, hi_h=30, hi_s=60)
    assert np.all(out[..., 0] > out[..., 2])


def test_global_wheel_adds_to_tonal_wheels(grading: Grading, adobe_icc: bytes, image: np.ndarray) -> None:
    """The Global wheel changes the result of the tonal wheels."""
    tonal = grading.apply(image, adobe_icc, **TEAL_ORANGE)
    both = grading.apply(image, adobe_icc, gl_h=120, gl_s=25, gl_l=10, **TEAL_ORANGE)
    assert np.abs(both - tonal).max() > 1e-3


def test_neutral_ramp_stays_monotonic(grading: Grading, adobe_icc: bytes, ramp: Callable[..., np.ndarray]) -> None:
    """A graded neutral ramp still rises in every channel."""
    out = grading.apply(ramp((1, 1, 1)), adobe_icc, balance=-50, blend=50, **TEAL_ORANGE)
    assert np.all(np.diff(out, axis=0) >= 0.0)


def test_colour_ramps_have_no_dark_jumps(grading: Grading, adobe_icc: bytes, ramp: Callable[..., np.ndarray]) -> None:
    """Black -> pure colour ramps: no step above 400 DN (16-bit) between neighbours over the first 400 pixels.
    Dividing the cap window by a near-zero reference mask once produced 2000-3000 DN jumps there."""
    ramps = np.stack([ramp(c) for c in PURE_COLOURS.values()])          # (6, 2048, 3)
    out = grading.apply(ramps, adobe_icc, balance=-50, blend=50, sh_h=210, sh_s=60, sh_l=-15, mi_h=30, mi_s=40, mi_l=10, hi_h=45, hi_s=25, hi_l=25)
    steps = np.abs(np.diff(out[:, :400] * 65535.0, axis=1)).max(axis=(1, 2))
    worst = dict(zip(PURE_COLOURS, steps.round(1)))
    assert np.all(steps <= 400.0), worst


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_global_only_dtype_preserved(grading: Grading, adobe_icc: bytes, dtype: type) -> None:
    """Global alone keeps the dtype and shape and changes the image."""
    img = np.random.default_rng(5).random((13, 12, 3)).astype(dtype) ** 3
    out = grading.apply(img, adobe_icc, gl_h=200, gl_s=60, gl_l=-20)
    assert out.dtype == dtype
    assert out.shape == (13, 12, 3)
    assert np.abs(out - img).max() > 1e-3
