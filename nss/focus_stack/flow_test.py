"""Tests for nss.focus_stack.flow."""
import cv2
import numpy as np

from nss.focus_stack.flow import estimate_flow, warp_with_flow
from nss.focus_stack.register_test import make_texture

SHAPE = (480, 640)  # rows, cols: non-square
IDENTITY = np.eye(3)


def make_displacement(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """A smooth (dx, dy) field: dx runs 1..5 px across the columns, dy about 2 px with a gentle ripple."""
    rows, cols = shape
    col = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :]
    row = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    dx = np.broadcast_to(1.0 + 4.0 * col, shape).astype(np.float32)
    dy = (2.0 + 1.0 * np.sin(2.0 * np.pi * row) + 0.0 * col).astype(np.float32)
    return dx, dy


def shifted_target(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The target a camera would see: target(x) = source(x + displacement)."""
    dx, dy = make_displacement(source.shape[:2])
    cols, rows = np.meshgrid(np.arange(source.shape[1], dtype=np.float32), np.arange(source.shape[0], dtype=np.float32))
    target = cv2.remap(source, cols + dx, rows + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return target, dx, dy


def test_estimate_flow_recovers_known_displacement() -> None:
    """The flow is (rows, cols, 2) and matches the (dx, dy) used to build the target."""
    source = make_texture(SHAPE, seed=5)
    target, dx, dy = shifted_target(source)
    flow = estimate_flow(target, source, max_dim=2000)
    assert flow.shape == (*SHAPE, 2)
    inner = (slice(40, -40), slice(40, -40))
    assert np.median(np.abs(flow[..., 0] - dx)[inner]) < 0.5
    assert np.median(np.abs(flow[..., 1] - dy)[inner]) < 0.5


def test_estimate_flow_ignores_gain_and_offset() -> None:
    """A brighter, lower-contrast target is still matched to the same displacement."""
    source = make_texture(SHAPE, seed=6)
    target, dx, dy = shifted_target(source)
    target = 0.6 * target + 0.2
    flow = estimate_flow(target, source, max_dim=2000)
    inner = (slice(40, -40), slice(40, -40))
    assert np.median(np.abs(flow[..., 0] - dx)[inner]) < 0.5
    assert np.median(np.abs(flow[..., 1] - dy)[inner]) < 0.5


def test_estimate_flow_at_reduced_size_is_in_full_size_pixels() -> None:
    """Flow computed on downsampled copies is upsampled and rescaled to full-size pixels."""
    source = make_texture((960, 1280), seed=7)
    target, dx, dy = shifted_target(source)
    flow = estimate_flow(target, source, max_dim=640)
    assert flow.shape == (960, 1280, 2)
    inner = (slice(80, -80), slice(80, -80))
    assert np.median(np.abs(flow[..., 0] - dx)[inner]) < 1.0
    assert np.median(np.abs(flow[..., 1] - dy)[inner]) < 1.0


def test_warp_with_flow_moves_source_closer_to_target() -> None:
    """Warping by the estimated flow leaves a smaller error than the unwarped source."""
    source = make_texture(SHAPE, seed=8)
    target, _, _ = shifted_target(source)
    flow = estimate_flow(target, source, max_dim=2000)
    warped = warp_with_flow(source, IDENTITY, flow)
    inner = (slice(40, -40), slice(40, -40))
    before = np.abs(source - target)[inner].mean()
    after = np.abs(warped - target)[inner].mean()
    assert warped.shape == target.shape
    assert after < 0.5 * before


def test_warp_with_flow_applies_matrix_then_flow() -> None:
    """With a (+5 col, +3 row) matrix and a flow of (-1, +2), source (col 8, row 6) lands at target (col 14, row 7)."""
    source = np.zeros((13, 12, 3), dtype=np.float32)
    source[6, 8] = (0.25, 0.5, 0.75)
    matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
    flow = np.zeros((9, 20, 2), dtype=np.float32)
    flow[..., 0] = -1.0
    flow[..., 1] = 2.0
    out = warp_with_flow(source, matrix, flow)
    assert out.shape == (9, 20, 3)
    assert np.allclose(out[7, 14], (0.25, 0.5, 0.75))
    assert np.count_nonzero(out) == 3


def test_warp_with_flow_strip_equals_the_same_rows_of_the_whole_warp() -> None:
    """Warping rows 100..260 with row_offset=100 gives exactly those rows of the whole-picture warp."""
    source = make_texture(SHAPE, seed=9)
    dx, dy = make_displacement(SHAPE)
    flow = np.stack([dx, dy], axis=-1)
    matrix = np.array([[0.9, -0.05, 12.0], [0.04, 0.95, -7.0], [1e-5, 2e-5, 1.0]])
    whole = warp_with_flow(source, matrix, flow)
    strip = warp_with_flow(source, matrix, flow[100:260], row_offset=100)
    assert strip.shape == (160, SHAPE[1], 3)
    assert np.abs(whole[100:260]).max() > 0.0
    assert np.array_equal(strip, whole[100:260])
