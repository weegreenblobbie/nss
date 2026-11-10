"""Tests for nss.focus_stack.stack and the command line in nss.focus_stack.__main__."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from nss.focus_stack.__main__ import main
from nss.focus_stack.register import RegistrationError
from nss.focus_stack.register_test import make_homography, make_texture
from nss.focus_stack.sharpness import sharpness_map
from nss.focus_stack.stack import render_full, render_strip, resample_rows, stack_frames
from nss.icc import gamma_icc_bytes
from nss.tiff_file import TiffFile

SHAPE = (960, 1280)  # rows, cols: non-square
THIRD = SHAPE[1] // 3
ICC = gamma_icc_bytes(
    "Adobe RGB (1998)",
    [(0.60974, 0.31111, 0.01947), (0.20528, 0.62567, 0.06087), (0.14919, 0.06322, 0.74457)],
    563 / 256,
)


def sharp_only_in(scene: np.ndarray, columns: slice) -> np.ndarray:
    """`scene` blurred everywhere except the given columns."""
    out = cv2.GaussianBlur(scene, (0, 0), 2.5)
    out[:, columns] = scene[:, columns]
    return out


def make_scene(seed: int) -> np.ndarray:
    """A textured float32 RGB scene with mid-range brightness."""
    return make_texture(SHAPE, seed=seed) * 0.6 + 0.3


def make_pair(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A target whose right half is blurred and a darker, sharp source of the same scene seen through a homography."""
    scene = make_scene(seed)
    target = cv2.warpPerspective(scene, make_homography(), (SHAPE[1], SHAPE[0]), flags=cv2.INTER_LINEAR)
    half = SHAPE[1] // 2
    target[:, half:] = cv2.GaussianBlur(target, (0, 0), 2.5)[:, half:]
    return target, 0.8 * scene


def make_thirds(seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """A target sharp only in the middle third and two darker frames: one sharp in the left and middle thirds, one in the middle and right."""
    scene = make_scene(seed)
    target = cv2.warpPerspective(
        sharp_only_in(scene, slice(THIRD, 2 * THIRD)), make_homography(), (SHAPE[1], SHAPE[0]), flags=cv2.INTER_LINEAR
    )
    return target, [0.8 * sharp_only_in(scene, slice(0, 2 * THIRD)), 0.9 * sharp_only_in(scene, slice(THIRD, SHAPE[1]))]


def test_stack_frames_sharpens_the_blurred_half_and_leaves_the_rest() -> None:
    """With one frame the right half gains sharpness, the left half stays as in the target, and the sizes are the working size."""
    target, source = make_pair(seed=1)
    out = stack_frames(target, [source], work_dim=1280)
    assert out.result.shape == target.shape
    assert out.labels.shape == SHAPE
    assert out.weights.shape == (2, *SHAPE)
    assert out.frames[0].inliers >= 30
    left = (slice(100, -100), slice(150, 500))
    right = (slice(100, -100), slice(800, 1150))
    assert sharpness_map(out.result)[right].mean() > 1.5 * sharpness_map(target)[right].mean()
    assert np.abs(out.result - target)[left].max() < 0.02
    assert 0.0 < out.replaced_fraction < 1.0


def test_stack_frames_takes_each_region_from_the_frame_that_is_sharp_there() -> None:
    """Target sharp in the middle, frame 1 sharp on the left, frame 2 on the right: labels follow and all three regions are sharp."""
    target, sources = make_thirds(seed=9)
    out = stack_frames(target, sources, work_dim=1280)
    left = (slice(100, 600), slice(100, 300))
    middle = (slice(100, 600), slice(450, 640))
    right = (slice(100, 600), slice(780, 1000))
    assert (out.labels[left] == 1).mean() > 0.8
    assert (out.labels[middle] == 0).mean() > 0.8
    assert (out.labels[right] == 2).mean() > 0.8
    sharp_target, sharp_result = sharpness_map(target), sharpness_map(out.result)
    assert sharp_result[left].mean() > 1.5 * sharp_target[left].mean()
    assert sharp_result[right].mean() > 1.5 * sharp_target[right].mean()
    assert np.abs(out.result - target)[middle].max() < 0.02


def test_stack_frames_raises_for_unrelated_pictures() -> None:
    """Two unrelated textures abort the run with a RegistrationError."""
    with pytest.raises(RegistrationError):
        stack_frames(make_texture(SHAPE, seed=2), [make_texture(SHAPE, seed=3)], work_dim=1280)


def test_render_full_matches_the_working_result_at_the_target_size() -> None:
    """Rendered at the target's size, the result is sharper on the blurred halves and, shrunk to the working size, equals the preview."""
    target, sources = make_thirds(seed=7)
    plan = stack_frames(target, sources, work_dim=960)
    assert plan.result.shape == (720, 960, 3)
    full = render_full(target, sources, plan.frames, plan.weights, tile_rows=300)
    assert full.shape == target.shape
    left = (slice(100, 600), slice(100, 300))
    assert sharpness_map(full)[left].mean() > 1.5 * sharpness_map(target)[left].mean()
    shrunk = cv2.resize(full, (960, 720), interpolation=cv2.INTER_AREA)
    inner = (slice(40, -40), slice(40, -40))
    assert np.abs(shrunk - plan.result)[inner].mean() < 0.01
    assert np.abs(full - target)[100:600, 450:640].max() < 0.02


def write_tif(path: Path, image: np.ndarray) -> None:
    """Saves a 0..1 float image as 16-bit TIFF with the Adobe RGB profile."""
    TiffFile.from_array(image, icc=ICC).saveas(str(path))


def test_main_writes_result_with_target_profile_and_debug_images(tmp_path: Path) -> None:
    """A run with two frames returns 0, writes the picture at the working size with the target's ICC profile, and the debug images."""
    target, sources = make_thirds(seed=9)
    write_tif(tmp_path / "target.tif", target)
    write_tif(tmp_path / "one.tif", sources[0])
    write_tif(tmp_path / "two.tif", sources[1])
    code = main([
        str(tmp_path / "target.tif"), str(tmp_path / "one.tif"), str(tmp_path / "two.tif"),
        "-o", str(tmp_path / "out.tif"), "--work-dim", "1280", "--debug-dir", str(tmp_path / "debug"),
    ])
    assert code == 0
    out = TiffFile().read(str(tmp_path / "out.tif"))
    assert out.array.shape == (*SHAPE, 3)
    assert out.icc_name == "Adobe RGB (1998)"
    assert not np.allclose(out.array, TiffFile().read(str(tmp_path / "target.tif")).array, atol=1e-3)
    assert sorted(p.name for p in (tmp_path / "debug").iterdir()) == [
        "agreement_1.png", "agreement_2.png", "aligned_1.png", "aligned_2.png", "labels.png", "target.png",
    ]


def test_main_full_res_writes_the_target_size(tmp_path: Path) -> None:
    """--full-res writes the result at the target's own size with its ICC profile."""
    target, source = make_pair(seed=8)
    write_tif(tmp_path / "target.tif", target)
    write_tif(tmp_path / "source.tif", source)
    code = main([
        str(tmp_path / "target.tif"), str(tmp_path / "source.tif"),
        "-o", str(tmp_path / "out.tif"), "--work-dim", "960", "--full-res",
    ])
    assert code == 0
    out = TiffFile().read(str(tmp_path / "out.tif"))
    assert out.array.shape == (*SHAPE, 3)
    assert out.icc_name == "Adobe RGB (1998)"


def test_main_aborts_without_writing_when_registration_fails(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Unrelated pictures return 1, say why, and leave no output file."""
    write_tif(tmp_path / "target.tif", make_texture(SHAPE, seed=5))
    write_tif(tmp_path / "source.tif", make_texture(SHAPE, seed=6))
    code = main([str(tmp_path / "target.tif"), str(tmp_path / "source.tif"), "-o", str(tmp_path / "out.tif")])
    assert code == 1
    assert not (tmp_path / "out.tif").exists()
    assert "registration failed" in capsys.readouterr().err


def test_resample_rows_matches_cv2_resize() -> None:
    """The rows of a strip equal the same rows of a whole bilinear resize, for 1, 2 and 3 channel arrays."""
    rng = np.random.default_rng(3)
    for channels in (None, 2, 3):
        small = rng.random((37, 53) if channels is None else (37, 53, channels)).astype(np.float32)
        whole = cv2.resize(small, (211, 149), interpolation=cv2.INTER_LINEAR)
        strip = resample_rows(small, (149, 211), 40, 97)
        assert strip.shape == whole[40:97].shape
        assert np.allclose(strip, whole[40:97], atol=1e-5)


def test_render_full_tiles_give_the_same_picture_as_one_tile() -> None:
    """Strips of 250 rows (the last one shorter) reproduce the single-strip render to within rounding."""
    target, sources = make_thirds(seed=7)
    plan = stack_frames(target, sources, work_dim=960)
    untiled = render_full(target, sources, plan.frames, plan.weights, tile_rows=SHAPE[0])
    tiled = render_full(target, sources, plan.frames, plan.weights, tile_rows=250)
    assert tiled.shape == untiled.shape
    assert np.abs(untiled - target).max() > 0.05
    assert np.abs(tiled - untiled).max() < 1e-4


def test_main_tile_rows_does_not_change_the_result(tmp_path: Path) -> None:
    """--tile-rows 300 and --tile-rows 960 write the same picture to within one 16-bit step."""
    target, sources = make_thirds(seed=9)
    write_tif(tmp_path / "target.tif", target)
    write_tif(tmp_path / "one.tif", sources[0])
    write_tif(tmp_path / "two.tif", sources[1])
    for rows in ("300", "960"):
        code = main([
            str(tmp_path / "target.tif"), str(tmp_path / "one.tif"), str(tmp_path / "two.tif"),
            "-o", str(tmp_path / f"out{rows}.tif"), "--work-dim", "960", "--full-res", "--tile-rows", rows,
        ])
        assert code == 0
    a = TiffFile().read(str(tmp_path / "out300.tif")).array
    b = TiffFile().read(str(tmp_path / "out960.tif")).array
    assert np.abs(a - b).max() <= 1.5 / 65535


def test_render_strip_is_those_rows_of_the_full_render() -> None:
    """A strip rendered on its own equals the same rows of the whole render."""
    target, sources = make_thirds(seed=7)
    plan = stack_frames(target, sources, work_dim=960)
    full = render_full(target, sources, plan.frames, plan.weights, tile_rows=SHAPE[0])
    strip = render_strip(target, sources, plan.frames, plan.weights, 200, 431)
    assert strip.shape == (231, SHAPE[1], 3)
    assert np.array_equal(strip, full[200:431])


def test_render_full_uses_the_weights_it_is_given() -> None:
    """Weights that give every pixel to the target return the target; the plan's weights do not."""
    target, sources = make_thirds(seed=7)
    plan = stack_frames(target, sources, work_dim=960)
    only_target = np.zeros_like(plan.weights)
    only_target[0] = 1.0
    assert np.allclose(render_full(target, sources, plan.frames, only_target, tile_rows=400), target, atol=1e-6)
    assert np.abs(render_full(target, sources, plan.frames, plan.weights, tile_rows=400) - target).max() > 0.05


def test_render_full_with_the_median_layer_tiles_shrinks_and_removes_the_person() -> None:
    """Showing the median over the person: the strips agree with one strip, the shrunk render matches the working composite, and the person is gone."""
    from nss.focus_stack.blend import composite
    from nss.focus_stack.clean import stack_clean
    from nss.focus_stack.clean_test import PERSON_SPOTS, make_background, make_people_pair
    target, frames = make_people_pair()
    plan = stack_clean(target, frames, work_dim=480)
    assert plan.weights.shape[0] == len(frames) + 2
    weights = np.zeros_like(plan.weights)
    weights[0] = 1.0
    weights[0, 150:202, 225:255] = 0.0     # the person's box at the 0.75 working scale
    weights[-1, 150:202, 225:255] = 1.0    # shows the median there
    untiled = render_full(target, frames, plan.frames, weights, tile_rows=target.shape[0])
    tiled = render_full(target, frames, plan.frames, weights, tile_rows=130)
    assert np.abs(tiled - untiled).max() < 1e-4
    working = composite(plan.target, [frame.aligned for frame in plan.frames] + [plan.median], weights)
    shrunk = cv2.resize(untiled, (working.shape[1], working.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.abs(shrunk - working)[20:-20, 20:-20].mean() < 0.02
    row, col = PERSON_SPOTS[0]
    person = (slice(row + 5, row + 65), slice(col + 5, col + 35))
    assert np.abs(target - make_background())[person].mean() > 0.2
    assert np.abs(untiled - make_background())[person].mean() < 0.08


def test_render_full_shows_a_given_plate_instead_of_recomputing_the_median() -> None:
    """With a working-size plate the last layer is that plate resampled, the strips agree, and nothing else changes."""
    from nss.focus_stack.clean import stack_clean
    from nss.focus_stack.clean_test import make_people_pair
    target, frames = make_people_pair()
    plan = stack_clean(target, frames, work_dim=480)
    weights = np.zeros_like(plan.weights)
    weights[0] = 1.0
    weights[0, 150:202, 225:255] = 0.0
    weights[-1, 150:202, 225:255] = 1.0
    plate = np.full_like(plan.median, 0.25)
    untiled = render_full(target, frames, plan.frames, weights, tile_rows=target.shape[0], plate=plate)
    tiled = render_full(target, frames, plan.frames, weights, tile_rows=130, plate=plate)
    assert np.abs(tiled - untiled).max() < 1e-4
    assert np.allclose(untiled[230:260, 310:330], 0.25, atol=1e-4)           # inside the box at the full size
    assert np.allclose(untiled[:100], target[:100], atol=1e-6)


def test_render_strip_skips_layers_with_no_weight_in_the_strip() -> None:
    """A frame with zero weight is not even read: a source full of NaN does not reach the picture, and the median is skipped likewise."""
    from nss.focus_stack.clean import stack_clean
    from nss.focus_stack.clean_test import make_people_pair
    target, frames = make_people_pair()
    plan = stack_clean(target, frames, work_dim=480)
    weights = np.zeros_like(plan.weights)
    weights[0] = 1.0
    weights[1] = 0.0
    poisoned = [np.full_like(frame, np.nan) for frame in frames]
    out = render_full(target, poisoned, plan.frames, weights, tile_rows=200)
    assert not np.isnan(out).any() and np.allclose(out, target, atol=1e-6)


def clean_files(tmp_path: Path) -> list[str]:
    """The people scene written as TIFFs: the target first, then its three frames."""
    from nss.focus_stack.clean_test import make_people_pair
    target, frames = make_people_pair()
    paths = []
    for name, image in zip(["target", "one", "two", "three"], [target, *frames]):
        write_tif(tmp_path / f"{name}.tif", image)
        paths.append(str(tmp_path / f"{name}.tif"))
    return paths


def test_main_clean_mode_removes_the_person_and_writes_clean_debug_images(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """--mode clean returns 0, replaces the target's person with background, reports the median and writes a median image, no agreement maps."""
    from nss.focus_stack.clean_test import PERSON_SPOTS, make_background
    paths = clean_files(tmp_path)
    code = main([*paths, "-o", str(tmp_path / "out.tif"), "--work-dim", "640", "--mode", "clean", "--debug-dir", str(tmp_path / "debug")])
    assert code == 0
    out = TiffFile().read(str(tmp_path / "out.tif")).array
    row, col = PERSON_SPOTS[0]
    person = (slice(row + 5, row + 65), slice(col + 5, col + 35))
    assert np.abs(TiffFile().read(paths[0]).array - make_background())[person].mean() > 0.2
    assert np.abs(out - make_background())[person].mean() < 0.08
    assert sorted(p.name for p in (tmp_path / "debug").iterdir()) == [
        "aligned_1.png", "aligned_2.png", "aligned_3.png", "labels.png", "median.png", "target.png",
    ]
    assert "clean plate (median)" in capsys.readouterr().out


def test_main_clean_mode_full_res_writes_the_target_size(tmp_path: Path) -> None:
    """--mode clean --full-res renders the target's own size, with the median computed strip by strip."""
    paths = clean_files(tmp_path)
    code = main([*paths, "-o", str(tmp_path / "out.tif"), "--work-dim", "480", "--mode", "clean", "--full-res", "--tile-rows", "200"])
    assert code == 0
    assert TiffFile().read(str(tmp_path / "out.tif")).array.shape == (480, 640, 3)


def test_main_clean2_mode_full_res_writes_the_target_size(tmp_path: Path) -> None:
    """--mode clean2 --full-res renders the target's own size, resampling the plate."""
    paths = clean_files(tmp_path)
    code = main([*paths, "-o", str(tmp_path / "out.tif"), "--work-dim", "480", "--mode", "clean2", "--full-res", "--tile-rows", "200"])
    assert code == 0
    assert TiffFile().read(str(tmp_path / "out.tif")).array.shape == (480, 640, 3)
