"""Tests for nss.focus_stack.edit."""
import numpy as np
import pytest

from nss.focus_stack.blend import composite
from nss.focus_stack.edit import EditState
from nss.focus_stack.labels import label_weights
from nss.focus_stack.stack import MIN_AGREEMENT, AlignedFrame, StackResult

SHAPE = (30, 50)  # rows, cols: non-square
FEATHER = 1.0
FRAME_COLORS = (0.8, 0.2)  # frame 1 and frame 2 are flat grey at these levels; the target is 0.5


def make_frame(color: float, shape: tuple[int, int] = SHAPE) -> AlignedFrame:
    """A flat frame that covers everything and agrees everywhere."""
    return AlignedFrame(
        aligned=np.full((*shape, 3), color, dtype=np.float32),
        coverage=np.ones(shape, dtype=bool),
        agreement=np.ones(shape, dtype=np.float32),
        sharpness=np.ones(shape, dtype=np.float32),
        inliers=100,
        matrix=np.eye(3),
        flow=np.zeros((*shape, 2), dtype=np.float32),
        gain=np.ones((*shape, 3), dtype=np.float32),
    )


def make_plan(
    frames: list[AlignedFrame] | None = None,
    shape: tuple[int, int] = SHAPE,
    feather: float = FEATHER,
    median_level: float | None = None,
) -> StackResult:
    """A plan on a flat 0.5 target whose automatic labels put frame 1 on the left third and leave the rest on the target.

    With `median_level` the plan also has a flat median layer at that level, after the frames.
    """
    frames = frames if frames is not None else [make_frame(c, shape) for c in FRAME_COLORS]
    target = np.full((*shape, 3), 0.5, dtype=np.float32)
    labels = np.zeros(shape, dtype=np.int16)
    labels[:, :shape[1] // 3] = 1
    eligible = [f.coverage & (f.agreement >= MIN_AGREEMENT) for f in frames]
    layers = [f.aligned for f in frames]
    median = None
    if median_level is not None:
        median = np.full((*shape, 3), median_level, dtype=np.float32)
        eligible.append(np.ones(shape, dtype=bool))
        layers.append(median)
    weights = label_weights(labels, eligible, feather)
    return StackResult(
        target=target,
        target_sharpness=np.ones(shape, dtype=np.float32),
        frames=frames,
        labels=labels,
        weights=weights,
        result=composite(target, layers, weights),
        replaced_fraction=0.3,
        feather_sigma=feather,
        median=median,
    )


def stroke(state: EditState, points: list[tuple[float, float]], radius: float, erase: bool = False) -> None:
    """One mouse stroke: a dab (or an erase) at each point."""
    state.begin_stroke()
    for x, y in points:
        state.erase(x, y, radius) if erase else state.dab(x, y, radius)
    state.end_stroke()


def test_no_patches_reproduces_the_plan() -> None:
    """Without patches the labels, weights and result are the plan's own."""
    plan = make_plan()
    state = EditState(plan)
    assert np.array_equal(state.labels(), plan.labels)
    assert np.allclose(state.weights(), plan.weights)
    assert np.allclose(state.result(), plan.result)
    assert state.patch_count == 0 and state.active is None and state.active_mask() is None


def test_a_stroke_makes_a_disk_patch_clipped_at_the_border() -> None:
    """A dab of radius 5 selects the 81 lattice points of the disk; at a corner only the 26 inside the picture."""
    state = EditState(make_plan())
    stroke(state, [(30.0, 15.0)], 5.0)
    mask = state.active_mask()
    assert mask.shape == SHAPE
    assert mask.sum() == 81
    assert mask[15, 30] and mask[15, 35] and not mask[15, 36]
    state.deselect()
    stroke(state, [(0.0, 0.0)], 5.0)
    corner = state.active_mask()
    assert corner.sum() == 26
    assert corner[0, 0] and corner[4, 0] and not corner[0, 6]
    assert state.patch_count == 2


def test_a_new_patch_shows_the_next_layer_straight_away() -> None:
    """Painted on the target (label 0) the patch shows frame 1; painted on frame 1's area it shows frame 2."""
    plan = make_plan()
    state = EditState(plan)
    stroke(state, [(40.0, 15.0)], 4.0)
    assert state.patch_label(0) == 1
    assert np.allclose(state.result()[15, 40], FRAME_COLORS[0], atol=0.02)
    state.deselect()
    stroke(state, [(5.0, 15.0)], 4.0)
    assert state.patch_label(1) == 2
    assert np.allclose(state.result()[15, 5], FRAME_COLORS[1], atol=0.02)
    assert np.allclose(state.result()[15, 48], 0.5, atol=0.01)


def test_painting_adds_to_the_active_patch_across_strokes_until_deselected() -> None:
    """Two strokes far apart are one patch; after deselect the next stroke is another patch."""
    state = EditState(make_plan())
    stroke(state, [(30.0, 8.0)], 3.0)
    stroke(state, [(30.0, 22.0)], 3.0)
    assert state.patch_count == 1
    assert state.active_mask()[8, 30] and state.active_mask()[22, 30] and not state.active_mask()[15, 30]
    state.deselect()
    assert state.active is None
    stroke(state, [(45.0, 15.0)], 3.0)
    assert state.patch_count == 2 and state.active == 1


def test_erase_removes_pixels_and_gives_them_back_their_automatic_labels() -> None:
    """Erasing from a patch drops those pixels, which return to the automatic label and stop being manual."""
    plan = make_plan()
    state = EditState(plan)
    stroke(state, [(35.0, 15.0)], 6.0)
    stroke(state, [(38.0, 15.0)], 2.0, erase=True)
    mask = state.active_mask()
    assert not mask[15, 38] and mask[15, 31]
    assert state.labels()[15, 38] == plan.labels[15, 38]
    assert not state.manual()[15, 38] and state.manual()[15, 31]
    assert state.labels()[15, 31] == state.patch_label(0)


def test_erasing_the_whole_patch_deletes_it() -> None:
    """A patch with no pixels left is gone and nothing is active."""
    state = EditState(make_plan())
    stroke(state, [(35.0, 15.0)], 3.0)
    stroke(state, [(35.0, 15.0)], 5.0, erase=True)
    assert state.patch_count == 0 and state.active is None


def test_cycle_steps_through_target_and_frames_and_wraps() -> None:
    """From frame 1 the active patch steps 2, 0, 1 forwards and, from 1, 0 then 2 backwards."""
    state = EditState(make_plan())
    stroke(state, [(40.0, 15.0)], 4.0)
    seen = []
    for _ in range(3):
        state.cycle(1)
        seen.append(state.patch_label(0))
    assert seen == [2, 0, 1]
    state.cycle(-1)
    state.cycle(-1)
    assert state.patch_label(0) == 2
    assert np.allclose(state.result()[15, 40], FRAME_COLORS[1], atol=0.02)
    assert np.allclose(state.result()[15, 48], 0.5, atol=0.01)


def test_set_label_picks_a_layer_directly() -> None:
    """set_label(0) gives the patch back to the target, set_label(2) to frame 2."""
    state = EditState(make_plan())
    stroke(state, [(40.0, 15.0)], 4.0)
    state.set_label(2)
    assert np.allclose(state.result()[15, 40], FRAME_COLORS[1], atol=0.02)
    state.set_label(0)
    assert np.allclose(state.result()[15, 40], 0.5, atol=0.02)


def test_select_at_picks_the_topmost_patch_and_edits_that_one() -> None:
    """Where two patches overlap the newer is on top and is the one selected; cycling an older patch leaves the newer alone."""
    state = EditState(make_plan())
    stroke(state, [(30.0, 15.0)], 8.0)
    state.deselect()
    stroke(state, [(34.0, 15.0)], 8.0)
    state.deselect()
    assert state.labels()[15, 32] == state.patch_label(1)
    assert state.select_at(32.0, 15.0) and state.active == 1
    assert state.select_at(23.0, 15.0) and state.active == 0
    assert not state.select_at(2.0, 25.0) and state.active == 0
    before = state.labels()[15, 40]
    state.cycle(1)
    assert state.labels()[15, 40] == before
    assert state.labels()[15, 23] == state.patch_label(0)


def test_delete_active_removes_the_patch_and_restores_the_automatic_result() -> None:
    """Deleting the active patch leaves no patch and the plan's own result."""
    plan = make_plan()
    state = EditState(plan)
    stroke(state, [(40.0, 15.0)], 4.0)
    state.delete_active()
    assert state.patch_count == 0 and state.active is None
    assert np.allclose(state.result(), plan.result)


def test_manual_choice_overrides_agreement_but_not_coverage() -> None:
    """A frame that disagrees is used once painted by hand; a frame that does not cover a spot never is."""
    frames = [make_frame(c) for c in FRAME_COLORS]
    frames[1].agreement[:, 30:40] = 0.0
    frames[1].coverage[:, 40:] = False
    state = EditState(make_plan(frames))
    stroke(state, [(35.0, 15.0), (45.0, 15.0)], 4.0)
    state.set_label(2)
    result = state.result()
    assert np.allclose(result[15, 35], FRAME_COLORS[1], atol=0.02)
    assert np.allclose(result[15, 45], 0.5)
    assert state.manual()[15, 35] and state.manual()[15, 45]


def test_centers_are_the_middle_of_each_patch() -> None:
    """The pin of a disk patch sits at its centre, as (column, row)."""
    state = EditState(make_plan())
    stroke(state, [(30.0, 12.0)], 5.0)
    state.deselect()
    stroke(state, [(44.0, 20.0)], 3.0)
    centers = state.centers()
    assert len(centers) == 2
    assert np.allclose(centers[0], (30.0, 12.0), atol=0.5)
    assert np.allclose(centers[1], (44.0, 20.0), atol=0.5)


def test_undo_and_redo_cover_creating_painting_cycling_and_deleting() -> None:
    """Each kind of edit undoes and redoes; cycling one patch several times is a single step; a new edit clears redo."""
    plan = make_plan()
    state = EditState(plan)
    stroke(state, [(40.0, 15.0)], 4.0)
    created = state.result().copy()
    for _ in range(3):
        state.cycle(1)
    cycled = state.result().copy()
    state.undo()
    assert state.patch_count == 1 and np.allclose(state.result(), created)
    state.redo()
    assert np.allclose(state.result(), cycled)
    stroke(state, [(40.0, 22.0)], 3.0)
    assert state.active_mask()[22, 40]
    state.undo()
    assert not state.active_mask()[22, 40]
    state.delete_active()
    assert state.patch_count == 0
    state.undo()
    assert state.patch_count == 1
    state.undo()
    state.undo()
    assert state.patch_count == 0 and np.allclose(state.result(), plan.result)
    state.redo()
    assert state.patch_count == 1
    state.deselect()
    stroke(state, [(5.0, 5.0)], 2.0)
    state.redo()
    assert state.patch_count == 2


def test_a_stroke_that_changes_nothing_takes_no_undo_step() -> None:
    """An erase stroke with no active patch leaves history alone, so one undo still removes the earlier patch."""
    state = EditState(make_plan())
    stroke(state, [(40.0, 15.0)], 4.0)
    state.deselect()
    stroke(state, [(10.0, 10.0)], 3.0, erase=True)
    state.undo()
    assert state.patch_count == 0


BIG = (70, 110)  # rows, cols: a picture larger than the feather's reach


def painted_state(feather: float) -> EditState:
    """A state on a BIG plan with three overlapping patches near the middle and one at the corner."""
    state = EditState(make_plan(shape=BIG, feather=feather))
    stroke(state, [(50.0, 30.0), (56.0, 34.0)], 7.0)
    state.deselect()
    stroke(state, [(60.0, 36.0)], 8.0)
    state.set_label(2)
    state.deselect()
    stroke(state, [(2.0, 3.0)], 6.0)
    state.deselect()
    return state


def test_result_region_equals_the_same_crop_of_the_full_result() -> None:
    """For crops inside, on the border and across patches, and for feathers of 1 and 3 px, the crop is exactly the full render's."""
    for feather in (1.0, 3.0):
        state = painted_state(feather)
        full = state.result()
        for rows, cols in [
            (slice(25, 45), slice(45, 70)), (slice(0, 12), slice(0, 15)), (slice(60, 70), slice(100, 110)),
            (slice(0, 70), slice(0, 110)), (slice(33, 34), slice(57, 58)),
        ]:
            region = state.result_region(rows, cols)
            assert region.shape == full[rows, cols].shape
            assert np.allclose(region, full[rows, cols], atol=1e-6)
        assert np.abs(full - state.result_region(slice(0, 70), slice(0, 110))).max() < 1e-6
        assert np.abs(full - make_plan(shape=BIG, feather=feather).result).max() > 0.05


def test_dirty_boxes_cover_every_pixel_an_edit_changes() -> None:
    """After each of a long random run of edits, patching the cached picture with the dirty box's region equals a full render."""
    rng = np.random.default_rng(11)
    state = EditState(make_plan(shape=BIG, feather=2.0))
    image = state.result()
    assert state.take_dirty() is None
    for step in range(120):
        x, y, radius = float(rng.uniform(0, 110)), float(rng.uniform(0, 70)), float(rng.uniform(2, 12))
        kind = int(rng.integers(0, 10))
        if kind <= 3:
            stroke(state, [(x, y), (x + 4.0, y + 3.0)], radius)
        elif kind == 4:
            stroke(state, [(x, y)], radius, erase=True)
        elif kind == 5:
            state.cycle(int(rng.choice([-1, 1])))
        elif kind == 6:
            state.set_label(int(rng.integers(0, 3)))
        elif kind == 7:
            state.select_at(x, y) or state.deselect()
        elif kind == 8:
            state.undo() if rng.random() < 0.7 else state.redo()
        else:
            state.delete_active() if rng.random() < 0.3 else state.deselect()
        box = state.take_dirty()
        if box is not None:
            image[box] = state.result_region(*box)
        assert np.abs(image - state.result()).max() < 1e-6, f"step {step}"
        assert state.take_dirty() is None
    assert state.patch_count > 0


MEDIAN_GREY = 0.65


def test_the_median_is_one_more_layer_after_the_frames() -> None:
    """With a median the editor has the target, the frames and the median as layers; without one it has no median layer."""
    assert EditState(make_plan()).layer_count == 3
    state = EditState(make_plan(median_level=MEDIAN_GREY))
    assert state.layer_count == 4 and state.frame_count == 2
    assert state.weights().shape == (4, *SHAPE)
    assert np.allclose(state.weights().sum(axis=0), 1.0, atol=1e-5)


def test_painting_and_cycling_reach_the_median_and_wrap() -> None:
    """From frame 2 a patch cycles on to the median, then the target; a new patch on frame 2's area starts on the median."""
    plan = make_plan(median_level=MEDIAN_GREY)
    plan.labels[:, 40:] = 2
    state = EditState(plan)
    stroke(state, [(45.0, 15.0)], 4.0)
    assert state.patch_label(0) == 3
    assert np.allclose(state.result()[15, 45], MEDIAN_GREY, atol=0.02)
    state.cycle(1)
    assert state.patch_label(0) == 0
    state.cycle(-1)
    state.cycle(-1)
    assert state.patch_label(0) == 2
    state.set_label(3)
    assert np.allclose(state.result()[15, 45], MEDIAN_GREY, atol=0.02)


def test_result_regions_include_the_median_layer() -> None:
    """A crop of the result with patches showing the median equals the same crop of the full result."""
    state = EditState(make_plan(shape=BIG, feather=2.0, median_level=MEDIAN_GREY))
    stroke(state, [(50.0, 30.0), (56.0, 34.0)], 7.0)
    state.set_label(3)
    full = state.result()
    assert np.allclose(full[30, 50], MEDIAN_GREY, atol=0.02)
    for rows, cols in [(slice(20, 45), slice(40, 70)), (slice(0, 70), slice(0, 110))]:
        assert np.abs(state.result_region(rows, cols) - full[rows, cols]).max() < 1e-6


def test_a_new_patch_in_a_clean_plan_starts_on_the_layer_the_rules_suggest() -> None:
    """With a median the first dab picks the frame that is clean there, else the median, instead of the next layer."""
    plan = make_plan(median_level=MEDIAN_GREY)
    plan.frames[1].aligned[:, 40:] = MEDIAN_GREY            # frame 2 is clean on the right, frame 1 nowhere
    state = EditState(plan)
    stroke(state, [(45.0, 15.0)], 4.0)
    assert state.patch_label(0) == 2
    assert np.allclose(state.result()[15, 45], MEDIAN_GREY, atol=0.02)
    state.deselect()
    stroke(state, [(25.0, 15.0)], 4.0)
    assert state.patch_label(1) == 3                         # no frame is clean here: the median
    state.cycle(1)
    assert state.patch_label(1) == 0


def test_a_small_patch_hides_what_is_under_it_completely() -> None:
    """A patch no wider than the feather's blur still shows its frame at full strength in the middle: no half-faded target shows through."""
    state = EditState(make_plan(feather=4.0))
    stroke(state, [(40.0, 15.0)], 5.0)
    state.set_label(2)
    assert np.allclose(state.result()[15, 40], FRAME_COLORS[1], atol=0.03)
    assert np.allclose(state.result()[15, 40 - 4:40 + 5], FRAME_COLORS[1], atol=0.06)


def test_layer_image_is_the_whole_picture_of_a_layer() -> None:
    """Layer 0 is the target, layer k the aligned frame k, and with a median the layer after the frames is the median."""
    state = EditState(make_plan(median_level=MEDIAN_GREY))
    assert np.allclose(state.layer_image(0), 0.5)
    assert np.allclose(state.layer_image(1), FRAME_COLORS[0]) and np.allclose(state.layer_image(2), FRAME_COLORS[1])
    assert np.allclose(state.layer_image(3), MEDIAN_GREY)
    for bad in (-1, 4):
        with pytest.raises(IndexError):
            state.layer_image(bad)
    with pytest.raises(IndexError):
        EditState(make_plan()).layer_image(3)


def test_a_painted_patch_ends_at_the_brush_circle_not_a_feather_beyond_it() -> None:
    """With a 6 px automatic feather a radius 8 patch is a full frame inside the circle and the target again 4 px outside it."""
    state = EditState(make_plan(shape=BIG, feather=6.0))
    stroke(state, [(80.0, 35.0)], 8.0)
    weights = state.weights()
    assert state.patch_label(0) == 1
    assert weights[1, 35, 80] > 0.99 and weights[1, 35, 80 + 7] > 0.99
    assert weights[1, 35, 80 + 12] < 0.01 and weights[1, 35 - 12, 80] < 0.01
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-4)
