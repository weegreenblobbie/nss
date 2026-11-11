"""Tests for nss.focus_stack.gui.canvas, driven with real mouse, wheel and key events on an offscreen widget."""
import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QNativeGestureEvent, QPointingDevice, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from nss.focus_stack.edit import EditState
from nss.focus_stack.edit_test import make_plan
from nss.focus_stack.gui.canvas import CanvasView
from nss.image_utils import to_uint8_display

APP = QApplication.instance() or QApplication([])  # kept alive for every test in this module
VIEWS: list[CanvasView] = []
SHAPE = (60, 100)  # rows, cols: non-square; fits a 400 x 300 widget at 4 widget pixels per picture pixel
NAMES = ["target", "one.tif", "two.tif"]
TARGET_GREY, FRAME1_GREY, FRAME2_GREY = (int(round(v * 255)) for v in (0.5, 0.8, 0.2))


@pytest.fixture(autouse=True)
def delete_views():
    """Deletes every canvas a test made, so Python's garbage collector never frees one in the middle of a paint."""
    yield
    for view in VIEWS:
        view.hide()
        view.deleteLater()
    VIEWS.clear()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def make_view() -> tuple[CanvasView, EditState]:
    """A shown 400 x 300 canvas on a 60 x 100 plan: frame 1 on the left third, the target elsewhere."""
    state = EditState(make_plan(shape=SHAPE, feather=1.0))
    view = CanvasView()
    VIEWS.append(view)
    view.resize(400, 300)
    view.set_state(state, NAMES)
    view.show()
    QApplication.processEvents()
    return view, state


def point(view: CanvasView, x: float, y: float) -> QPoint:
    """The widget position of picture pixel (column x, row y)."""
    return view.to_widget(x, y).toPoint()


def drag(view: CanvasView, points: list[tuple[float, float]], modifier=Qt.KeyboardModifier.NoModifier) -> None:
    """Presses at the first picture position, moves through the rest and releases."""
    QTest.mousePress(view, Qt.MouseButton.LeftButton, modifier, point(view, *points[0]))
    for x, y in points[1:]:
        QTest.mouseMove(view, point(view, x, y))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, modifier, point(view, *points[-1]))


def test_fit_shows_the_whole_picture_and_pixels_are_the_result() -> None:
    """The picture is scaled to fit, mapping round-trips, and the shown pixels are the 8-bit result."""
    view, state = make_view()
    assert view.scale == 4.0
    assert np.array_equal(view.pixels, to_uint8_display(state.result()))
    assert np.array_equal(view.shown_pixels(), view.pixels)
    centre = view.to_widget(10.0, 20.0)
    x, y = view.to_image(centre)
    assert abs(x - 10.0) < 1e-9 and abs(y - 20.0) < 1e-9
    assert abs(view.to_widget(0.0, 0.0).x() - 2.0) < 1e-9   # pixel (0, 0) is centred half a pixel in


def test_dragging_paints_a_patch_with_live_feedback() -> None:
    """One drag over the target makes one patch showing frame 1; the shown pixels change at once; the stroke is one undo step."""
    view, state = make_view()
    view.radius = 3.0
    edits = []
    view.edited.connect(lambda: edits.append(1))
    drag(view, [(60.0, 30.0), (75.0, 30.0)])
    assert state.patch_count == 1 and state.active == 0
    assert np.all(view.pixels[30, 62:74] == FRAME1_GREY)
    assert np.all(view.pixels[30, 85] == TARGET_GREY)
    assert np.array_equal(view.pixels, to_uint8_display(state.result()))
    assert len(edits) >= 1
    state.undo()
    assert state.patch_count == 0


def test_shift_drag_and_alt_drag_erase_from_the_active_patch() -> None:
    """Shift-dragging or Alt-dragging over the patch removes those pixels, which show the target again."""
    view, state = make_view()
    view.radius = 4.0
    drag(view, [(60.0, 30.0), (80.0, 30.0)])
    drag(view, [(66.0, 30.0), (68.0, 30.0)], Qt.KeyboardModifier.ShiftModifier)
    assert not state.active_mask()[30, 67] and state.active_mask()[30, 78]
    assert np.all(view.pixels[30, 67] == TARGET_GREY) and np.all(view.pixels[30, 78] == FRAME1_GREY)
    drag(view, [(74.0, 30.0), (76.0, 30.0)], Qt.KeyboardModifier.AltModifier)
    assert not state.active_mask()[30, 75]
    assert np.array_equal(view.pixels, to_uint8_display(state.result()))


def test_keys_cycle_set_layer_deselect_and_delete() -> None:
    """Tab and Shift+Tab cycle the layer, digit keys do nothing, Escape deselects, Delete removes the patch."""
    view, state = make_view()
    view.radius = 4.0
    drag(view, [(70.0, 30.0), (72.0, 30.0)])
    assert state.patch_label(0) == 1
    QTest.keyClick(view, Qt.Key.Key_Tab)
    assert state.patch_label(0) == 2 and np.all(view.pixels[30, 71] == FRAME2_GREY)
    QTest.keyClick(view, Qt.Key.Key_Backtab, Qt.KeyboardModifier.ShiftModifier)
    assert state.patch_label(0) == 1
    for digit in (Qt.Key.Key_0, Qt.Key.Key_2, Qt.Key.Key_5):
        QTest.keyClick(view, digit)
        assert state.patch_label(0) == 1
    QTest.keyClick(view, Qt.Key.Key_Backtab, Qt.KeyboardModifier.ShiftModifier)
    assert state.patch_label(0) == 0 and np.all(view.pixels[30, 71] == TARGET_GREY)
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert state.active is None
    state.select(0)
    QTest.keyClick(view, Qt.Key.Key_Delete)
    assert state.patch_count == 0
    assert np.array_equal(view.pixels, to_uint8_display(state.result()))


def test_bracket_keys_change_the_brush_size() -> None:
    """] grows and [ shrinks the brush by a quarter, within 1 and 500 picture pixels."""
    view, _ = make_view()
    view.radius = 20.0
    QTest.keyClick(view, Qt.Key.Key_BracketRight)
    assert abs(view.radius - 25.0) < 1e-9
    QTest.keyClick(view, Qt.Key.Key_BracketLeft)
    QTest.keyClick(view, Qt.Key.Key_BracketLeft)
    assert abs(view.radius - 16.0) < 1e-9
    for _ in range(40):
        QTest.keyClick(view, Qt.Key.Key_BracketLeft)
    assert view.radius == 1.0
    for _ in range(60):
        QTest.keyClick(view, Qt.Key.Key_BracketRight)
    assert view.radius == 500.0


def send_wheel(view: CanvasView, at: QPointF, pixels: QPoint, angle: QPoint, modifier=Qt.KeyboardModifier.NoModifier) -> None:
    """Delivers one wheel event: `pixels` is a touchpad's pixel delta (zero for a mouse), `angle` the delta in eighths of a degree."""
    wheel = QWheelEvent(
        at, view.mapToGlobal(at), pixels, angle, Qt.MouseButton.NoButton, modifier, Qt.ScrollPhase.NoScrollPhase, False
    )
    QApplication.sendEvent(view, wheel)


def test_ctrl_wheel_zooms_about_the_cursor_and_middle_drag_pans() -> None:
    """Ctrl+wheel keeps the picture point under the cursor still; a middle-button drag moves the picture by the drag."""
    view, _ = make_view()
    cursor = QPointF(300.0, 120.0)
    before = view.to_image(cursor)
    send_wheel(view, cursor, QPoint(0, 0), QPoint(0, 120), Qt.KeyboardModifier.ControlModifier)
    assert view.scale > 4.0
    after = view.to_image(cursor)
    assert abs(after[0] - before[0]) < 1e-6 and abs(after[1] - before[1]) < 1e-6
    start = view.to_widget(0.0, 0.0)
    QTest.mousePress(view, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(100, 100))
    QTest.mouseMove(view, QPoint(130, 90))
    QTest.mouseRelease(view, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(130, 90))
    moved = view.to_widget(0.0, 0.0)
    assert abs(moved.x() - start.x() - 30.0) < 1e-6 and abs(moved.y() - start.y() + 10.0) < 1e-6


def test_clicking_a_pin_selects_that_patch_instead_of_painting() -> None:
    """After two patches, clicking the first one's pin makes it active and adds no patch."""
    view, state = make_view()
    view.radius = 3.0
    drag(view, [(60.0, 20.0), (64.0, 20.0)])
    QTest.keyClick(view, Qt.Key.Key_Escape)
    drag(view, [(60.0, 45.0), (64.0, 45.0)])
    assert state.patch_count == 2 and state.active == 1
    cx, cy = state.centers()[0]
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(view, cx, cy))
    assert state.patch_count == 2 and state.active == 0


def test_holding_t_shows_the_original_target() -> None:
    """While T is held the shown pixels are the target's; releasing it shows the result again."""
    view, state = make_view()
    target = to_uint8_display(state.target)
    assert not np.array_equal(view.pixels, target)
    QTest.keyPress(view, Qt.Key.Key_T)
    assert np.array_equal(view.shown_pixels(), target)
    QTest.keyRelease(view, Qt.Key.Key_T)
    assert np.array_equal(view.shown_pixels(), view.pixels)


def test_the_outline_follows_the_active_patch_only() -> None:
    """The outline's bounding box matches the active patch's, and it is empty once nothing is active."""
    view, state = make_view()
    view.radius = 5.0
    drag(view, [(60.0, 30.0), (75.0, 30.0)])
    contours = view.outline()
    assert len(contours) == 1
    xs, ys = contours[0][:, 0], contours[0][:, 1]
    rows, cols = np.nonzero(state.active_mask())
    assert xs.min() == cols.min() and xs.max() == cols.max() and ys.min() == rows.min() and ys.max() == rows.max()
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert view.outline() == []


def test_many_strokes_edits_and_undos_keep_the_pixels_equal_to_a_full_render() -> None:
    """After a mixed run of strokes, erases, layer changes, undos and redos, the incrementally updated pixels equal a full render."""
    view, state = make_view()
    rng = np.random.default_rng(5)
    for step in range(25):
        view.radius = float(rng.uniform(2, 9))
        x0, y0 = float(rng.uniform(5, 95)), float(rng.uniform(5, 55))
        kind = step % 5
        if kind in (0, 1):
            QTest.keyClick(view, Qt.Key.Key_Escape)
            drag(view, [(x0, y0), (x0 + 6.0, y0 + 4.0)])
        elif kind == 2:
            drag(view, [(x0, y0), (x0 + 3.0, y0)], Qt.KeyboardModifier.ShiftModifier)
        elif kind == 3:
            QTest.keyClick(view, Qt.Key.Key_Tab)
        else:
            view.undo() if step % 2 else view.redo()
        assert np.array_equal(view.pixels, to_uint8_display(state.result())), f"step {step}"


def test_status_text_names_the_brush_and_the_active_layer() -> None:
    """The status line gives the brush size, and the active patch's number and layer name once there is one."""
    view, _ = make_view()
    view.radius = 12.0
    assert "brush 12" in view.describe() and "patch" not in view.describe()
    drag(view, [(70.0, 30.0), (72.0, 30.0)])
    assert "patch 1 of 1" in view.describe() and "one.tif" in view.describe()
    QTest.keyClick(view, Qt.Key.Key_Tab)
    assert "two.tif" in view.describe()


def test_tint_and_pins_paint_without_changing_the_pixels() -> None:
    """Toggling the label tint changes what is drawn but not the picture data, and grabbing the widget works with patches present."""
    view, state = make_view()
    view.radius = 4.0
    drag(view, [(70.0, 30.0), (72.0, 30.0)])
    plain = view.grab().toImage()
    before = view.pixels.copy()
    view.toggle_tint()
    assert view.show_tint
    tinted = view.grab().toImage()
    assert np.array_equal(view.pixels, before)
    spot = point(view, 10.0, 30.0)
    assert plain.pixelColor(spot) != tinted.pixelColor(spot)


def test_zoom_presets_keep_the_middle_of_the_view_still() -> None:
    """set_zoom changes the scale about the widget centre, clamps to the range, and fit goes back to filling the widget."""
    view, _ = make_view()
    centre = QPointF(view.width() / 2.0, view.height() / 2.0)
    before = view.to_image(centre)
    view.set_zoom(2.0)
    assert view.scale == 2.0
    after = view.to_image(centre)
    assert abs(after[0] - before[0]) < 1e-9 and abs(after[1] - before[1]) < 1e-9
    view.set_zoom(1000.0)
    assert view.scale == 64.0
    view.set_zoom(0.0001)
    assert view.scale == 0.02
    view.zoom_fit()
    assert view.scale == 4.0
    view.resize(200, 150)
    QApplication.processEvents()
    assert view.scale == 2.0


def test_zoom_in_and_out_step_through_the_levels() -> None:
    """From the fitted 400% zoom in goes to 800% and out to 200%, 100%, 50%; the ends do not overshoot."""
    view, _ = make_view()
    view.zoom_in()
    assert view.scale == 8.0
    view.zoom_out()
    view.zoom_out()
    assert view.scale == 2.0
    view.zoom_out()
    view.zoom_out()
    assert view.scale == 0.5
    for _ in range(20):
        view.zoom_out()
    assert view.scale == 0.125
    for _ in range(20):
        view.zoom_in()
    assert view.scale == 32.0


def test_status_gives_the_zoom_relative_to_the_preview_and_to_the_full_size() -> None:
    """With the full size known the status says what share of it the zoom shows; without it only the preview percentage."""
    view, state = make_view()
    assert "zoom 400%" in view.describe() and "full size" not in view.describe()
    view.set_state(state, NAMES, full_shape=(600, 1000))
    assert "zoom 400% (40% of full size)" in view.describe()
    view.set_zoom(1.0)
    assert "zoom 100% (10% of full size)" in view.describe()


def test_space_toggles_pan_mode_and_escape_leaves_it_without_deselecting() -> None:
    """Space enters and leaves pan mode; the first Esc in pan mode only leaves it, the next one deselects the patch."""
    view, state = make_view()
    view.radius = 3.0
    drag(view, [(60.0, 30.0), (64.0, 30.0)])
    assert state.active == 0
    QTest.keyClick(view, Qt.Key.Key_Space)
    assert view.pan_mode
    QTest.keyClick(view, Qt.Key.Key_Space)
    assert not view.pan_mode
    QTest.keyClick(view, Qt.Key.Key_Space)
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert not view.pan_mode and state.active == 0
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert state.active is None


def test_left_drag_pans_in_pan_mode_and_paints_again_after_leaving_it() -> None:
    """In pan mode a left-drag moves the picture by the drag and makes no patch; leaving pan mode, a drag paints."""
    view, state = make_view()
    view.radius = 3.0
    start = view.to_widget(0.0, 0.0)
    QTest.keyClick(view, Qt.Key.Key_Space)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(100, 100))
    QTest.mouseMove(view, QPoint(130, 90))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(130, 90))
    moved = view.to_widget(0.0, 0.0)
    assert abs(moved.x() - start.x() - 30.0) < 1e-6 and abs(moved.y() - start.y() + 10.0) < 1e-6
    assert state.patch_count == 0 and view.scale == 4.0
    QTest.keyClick(view, Qt.Key.Key_Space)
    drag(view, [(60.0, 30.0), (64.0, 30.0)])
    assert state.patch_count == 1


def test_clicking_a_pin_does_nothing_in_pan_mode() -> None:
    """A pin is not selected by a click while panning."""
    view, state = make_view()
    view.radius = 3.0
    drag(view, [(60.0, 30.0), (64.0, 30.0)])
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert state.active is None
    QTest.keyClick(view, Qt.Key.Key_Space)
    cx, cy = state.centers()[0]
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(view, cx, cy))
    assert state.active is None and state.patch_count == 1


def test_the_cursor_shows_whether_the_picture_can_be_dragged() -> None:
    """Blank (the brush circle shows) normally, an open hand in pan mode, a closed hand while dragging, by pan mode or middle button."""
    view, _ = make_view()
    assert view.cursor().shape() == Qt.CursorShape.BlankCursor
    QTest.keyClick(view, Qt.Key.Key_Space)
    assert view.cursor().shape() == Qt.CursorShape.OpenHandCursor
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(100, 100))
    assert view.cursor().shape() == Qt.CursorShape.ClosedHandCursor
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(100, 100))
    assert view.cursor().shape() == Qt.CursorShape.OpenHandCursor
    QTest.keyClick(view, Qt.Key.Key_Space)
    assert view.cursor().shape() == Qt.CursorShape.BlankCursor
    QTest.mousePress(view, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(100, 100))
    assert view.cursor().shape() == Qt.CursorShape.ClosedHandCursor
    QTest.mouseRelease(view, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(100, 100))
    assert view.cursor().shape() == Qt.CursorShape.BlankCursor


def test_pan_mode_is_in_the_status_line_and_announced_once() -> None:
    """The status line says so in pan mode only, and pan_mode_changed fires only when the mode really changes."""
    view, _ = make_view()
    changes = []
    view.pan_mode_changed.connect(changes.append)
    assert "pan mode" not in view.describe()
    view.set_pan_mode(True)
    view.set_pan_mode(True)
    assert "pan mode" in view.describe() and changes == [True]
    view.set_pan_mode(False)
    assert "pan mode" not in view.describe() and changes == [True, False]


def test_two_finger_scroll_pans_by_its_pixel_delta_and_does_not_zoom() -> None:
    """A touchpad scroll (a pixel delta, no Ctrl) moves the picture with the fingers and leaves the zoom alone."""
    view, _ = make_view()
    start = view.to_widget(0.0, 0.0)
    send_wheel(view, QPointF(200.0, 150.0), QPoint(30, -20), QPoint(30, -20))
    moved = view.to_widget(0.0, 0.0)
    assert abs(moved.x() - start.x() - 30.0) < 1e-9 and abs(moved.y() - start.y() + 20.0) < 1e-9
    assert view.scale == 4.0


def test_a_mouse_wheel_notch_pans_by_a_fixed_step_in_either_direction() -> None:
    """Without a pixel delta a notch (120) moves the picture 60 pixels, vertically for a wheel and horizontally for a sideways scroll."""
    view, _ = make_view()
    start = view.to_widget(0.0, 0.0)
    send_wheel(view, QPointF(200.0, 150.0), QPoint(0, 0), QPoint(0, 120))
    send_wheel(view, QPointF(200.0, 150.0), QPoint(0, 0), QPoint(-120, 0))
    moved = view.to_widget(0.0, 0.0)
    assert abs(moved.x() - start.x() + 60.0) < 1e-9 and abs(moved.y() - start.y() - 60.0) < 1e-9
    assert view.scale == 4.0


def test_pinch_zooms_about_the_gesture_position() -> None:
    """A pinch of 0.5 grows the zoom by half about the fingers; the picture point under them stays put."""
    view, _ = make_view()
    at = QPointF(120.0, 200.0)
    before = view.to_image(at)
    pinch = QNativeGestureEvent(
        Qt.NativeGestureType.ZoomNativeGesture, QPointingDevice.primaryPointingDevice(), 2, at, at, view.mapToGlobal(at), 0.5, QPointF(0.0, 0.0)
    )
    QApplication.sendEvent(view, pinch)
    assert abs(view.scale - 6.0) < 1e-9
    after = view.to_image(at)
    assert abs(after[0] - before[0]) < 1e-9 and abs(after[1] - before[1]) < 1e-9


def test_viewing_a_layer_shows_its_whole_picture_and_painting_is_off() -> None:
    """view_layer shows that layer's 8-bit picture; strokes then make no patch; Escape goes back to the merged picture."""
    view, state = make_view()
    changes = []
    view.view_changed.connect(changes.append)
    view.view_layer(2)
    assert view.viewed_layer == 2 and np.all(view.shown_pixels() == FRAME2_GREY)
    drag(view, [(60.0, 30.0), (70.0, 30.0)])
    assert state.patch_count == 0
    view.view_layer(0)
    assert np.all(view.shown_pixels() == TARGET_GREY)
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert view.viewed_layer is None and np.array_equal(view.shown_pixels(), view.pixels)
    assert changes == [2, 0, None]
    drag(view, [(60.0, 30.0), (70.0, 30.0)])
    assert state.patch_count == 1


def test_the_status_line_names_the_layer_being_viewed_and_a_new_state_stops_viewing() -> None:
    """While viewing, the status line says which layer and how to leave; a new edit state ends the viewing."""
    view, state = make_view()
    assert "viewing" not in view.describe()
    view.view_layer(1)
    assert "viewing one.tif" in view.describe() and "Esc" in view.describe()
    view.set_state(state, NAMES)
    assert view.viewed_layer is None and "viewing" not in view.describe()


def test_hovering_a_pin_shows_a_pointing_hand_and_leaving_it_restores_the_brush() -> None:
    """Over a pin the cursor is a pointing hand and no brush circle is drawn; elsewhere the cursor is blank again."""
    view, state = make_view()
    view.radius = 3.0
    drag(view, [(60.0, 20.0), (64.0, 20.0)])
    QTest.keyClick(view, Qt.Key.Key_Escape)
    cx, cy = state.centers()[0]
    QTest.mouseMove(view, point(view, cx, cy))
    assert view.cursor().shape() == Qt.CursorShape.PointingHandCursor
    assert view.hovered_pin() == 0
    QTest.mouseMove(view, point(view, 20.0, 45.0))
    assert view.cursor().shape() == Qt.CursorShape.BlankCursor
    assert view.hovered_pin() is None


def test_tint_is_clear_under_the_active_patch_until_it_is_deselected() -> None:
    """With the tint on, the active patch is untinted and the rest keeps its tint; Esc tints the patch again."""
    view, state = make_view()
    view.radius = 4.0
    view.toggle_tint()
    drag(view, [(70.0, 30.0), (72.0, 30.0)])
    view._tint()
    inside = state.active_mask()
    assert inside.any() and not (~inside).all()
    assert not view._tint_array[inside].any()
    assert view._tint_array[~inside].any()
    QTest.keyClick(view, Qt.Key.Key_Escape)
    view._tint()
    assert view._tint_array[inside].any()
