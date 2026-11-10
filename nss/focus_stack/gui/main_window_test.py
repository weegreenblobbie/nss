"""Tests for nss.focus_stack.gui.main_window: loading, editing through the menu, saving and quitting."""
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from nss.focus_stack.gui.main_window import MainWindow
from nss.focus_stack.register_test import make_texture
from nss.focus_stack.stack import render_full, stack_frames
from nss.focus_stack.stack_test import SHAPE, make_thirds, write_tif
from nss.image_utils import downsample_image, to_uint8_display
from nss.tiff_file import TiffFile
from plugin import stack_bridge
from plugin.handover import read_values

APP = QApplication.instance() or QApplication([])  # kept alive for every test in this module
WORK_DIM = 960
WINDOWS: list[MainWindow] = []


@pytest.fixture(autouse=True)
def delete_windows():
    """Deletes every window a test made, so Python's garbage collector never frees one in the middle of a paint."""
    yield
    for window in WINDOWS:
        window._unsaved = False
        window.close()
        window.deleteLater()
    WINDOWS.clear()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def new_window() -> MainWindow:
    """An empty window that the test cleanup will delete."""
    window = MainWindow()
    WINDOWS.append(window)
    return window


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Three TIFFs on disk (target, two frames) and the stack computed from them once."""
    folder = tmp_path_factory.mktemp("stack")
    target, sources = make_thirds(seed=9)
    paths = [str(folder / name) for name in ("target.tif", "one.tif", "two.tif")]
    for path, image in zip(paths, [target, *sources]):
        write_tif(Path(path), image)
    target_file = TiffFile().read(paths[0])
    frames = [TiffFile().read(path).array for path in paths[1:]]
    return {"paths": paths, "target_file": target_file, "frames": frames, "plan": stack_frames(target_file.array, frames, WORK_DIM), "folder": folder}


def loaded_window(files: dict) -> MainWindow:
    """A window with the shared stack set directly, as if the load thread had finished."""
    window = new_window()
    window.resize(500, 700)
    window.set_stack(files["paths"][0], files["target_file"], files["paths"][1:], files["frames"], files["plan"])
    window.show()
    QApplication.processEvents()
    return window


def paint(window: MainWindow, x: float, y: float) -> None:
    """One short stroke at picture position (column x, row y)."""
    canvas = window.canvas
    canvas.radius = 12.0
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, canvas.to_widget(x, y).toPoint())
    QTest.mouseMove(canvas, canvas.to_widget(x + 6.0, y).toPoint())
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, canvas.to_widget(x + 6.0, y).toPoint())


def test_window_starts_empty_and_a_stack_enables_saving(files: dict) -> None:
    """Before a stack is set saving is off; after it the canvas shows the picture, the title names the target, and the status line shows the brush."""
    empty = new_window()
    assert not empty.save_action.isEnabled()
    window = loaded_window(files)
    assert window.save_action.isEnabled()
    assert "target.tif" in window.windowTitle()
    assert window.canvas.pixels.shape == (*window.canvas.state.target.shape[:2], 3)
    assert "brush" in window.status_label.text()


def test_undo_and_redo_actions_drive_the_canvas(files: dict) -> None:
    """A stroke creates a patch; the Undo action removes it and the Redo action brings it back."""
    window = loaded_window(files)
    paint(window, 600.0, 400.0)
    state = window.canvas.state
    assert state.patch_count == 1
    window.undo_action.trigger()
    assert state.patch_count == 0
    window.redo_action.trigger()
    assert state.patch_count == 1
    assert "patch 1 of 1" in window.status_label.text()


def test_save_writes_the_full_size_render_of_the_edited_patches(files: dict) -> None:
    """Saving renders the full-size picture with the current weights, keeping the target's size and ICC profile, and shows progress."""
    window = loaded_window(files)
    paint(window, 600.0, 400.0)
    weights = window.canvas.state.weights()
    out = str(files["folder"] / "saved.tif")
    window.save_to(out)
    window._save_thread.wait(60000)
    QApplication.processEvents()
    saved = TiffFile().read(out)
    expected = render_full(files["target_file"].array, files["frames"], files["plan"].frames, weights, 256)
    assert saved.array.shape == (*SHAPE, 3)
    assert saved.icc_name == "Adobe RGB (1998)"
    assert np.abs(saved.array - expected).max() <= 1.5 / 65535
    unedited = render_full(files["target_file"].array, files["frames"], files["plan"].frames, files["plan"].weights, 256)
    assert np.abs(saved.array - unedited).max() > 0.02
    assert "Saved" in window.statusBar().currentMessage()
    assert not window.progress_bar.isVisible()


def test_quitting_with_unsaved_patches_asks_first(files: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing with patches asks; No keeps the window open and Yes closes it. With no patches it closes without asking."""
    window = loaded_window(files)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: asked.append(1) or QMessageBox.StandardButton.No)
    assert window.close()
    assert asked == []
    window = loaded_window(files)
    paint(window, 600.0, 400.0)
    assert not window.close()
    assert asked == [1] and window.isVisible()
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    assert window.close()


def test_open_aligns_in_a_thread_and_then_shows_the_stack(files: dict) -> None:
    """open() starts the alignment thread; when it finishes the window holds a stack with one layer per frame."""
    window = new_window()
    window.open(files["paths"][0], files["paths"][1:], WORK_DIM, "focus")
    assert window.progress_bar.isVisible() or window.progress_bar.maximum() == 0
    window._load_thread.wait(120000)
    QApplication.processEvents()
    assert window.save_action.isEnabled()
    assert window.canvas.state.frame_count == 2
    assert window.exit_code == 0
    assert not window.progress_bar.isVisible()


def test_open_reports_a_failed_registration_and_sets_the_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrelated pictures show an error, leave saving off and make the exit code 1."""
    for name, seed in (("target", 5), ("frame", 6)):
        write_tif(tmp_path / f"{name}.tif", make_texture(SHAPE, seed=seed))
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda parent, title, text, *args: shown.append(text))
    window = new_window()
    window.open(str(tmp_path / "target.tif"), [str(tmp_path / "frame.tif")], WORK_DIM, "focus")
    window._load_thread.wait(120000)
    QApplication.processEvents()
    assert window.exit_code == 1
    assert not window.save_action.isEnabled()
    assert len(shown) == 1 and "registration failed" in shown[0]


def test_zoom_actions_set_the_levels_and_the_status_gives_the_full_size_share(files: dict) -> None:
    """The toolbar and menu actions set 100%, 200% and 400% and step in and out; the status line relates them to the full size."""
    window = loaded_window(files)
    canvas = window.canvas
    window.zoom_100_action.trigger()
    assert canvas.scale == 1.0
    assert "zoom 100% (75% of full size)" in window.status_label.text()   # the 720 x 960 preview of a 960 x 1280 picture
    window.zoom_200_action.trigger()
    assert canvas.scale == 2.0
    window.zoom_400_action.trigger()
    assert canvas.scale == 4.0
    window.zoom_out_action.trigger()
    assert canvas.scale == 2.0
    window.zoom_in_action.trigger()
    window.zoom_in_action.trigger()
    assert canvas.scale == 8.0
    window.zoom_fit_action.trigger()
    assert canvas.scale == min(canvas.width() / 960, canvas.height() / 720)
    assert {action.text() for action in window.toolbar.actions()} >= {"Fit", "100%", "200%", "400%"}


def test_layers_panel_lists_the_target_and_frames_and_follows_the_active_patch(files: dict) -> None:
    """The panel shows the target then each frame; it highlights the active patch's layer through painting, Tab and Escape."""
    window = loaded_window(files)
    panel = window.layers
    assert [panel.list.item(i).text() for i in range(panel.list.count())] == ["Target", "one.tif", "two.tif"]
    assert panel.list.currentRow() == -1
    paint(window, 600.0, 400.0)
    start = window.canvas.state.patch_label(0)
    assert panel.list.currentRow() == start
    QTest.keyClick(window.canvas, Qt.Key.Key_Tab)
    assert panel.list.currentRow() == (start + 1) % 3
    QTest.keyClick(window.canvas, Qt.Key.Key_Tab)
    assert panel.list.currentRow() == (start + 2) % 3
    QTest.keyClick(window.canvas, Qt.Key.Key_Escape)
    assert panel.list.currentRow() == -1


def test_clicking_a_layer_gives_it_to_the_active_patch(files: dict) -> None:
    """Clicking a row changes the active patch's layer and the picture; with no active patch it only shows a hint."""
    window = loaded_window(files)
    panel = window.layers
    click = lambda row: QTest.mouseClick(
        panel.list.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, panel.list.visualItemRect(panel.list.item(row)).center()
    )
    click(1)
    assert window.canvas.state.patch_count == 0
    assert "Paint a patch" in window.statusBar().currentMessage()
    paint(window, 600.0, 400.0)
    first = (window.canvas.state.patch_label(0) + 1) % 3
    second = (first + 1) % 3
    before = window.canvas.pixels.copy()
    click(first)
    assert window.canvas.state.patch_label(0) == first
    assert panel.list.currentRow() == first
    assert not np.array_equal(window.canvas.pixels, before)
    click(second)
    assert window.canvas.state.patch_label(0) == second
    assert panel.list.currentRow() == second
    assert np.array_equal(window.canvas.shown_pixels(), to_uint8_display(window.canvas.state.result()))


def test_pan_button_and_space_key_stay_in_step(files: dict) -> None:
    """The toolbar's Pan button toggles pan mode and follows the Space and Esc keys."""
    window = loaded_window(files)
    assert window.pan_action.isCheckable() and not window.pan_action.isChecked()
    assert "Pan" in {action.text() for action in window.toolbar.actions()}
    window.pan_action.trigger()
    assert window.canvas.pan_mode and window.pan_action.isChecked()
    QTest.keyClick(window.canvas, Qt.Key.Key_Space)
    assert not window.canvas.pan_mode and not window.pan_action.isChecked()
    QTest.keyClick(window.canvas, Qt.Key.Key_Space)
    assert window.pan_action.isChecked()
    QTest.keyClick(window.canvas, Qt.Key.Key_Escape)
    assert not window.canvas.pan_mode and not window.pan_action.isChecked()


def test_open_in_clean_mode_adds_the_median_as_the_last_layer(files: dict) -> None:
    """In clean mode the panel lists the target, the frames and then the clean plate, and the editor has one more layer."""
    window = new_window()
    window.open(files["paths"][0], files["paths"][1:], WORK_DIM, "clean")
    window._load_thread.wait(120000)
    QApplication.processEvents()
    panel = window.layers
    assert [panel.list.item(i).text() for i in range(panel.list.count())] == ["Target", "one.tif", "two.tif", "Clean plate (median)"]
    assert all(not panel.list.item(i).icon().isNull() for i in range(4))
    assert window.canvas.state.layer_count == 4 and window.canvas.state.frame_count == 2
    paint(window, 600.0, 400.0)
    assert window.layers.list.currentRow() == window.canvas.state.patch_label(0)


def lightroom_window(files: dict, folder: Path, output: str | None = None) -> tuple[MainWindow, "stack_bridge.Request"]:
    """A loaded window switched to Lightroom mode, with its request pointing into `folder`."""
    window = loaded_window(files)
    first = output or str(folder / "out.tif")
    request = stack_bridge.Request(
        "focus", files["paths"][0], files["paths"][1:], first, str(folder / "result.txt"), WORK_DIM, [first, str(folder / "out1.tif"), str(folder / "out2.tif")]
    )
    window.use_lightroom(request)
    return window, request


def test_lightroom_mode_puts_save_and_cancel_on_the_toolbar(files: dict, tmp_path: Path) -> None:
    """Lightroom mode renames the save action and adds a Cancel button; both are on the toolbar."""
    window, _ = lightroom_window(files, tmp_path)
    assert window.save_action.text() == "Save to Lightroom" and window.save_action.isEnabled()
    assert {"Save to Lightroom", "Cancel"} <= {action.text() for action in window.toolbar.actions()}


def test_saving_to_lightroom_writes_the_picture_and_the_saved_result_and_closes(files: dict, tmp_path: Path) -> None:
    """Save writes the full-size picture to the request's output, answers saved, closes the window and the exit code is 0."""
    window, request = lightroom_window(files, tmp_path)
    paint(window, 600.0, 400.0)
    window.save_action.trigger()
    window._save_thread.wait(60000)
    QApplication.processEvents()
    assert TiffFile().read(request.output).array.shape == (*SHAPE, 3)
    assert read_values(request.result) == {"status": "saved", "output": request.output, "target": "0"}
    assert not window.isVisible()
    assert window.finish_lightroom() == 0
    assert read_values(request.result)["status"] == "saved"


def test_cancel_without_patches_closes_and_reports_cancelled(files: dict, tmp_path: Path) -> None:
    """Cancel closes at once when nothing was painted: status cancelled, exit code 1, no picture written."""
    window, request = lightroom_window(files, tmp_path)
    window.cancel_action.trigger()
    assert not window.isVisible()
    assert window.finish_lightroom() == 1
    assert read_values(request.result) == {"status": "cancelled"}
    assert not Path(request.output).exists()


def test_cancel_with_patches_asks_before_discarding(files: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With patches Cancel asks; No keeps the window open, Yes discards: cancelled, exit code 1."""
    window, request = lightroom_window(files, tmp_path)
    paint(window, 600.0, 400.0)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)
    window.cancel_action.trigger()
    assert window.isVisible() and not Path(request.result).exists()
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    window.cancel_action.trigger()
    assert not window.isVisible() and window.finish_lightroom() == 1
    assert read_values(request.result) == {"status": "cancelled"}


def test_alignment_failure_is_an_error_result_with_the_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrelated pictures: the result says error with the one-line reason and the exit code is 2."""
    for name, seed in (("target", 5), ("frame", 6)):
        write_tif(tmp_path / f"{name}.tif", make_texture(SHAPE, seed=seed))
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: None)
    window = new_window()
    request = stack_bridge.Request("clean", str(tmp_path / "target.tif"), [str(tmp_path / "frame.tif")], str(tmp_path / "o.tif"), str(tmp_path / "r.txt"), WORK_DIM, [str(tmp_path / "o.tif"), str(tmp_path / "o1.tif")])
    window.use_lightroom(request)
    window.open(request.target, request.frames, request.work_dim, request.mode)
    window._load_thread.wait(120000)
    QApplication.processEvents()
    assert window.finish_lightroom() == 2
    values = read_values(request.result)
    assert values["status"] == "error" and "registration failed" in values["message"]


def test_a_failed_save_keeps_the_window_open_and_ends_as_an_error(files: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the output cannot be written the window stays open to try again; closing it then reports the error, exit code 2."""
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda parent, title, text, *args: shown.append(text))
    window, request = lightroom_window(files, tmp_path, output=str(tmp_path / "missing_folder" / "out.tif"))
    window.save_action.trigger()
    window._save_thread.wait(60000)
    QApplication.processEvents()
    assert window.isVisible() and len(shown) == 1 and "Could not save" in shown[0]
    window._unsaved = False
    window.close()
    assert window.finish_lightroom() == 2
    assert read_values(request.result)["status"] == "error"


def ctrl_click(window: MainWindow, row: int) -> None:
    """Ctrl+clicks a row of the Layers panel."""
    panel = window.layers
    QTest.mouseClick(
        panel.list.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, panel.list.visualItemRect(panel.list.item(row)).center()
    )


def test_ctrl_click_views_a_layer_and_ctrl_click_again_goes_back(files: dict) -> None:
    """Ctrl+click shows that layer's picture and marks its row bold; the same again, or Esc, returns to the merged picture."""
    window = loaded_window(files)
    ctrl_click(window, 2)
    assert window.canvas.viewed_layer == 2 and window.layers.list.item(2).font().bold()
    assert np.array_equal(window.canvas.shown_pixels(), to_uint8_display(window.canvas.state.layer_image(2)))
    assert "viewing two.tif" in window.status_label.text()
    ctrl_click(window, 2)
    assert window.canvas.viewed_layer is None and not window.layers.list.item(2).font().bold()
    ctrl_click(window, 1)
    QTest.keyClick(window.canvas, Qt.Key.Key_Escape)
    assert window.canvas.viewed_layer is None and not window.layers.list.item(1).font().bold()


def wait_for_restack(window: MainWindow) -> None:
    """Waits for the thread that re-aligns after the target changed and delivers its answer."""
    window._restack_thread.wait(120000)
    QApplication.processEvents()


def test_setting_a_frame_as_the_target_restacks_with_it_first(files: dict) -> None:
    """The chosen frame becomes the target, the old target takes its place among the frames, and the title and panel follow."""
    window = loaded_window(files)
    window.layers.target_requested.emit(2)
    wait_for_restack(window)
    assert [window.layers.list.item(i).text() for i in range(3)] == ["Target", "one.tif", "target.tif"]
    assert "two.tif" in window.windowTitle()
    assert np.allclose(window.canvas.state.target, downsample_image(files["frames"][1], WORK_DIM))
    assert window.canvas.state.frame_count == 2 and window.canvas.state.patch_count == 0
    assert window.layers.isEnabled() and window.canvas.isEnabled()


def test_setting_the_target_with_patches_asks_first_and_discards_them(files: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """With patches painted the change asks; No leaves everything alone, Yes re-stacks and the patches are gone."""
    window = loaded_window(files)
    paint(window, 600.0, 400.0)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)
    window.layers.target_requested.emit(1)
    assert window.canvas.state.patch_count == 1 and window._restack_thread is None
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    window.layers.target_requested.emit(1)
    wait_for_restack(window)
    assert window.canvas.state.patch_count == 0 and "one.tif" in window.windowTitle()


def test_a_failed_restack_keeps_the_old_stack_and_reenables_the_window(files: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the new target cannot be aligned with the others, a dialog says so and nothing else changes."""
    window = loaded_window(files)
    window._frame_arrays = [make_texture(SHAPE, seed=31)] + list(window._frame_arrays[1:])
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda parent, title, text, *args: shown.append(text))
    window.layers.target_requested.emit(1)
    wait_for_restack(window)
    assert len(shown) == 1 and "registration failed" in shown[0]
    assert "target.tif" in window.windowTitle() and window.layers.isEnabled() and window.canvas.isEnabled()


def test_lightroom_save_after_changing_the_target_writes_that_photos_output_and_reports_it(files: dict, tmp_path: Path) -> None:
    """After frame 2 became the target the picture goes to output 2 and the result says target=2."""
    window, request = lightroom_window(files, tmp_path)
    window.layers.target_requested.emit(2)
    wait_for_restack(window)
    window.save_action.trigger()
    window._save_thread.wait(60000)
    QApplication.processEvents()
    assert Path(request.outputs[2]).exists() and not Path(request.outputs[0]).exists()
    assert read_values(request.result) == {"status": "saved", "output": request.outputs[2], "target": "2"}


def test_open_in_clean2_mode_lists_the_plate_and_saves_with_it(files: dict) -> None:
    """clean2's last layer is called "Clean plate", not a median, and the saved picture is rendered with that plate."""
    window = new_window()
    window.open(files["paths"][0], files["paths"][1:], WORK_DIM, "clean2")
    window._load_thread.wait(120000)
    QApplication.processEvents()
    panel = window.layers
    assert [panel.list.item(i).text() for i in range(panel.list.count())][-1] == "Clean plate"
    assert not window._plan.plate_is_median
    paint(window, 600.0, 400.0)
    weights = window.canvas.state.weights()
    out = str(files["folder"] / "saved2.tif")
    window.save_to(out)
    window._save_thread.wait(60000)
    QApplication.processEvents()
    expected = render_full(files["target_file"].array, files["frames"], window._plan.frames, weights, 256, window._plan.median)
    assert np.abs(TiffFile().read(out).array - expected).max() <= 1.5 / 65535
