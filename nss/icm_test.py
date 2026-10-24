"""Tests of nss.icm: the background preview, the full-size save and the Lightroom mode's saved, cancelled and error answers."""
import time
from pathlib import Path
from typing import Callable, Iterator
from unittest.mock import patch
import numpy as np
import pytest
import tifffile
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox
from nss.icc import gamma_icc_bytes
from nss.icm import ICMWindow
from nss.icm_render import render
from nss.tiff_file import TiffFile
from plugin import icm_bridge
from plugin.handover import read_values

ICC = gamma_icc_bytes("Adobe RGB (1998)", [(0.60974, 0.31111, 0.01947), (0.20528, 0.62567, 0.06087), (0.14919, 0.06322, 0.74457)], 563 / 256)
SHAPE = (650, 1100, 3)       # larger than the offscreen screen, so the preview is smaller than the picture
_KEEP = []                   # keeps the application alive until the end of the test session
_OPEN = []                   # the current test's windows, closed after it


def _wait(condition: Callable[[], bool], timeout: float = 30.0) -> None:
    """Processes Qt events until condition() holds."""
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, "timed out"
        QApplication.processEvents()
        time.sleep(0.005)


@pytest.fixture(autouse=True)
def _close_windows() -> Iterator[None]:
    """Closes the test's windows afterwards (discarding), so that no preview timer or render thread outlives the test."""
    yield
    with _answer(QMessageBox.StandardButton.Discard):
        for window in _OPEN:
            window.close()
    _OPEN.clear()


def _answer(button: QMessageBox.StandardButton) -> patch:
    """Answers the save-before-close question with button."""
    return patch.object(ICMWindow, "ask_save_before_close", return_value=button)


def _window(tmp_path: Path, output: str = "", draw: bool = True, lightroom: bool = True) -> ICMWindow:
    """A shown ICM window (6 frames) on a 16-bit Adobe RGB TIFF, with a Lightroom request when lightroom and a motion path when draw."""
    _KEEP.append(QApplication.instance() or QApplication([]))     # an unreferenced QApplication is deleted at once
    raw = np.linspace(1000, 60000, np.prod(SHAPE)).reshape(SHAPE)
    raw[200:300, 400:500] = 65000                                  # a bright block for the motion to smear
    TiffFile.from_array(raw / 65535.0, icc=ICC).saveas(str(tmp_path / "in.tif"))
    request = icm_bridge.Request(str(tmp_path / "in.tif"), output or str(tmp_path / "in_icm.tif"), str(tmp_path / "result.txt"))
    window = ICMWindow(TiffFile().read(request.tiff), None)
    if lightroom:
        window.add_lightroom_button(request)
    window.shutter_dropdown.setCurrentText("1/10")
    window.show()
    window.display_image()
    _OPEN.append(window)
    if draw:
        rect = window.canvas.image_rect()
        stroke = [QtCore.QPoint(int(rect.x() + rect.width() * (0.3 + 0.03 * i)), int(rect.y() + rect.height() * (0.4 + 0.002 * i * i)))
                  for i in range(12)]
        window.canvas.fit_stroke(stroke)
    return window


def _save(window: ICMWindow) -> None:
    """Presses Save to Lightroom and waits for the full-size render to end."""
    window.save_to_lightroom()
    _wait(lambda: window.progress_bar.isHidden())


def test_preview_renders_in_the_background_at_screen_size(tmp_path: Path) -> None:
    """After a path is drawn the preview of the smaller preview picture is rendered off the GUI thread and shown."""
    window = _window(tmp_path)
    assert window.preview_source.shape[1] < SHAPE[1] and window.save_button.isEnabled()
    _wait(lambda: window.status_label.text() == "Preview ready.")
    assert window.shown_array.shape == window.preview_source.shape
    assert np.array_equal(window.shown_array, render(window.preview_source, window.motion()))
    assert np.abs(window.shown_array - window.preview_source).max() > 0.01


def test_save_writes_the_full_size_render_with_the_profile_and_closes(tmp_path: Path) -> None:
    """The output is the full-size render, uncompressed 16-bit with the input's ICC profile; status=saved, exit code 0."""
    window = _window(tmp_path)
    motion = window.motion()
    _save(window)
    out = tmp_path / "in_icm.tif"
    assert not window.isVisible() and window.lightroom_status == icm_bridge.SAVED
    with tifffile.TiffFile(out) as f:
        page = f.pages.first
        assert page.dtype == np.uint16 and page.shape == SHAPE and page.compression == tifffile.COMPRESSION.NONE
    saved = TiffFile().read(str(out))
    assert saved.iccprofile == ICC
    assert np.abs(saved.array - render(window.original_array, motion)).max() <= 1 / 65535
    assert np.abs(saved.array - window.original_array).max() > 0.01   # the motion did change the picture
    assert read_values(str(tmp_path / "result.txt")) == {"status": "saved", "output": str(out)}
    assert window.finish_lightroom() == 0
    assert read_values(str(tmp_path / "result.txt"))["status"] == "saved"


def test_path_survives_a_resize(tmp_path: Path) -> None:
    """The path is kept in fractions of the picture, so resizing the window keeps the same motion."""
    window = _window(tmp_path)
    before = window.motion()
    window.resize(900, 700)
    window.display_image()
    assert np.array_equal(window.motion().path, before.path)


def test_failed_save_stays_open_and_closing_answers_error(tmp_path: Path) -> None:
    """An unwritable output shows the error and keeps the window open; closing it then writes status=error and exits 2."""
    (tmp_path / "a_folder").mkdir()
    window = _window(tmp_path, output=str(tmp_path / "a_folder"))
    _save(window)
    assert window.isVisible() and window.control_panel.isEnabled() and window.status_label.text().startswith("Error: ")
    assert not (tmp_path / "result.txt").exists()
    with _answer(QMessageBox.StandardButton.Discard):
        window.close()
    assert window.finish_lightroom() == 2
    result = read_values(str(tmp_path / "result.txt"))
    assert result["status"] == "error" and result["message"] == window.lightroom_message and "a_folder" in result["message"]


def test_close_without_save_answers_cancelled(tmp_path: Path) -> None:
    """Closing with the motion unsaved asks first; Discard writes status=cancelled, no output, and exits 1."""
    window = _window(tmp_path)
    with _answer(QMessageBox.StandardButton.Discard) as ask:
        window.close()
    assert ask.call_count == 1 and not window.isVisible()
    assert window.finish_lightroom() == 1
    assert read_values(str(tmp_path / "result.txt")) == {"status": "cancelled"}
    assert not (tmp_path / "in_icm.tif").exists()


def test_close_during_the_full_size_render_cancels_it(tmp_path: Path) -> None:
    """Closing while the full-size picture renders stops the render and writes nothing but the cancelled answer."""
    window = _window(tmp_path)
    window.samples_spin.setValue(300)                 # 30 frames
    window.save_to_lightroom()
    thread = window.full_thread
    window.close()
    assert not thread.isRunning()
    _wait(lambda: thread.isFinished())
    QApplication.processEvents()                      # a late result of the render is dropped
    assert window.finish_lightroom() == 1 and not (tmp_path / "in_icm.tif").exists()


def test_save_without_a_path_asks_for_one(tmp_path: Path) -> None:
    """Without a drawn motion path nothing is rendered or written and the window stays open."""
    window = _window(tmp_path, draw=False)
    window.save_to_lightroom()
    assert window.isVisible() and window.lightroom_status == icm_bridge.CANCELLED and "motion path" in window.status_label.text()
    assert not (tmp_path / "in_icm.tif").exists() and not (tmp_path / "result.txt").exists()


def test_close_cancelled_at_the_question_stays_open(tmp_path: Path) -> None:
    """Cancel at the save-before-close question keeps the window open and writes nothing."""
    window = _window(tmp_path)
    with _answer(QMessageBox.StandardButton.Cancel):
        window.close()
    assert window.isVisible() and not (tmp_path / "result.txt").exists()


def test_close_saved_at_the_question_saves_to_lightroom_then_closes(tmp_path: Path) -> None:
    """Save at the question in Lightroom mode renders, saves to Lightroom and then closes without asking again."""
    window = _window(tmp_path)
    with _answer(QMessageBox.StandardButton.Save) as ask:
        window.close()
        _wait(lambda: window.progress_bar.isHidden())
    assert ask.call_count == 1 and not window.isVisible() and window.finish_lightroom() == 0
    assert (tmp_path / "in_icm.tif").exists()


def test_close_saved_at_the_question_saves_the_image_then_closes(tmp_path: Path) -> None:
    """Save at the question without Lightroom asks for the file, writes the full-size picture and closes."""
    window = _window(tmp_path, lightroom=False)
    out = tmp_path / "out" / "saved.tif"
    with _answer(QMessageBox.StandardButton.Save), patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(out), "")):
        window.close()
        _wait(lambda: window.progress_bar.isHidden())
    assert not window.isVisible() and TiffFile().read(str(out)).array.shape == SHAPE


def test_close_without_a_motion_does_not_ask(tmp_path: Path) -> None:
    """Without a path there is nothing to save: the window closes at once."""
    window = _window(tmp_path, draw=False)
    with _answer(QMessageBox.StandardButton.Cancel) as ask:
        window.close()
    assert ask.call_count == 0 and not window.isVisible()


def test_reset_clears_the_path_the_curves_and_the_preview(tmp_path: Path) -> None:
    """Reset drops the path and its preview and flattens the curves; the exposure settings stay and closing no longer asks."""
    window = _window(tmp_path)
    window.rotation_curve.points[1] = [0.33, 0.9]
    window.zoom_curve.points[2] = [0.66, 0.2]
    window.speed_curve.points[3] = [1.0, 0.7]
    window.schedule_preview()
    _wait(lambda: window.status_label.text() == "Preview ready.")
    assert window.shown_array is not window.preview_source
    window.reset()
    assert window.canvas.tck is None and window.shown_array is window.preview_source and not window.save_button.isEnabled()
    for curve in (window.rotation_curve, window.zoom_curve, window.speed_curve):
        assert np.allclose(curve.get_curve_values(7), 0.5)
    assert window.shutter_dropdown.currentText() == "1/10" and not window.unsaved_changes
    with _answer(QMessageBox.StandardButton.Cancel) as ask:
        window.close()
    assert ask.call_count == 0 and not window.isVisible()


def test_escape_asks_before_closing(tmp_path: Path) -> None:
    """Escape closes the window like its close button: with the motion unsaved it asks first, and Discard closes it."""
    window = _window(tmp_path)
    window.activateWindow()
    _wait(lambda: window.isActiveWindow())
    with _answer(QMessageBox.StandardButton.Discard) as ask:
        QTest.keyClick(window.canvas, QtCore.Qt.Key.Key_Escape)
    assert ask.call_count == 1 and not window.isVisible()


def test_multi_exposure_disables_the_shutter(tmp_path: Path) -> None:
    """A multi-exposure counts total frames, so the shutter is disabled while it is on and enabled again after."""
    window = _window(tmp_path)
    assert window.shutter_dropdown.isEnabled()
    window.multi_expo_check.setChecked(True)
    assert not window.shutter_dropdown.isEnabled() and window.samples_label.text() == "Total Frames:"
    window.multi_expo_check.setChecked(False)
    assert window.shutter_dropdown.isEnabled() and window.samples_label.text() == "Samples/sec:"


def test_two_frames_in_both_modes(tmp_path: Path) -> None:
    """Samples/sec and Total Frames both go down to 2, and Total Frames keeps even counts and steps by 1."""
    window = _window(tmp_path)
    window.samples_spin.setValue(2)
    assert window.samples_spin.value() == 2
    window.samples_spin.setValue(20)
    window.multi_expo_check.setChecked(True)
    assert window.samples_spin.value() == 20 and len(window.motion().angle) == 20
    window.samples_spin.setValue(2)
    assert window.samples_spin.value() == 2 and len(window.motion().angle) == 2
    window.samples_spin.stepUp()
    assert window.samples_spin.value() == 3
