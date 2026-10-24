#!/usr/bin/env python3
"""
Intentional Camera Movement (ICM) simulator: draw a motion path on the picture and shape it with the rotation, zoom and speed curves.  A
screen-size preview is rendered in a background thread after every change; saving renders the full-size picture (nss.icm_render).

    python -m nss.icm picture.tif [-o out.tif]
    python -m nss.icm --lightroom <request file>      from the Lightroom plug-in (plugin/icm_bridge.py)
"""
import argparse
import os
import sys
from typing import Callable

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from scipy.interpolate import PchipInterpolator, splev

from nss import utils
from nss.icm_render import Motion, fit_path, frame_count, render, sample_motion, shutter_seconds
from nss.image_utils import downsample_image, to_uint8_display
from nss.tiff_file import TiffFile
from plugin import icm_bridge

SHUTTER_SPEEDS = ("1/100", "1/80", "1/60", "1/50", "1/40", "1/30", "1/25", "1/20", "1/15", "1/13", "1/10", "1/8", "1/6", "1/5", "1/4", "1/3",
                  "1/2.5", "1/2", "1/1.6", "1/1.3", "1s", "1.3s", "1.6s", "2s", "2.5s", "3s", "4s", "5s")      # 1/3 stops
PREVIEW_DELAY_MS = 200       # the preview renders this long after the last change
RESIZE_DELAY_MS = 200        # the picture is refitted this long after the last resize


class CurveEditor(QtWidgets.QWidget):
    """A curve over the exposure (x: time 0..1, y: value 0..1) through four draggable points, interpolated with PCHIP."""

    changed = QtCore.pyqtSignal()        # a point was dragged and let go

    def __init__(self, default_val: float = 0.5) -> None:
        """A flat curve at default_val."""
        super().__init__()
        self.setMinimumHeight(150)
        self.default_val = default_val
        self.points = [[0.0, default_val], [0.33, default_val], [0.66, default_val], [1.0, default_val]]
        self.active_idx: int | None = None
        self.margin = 15

    def reset(self) -> None:
        """Makes the curve flat at its default value again."""
        self.points = [[0.0, self.default_val], [0.33, self.default_val], [0.66, self.default_val], [1.0, self.default_val]]
        self.update()

    def get_curve_values(self, num_steps: int = 60) -> np.ndarray:
        """The curve at num_steps evenly spaced times from 0 to 1."""
        pts = sorted(self.points, key=lambda p: p[0])
        return PchipInterpolator([p[0] for p in pts], [p[1] for p in pts])(np.linspace(0, 1, num_steps))

    def _to_screen(self, norm_x: float, norm_y: float) -> tuple[int, int]:
        """Widget pixels of a curve point."""
        w, h = self.width() - 2 * self.margin, self.height() - 2 * self.margin
        return int(self.margin + norm_x * w), int(self.margin + (1.0 - norm_y) * h)

    def _to_norm(self, screen_x: float, screen_y: float) -> tuple[float, float]:
        """The curve point (clipped to 0..1) at widget pixels."""
        w, h = self.width() - 2 * self.margin, self.height() - 2 * self.margin
        nx = max(0.0, min(1.0, (screen_x - self.margin) / w))
        ny = max(0.0, min(1.0, 1.0 - (screen_y - self.margin) / h))
        return nx, ny

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Picks the point within 10 pixels of a left click."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            mx, my = event.pos().x(), event.pos().y()
            for i, (nx, ny) in enumerate(self.points):
                sx, sy = self._to_screen(nx, ny)
                if (sx - mx) ** 2 + (sy - my) ** 2 < 100:
                    self.active_idx = i
                    break

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Drags the picked point; the first and last points stay at time 0 and 1."""
        if self.active_idx is not None:
            nx, ny = self._to_norm(event.pos().x(), event.pos().y())
            if self.active_idx == 0:
                nx = 0.0
            elif self.active_idx == len(self.points) - 1:
                nx = 1.0
            self.points[self.active_idx] = [nx, ny]
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Lets go of the picked point and emits changed."""
        if self.active_idx is not None:
            self.active_idx = None
            self.changed.emit()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Draws the middle line, the curve and its points."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200), 1, QtCore.Qt.PenStyle.DashLine))
        painter.drawLine(self.margin, self.height() // 2, self.width() - self.margin, self.height() // 2)

        curve_vals = self.get_curve_values(60)
        pen_curve = QtGui.QPen(QtGui.QColor(0, 150, 255))
        pen_curve.setWidth(2)
        painter.setPen(pen_curve)
        for i in range(59):
            x1, y1 = self._to_screen(i / 59.0, curve_vals[i])
            x2, y2 = self._to_screen((i + 1) / 59.0, curve_vals[i + 1])
            painter.drawLine(x1, y1, x2, y2)

        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 100, 100)))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for nx, ny in self.points:
            sx, sy = self._to_screen(nx, ny)
            painter.drawEllipse(QtCore.QPoint(sx, sy), 5, 5)


class ImageCanvas(QtWidgets.QLabel):
    """Shows the picture fitted to the widget and records the motion path drawn on it with the left mouse button."""

    path_started = QtCore.pyqtSignal()   # a new stroke began (the old path is gone)
    path_changed = QtCore.pyqtSignal()   # the stroke ended: tck is its path, or None when it was too short

    def __init__(self) -> None:
        """No picture and no path yet."""
        super().__init__()
        self.stroke: list[QtCore.QPoint] = []    # the stroke being drawn, widget pixels
        self.tck: tuple | None = None            # the motion path, in fractions of the picture (nss.icm_render.fit_path)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)

    def image_rect(self) -> QtCore.QRectF:
        """Where the (centred) picture is in widget pixels."""
        size = self.pixmap().size()
        return QtCore.QRectF((self.width() - size.width()) / 2, (self.height() - size.height()) / 2, size.width(), size.height())

    def fit_stroke(self, stroke: list[QtCore.QPoint]) -> None:
        """Makes the path of a stroke drawn in widget pixels (none when it is too short) and emits path_changed."""
        rect = self.image_rect()
        points = np.array([(p.x(), p.y()) for p in stroke], dtype=float).reshape(-1, 2)
        self.tck = fit_path(points, (rect.x(), rect.y(), rect.width(), rect.height()))
        self.update()
        self.path_changed.emit()

    def clear_path(self) -> None:
        """Drops the path and any stroke being drawn."""
        self.stroke = []
        self.tck = None
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Starts a new stroke, dropping the path."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.stroke = [event.pos()]
            self.tck = None
            self.path_started.emit()
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Extends the stroke."""
        if self.stroke and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.stroke.append(event.pos())
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Ends the stroke and fits the path to it."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.stroke:
            stroke, self.stroke = self.stroke, []
            self.fit_stroke(stroke)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Draws the picture, then the stroke being drawn (red) or the path (green)."""
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if len(self.stroke) >= 2:
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawPolyline(QtGui.QPolygon(self.stroke))
        if self.tck is not None:
            rect = self.image_rect()
            x, y = splev(np.linspace(0, 1, 200), self.tck)
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(rect.x() + px * rect.width(), rect.y() + py * rect.height()) for px, py in zip(x, y)]))


class RenderThread(QtCore.QThread):
    """Renders one ICM picture (nss.icm_render.render) off the GUI thread; cancel() stops it before its next frame."""

    progress = QtCore.pyqtSignal(int)        # frames done
    rendered = QtCore.pyqtSignal(object)     # the picture, or None when cancelled
    failed = QtCore.pyqtSignal(str)          # one line: the exception

    def __init__(self, image: np.ndarray, motion: Motion) -> None:
        """A thread that will render image with motion once started."""
        super().__init__()
        self.image = image
        self.motion = motion
        self.cancelled = False

    def cancel(self) -> None:
        """Asks the render to stop before its next frame."""
        self.cancelled = True

    def run(self) -> None:
        """Renders and emits rendered, or failed on an exception."""
        try:
            result = render(self.image, self.motion, lambda done, total: self.progress.emit(done), lambda: self.cancelled)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.rendered.emit(result)


class ICMWindow(QtWidgets.QMainWindow):
    """The ICM simulator window: exposure settings and curves on the left, the picture with its motion path and preview on the right."""

    def __init__(self, tiff_source: TiffFile, output_filename: str | None) -> None:
        """Shows tiff_source; Save Image writes to output_filename (asking when it is None or exists)."""
        super().__init__()
        self.setWindowTitle("ICM Motion Recorder")
        self.tiff_source = tiff_source
        self.original_array = tiff_source.array
        self.output_filename = output_filename
        screen = QtWidgets.QApplication.primaryScreen()
        screen_size = screen.availableGeometry().size() * screen.devicePixelRatio()
        self.preview_source = downsample_image(self.original_array, max(screen_size.width(), screen_size.height()))
        self.shown_array = self.preview_source          # what the canvas shows: the preview source or its ICM preview
        self.preview_thread: RenderThread | None = None
        self.full_thread: RenderThread | None = None
        self.lightroom_request: icm_bridge.Request | None = None    # set by add_lightroom_button
        self.unsaved_changes = False                    # a motion path is set and changed since the last save; closing asks first

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)

        self.control_panel = QtWidgets.QWidget()
        self.control_panel.setFixedWidth(250)
        self.control_layout = QtWidgets.QVBoxLayout(self.control_panel)

        exposure_group = QtWidgets.QGroupBox("Exposure Settings")
        exposure_layout = QtWidgets.QFormLayout(exposure_group)
        self.shutter_dropdown = QtWidgets.QComboBox()
        self.shutter_dropdown.addItems(SHUTTER_SPEEDS)
        self.shutter_dropdown.setCurrentText("1s")
        exposure_layout.addRow("Shutter:", self.shutter_dropdown)
        self.samples_spin = QtWidgets.QSpinBox()
        self.samples_spin.setRange(2, 300)
        self.samples_spin.setValue(60)
        self.samples_label = QtWidgets.QLabel("Samples/sec:")
        exposure_layout.addRow(self.samples_label, self.samples_spin)
        self.multi_expo_check = QtWidgets.QCheckBox("Enable")
        self.multi_expo_check.toggled.connect(self.toggle_multi_exposure)
        exposure_layout.addRow("Multi-Expo:", self.multi_expo_check)
        self.control_layout.addWidget(exposure_group)

        self.rotation_curve = CurveEditor(default_val=0.5)
        self.zoom_curve = CurveEditor(default_val=0.5)
        self.speed_curve = CurveEditor(default_val=0.5)
        for title, curve in (("Rotation Curve", self.rotation_curve), ("Zoom/Scale Curve", self.zoom_curve), ("Motion Speed Curve", self.speed_curve)):
            group = QtWidgets.QGroupBox(title)
            QtWidgets.QVBoxLayout(group).addWidget(curve)
            self.control_layout.addWidget(group)
            curve.changed.connect(self.schedule_preview)

        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        self.control_layout.addWidget(self.reset_button)
        self.save_button = QtWidgets.QPushButton("Save Image")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(lambda: self.save_image())     # not clicked's checked argument as close_after
        self.control_layout.addWidget(self.save_button)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.hide()
        self.control_layout.addWidget(self.progress_bar)
        self.status_label = QtWidgets.QLabel("Draw a motion path on the image.")
        self.status_label.setWordWrap(True)
        self.control_layout.addWidget(self.status_label)
        self.control_layout.addStretch()

        self.canvas = ImageCanvas()
        self.canvas.path_started.connect(self.show_preview_source)
        self.canvas.path_changed.connect(self.schedule_preview)
        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.canvas)

        self.shutter_dropdown.currentTextChanged.connect(self.schedule_preview)
        self.samples_spin.valueChanged.connect(self.schedule_preview)
        self.multi_expo_check.toggled.connect(self.schedule_preview)

        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.start_preview)
        self.resize_timer = QtCore.QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.display_image)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self, self.close)    # like the close button, asks first when unsaved

        self.resize(1200, 800)
        self.display_image()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Refits the picture once the resizing pauses."""
        super().resizeEvent(event)
        self.resize_timer.start(RESIZE_DELAY_MS)

    def display_image(self) -> None:
        """Shows shown_array fitted to the canvas."""
        display_array = np.ascontiguousarray(to_uint8_display(self.shown_array))
        height, width = display_array.shape[:2]
        if display_array.ndim == 2:
            q_img = QtGui.QImage(display_array.data, width, height, width, QtGui.QImage.Format.Format_Grayscale8)
        else:
            q_img = QtGui.QImage(display_array.data, width, height, 3 * width, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)
        self.canvas.setPixmap(pixmap.scaled(self.canvas.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                            QtCore.Qt.TransformationMode.SmoothTransformation))

    def show_array(self, array: np.ndarray) -> None:
        """Makes array the shown picture."""
        self.shown_array = array
        self.display_image()

    def toggle_multi_exposure(self, is_checked: bool) -> None:
        """Multi-exposure counts total frames and disables the shutter; otherwise samples per second of shutter time."""
        self.shutter_dropdown.setEnabled(not is_checked)
        if is_checked:
            self.samples_label.setText("Total Frames:")
        else:
            self.samples_label.setText("Samples/sec:")

    def motion(self) -> Motion | None:
        """The camera motion of the drawn path and the current settings; None without a path."""
        if self.canvas.tck is None:
            return None
        frames = frame_count(shutter_seconds(self.shutter_dropdown.currentText()), self.samples_spin.value(), self.multi_expo_check.isChecked())
        return sample_motion(self.canvas.tck, self.speed_curve.get_curve_values(frames), self.rotation_curve.get_curve_values(frames),
                             self.zoom_curve.get_curve_values(frames))

    def schedule_preview(self) -> None:
        """Renders the preview once the changes pause; with a path, Save is possible and the change is unsaved."""
        self.save_button.setEnabled(self.canvas.tck is not None)
        self.unsaved_changes = self.canvas.tck is not None
        self.preview_timer.start(PREVIEW_DELAY_MS)

    def stop_preview(self) -> None:
        """Cancels a running preview render and waits for it (at most one frame)."""
        if self.preview_thread is not None:
            self.preview_thread.cancel()
            self.preview_thread.wait()
            self.preview_thread = None

    def show_preview_source(self) -> None:
        """Shows the picture without motion (while a new path is drawn, or after Reset); nothing is left to save."""
        self.preview_timer.stop()
        self.stop_preview()
        self.save_button.setEnabled(False)
        self.unsaved_changes = False
        self.show_array(self.preview_source)

    def reset(self) -> None:
        """Drops the motion path and its preview and makes the rotation, zoom and speed curves flat again (the exposure settings stay)."""
        self.canvas.clear_path()
        for curve in (self.rotation_curve, self.zoom_curve, self.speed_curve):
            curve.reset()
        self.show_preview_source()
        self.status_label.setText("Draw a motion path on the image.")

    def start_preview(self) -> None:
        """Renders the screen-size preview of the current motion in the background (without a path, shows the picture as it is)."""
        self.stop_preview()
        motion = self.motion()
        if motion is None:
            self.show_array(self.preview_source)
            self.status_label.setText("Draw a motion path on the image.")
            return
        self.status_label.setText(f"Rendering the preview ({len(motion.angle)} frames) ...")
        self.preview_thread = RenderThread(self.preview_source, motion)
        self.preview_thread.rendered.connect(self.on_preview_rendered)
        self.preview_thread.failed.connect(self.show_error)
        self.preview_thread.start()

    def on_preview_rendered(self, result: np.ndarray | None) -> None:
        """Shows a preview unless it was cancelled or superseded."""
        if result is not None and self.sender() is self.preview_thread:
            self.show_array(result)
            self.status_label.setText("Preview ready.")

    def show_error(self, message: str) -> None:
        """Shows an error in the status line."""
        self.status_label.setText(f"Error: {message}")

    def render_full_size(self, on_done: Callable[[np.ndarray], None], on_error: Callable[[str], None]) -> None:
        """
        Renders the full-size picture with the current motion in the background, with a progress bar and the controls disabled, then calls
        on_done(picture) or on_error(message).
        """
        motion = self.motion()
        if motion is None:
            self.status_label.setText("Draw a motion path on the image first.")
            return
        self.control_panel.setEnabled(False)
        self.canvas.setEnabled(False)
        self.progress_bar.setRange(0, len(motion.angle))
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status_label.setText("Rendering the full-size image ...")

        thread = RenderThread(self.original_array, motion)

        def finish() -> bool:
            """Puts the controls back; False when the window was closed meanwhile (closeEvent dropped the render)."""
            if self.full_thread is not thread:
                return False
            self.full_thread = None
            self.progress_bar.hide()
            self.control_panel.setEnabled(True)
            self.canvas.setEnabled(True)
            return True

        def done(result: np.ndarray | None) -> None:
            """Hands the picture on."""
            if finish() and result is not None:
                on_done(result)

        def failed(message: str) -> None:
            """Hands the error on."""
            if finish():
                on_error(message)

        self.full_thread = thread
        thread.progress.connect(self.progress_bar.setValue)
        thread.rendered.connect(done)
        thread.failed.connect(failed)
        thread.start()

    def write_tiff(self, picture: np.ndarray, path: str) -> None:
        """Writes picture to path with the source's ICC profile and bit depth."""
        self.tiff_source.array = picture
        self.tiff_source.saveas(path)

    def save_image(self, close_after: bool = False) -> None:
        """
        Asks where to save (unless -o names a file that does not exist yet), then renders the full-size picture and writes it there; with
        close_after the window closes once it is written.
        """
        save_path = self.output_filename
        if save_path is None or os.path.exists(save_path):
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ICM Image", save_path or "", "TIFF Images (*.tif *.tiff)")
        if save_path:
            self.render_full_size(lambda picture: self.write_and_report(picture, save_path, close_after), self.show_error)

    def write_and_report(self, picture: np.ndarray, path: str, close_after: bool) -> None:
        """Writes picture to path (creating its folder) and says so, or closes the window with close_after; path is the next save's default."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self.write_tiff(picture, path)
        except Exception as e:
            self.show_error(f"{type(e).__name__}: {e}")
            return
        self.output_filename = path
        self.unsaved_changes = False
        self.status_label.setText(f"Saved {path}")
        if close_after:
            self.close()

    def add_lightroom_button(self, request: icm_bridge.Request) -> None:
        """Adds the Save to Lightroom button under Save Image; until it saves, lightroom_status is cancelled."""
        self.lightroom_request = request
        self.lightroom_status = icm_bridge.CANCELLED
        self.lightroom_message = ""
        self.lightroom_button = QtWidgets.QPushButton("Save to Lightroom")
        self.lightroom_button.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2d7d2d; color: white;")
        self.lightroom_button.clicked.connect(self.save_to_lightroom)
        self.control_layout.insertWidget(self.control_layout.indexOf(self.save_button) + 1, self.lightroom_button)

    def save_to_lightroom(self) -> None:
        """Renders the full-size picture, then writes it to the request's output, writes the saved result and closes (write_lightroom_output)."""
        self.render_full_size(self.write_lightroom_output, self.fail_lightroom)

    def write_lightroom_output(self, picture: np.ndarray) -> None:
        """Writes picture and the saved result and closes the window; on a failure keeps the window open (fail_lightroom)."""
        try:
            self.write_tiff(picture, self.lightroom_request.output)
            icm_bridge.write_result(self.lightroom_request.result, icm_bridge.SAVED, self.lightroom_request.output)
        except Exception as e:
            self.fail_lightroom(f"{type(e).__name__}: {e}")
            return
        self.lightroom_status = icm_bridge.SAVED
        self.unsaved_changes = False
        self.close()

    def fail_lightroom(self, message: str) -> None:
        """Shows the error and keeps it as the answer should the window be closed without another try."""
        self.lightroom_status, self.lightroom_message = icm_bridge.ERROR, message
        self.show_error(message)

    def finish_lightroom(self) -> int:
        """After the window closed: writes the cancelled or error result (if it can; saved was written by Save to Lightroom) and returns the exit code."""
        if self.lightroom_status != icm_bridge.SAVED:
            try:
                icm_bridge.write_result(self.lightroom_request.result, self.lightroom_status, message=self.lightroom_message)
            except OSError as e:
                print(f"Cannot write the result file {self.lightroom_request.result}: {e}")
        return icm_bridge.EXIT_CODES[self.lightroom_status]

    def ask_save_before_close(self) -> QtWidgets.QMessageBox.StandardButton:
        """Asks whether to save the unsaved motion first: Save (Save to Lightroom in Lightroom mode), Discard or Cancel."""
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Question, "ICM Motion Recorder",
                                    "The ICM motion is not saved.  Save it before closing?", parent=self)
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard
                               | QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Save)
        if self.lightroom_request is not None:
            box.button(QtWidgets.QMessageBox.StandardButton.Save).setText("Save to Lightroom")
        return QtWidgets.QMessageBox.StandardButton(box.exec())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """
        With unsaved changes (and no save running) asks first: Save starts the save, which closes the window once written, Cancel keeps it
        open.  Closing cancels the running renders and waits for them; a full-size render's result is dropped.
        """
        if self.unsaved_changes and self.full_thread is None:
            answer = self.ask_save_before_close()
            if answer != QtWidgets.QMessageBox.StandardButton.Discard:
                event.ignore()
                if answer == QtWidgets.QMessageBox.StandardButton.Save:
                    if self.lightroom_request is not None:
                        self.save_to_lightroom()
                    else:
                        self.save_image(close_after=True)
                return
        self.preview_timer.stop()
        self.stop_preview()
        if self.full_thread is not None:
            self.full_thread.cancel()
            self.full_thread.wait()
            self.full_thread = None
        super().closeEvent(event)


def main() -> None:
    """
    Opens the source TIFF in the ICM window.  With --lightroom it opens the request's TIFF with a Save to Lightroom button, answers the
    request's result file and exits 0 (saved), 1 (closed without saving) or 2 (closed after an error), see plugin/icm_bridge.py.
    """
    parser = argparse.ArgumentParser(description="Intentional Camera Movement Simulator")
    parser.add_argument("-o", "--output", default=None, type=str, help="The output image to write.")
    source_or_request = parser.add_mutually_exclusive_group(required=True)
    source_or_request.add_argument("source", nargs="?", default=None, type=str, help="The source image to be processed")
    source_or_request.add_argument("--lightroom", metavar="REQUEST", help="request file written by the Lightroom plug-in (plugin/icm_bridge.py)")
    args = parser.parse_args()
    request = None if args.lightroom is None else icm_bridge.read_request(args.lightroom)
    source_fn = args.source if request is None else request.tiff

    with utils.timeit("Reading image ...\n"):
        source = TiffFile().read(source_fn)
        print(f"    Source: {source_fn}")

    app = QtWidgets.QApplication(sys.argv)
    window = ICMWindow(source, args.output)
    if request is not None:
        window.add_lightroom_button(request)
    window.show()
    code = app.exec()
    if request is not None:
        code = window.finish_lightroom()
    sys.exit(code)


if __name__ == "__main__":
    main()
