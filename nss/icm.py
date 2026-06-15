#!/usr/bin/env python3
"""
Intentional Camera Movement (ICM) simulator.
"""
import argparse
import os
import sys

import cv2
import numpy as np
from scipy.interpolate import splprep, splev, PchipInterpolator

from PyQt6 import QtWidgets, QtGui, QtCore

from nss import utils


class CurveEditor(QtWidgets.QWidget):
    def __init__(self, default_val=0.5):
        super().__init__()
        self.setMinimumHeight(150)
        # Store points as normalized values (x: time 0 to 1, y: value 0 to 1)
        self.points = [[0.0, default_val], [0.33, default_val], [0.66, default_val], [1.0, default_val]]
        self.active_idx = None
        self.margin = 15

    def get_curve_values(self, num_steps=60):
        # Sort points by X to prevent interpolation errors
        pts = sorted(self.points, key=lambda p: p[0])
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        interpolator = PchipInterpolator(x, y)
        return interpolator(np.linspace(0, 1, num_steps))

    def _to_screen(self, norm_x, norm_y):
        w, h = self.width() - 2 * self.margin, self.height() - 2 * self.margin
        return int(self.margin + norm_x * w), int(self.margin + (1.0 - norm_y) * h)

    def _to_norm(self, screen_x, screen_y):
        w, h = self.width() - 2 * self.margin, self.height() - 2 * self.margin
        nx = max(0.0, min(1.0, (screen_x - self.margin) / w))
        ny = max(0.0, min(1.0, 1.0 - (screen_y - self.margin) / h))
        return nx, ny

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Find closest point
            mx, my = event.pos().x(), event.pos().y()
            for i, (nx, ny) in enumerate(self.points):
                sx, sy = self._to_screen(nx, ny)
                if (sx - mx)**2 + (sy - my)**2 < 100: # 10 pixel radius
                    self.active_idx = i
                    break

    def mouseMoveEvent(self, event):
        if self.active_idx is not None:
            nx, ny = self._to_norm(event.pos().x(), event.pos().y())
            # Lock the X axis for the first and last points to keep the timeline 0 to 1
            if self.active_idx == 0: nx = 0.0
            elif self.active_idx == len(self.points) - 1: nx = 1.0

            self.points[self.active_idx] = [nx, ny]
            self.update()

    def mouseReleaseEvent(self, event):
        self.active_idx = None

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Draw grid
        painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200), 1, QtCore.Qt.PenStyle.DashLine))
        painter.drawLine(self.margin, self.height() // 2, self.width() - self.margin, self.height() // 2)

        # Draw curve
        curve_vals = self.get_curve_values(60)
        pen_curve = QtGui.QPen(QtGui.QColor(0, 150, 255))
        pen_curve.setWidth(2)
        painter.setPen(pen_curve)

        for i in range(59):
            x1, y1 = self._to_screen(i / 59.0, curve_vals[i])
            x2, y2 = self._to_screen((i + 1) / 59.0, curve_vals[i+1])
            painter.drawLine(x1, y1, x2, y2)

        # Draw control points
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 100, 100)))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for nx, ny in self.points:
            sx, sy = self._to_screen(nx, ny)
            painter.drawEllipse(QtCore.QPoint(sx, sy), 5, 5)

class ImageCanvas(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.path_points = []
        self.smoothed_points = []
        self.is_recording = False
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.path_points = [event.pos()]
            self.smoothed_points = []
            self.is_recording = True

            # Clear the underlying math object if it exists
            if hasattr(self, 'tck'):
                delattr(self, 'tck')

            # Disable buttons and restore the pristine image
            if isinstance(self.window(), ICMWindow):
                self.window().process_button.setEnabled(False)
                self.window().save_button.setEnabled(False)

                if self.window().is_rendered:
                    self.window().restore_original_image()
                    self.window().is_rendered = False

            self.update()

    def mouseMoveEvent(self, event):
        if self.is_recording and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.path_points.append(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.is_recording = False
            print(f"Mouse released. Total raw points: {len(self.path_points)}")

            if len(self.path_points) > 3:
                x = [p.x() for p in self.path_points]
                y = [p.y() for p in self.path_points]

                x_clean, y_clean = [x[0]], [y[0]]
                for i in range(1, len(x)):
                    if x[i] != x[i-1] or y[i] != y[i-1]:
                        x_clean.append(x[i])
                        y_clean.append(y[i])

                print(f"Cleaned points: {len(x_clean)}")

                if len(x_clean) < 4:
                    print("Path too short! Please draw a longer line (needs at least 4 points).")
                else:
                    try:
                        print("Attempting to calculate spline...")
                        smoothing_factor = 5000
                        self.tck, self.u = splprep([x_clean, y_clean], s=smoothing_factor)

                        print("Spline calculated. Generating 200 visual points...")
                        u_fine = np.linspace(0, 1, 200)
                        x_smooth, y_smooth = splev(u_fine, self.tck)

                        print("Points generated. Updating UI...")
                        self.smoothed_points = [QtCore.QPoint(int(mx), int(my)) for mx, my in zip(x_smooth, y_smooth)]

                        if isinstance(self.window(), ICMWindow):
                            self.window().process_button.setEnabled(True)
                            print("Button enabled successfully.")
                        else:
                            print("Warning: Could not find parent ICMWindow to enable button.")

                    except Exception as e:
                        print(f"!!! SPLINE ERROR !!! : {e}")

            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        if len(self.path_points) >= 2:
            pen_raw = QtGui.QPen(QtGui.QColor(255, 0, 0))
            pen_raw.setWidth(2)
            painter.setPen(pen_raw)
            for i in range(len(self.path_points) - 1):
                painter.drawLine(self.path_points[i], self.path_points[i + 1])

        if len(self.smoothed_points) >= 2:
            pen_smooth = QtGui.QPen(QtGui.QColor(0, 255, 0))
            pen_smooth.setWidth(3)
            painter.setPen(pen_smooth)
            for i in range(len(self.smoothed_points) - 1):
                painter.drawLine(self.smoothed_points[i], self.smoothed_points[i + 1])


class ICMWindow(QtWidgets.QMainWindow):

    def __init__(self, tiff_source, output_filename):
        super().__init__()
        self.setWindowTitle("ICM Motion Recorder")

        # Store the object and path, and extract the array
        self.tiff_source = tiff_source
        self.image_array = np.copy(tiff_source.array)
        self.original_array = np.copy(tiff_source.array)
        self.is_rendered = False
        self.output_filename = output_filename

        # Main layout container
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)

        # --- Left Control Panel ---
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_panel.setFixedWidth(250)

        # --- Left Control Panel ---
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_panel.setFixedWidth(250)

        # --- NEW: Exposure Settings ---
        exposure_group = QtWidgets.QGroupBox("Exposure Settings")
        exposure_layout = QtWidgets.QFormLayout(exposure_group)

        # 1. Shutter Speed Dropdown (1/3 stop increments)
        self.shutter_dropdown = QtWidgets.QComboBox()
        shutter_speeds = [
            "1/100", "1/80", "1/60", "1/50", "1/40", "1/30", "1/25", "1/20",
            "1/15", "1/13", "1/10", "1/8", "1/6", "1/5", "1/4", "1/3", "1/2.5",
            "1/2", "1/1.6", "1/1.3", "1s", "1.3s", "1.6s", "2s", "2.5s", "3s", "4s", "5s"
        ]
        self.shutter_dropdown.addItems(shutter_speeds)
        self.shutter_dropdown.setCurrentText("1s")
        exposure_layout.addRow("Shutter:", self.shutter_dropdown)

        # 2. Samples / Frames Input
        self.samples_spin = QtWidgets.QSpinBox()
        self.samples_spin.setRange(3, 300)
        self.samples_spin.setValue(60)

        # We save a reference to the label so we can rename it when toggling modes
        self.samples_label = QtWidgets.QLabel("Samples/sec:")
        exposure_layout.addRow(self.samples_label, self.samples_spin)

        # 3. Multi-Exposure Toggle
        self.multi_expo_check = QtWidgets.QCheckBox("Enable")
        self.multi_expo_check.toggled.connect(self.toggle_multi_exposure)
        exposure_layout.addRow("Multi-Expo:", self.multi_expo_check)

        control_layout.addWidget(exposure_group)

        # Rotation Curve Editor
        rotation_group = QtWidgets.QGroupBox("Rotation Curve")
        rotation_layout = QtWidgets.QVBoxLayout(rotation_group)
        self.rotation_curve = CurveEditor(default_val=0.5)
        rotation_layout.addWidget(self.rotation_curve)
        control_layout.addWidget(rotation_group)

        # Zoom/Scale Curve Editor
        zoom_group = QtWidgets.QGroupBox("Zoom/Scale Curve")
        zoom_layout = QtWidgets.QVBoxLayout(zoom_group)
        self.zoom_curve = CurveEditor(default_val=0.5)
        zoom_layout.addWidget(self.zoom_curve)
        control_layout.addWidget(zoom_group)

        # Speed Curve Editor
        speed_group = QtWidgets.QGroupBox("Motion Speed Curve")
        speed_layout = QtWidgets.QVBoxLayout(speed_group)
        self.speed_curve = CurveEditor(default_val=0.5)
        speed_layout.addWidget(self.speed_curve)
        control_layout.addWidget(speed_group)

        # Process Button
        self.process_button = QtWidgets.QPushButton("Render ICM Image")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.render_icm)
        control_layout.addWidget(self.process_button)

        # Save Button
        self.save_button = QtWidgets.QPushButton("Save Image")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_image)
        control_layout.addWidget(self.save_button)

        # Add stretching spacer to keep elements tidy
        control_layout.addStretch()

        # --- Right Image Canvas ---
        self.label = ImageCanvas()

        # Assemble the main window
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.label)

        self.resize(1200, 800)
        self.display_image()

        # Add a timer to debounce the resize event
        self.resize_timer = QtCore.QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.on_resize_done)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Restart the 200ms timer every time the window shifts by a pixel
        self.resize_timer.start(200)

    def on_resize_done(self):
        if hasattr(self, 'image_array'):
            self.display_image()

            # Clear the paths to prevent coordinate misalignment
            self.label.path_points = []
            self.label.smoothed_points = []
            if hasattr(self.label, 'tck'):
                delattr(self.label, 'tck')

            self.process_button.setEnabled(False)
            self.label.update()

    def display_image(self):
        display_array = (self.image_array * 255).astype(np.uint8)
        height, width = display_array.shape[:2]

        if len(display_array.shape) == 2:
            bytes_per_line = width
            q_img = QtGui.QImage(display_array.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_Grayscale8)
        else:
            bytes_per_line = 3 * width
            q_img = QtGui.QImage(display_array.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)

        pixmap = QtGui.QPixmap.fromImage(q_img)

        scaled_pixmap = pixmap.scaled(self.label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)

    def parse_shutter_speed(self, text):
        if "s" in text:
            text = text.replace("s", "")
        if "/" in text:
            num, den = text.split("/")
            return float(num) / float(den)
        return float(text)

    def toggle_multi_exposure(self, is_checked):
        if is_checked:
            self.samples_label.setText("Total Frames:")
            # Force current value to be odd
            current_val = self.samples_spin.value()
            if current_val % 2 == 0:
                self.samples_spin.setValue(current_val + 1)
            self.samples_spin.setSingleStep(2) # Jump by 2 to stay odd
        else:
            self.samples_label.setText("Samples/sec:")
            self.samples_spin.setSingleStep(1)

    def render_icm(self):
        if not hasattr(self.label, 'tck'):
            print("No motion path drawn. Draw a line on the image first!")
            return

        self.process_button.setEnabled(False)
        self.process_button.setText("Rendering...")
        QtWidgets.QApplication.processEvents()

        # 1. Determine Total Frames
        if self.multi_expo_check.isChecked():
            num_steps = self.samples_spin.value()
        else:
            shutter_sec = self.parse_shutter_speed(self.shutter_dropdown.currentText())
            num_steps = max(2, int(shutter_sec * self.samples_spin.value()))

        print(f"Rendering {num_steps} exposure steps...")

        # 2. Map Speed Curve to Path Progress (0.0 to 1.0)
        speed_vals = self.speed_curve.get_curve_values(num_steps)
        progress = np.cumsum(speed_vals)
        progress -= progress[0]
        if progress[-1] > 0:
            progress /= progress[-1]
        else:
            progress = np.linspace(0, 1, num_steps)

        # 3. Extract specific coordinates from the saved spline math
        x_path, y_path = splev(progress, self.label.tck)

        rot_norm = self.rotation_curve.get_curve_values(num_steps)
        zoom_norm = self.zoom_curve.get_curve_values(num_steps)

        h, w = self.image_array.shape[:2]
        ui_w = self.label.width()
        ui_h = self.label.height()

        # 1. Calculate aspect ratio scaling and offsets of the image inside the QLabel widget
        scale_factor = min(ui_w / w, ui_h / h)
        display_w = w * scale_factor
        display_h = h * scale_factor
        offset_x = (ui_w - display_w) / 2.0
        offset_y = (ui_h - display_h) / 2.0

        # 2. Map the UI-space path coordinates directly into high-res image pixel coordinates
        x_img = [(pt - offset_x) / scale_factor for pt in x_path]
        y_img = [(pt - offset_y) / scale_factor for pt in y_path]

        # 3. Compute the center of rotation/zoom in true image pixel space
        center_x = sum(x_img) / len(x_img)
        center_y = sum(y_img) / len(y_img)
        center = (center_x, center_y)

        accumulator = np.zeros_like(self.image_array, dtype=np.float32)

        # 4. Process each frame using calibrated pixel translations
        for i in range(num_steps):
            tx = x_img[i] - x_img[0]
            ty = y_img[i] - y_img[0]

            angle = (rot_norm[i] - 0.5) * 90.0
            scale = 0.5 + zoom_norm[i]

            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty

            frame = cv2.warpAffine(self.image_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            accumulator += frame

            print(f"Rendered frame {i+1}/{num_steps}", end='\r')

        print("\nNormalizing result...")

        final_image = accumulator / num_steps
        self.image_array = final_image
        self.is_rendered = True
        self.display_image()

        self.process_button.setEnabled(True)
        self.process_button.setText("Render ICM Image")
        self.save_button.setEnabled(True)
        print("Done!")

    def save_image(self):
        save_path = self.output_filename

        # If the file already exists, or no output arg was provided, prompt the user
        if save_path is None or os.path.exists(save_path):
            dialog_path = save_path if save_path else ""
            # Open the Save As dialog
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save ICM Image", dialog_path, "TIFF Images (*.tif *.tiff)"
            )

        if save_path:
            print(f"Saving to {save_path}...")
            # Update the original TIFF object with the new processed array
            self.tiff_source.array = self.image_array

            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

            self.tiff_source.saveas(save_path)
            self.output_filename = save_path # Update the default for subsequent saves
            print("Saved successfully!")

    def restore_original_image(self):
        # Swap the pristine backup back into the active viewer
        self.image_array = np.copy(self.original_array)
        self.display_image()


def make_icm_filename(filename):
    r"""
    In:  C:\some\path\to\image.tif
    OUT: C:\some\path\to\icm\image.tif
    """
    basename = os.path.basename(filename)
    dir_prefix = os.path.dirname(filename)
    return os.path.join(dir_prefix, "icm", basename)

def main():
    parser = argparse.ArgumentParser(description="Intentional Camera Movement Simulator")

    parser.add_argument(
        "-o",
        "--output",
        default = None,
        type = str,
        help = "The output image to write."
    )

    parser.add_argument(
        "source",
        default = None,
        type = str,
        help = "The source image to be processed"
    )

    args = parser.parse_args()
    source_fn = args.source

    with utils.timeit("Reading image ...\n"):
        source = utils.TiffFile().read(source_fn)
        print(f"    Source: {source_fn}")

    # Launch the PyQt6 UI and pass the image data to it
    app = QtWidgets.QApplication(sys.argv)

    window = ICMWindow(source, args.output)
    window.show()

    # Block execution until the window is closed
    sys.exit(app.exec())

if __name__ == "__main__":
    main()