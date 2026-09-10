import numpy as np
import json
import math
from typing import List, Optional
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QGridLayout,
    QLabel,
    QFrame,
    QToolBar,
    QStatusBar,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
    QTabWidget,
    QColorDialog,
    QMenu,
    QDialog,
    QLineEdit,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSettings, QPointF, QMimeData
from PyQt6.QtGui import (
    QMouseEvent,
    QAction,
    QImage,
    QPixmap,
    QColor,
    QPainter,
    QConicalGradient,
    QRadialGradient,
    QPen,
    QDrag,
    QIntValidator,
)

from nss.utils import TiffFile
from nss.color_math import (
    StateNode,
    MutationAxis,
    create_default_state,
    apply_grading,
    generate_mutations,
)
from nss.history import HistoryManager
from nss.mru import MruManager

class ImageContainer(QLabel):
    """
    A custom clickable image container for the 3x3 color grading explorer.
    """
    clicked = pyqtSignal(int)  # Emits the index of the container (0-8)

    def __init__(self, index: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.index: int = index
        self.master_pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(150, 150)
        
        # Initial aesthetic styling
        self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        self.setLineWidth(2)
        self.setStyleSheet(
            "background-color: #2e2e2e; color: #d0d0d0; "
            "border: 1px solid #555555; border-radius: 4px;"
        )
        self.setText(f"Image {index}")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)

    def set_active(self, is_active: bool) -> None:
        """
        Sets active (center) or inactive style for the container.
        """
        if is_active:
            self.setStyleSheet(
                "background-color: #1a3a1a; color: #40ff40; "
                "border: 3px solid #00ff00; border-radius: 4px;"
            )
            self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Raised)
        else:
            self.setStyleSheet(
                "background-color: #2e2e2e; color: #d0d0d0; "
                "border: 1px solid #555555; border-radius: 4px;"
            )
            self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)

    def set_image(self, arr: Optional[np.ndarray]) -> None:
        """
        Converts a NumPy array (Grayscale, RGB, or float32) to QPixmap and displays it.
        If arr is None, clears the image and displays placeholder text.
        """
        if arr is None:
            self.master_pixmap = None
            self.clear()
            if self.index == 4:
                self.setText("Center (Active)")
            else:
                self.setText(f"Mutation {self.index}")
            return

        try:
            # Handle float32 automatic display mapping
            if arr.dtype == np.float32:
                from nss.image_utils import to_uint8_display
                arr = to_uint8_display(arr)
                
            if arr.ndim == 2:
                h, w = arr.shape
                # Use tobytes() to ensure memory safety
                qimg = QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
            elif arr.ndim == 3:
                h, w, c = arr.shape
                if c == 3:
                    qimg = QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
                elif c == 4:
                    qimg = QImage(arr.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
                else:
                    raise ValueError(f"Unsupported number of channels: {c}")
            else:
                raise ValueError(f"Unsupported dimensions: {arr.ndim}")

            self.master_pixmap = QPixmap.fromImage(qimg)
            self.update_display_pixmap()
        except Exception as e:
            self.setText(f"Error: {str(e)}")

    def update_display_pixmap(self) -> None:
        """
        Re-scales and displays the master pixmap inside the label, maintaining aspect ratio.
        """
        if self.master_pixmap is not None and not self.master_pixmap.isNull():
            scaled = self.master_pixmap.scaled(
                self.width() - 8,
                self.height() - 8,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        """
        Handle container resizing by scaling the display image.
        """
        super().resizeEvent(event)
        self.update_display_pixmap()


class ResetLabel(QLabel):
    """
    A custom QLabel that detects double clicks and emits a signal to reset controls.
    """
    doubleClicked = pyqtSignal()

    def mouseDoubleClickEvent(self, event) -> None:
        self.doubleClicked.emit()


class ClickableSwatchLabel(QLabel):
    """
    A custom QLabel that acts as a draggable active color swatch.
    """
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.hue: float = 0.0
        self.sat: float = 0.0
        self.drag_start_position = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_position = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if self.drag_start_position is None:
            return
        from PyQt6.QtWidgets import QApplication
        if (event.position() - self.drag_start_position).manhattanLength() < QApplication.startDragDistance():
            return

        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(f"{self.hue},{self.sat}")
        drag.setMimeData(mime_data)

        pixmap = self.pixmap()
        if pixmap and not pixmap.isNull():
            drag.setPixmap(pixmap)
        drag.exec(Qt.DropAction.CopyAction)


class ColorWheel(QWidget):
    """
    A custom circular color wheel widget.
    Paints Hue using a conical gradient and Saturation using a radial gradient.
    """
    colorChanged = pyqtSignal(float, float)  # Emits (hue, saturation)
    interactionFinished = pyqtSignal(float, float)  # Emits (hue, saturation) when mouse is released

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.hue: float = 0.0  # Range: [0.0, 360.0]
        self.sat: float = 0.0  # Range: [0.0, 1.0]
        self.setMinimumSize(160, 160)
        self.setMaximumSize(240, 240)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_color(self, hue: float, sat: float) -> None:
        """
        Updates the internal color state and schedules a repaint. Does not emit signals.
        """
        self.hue = float(hue) % 360.0
        self.sat = float(np.clip(sat, 0.0, 1.0))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        size = min(rect.width(), rect.height()) - 10
        cx = rect.width() / 2.0
        cy = rect.height() / 2.0
        radius = size / 2.0

        if radius <= 0:
            return

        # 1. Paint conical gradient (Hue spectrum)
        conical = QConicalGradient(cx, cy, 0.0)
        for i in range(361):
            conical.setColorAt(i / 360.0, QColor.fromHslF(i / 360.0, 1.0, 0.5))
        
        painter.setBrush(conical)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))

        # 2. Paint radial gradient (Saturation overlay)
        radial = QRadialGradient(cx, cy, radius)
        radial.setColorAt(0.0, QColor(255, 255, 255, 255))
        radial.setColorAt(1.0, QColor(255, 255, 255, 0))

        painter.setBrush(radial)
        painter.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))

        # 3. Draw indicator/handle
        angle_rad = math.radians(self.hue)
        d = self.sat * radius
        px = cx + d * math.cos(angle_rad)
        py = cy - d * math.sin(angle_rad)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawEllipse(QPointF(px, py), 5.0, 5.0)
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.drawEllipse(QPointF(px, py), 4.0, 4.0)

    def _update_color_from_mouse(self, pos: QPointF) -> None:
        rect = self.rect()
        cx = rect.width() / 2.0
        cy = rect.height() / 2.0
        size = min(rect.width(), rect.height()) - 10
        radius = size / 2.0

        if radius <= 0:
            return

        dx = pos.x() - cx
        dy = pos.y() - cy
        d = math.sqrt(dx*dx + dy*dy)

        sat = min(1.0, d / radius)
        if d == 0:
            hue = self.hue
        else:
            angle_rad = math.atan2(-dy, dx)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360.0
            hue = angle_deg

        self.hue = hue
        self.sat = sat
        self.update()
        self.colorChanged.emit(self.hue, self.sat)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._update_color_from_mouse(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._update_color_from_mouse(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._update_color_from_mouse(event.position())
            self.interactionFinished.emit(self.hue, self.sat)


class ColorModifyDialog(QDialog):
    """
    A dialog for editing a color with real-time bi-directional sync
    between RGB, HSL, HSV, and Hex representation fields (0-255 values).
    """
    def __init__(self, color: QColor, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Saved Color")
        self.setStyleSheet("background-color: #2b2b2b; color: #eee;")
        self.current_color = QColor(color)
        self.updating = False

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Preview row
        preview_layout = QHBoxLayout()
        preview_label = QLabel("Preview:")
        preview_label.setStyleSheet("font-weight: bold;")
        self.preview_swatch = QLabel()
        self.preview_swatch.setFixedSize(80, 24)
        preview_layout.addWidget(preview_label)
        preview_layout.addWidget(self.preview_swatch)
        preview_layout.addStretch()
        layout.addLayout(preview_layout)
        
        # Fields grid
        grid = QGridLayout()
        grid.setSpacing(8)
        
        # Headers
        rgb_hdr = QLabel("RGB (0-255)")
        rgb_hdr.setStyleSheet("font-weight: bold; color: #aaa;")
        hsl_hdr = QLabel("HSL (0-255)")
        hsl_hdr.setStyleSheet("font-weight: bold; color: #aaa;")
        hsv_hdr = QLabel("HSV (0-255)")
        hsv_hdr.setStyleSheet("font-weight: bold; color: #aaa;")
        
        grid.addWidget(rgb_hdr, 0, 0, 1, 2)
        grid.addWidget(hsl_hdr, 0, 2, 1, 2)
        grid.addWidget(hsv_hdr, 0, 4, 1, 2)
        
        # R / H / H
        grid.addWidget(QLabel("R:"), 1, 0)
        self.r_edit = QLineEdit()
        self.r_edit.setValidator(QIntValidator(0, 255))
        grid.addWidget(self.r_edit, 1, 1)
        
        grid.addWidget(QLabel("H:"), 1, 2)
        self.hsl_h_edit = QLineEdit()
        self.hsl_h_edit.setValidator(QIntValidator(0, 255))
        grid.addWidget(self.hsl_h_edit, 1, 3)
        
        grid.addWidget(QLabel("H:"), 1, 4)
        self.hsv_h_edit = QLineEdit()
        self.hsv_h_edit.setValidator(QIntValidator(0, 255))
        grid.addWidget(self.hsv_h_edit, 1, 5)
        
        # G / S / S
        grid.addWidget(QLabel("G:"), 2, 0)
        self.g_edit = QLineEdit()
        self.g_edit.setValidator(QIntValidator(0, 255))
        grid.addWidget(self.g_edit, 2, 1)
        
        grid.addWidget(QLabel("S:"), 2, 2)
        self.hsl_s_edit = QLineEdit()
        self.hsl_s_edit.setValidator(QIntValidator(0, 255))
        grid.addWidget(self.hsl_s_edit, 2, 3)
        
        grid.addWidget(QLabel("S:"), 2, 4)
        self.hsv_s_edit = QLineEdit()
        self.hsv_s_edit.setValidator(QIntValidator(0, 255))
        grid.addWidget(self.hsv_s_edit, 2, 5)
        
        # B / L / V
        grid.addWidget(QLabel("B:"), 3, 0)
        self.b_edit = QLineEdit()
        self.b_edit.setValidator(QIntValidator(0, 255))
        grid.addWidget(self.b_edit, 3, 1)
        
        grid.addWidget(QLabel("L:"), 3, 2)
        self.hsl_l_edit = QLineEdit()
        self.hsl_l_edit.setValidator(QIntValidator(0, 255))
        grid.addWidget(self.hsl_l_edit, 3, 3)
        
        grid.addWidget(QLabel("V:"), 3, 4)
        self.hsv_v_edit = QLineEdit()
        self.hsv_v_edit.setValidator(QIntValidator(0, 255))
        grid.addWidget(self.hsv_v_edit, 3, 5)
        
        # Hex field spanning at the bottom
        grid.addWidget(QLabel("Hex:"), 4, 0)
        self.hex_edit = QLineEdit()
        grid.addWidget(self.hex_edit, 4, 1, 1, 2)
        
        layout.addLayout(grid)
        
        # Set text box styles for a polished look
        for edit in [self.r_edit, self.g_edit, self.b_edit, 
                     self.hsl_h_edit, self.hsl_s_edit, self.hsl_l_edit,
                     self.hsv_h_edit, self.hsv_s_edit, self.hsv_v_edit, self.hex_edit]:
            edit.setStyleSheet("background-color: #3a3a3a; color: #fff; border: 1px solid #555; border-radius: 2px; padding: 2px;")
            edit.setFixedWidth(50)
        self.hex_edit.setFixedWidth(80) # Hex needs a bit more space
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Connect text signals
        self.r_edit.textChanged.connect(self.on_rgb_edited)
        self.g_edit.textChanged.connect(self.on_rgb_edited)
        self.b_edit.textChanged.connect(self.on_rgb_edited)
        
        self.hsl_h_edit.textChanged.connect(self.on_hsl_edited)
        self.hsl_s_edit.textChanged.connect(self.on_hsl_edited)
        self.hsl_l_edit.textChanged.connect(self.on_hsl_edited)
        
        self.hsv_h_edit.textChanged.connect(self.on_hsv_edited)
        self.hsv_s_edit.textChanged.connect(self.on_hsv_edited)
        self.hsv_v_edit.textChanged.connect(self.on_hsv_edited)
        
        self.hex_edit.textChanged.connect(self.on_hex_edited)
        
        # Initialize values
        self.update_all_except(None)

    def update_all_except(self, exclude: Optional[str]) -> None:
        self.updating = True
        color = self.current_color
        
        # Update RGB
        if exclude != "rgb":
            self.r_edit.setText(str(color.red()))
            self.g_edit.setText(str(color.green()))
            self.b_edit.setText(str(color.blue()))
            
        # Update HSL
        if exclude != "hsl":
            h_deg = max(0, color.hslHue())
            h_255 = int(h_deg * 255.0 / 360.0)
            self.hsl_h_edit.setText(str(h_255))
            self.hsl_s_edit.setText(str(color.hslSaturation()))
            self.hsl_l_edit.setText(str(color.lightness()))
            
        # Update HSV
        if exclude != "hsv":
            h_deg = max(0, color.hsvHue())
            h_255 = int(h_deg * 255.0 / 360.0)
            self.hsv_h_edit.setText(str(h_255))
            self.hsv_s_edit.setText(str(color.hsvSaturation()))
            self.hsv_v_edit.setText(str(color.value()))
            
        # Update Hex
        if exclude != "hex":
            self.hex_edit.setText(color.name().upper())
            
        # Update preview color
        self.preview_swatch.setStyleSheet(
            f"border: 1px solid #555; border-radius: 4px; background-color: {color.name()};"
        )
        self.updating = False

    def on_rgb_edited(self) -> None:
        if self.updating:
            return
        try:
            r = int(self.r_edit.text() or 0)
            g = int(self.g_edit.text() or 0)
            b = int(self.b_edit.text() or 0)
            r = min(255, max(0, r))
            g = min(255, max(0, g))
            b = min(255, max(0, b))
            
            self.current_color.setRgb(r, g, b)
            self.update_all_except("rgb")
        except ValueError:
            pass

    def on_hsl_edited(self) -> None:
        if self.updating:
            return
        try:
            h = int(self.hsl_h_edit.text() or 0)
            s = int(self.hsl_s_edit.text() or 0)
            l = int(self.hsl_l_edit.text() or 0)
            h = min(255, max(0, h))
            s = min(255, max(0, s))
            l = min(255, max(0, l))
            
            h_deg = int(h * 360.0 / 255.0) % 360
            self.current_color.setHsl(h_deg, s, l)
            self.update_all_except("hsl")
        except ValueError:
            pass

    def on_hsv_edited(self) -> None:
        if self.updating:
            return
        try:
            h = int(self.hsv_h_edit.text() or 0)
            s = int(self.hsv_s_edit.text() or 0)
            v = int(self.hsv_v_edit.text() or 0)
            h = min(255, max(0, h))
            s = min(255, max(0, s))
            v = min(255, max(0, v))
            
            h_deg = int(h * 360.0 / 255.0) % 360
            self.current_color.setHsv(h_deg, s, v)
            self.update_all_except("hsv")
        except ValueError:
            pass

    def on_hex_edited(self) -> None:
        if self.updating:
            return
        text = self.hex_edit.text().strip()
        if not text.startswith("#"):
            text = "#" + text
        if QColor.isValidColor(text):
            self.current_color.setNamedColor(text)
            self.update_all_except("hex")


class ColorSwatch(QLabel):
    """
    A custom color swatch that displays an HSL color, emits a clicked signal,
    and supports drag-and-drop to overwrite its color with the dragged color.
    """
    clicked = pyqtSignal(float, float)  # Emits (hue, saturation)
    overwritten = pyqtSignal(int, float, float)  # Emits (index, hue, saturation)
    forgotten = pyqtSignal(int)  # Emits (index)
    edited = pyqtSignal(int, float, float)  # Emits (index, hue, saturation)

    def __init__(self, hue: Optional[float] = 0.0, sat: Optional[float] = 0.0, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.hue: Optional[float] = hue
        self.sat: Optional[float] = sat
        self.index: int = 0
        self.setFixedSize(24, 24)
        self.setAcceptDrops(True)
        self.update_color(hue, sat)

    def update_color(self, hue: Optional[float], sat: Optional[float]) -> None:
        if hue is None or sat is None:
            self.hue = None
            self.sat = None
            self.setStyleSheet(
                "border: 1px dashed #555; border-radius: 4px; "
                "background-color: #1a1a1a;"
            )
            self.setToolTip("Empty Slot. Drag a color here to save.")
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.hue = float(hue)
            self.sat = float(sat)
            color = QColor.fromHslF(self.hue / 360.0, self.sat, 1.0 - 0.5 * self.sat)
            self.setStyleSheet(
                f"border: 1px solid #555; border-radius: 4px; "
                f"background-color: {color.name()};"
            )
            self.setToolTip(f"Hue: {self.hue:.0f}°, Sat: {self.sat:.2f}")
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self.hue is not None and self.sat is not None:
                self.clicked.emit(self.hue, self.sat)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #2b2b2b; color: #eee; border: 1px solid #555; }"
            "QMenu::item:selected { background-color: #444; }"
        )
        
        edit_action = QAction("Edit", self)
        forget_action = QAction("Forget", self)
        
        if self.hue is None or self.sat is None:
            edit_action.setEnabled(False)
            forget_action.setEnabled(False)
            
        menu.addAction(edit_action)
        menu.addAction(forget_action)
        
        edit_action.triggered.connect(self._on_edit_triggered)
        forget_action.triggered.connect(self._on_forget_triggered)
        
        menu.exec(event.globalPos())

    def _on_edit_triggered(self) -> None:
        if self.hue is None or self.sat is None:
            return
            
        color = QColor.fromHslF(self.hue / 360.0, self.sat, 1.0 - 0.5 * self.sat)
        dialog = ColorModifyDialog(color, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_color = dialog.current_color
            h_deg = max(0, new_color.hslHue())
            s_val = new_color.hslSaturationF()
            
            self.update_color(float(h_deg), float(s_val))
            self.edited.emit(self.index, float(h_deg), float(s_val))

    def _on_forget_triggered(self) -> None:
        self.forgotten.emit(self.index)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        text = event.mimeData().text()
        try:
            hue_str, sat_str = text.split(",")
            hue = float(hue_str)
            sat = float(sat_str)
            self.update_color(hue, sat)
            self.overwritten.emit(self.index, hue, sat)
            event.acceptProposedAction()
        except Exception:
            pass


class GradingWorker(QThread):
    """
    Background worker thread to compute color grading for mutations asynchronously
    to prevent blocking the main PyQt6 GUI event thread.
    """
    progress = pyqtSignal(int, np.ndarray)
    finished_all = pyqtSignal()

    def __init__(self, working_image: np.ndarray, states: List[StateNode], skip_center: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.working_image = working_image
        self.states = states
        self.skip_center = skip_center
        self.is_cancelled = False

    def cancel(self) -> None:
        """
        Signals the worker thread to stop processing.
        """
        self.is_cancelled = True

    def run(self) -> None:
        # Determine index processing sequence (render center index 4 first unless skipped)
        if self.skip_center:
            indices = [i for i in range(9) if i != 4]
        else:
            indices = [4] + [i for i in range(9) if i != 4]

        for i in indices:
            if self.is_cancelled:
                return
            if i >= len(self.states):
                continue
            try:
                # Core NumPy and OpenCV image processing grading happens in background
                graded = apply_grading(self.working_image, self.states[i])
                if self.is_cancelled:
                    return
                self.progress.emit(i, graded)
            except Exception:
                pass
        self.finished_all.emit()


class MainWindow(QMainWindow):
    """
    The Main Window for the 16-bit Color Grading Explorer with Lightroom-style controls.
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("16-bit Color Grading Explorer")
        self.resize(1300, 850)

        # State Variables
        self.master_image: Optional[np.ndarray] = None
        self.proxy_image: Optional[np.ndarray] = None
        self.tiff_obj: Optional[TiffFile] = None
        self.history_manager: HistoryManager = HistoryManager()
        self.mru_manager: MruManager = MruManager()
        self.grid_states: Optional[List[StateNode]] = None
        self.grading_worker: Optional[GradingWorker] = None

        # Setup UI Components
        self.init_ui()

    def init_ui(self) -> None:
        # Create Menu Bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open a 16-bit TIFF image")
        open_action.triggered.connect(lambda: self.open_file())
        file_menu.addAction(open_action)

        # Recent Directories Submenu
        self.recent_menu = file_menu.addMenu("Recent &Directories")
        self.update_recent_directories_menu()

        save_action = QAction("&Save As...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setStatusTip("Save the active color graded image")
        save_action.setEnabled(False)
        save_action.triggered.connect(self.save_file)
        self.save_action = save_action
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 1. Create Toolbar with Back/Forward, Step-size slider and History label
        toolbar = QToolBar("Navigation and Controls")
        self.addToolBar(toolbar)

        # "Back" button
        self.back_action = QAction("Back", self)
        self.back_action.setToolTip("Go to previous color state (Undo)")
        self.back_action.setEnabled(False)
        self.back_action.triggered.connect(self.on_back_clicked)
        toolbar.addAction(self.back_action)

        # History Index Display Label
        self.history_label = QLabel("Step: 0 of 0")
        self.history_label.setStyleSheet(
            "font-weight: bold; margin-left: 10px; margin-right: 10px; color: #a0a0a0;"
        )
        toolbar.addWidget(self.history_label)

        # "Forward" button
        self.forward_action = QAction("Forward", self)
        self.forward_action.setToolTip("Go to next color state (Redo)")
        self.forward_action.setEnabled(False)
        self.forward_action.triggered.connect(self.on_forward_clicked)
        toolbar.addAction(self.forward_action)

        toolbar.addSeparator()

        # Mutation Intensity (Step Size) Slider - Defaults to 0.20x magnitude (subtle steps)
        self.intensity_label = QLabel(" Intensity: 0.20x ")
        self.intensity_label.setStyleSheet("color: #a0a0a0; font-weight: bold;")
        toolbar.addWidget(self.intensity_label)

        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(10, 200)
        self.intensity_slider.setValue(20)  # Default value 20 represents 0.20
        self.intensity_slider.setFixedWidth(120)
        self.intensity_slider.setToolTip("Slide to scale mutation offset magnitude (0.10x to 2.00x)")
        self.intensity_slider.valueChanged.connect(self.on_intensity_changed)
        toolbar.addWidget(self.intensity_slider)

        # 2. Setup Central Widget with QHBoxLayout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Left Side: 3x3 Grid Layout
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(10)
        main_layout.addWidget(grid_widget, stretch=4)

        self.containers: List[ImageContainer] = []
        for i in range(9):
            container = ImageContainer(i, self)
            container.clicked.connect(self.on_container_clicked)
            self.containers.append(container)

            # Map index to row and col
            row = i // 3
            col = i % 3
            grid_layout.addWidget(container, row, col)

            # Mark center container (index 4) as active
            if i == 4:
                container.set_active(True)
                container.setText("Center (Active)")
            else:
                container.set_active(False)
                container.setText(f"Mutation {i}")

        # Right Side: Structured Lightroom-Style Manual 3-Way Color Grading Control Dock
        self.inspector_panel = QFrame()
        self.inspector_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.inspector_panel.setFrameShadow(QFrame.Shadow.Raised)
        self.inspector_panel.setFixedWidth(300)
        self.inspector_panel.setStyleSheet(
            "background-color: #252525; border: 1px solid #444; border-radius: 4px;"
        )
        
        inspector_layout = QVBoxLayout(self.inspector_panel)
        inspector_layout.setContentsMargins(15, 15, 15, 15)
        inspector_layout.setSpacing(12)
        
        main_layout.addWidget(self.inspector_panel, stretch=1)

        # Dock Title Header
        header = QLabel("3-WAY COLOR GRADING")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #40ff40; border: none; margin-bottom: 5px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inspector_layout.addWidget(header)

        # 3-Zone Tabs Setup
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #444; background-color: #2b2b2b; }"
            "QTabBar::tab { background-color: #333; color: #aaa; padding: 6px 12px; border: 1px solid #444; }"
            "QTabBar::tab:selected { background-color: #2b2b2b; color: #eee; font-weight: bold; }"
        )
        inspector_layout.addWidget(self.tabs)

        # Tab Helper Function
        def create_zone_tab(title_prefix: str, default_h: int) -> tuple[QWidget, ColorWheel, ResetLabel, QSlider, ResetLabel, QSlider, ResetLabel, ClickableSwatchLabel, QGridLayout]:
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            tab_layout.setContentsMargins(10, 10, 10, 10)
            tab_layout.setSpacing(5)

            swatch_lbl = ClickableSwatchLabel()
            swatch_lbl.setFixedSize(24, 24)
            swatch_lbl.setStyleSheet("border: 1px solid #555; border-radius: 4px;")
            swatch_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
            swatch_lbl.setToolTip("Click to open color picker wheel")

            # 1. Color Wheel
            wheel = ColorWheel()
            wheel_container = QHBoxLayout()
            wheel_container.addStretch()
            wheel_container.addWidget(wheel)
            wheel_container.addStretch()
            tab_layout.addLayout(wheel_container)

            # 2. Hue Control (Double click label to reset to default)
            hue_lbl = ResetLabel(f"Hue: {default_h}°")
            hue_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
            hue_lbl.setToolTip("Double-click to reset Hue to default")

            # 3. Saturation Control
            sat_lbl = ResetLabel("Sat: 0.00")
            sat_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
            sat_lbl.setToolTip("Double-click to reset Saturation to 0.00")
            sat_sld = QSlider(Qt.Orientation.Horizontal)
            sat_sld.setRange(0, 100)
            sat_sld.setValue(0)
            sat_sld.valueChanged.connect(self.on_manual_slider_changed)

            # 4. Luminance Control
            lum_lbl = ResetLabel("Luma: 0.00")
            lum_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
            lum_lbl.setToolTip("Double-click to reset Luminance to 0.00")
            lum_sld = QSlider(Qt.Orientation.Horizontal)
            lum_sld.setRange(-100, 100)
            lum_sld.setValue(0)
            lum_sld.valueChanged.connect(self.on_manual_slider_changed)

            tab_layout.addWidget(hue_lbl)
            tab_layout.addWidget(sat_lbl)
            tab_layout.addWidget(sat_sld)
            tab_layout.addWidget(lum_lbl)
            tab_layout.addWidget(lum_sld)

            # 5. Saved Colors Grid
            recent_lbl = QLabel("Saved Custom Colors:")
            recent_lbl.setStyleSheet("color: #aaa; font-size: 10px; font-weight: bold; margin-top: 5px;")
            tab_layout.addWidget(recent_lbl)

            # Position active color patch (swatch_lbl) and a "Drag to save" label directly above the grid widget
            patch_layout = QHBoxLayout()
            patch_layout.setContentsMargins(0, 0, 0, 0)
            patch_layout.setSpacing(6)
            
            drag_lbl = QLabel("Drag to save")
            drag_lbl.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
            
            patch_layout.addWidget(swatch_lbl)
            patch_layout.addWidget(drag_lbl)
            patch_layout.addStretch()
            
            tab_layout.addLayout(patch_layout)

            grid_widget = QWidget()
            grid_layout = QGridLayout(grid_widget)
            grid_layout.setContentsMargins(0, 2, 0, 2)
            grid_layout.setSpacing(4)
            tab_layout.addWidget(grid_widget)
            tab_layout.addStretch()

            return tab_widget, wheel, hue_lbl, sat_sld, sat_lbl, lum_sld, lum_lbl, swatch_lbl, grid_layout

        # Shadows Tab (default blue 240)
        sh_tab, self.sh_wheel, self.sh_hue_lbl, self.sh_sat_slider, self.sh_sat_lbl, self.sh_light_slider, self.sh_light_lbl, self.sh_swatch_label, self.sh_history_grid = create_zone_tab("Shadows", 240)
        self.tabs.addTab(sh_tab, "Shadows")
        self.sh_wheel.colorChanged.connect(self.on_sh_wheel_changed)
        self.sh_hue_lbl.doubleClicked.connect(lambda: self.reset_wheel_hue(self.sh_wheel, 240))
        self.sh_sat_lbl.doubleClicked.connect(lambda: self.reset_slider(self.sh_sat_slider, 0))
        self.sh_light_lbl.doubleClicked.connect(lambda: self.reset_slider(self.sh_light_slider, 0))

        # Midtones Tab (default green 120)
        mid_tab, self.mid_wheel, self.mid_hue_lbl, self.mid_sat_slider, self.mid_sat_lbl, self.mid_light_slider, self.mid_light_lbl, self.mid_swatch_label, self.mid_history_grid = create_zone_tab("Midtones", 120)
        self.tabs.addTab(mid_tab, "Midtones")
        self.mid_wheel.colorChanged.connect(self.on_mid_wheel_changed)
        self.mid_hue_lbl.doubleClicked.connect(lambda: self.reset_wheel_hue(self.mid_wheel, 120))
        self.mid_sat_lbl.doubleClicked.connect(lambda: self.reset_slider(self.mid_sat_slider, 0))
        self.mid_light_lbl.doubleClicked.connect(lambda: self.reset_slider(self.mid_light_slider, 0))

        # Highlights Tab (default yellow/gold 60)
        hi_tab, self.hi_wheel, self.hi_hue_lbl, self.hi_sat_slider, self.hi_sat_lbl, self.hi_light_slider, self.hi_light_lbl, self.hi_swatch_label, self.hi_history_grid = create_zone_tab("Highlights", 60)
        self.tabs.addTab(hi_tab, "Highlights")
        self.hi_wheel.colorChanged.connect(self.on_hi_wheel_changed)
        self.hi_hue_lbl.doubleClicked.connect(lambda: self.reset_wheel_hue(self.hi_wheel, 60))
        self.hi_sat_lbl.doubleClicked.connect(lambda: self.reset_slider(self.hi_sat_slider, 0))
        self.hi_light_lbl.doubleClicked.connect(lambda: self.reset_slider(self.hi_light_slider, 0))

        # Populate custom color history grids from settings
        self.update_history_swatches_ui("shadows", self.load_custom_colors("shadows"))
        self.update_history_swatches_ui("midtones", self.load_custom_colors("midtones"))
        self.update_history_swatches_ui("highlights", self.load_custom_colors("highlights"))

        # Master Global Controls Section
        master_widget = QWidget()
        master_widget.setStyleSheet("border: none; background: transparent;")
        master_widget.setContentsMargins(0, 0, 0, 0)
        master_layout = QVBoxLayout(master_widget)
        master_layout.setSpacing(10)
        inspector_layout.addWidget(master_widget)

        # Title
        m_title = QLabel("MASTER DOCK CONTROLS")
        m_title.setStyleSheet("font-weight: bold; color: #a0a0a0; border: none; border-bottom: 1px solid #444; padding-bottom: 2px;")
        master_layout.addWidget(m_title)

        # Master Blending
        self.blending_lbl = ResetLabel("Blending: 0.50")
        self.blending_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
        self.blending_lbl.setToolTip("Double-click to reset Blending to 0.50")
        self.blending_slider = QSlider(Qt.Orientation.Horizontal)
        self.blending_slider.setRange(0, 100)
        self.blending_slider.setValue(50)
        self.blending_slider.valueChanged.connect(self.on_manual_slider_changed)
        self.blending_lbl.doubleClicked.connect(lambda: self.reset_slider(self.blending_slider, 50))
        
        # Master Balance
        self.balance_lbl = ResetLabel("Balance: 0.00")
        self.balance_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
        self.balance_lbl.setToolTip("Double-click to reset Balance to 0.00")
        self.balance_slider = QSlider(Qt.Orientation.Horizontal)
        self.balance_slider.setRange(-100, 100)
        self.balance_slider.setValue(0)
        self.balance_slider.valueChanged.connect(self.on_manual_slider_changed)
        self.balance_lbl.doubleClicked.connect(lambda: self.reset_slider(self.balance_slider, 0))

        master_layout.addWidget(self.blending_lbl)
        master_layout.addWidget(self.blending_slider)
        master_layout.addWidget(self.balance_lbl)
        master_layout.addWidget(self.balance_slider)

        inspector_layout.addStretch()

        # 3. Setup Status Bar Message via built-in QMainWindow statusBar()
        self.statusBar().showMessage("Ready. Load a 16-bit TIFF image to begin.")

    def reset_slider(self, slider: QSlider, value: int) -> None:
        """
        Helper method to reset a slider back to its default value on label double-click.
        """
        slider.setValue(value)

    def on_manual_slider_changed(self) -> None:
        """
        Reads values from all manual grading controls (wheels, sliders), updates active center StateNode,
        re-renders center image preview immediately, and updates swatches.
        Isolates manual changes exclusively to the center image to preserve snappiness.
        """
        if self.master_image is None:
            return

        state = self.history_manager.get_current_state()
        if state is None:
            return

        # 1. Read values and update state node
        state["shadow_hue"] = float(self.sh_wheel.hue)
        state["shadow_sat"] = self.sh_sat_slider.value() / 100.0
        state["shadow_light"] = self.sh_light_slider.value() / 100.0

        state["midtone_hue"] = float(self.mid_wheel.hue)
        state["midtone_sat"] = self.mid_sat_slider.value() / 100.0
        state["midtone_light"] = self.mid_light_slider.value() / 100.0

        state["highlight_hue"] = float(self.hi_wheel.hue)
        state["highlight_sat"] = self.hi_sat_slider.value() / 100.0
        state["highlight_light"] = self.hi_light_slider.value() / 100.0

        state["blending"] = self.blending_slider.value() / 100.0
        state["balance"] = self.balance_slider.value() / 100.0

        # Ensure wheels are synchronized with the sliders
        self.sh_wheel.set_color(state["shadow_hue"], state["shadow_sat"])
        self.mid_wheel.set_color(state["midtone_hue"], state["midtone_sat"])
        self.hi_wheel.set_color(state["highlight_hue"], state["highlight_sat"])

        # Update text readouts
        self.sh_hue_lbl.setText(f"Hue: {state['shadow_hue']:.0f}°")
        self.sh_sat_lbl.setText(f"Sat: {state['shadow_sat']:.2f}")
        self.sh_light_lbl.setText(f"Luma: {state['shadow_light']:+.2f}")

        self.mid_hue_lbl.setText(f"Hue: {state['midtone_hue']:.0f}°")
        self.mid_sat_lbl.setText(f"Sat: {state['midtone_sat']:.2f}")
        self.mid_light_lbl.setText(f"Luma: {state['midtone_light']:+.2f}")

        self.hi_hue_lbl.setText(f"Hue: {state['highlight_hue']:.0f}°")
        self.hi_sat_lbl.setText(f"Sat: {state['highlight_sat']:.2f}")
        self.hi_light_lbl.setText(f"Luma: {state['highlight_light']:+.2f}")

        self.blending_lbl.setText(f"Blending: {state['blending']:.2f}")
        self.balance_lbl.setText(f"Balance: {state['balance']:+.2f}")

        # 2. Update visual color swatches live
        self.update_zone_swatches(state)

        # 3. Recalculate and update the center tile immediately!
        # Applying color grading math to proxy_image for real-time snappy feedback
        if self.proxy_image is not None:
            graded_center = apply_grading(self.proxy_image, state)
            self.containers[4].set_image(graded_center)

    def on_sh_wheel_changed(self, hue: float, sat: float) -> None:
        self.block_manual_signals(True)
        self.sh_sat_slider.setValue(int(sat * 100.0))
        self.block_manual_signals(False)
        self.on_manual_slider_changed()

    def on_mid_wheel_changed(self, hue: float, sat: float) -> None:
        self.block_manual_signals(True)
        self.mid_sat_slider.setValue(int(sat * 100.0))
        self.block_manual_signals(False)
        self.on_manual_slider_changed()

    def on_hi_wheel_changed(self, hue: float, sat: float) -> None:
        self.block_manual_signals(True)
        self.hi_sat_slider.setValue(int(sat * 100.0))
        self.block_manual_signals(False)
        self.on_manual_slider_changed()

    def reset_wheel_hue(self, wheel: ColorWheel, default_hue: float) -> None:
        """
        Helper method to reset a wheel back to default hue.
        """
        wheel.set_color(default_hue, wheel.sat)
        self.on_manual_slider_changed()

    def pick_shadow_color(self) -> None:
        """
        Opens QColorDialog to select a shadow tint color, and updates the state.
        """
        self.pick_zone_color("shadow_hue", "shadow_sat", self.sh_wheel, self.sh_sat_slider, 240.0)

    def pick_midtone_color(self) -> None:
        """
        Opens QColorDialog to select a midtone tint color, and updates the state.
        """
        self.pick_zone_color("midtone_hue", "midtone_sat", self.mid_wheel, self.mid_sat_slider, 120.0)

    def pick_highlight_color(self) -> None:
        """
        Opens QColorDialog to select a highlight tint color, and updates the state.
        """
        self.pick_zone_color("highlight_hue", "highlight_sat", self.hi_wheel, self.hi_sat_slider, 60.0)

    def pick_zone_color(self, hue_key: str, sat_key: str, wheel: ColorWheel, sat_slider: QSlider, default_hue: float) -> None:
        """
        Generic helper to open QColorDialog and apply HSL values to sliders and wheel.
        """
        state = self.history_manager.get_current_state()
        if state is None:
            return

        current_hue = state.get(hue_key, default_hue)
        current_sat = state.get(sat_key, 0.0)
        initial_color = QColor.fromHslF(current_hue / 360.0, current_sat, 1.0 - 0.5 * current_sat)

        color = QColorDialog.getColor(initial_color, self, "Select Zone Tint Color")
        if color.isValid():
            h, s, l, a = color.getHslF()
            hue = h * 360.0 if h >= 0.0 else current_hue
            sat = s
            
            # Sync wheel and slider
            self.block_manual_signals(True)
            wheel.set_color(hue, sat)
            sat_slider.setValue(int(sat * 100.0))
            self.block_manual_signals(False)
            
            # Recalculate
            self.on_manual_slider_changed()

    def load_custom_colors(self, zone: str) -> list:
        settings = QSettings("NSS", "ColorGradingExplorer")
        data = settings.value(f"history_grid_{zone}")
        if data:
            try:
                res = json.loads(str(data))
                if len(res) == 16:
                    return res
                if len(res) > 16:
                    return res[:16]
                else:
                    defaults = self.get_default_colors(zone)
                    return res + defaults[len(res):]
            except Exception:
                pass
                
        return self.get_default_colors(zone)

    def get_default_colors(self, zone: str) -> list:
        if zone == "shadows":
            return [
                [240.0, 0.0], [240.0, 0.1], [240.0, 0.2], [240.0, 0.3],
                [240.0, 0.4], [240.0, 0.5], [200.0, 0.1], [200.0, 0.2],
                [200.0, 0.3], [200.0, 0.4], [270.0, 0.1], [270.0, 0.2],
                [270.0, 0.3], [270.0, 0.4], [180.0, 0.1], [180.0, 0.2]
            ]
        elif zone == "midtones":
            return [
                [120.0, 0.0], [120.0, 0.1], [120.0, 0.2], [120.0, 0.3],
                [120.0, 0.4], [120.0, 0.5], [80.0, 0.1], [80.0, 0.2],
                [80.0, 0.3], [80.0, 0.4], [160.0, 0.1], [160.0, 0.2],
                [160.0, 0.3], [160.0, 0.4], [100.0, 0.1], [100.0, 0.2]
            ]
        else:
            return [
                [60.0, 0.0], [60.0, 0.1], [60.0, 0.2], [60.0, 0.3],
                [60.0, 0.4], [60.0, 0.5], [30.0, 0.1], [30.0, 0.2],
                [30.0, 0.3], [30.0, 0.4], [90.0, 0.1], [90.0, 0.2],
                [90.0, 0.3], [90.0, 0.4], [45.0, 0.1], [45.0, 0.2]
            ]

    def add_custom_color(self, zone: str, hue: float, sat: float) -> None:
        h = float(round(hue)) % 360
        s = float(round(sat, 2))
        
        history = self.load_custom_colors(zone)
                
        new_history = []
        for item in history:
            if item is None or not isinstance(item, list) or len(item) < 2:
                continue
            if abs(item[0] - h) < 1.0 and abs(item[1] - s) < 0.01:
                continue
            new_history.append(item)
            
        new_history.insert(0, [h, s])
        new_history = new_history[:16]
        
        settings = QSettings("NSS", "ColorGradingExplorer")
        settings.setValue(f"history_grid_{zone}", json.dumps(new_history))
        self.update_history_swatches_ui(zone, new_history)

    def update_history_swatches_ui(self, zone: str, history: list) -> None:
        if zone == "shadows":
            grid = self.sh_history_grid
        elif zone == "midtones":
            grid = self.mid_history_grid
        else:
            grid = self.hi_history_grid

        while grid.count():
            item = grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for i, item in enumerate(history):
            row = i // 8
            col = i % 8
            if item is None or not isinstance(item, list) or len(item) < 2:
                h, s = None, None
            else:
                h, s = item[0], item[1]
                
            swatch = ColorSwatch(h, s, self)
            swatch.index = i
            swatch.clicked.connect(lambda hue, sat, z=zone: self.on_swatch_clicked(z, hue, sat))
            swatch.overwritten.connect(lambda idx, hue, sat, z=zone: self.on_swatch_overwritten(z, idx, hue, sat))
            swatch.edited.connect(lambda idx, hue, sat, z=zone: self.on_swatch_overwritten(z, idx, hue, sat))
            swatch.forgotten.connect(lambda idx, z=zone: self.on_swatch_forgotten(z, idx))
            grid.addWidget(swatch, row, col)

    def on_swatch_clicked(self, zone: str, hue: float, sat: float) -> None:
        if zone == "shadows":
            wheel = self.sh_wheel
            sat_slider = self.sh_sat_slider
        elif zone == "midtones":
            wheel = self.mid_wheel
            sat_slider = self.mid_sat_slider
        else:
            wheel = self.hi_wheel
            sat_slider = self.hi_sat_slider

        self.block_manual_signals(True)
        wheel.set_color(hue, sat)
        sat_slider.setValue(int(sat * 100.0))
        self.block_manual_signals(False)

        self.on_manual_slider_changed()

    def on_swatch_overwritten(self, zone: str, index: int, hue: float, sat: float) -> None:
        history = self.load_custom_colors(zone)
        if index < len(history):
            history[index] = [float(round(hue)) % 360, float(round(sat, 2))]
            settings = QSettings("NSS", "ColorGradingExplorer")
            settings.setValue(f"history_grid_{zone}", json.dumps(history))

    def on_swatch_forgotten(self, zone: str, index: int) -> None:
        history = self.load_custom_colors(zone)
        if index < len(history):
            history[index] = None
            settings = QSettings("NSS", "ColorGradingExplorer")
            settings.setValue(f"history_grid_{zone}", json.dumps(history))
            self.update_history_swatches_ui(zone, history)

    def update_zone_swatches(self, state: StateNode) -> None:
        """
        Updates the three tab color swatches live based on HSL and updates draggable state.
        """
        from PyQt6.QtGui import QColor, QPixmap
        
        is_comp = state.get("harmony_mode") == "Complementary"

        # Shadows
        sh_hue = state.get("shadow_hue", 240.0)
        sh_sat = 0.15 if is_comp else state.get("shadow_sat", 0.0)
        sh_color = QColor.fromHslF(sh_hue / 360.0, sh_sat, 1.0 - 0.5 * sh_sat)
        sh_pix = QPixmap(24, 24)
        sh_pix.fill(sh_color)
        self.sh_swatch_label.setPixmap(sh_pix)
        self.sh_swatch_label.hue = sh_hue
        self.sh_swatch_label.sat = sh_sat
        
        # Midtones
        mid_hue = state.get("midtone_hue", 120.0)
        mid_sat = state.get("midtone_sat", 0.0)
        mid_color = QColor.fromHslF(mid_hue / 360.0, mid_sat, 1.0 - 0.5 * mid_sat)
        mid_pix = QPixmap(24, 24)
        mid_pix.fill(mid_color)
        self.mid_swatch_label.setPixmap(mid_pix)
        self.mid_swatch_label.hue = mid_hue
        self.mid_swatch_label.sat = mid_sat

        # Highlights
        hi_hue = state.get("highlight_hue", 60.0)
        hi_sat = 0.15 if is_comp else state.get("highlight_sat", 0.0)
        hi_color = QColor.fromHslF(hi_hue / 360.0, hi_sat, 1.0 - 0.5 * hi_sat)
        hi_pix = QPixmap(24, 24)
        hi_pix.fill(hi_color)
        self.hi_swatch_label.setPixmap(hi_pix)
        self.hi_swatch_label.hue = hi_hue
        self.hi_swatch_label.sat = hi_sat

    def sync_sliders_with_state(self, state: StateNode) -> None:
        """
        Synchronizes all UI sliders, wheels, and swatches to match the active StateNode parameters.
        Blocks signals to avoid triggering recursive render events during synchronization.
        """
        self.block_manual_signals(True)

        self.sh_wheel.set_color(state.get("shadow_hue", 240.0), state.get("shadow_sat", 0.0))
        self.sh_sat_slider.setValue(int(state.get("shadow_sat", 0.0) * 100.0))
        self.sh_light_slider.setValue(int(state.get("shadow_light", 0.0) * 100.0))

        self.mid_wheel.set_color(state.get("midtone_hue", 120.0), state.get("midtone_sat", 0.0))
        self.mid_sat_slider.setValue(int(state.get("midtone_sat", 0.0) * 100.0))
        self.mid_light_slider.setValue(int(state.get("midtone_light", 0.0) * 100.0))

        self.hi_wheel.set_color(state.get("highlight_hue", 60.0), state.get("highlight_sat", 0.0))
        self.hi_sat_slider.setValue(int(state.get("highlight_sat", 0.0) * 100.0))
        self.hi_light_slider.setValue(int(state.get("highlight_light", 0.0) * 100.0))

        self.blending_slider.setValue(int(state.get("blending", 0.5) * 100.0))
        self.balance_slider.setValue(int(state.get("balance", 0.0) * 100.0))

        # Update labels readouts
        self.sh_hue_lbl.setText(f"Hue: {state.get('shadow_hue', 240.0):.0f}°")
        self.sh_sat_lbl.setText(f"Sat: {state.get('shadow_sat', 0.0):.2f}")
        self.sh_light_lbl.setText(f"Luma: {state.get('shadow_light', 0.0):+.2f}")

        self.mid_hue_lbl.setText(f"Hue: {state.get('midtone_hue', 120.0):.0f}°")
        self.mid_sat_lbl.setText(f"Sat: {state.get('midtone_sat', 0.0):.2f}")
        self.mid_light_lbl.setText(f"Luma: {state.get('midtone_light', 0.0):+.2f}")

        self.hi_hue_lbl.setText(f"Hue: {state.get('highlight_hue', 60.0):.0f}°")
        self.hi_sat_lbl.setText(f"Sat: {state.get('highlight_sat', 0.0):.2f}")
        self.hi_light_lbl.setText(f"Luma: {state.get('highlight_light', 0.0):+.2f}")

        self.blending_lbl.setText(f"Blending: {state.get('blending', 0.5):.2f}")
        self.balance_lbl.setText(f"Balance: {state.get('balance', 0.0):+.2f}")

        self.update_zone_swatches(state)

        self.block_manual_signals(False)

    def block_manual_signals(self, block: bool) -> None:
        """
        Helper to block/unblock signals on all manual controls.
        """
        self.sh_wheel.blockSignals(block)
        self.sh_sat_slider.blockSignals(block)
        self.sh_light_slider.blockSignals(block)

        self.mid_wheel.blockSignals(block)
        self.mid_sat_slider.blockSignals(block)
        self.mid_light_slider.blockSignals(block)

        self.hi_wheel.blockSignals(block)
        self.hi_sat_slider.blockSignals(block)
        self.hi_light_slider.blockSignals(block)

        self.blending_slider.blockSignals(block)
        self.balance_slider.blockSignals(block)

    def update_recent_directories_menu(self) -> None:
        """
        Clears and repopulates the Recent Directories submenu from MruManager paths.
        """
        self.recent_menu.clear()
        paths = self.mru_manager.get_mru_directories()
        
        if not paths:
            no_recent_action = QAction("No Recent Directories", self)
            no_recent_action.setEnabled(False)
            self.recent_menu.addAction(no_recent_action)
            return

        for i, path in enumerate(paths):
            action = QAction(f"&{i + 1}: {path}", self)
            action.triggered.connect(lambda checked=False, p=path: self.open_file_dialog_at_dir(p))
            self.recent_menu.addAction(action)

    def open_file_dialog_at_dir(self, start_dir: str) -> None:
        """
        Convenience method to trigger the open file dialog at a specific path.
        """
        self.open_file(start_dir=start_dir)

    def open_file(self, start_dir: Optional[str] = None) -> None:
        """
        Opens a file dialog to load a 16-bit TIFF file.
        """
        if start_dir is None:
            start_dir = self.mru_manager.get_most_recent_directory()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 16-bit TIFF",
            start_dir,
            "TIFF Images (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            try:
                from nss.image_utils import load_tiff_to_float32, downsample_image
                
                # Load using the scaling pipeline (high resolution master)
                self.master_image, self.tiff_obj = load_tiff_to_float32(file_path)
                
                # Automatically generate a downsampled working proxy (1200px max dimension)
                self.proxy_image = downsample_image(self.master_image, max_dim=1200)
                
                # Update MRU directories
                self.mru_manager.add_path(file_path)
                self.update_recent_directories_menu()
                
                # Initialize history with default state
                self.history_manager.clear()
                initial_state = create_default_state("Monochromatic")
                self.history_manager.push_state(initial_state)
                
                # Render grid
                self.render_grid_from_current_state()
                
                # Update status bar
                shape_str = "x".join(map(str, self.master_image.shape))
                proxy_str = "x".join(map(str, self.proxy_image.shape))
                self.statusBar().showMessage(f"Loaded {file_path} (Master: {shape_str}, Proxy: {proxy_str})")
            except Exception as e:
                self.statusBar().showMessage(f"Error loading image: {str(e)}")

    def save_file(self) -> None:
        """
        Saves the current color graded image as a 16-bit TIFF.
        """
        if self.master_image is None or self.tiff_obj is None or self.grid_states is None:
            return

        start_dir = self.mru_manager.get_most_recent_directory()

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save 16-bit TIFF",
            start_dir,
            "TIFF Images (*.tif *.tiff);;All Files (*)"
        )
        if save_path:
            try:
                # Apply current center grading to the full-resolution master image!
                center_state = self.grid_states[4]
                graded_image = apply_grading(self.master_image, center_state)
                
                # Store the graded image onto our tiff object
                self.tiff_obj.array = graded_image
                
                # Save using the original metadata on tiff_obj
                self.tiff_obj.saveas(save_path)
                
                # Update MRU directories
                self.mru_manager.add_path(save_path)
                self.update_recent_directories_menu()
                
                self.statusBar().showMessage(f"Successfully saved graded image to {save_path}")
            except Exception as e:
                self.statusBar().showMessage(f"Error saving image: {str(e)}")

    def on_back_clicked(self) -> None:
        """
        Navigates back in history.
        """
        if self.history_manager.can_undo():
            self.history_manager.undo()
            self.render_grid_from_current_state()
            self.statusBar().showMessage("Undo action.")

    def on_forward_clicked(self) -> None:
        """
        Navigates forward in history.
        """
        if self.history_manager.can_redo():
            self.history_manager.redo()
            self.render_grid_from_current_state()
            self.statusBar().showMessage("Redo action.")

    def on_harmony_mode_changed(self, text: str) -> None:
        """
        Handler for when the user selects a different harmony mode from the dropdown.
        Updates active state's harmony mode and triggers re-generation of outer tiles,
        keeping the center image completely untouched.
        """
        if self.master_image is None:
            return
            
        state = self.history_manager.get_current_state()
        if state is not None:
            state["harmony_mode"] = text  # type: ignore
            # Trigger outer progressive render, skipping the center!
            self.start_asynchronous_render_for_state(state, skip_center=True)

    def on_axis_toggled(self, checked: bool) -> None:
        """
        Triggered when a mutation axis radio button is toggled.
        Only re-renders outer tiles (skipping center) if checked is True.
        Does NOT push a new state to history.
        """
        if not checked or self.master_image is None:
            return
            
        state = self.history_manager.get_current_state()
        if state is not None:
            self.start_asynchronous_render_for_state(state, skip_center=True)

    def on_intensity_changed(self, value: int) -> None:
        """
        Triggered when the user slides the mutation intensity control.
        Updates the intensity label, updates active state's step_size,
        and triggers a progressive re-render of the 8 surrounding outer tiles.
        """
        step_size = value / 100.0
        self.intensity_label.setText(f" Intensity: {step_size:.2f}x ")
        
        if self.master_image is None:
            return
            
        state = self.history_manager.get_current_state()
        if state is not None:
            state["step_size"] = step_size
            # Re-render outer tiles only, keeping the center invariant
            self.start_asynchronous_render_for_state(state, skip_center=True)

    def get_active_mutation_axis(self) -> MutationAxis:
        """
        Returns the active mutation axis. Defaults to 'All' now that controls are removed.
        """
        return "All"

    def render_grid_from_current_state(self) -> None:
        """
        Renders all 9 containers asynchronously from current active state in history.
        """
        state = self.history_manager.get_current_state()
        if state is not None:
            self.start_asynchronous_render_for_state(state, skip_center=False)
            self.sync_sliders_with_state(state)

    def start_asynchronous_render_for_state(self, state: StateNode, skip_center: bool = False) -> None:
        """
        Generates 9 mutations based on the input state, and fires up a background
        GradingWorker thread to compute and render the tiles progressively.
        """
        if self.proxy_image is None:
            return

        # 1. Sync intensity slider only if skip_center is False (load/undo/redo)
        if not skip_center:
            self.intensity_slider.blockSignals(True)
            self.intensity_slider.setValue(int(state.get("step_size", 0.2) * 100.0))
            self.intensity_slider.blockSignals(False)
            self.intensity_label.setText(f" Intensity: {state.get('step_size', 0.2):.2f}x ")

        # 2. Update navigation controls and history index display immediately
        self.back_action.setEnabled(self.history_manager.can_undo())
        self.forward_action.setEnabled(self.history_manager.can_redo())
        self.save_action.setEnabled(True)
        self.history_label.setText(self.history_manager.get_history_display_text())

        # 3. Generate the 9 state nodes for the grid using current state's active mode
        mutations = generate_mutations(state, state.get("harmony_mode", "Monochromatic"), axis="All")
        
        # Lock index 4 to be EXACTLY the original unchanged history state!
        mutations[4] = state.copy()
        self.grid_states = mutations

        # 4. Cancel any running background worker thread safely
        if self.grading_worker is not None:
            if self.grading_worker.isRunning():
                self.grading_worker.cancel()
                self.grading_worker.wait()
            self.grading_worker = None

        # 5. Create and launch the new worker thread using proxy_image for snappy UI calculation
        self.grading_worker = GradingWorker(
            self.proxy_image,
            self.grid_states,
            skip_center=skip_center,
            parent=self
        )
        self.grading_worker.progress.connect(self.on_worker_progress)
        self.grading_worker.finished_all.connect(self.on_worker_finished)
        self.grading_worker.start()

    def on_worker_progress(self, index: int, arr: np.ndarray) -> None:
        """
        Updates a specific container tile in the grid as soon as it is computed.
        """
        if self.proxy_image is not None:
            self.containers[index].set_image(arr)

    def on_worker_finished(self) -> None:
        """
        Updates the status bar when all outer mutations are finished processing.
        """
        self.statusBar().showMessage("Ready.")

    def on_container_clicked(self, index: int) -> None:
        """
        Handler for when an image container in the 3x3 grid is clicked.
        Promotes selected outer variation to center immediately, or re-rolls mutation proposals
        if the center tile itself is clicked.
        """
        if self.proxy_image is None or self.grid_states is None:
            return

        if index == 4:
            # Clicking the center image tile triggers a re-roll of the 8 outer tiles
            self.statusBar().showMessage("Re-rolling mutation proposals around center...")
            state = self.history_manager.get_current_state()
            if state is not None:
                # Clear all surrounding cells immediately to give instant visual feedback that a re-roll has begun
                for i in range(9):
                    if i != 4:
                        self.containers[i].set_image(None)
                # Fire up background progressive rendering of the 8 outer tiles, skipping center
                self.start_asynchronous_render_for_state(state, skip_center=True)
            return

        # 1. Promote clicked state parameters to center state
        chosen_state = self.grid_states[index]

        # 2. Update center tile immediately with pre-computed high-res pixmap of the clicked tile!
        clicked_container = self.containers[index]
        if clicked_container.master_pixmap is not None:
            self.containers[4].master_pixmap = clicked_container.master_pixmap
            self.containers[4].update_display_pixmap()
            self.containers[4].set_active(True)
        else:
            # Fallback if somehow pixmap is missing (e.g. still rendering)
            self.containers[4].set_image(apply_grading(self.proxy_image, chosen_state))

        # Clear other cells immediately to show we are generating new mutations around the new center
        for i in range(9):
            if i != 4:
                self.containers[i].set_image(None)

        # 3. Push the new state to history
        self.history_manager.push_state(chosen_state)
        self.sync_sliders_with_state(chosen_state)

        # 4. Fire up background asynchronous worker to render other 8 tiles progressively, skipping the center!
        self.start_asynchronous_render_for_state(chosen_state, skip_center=True)
        
        self.statusBar().showMessage(f"Promoted mutation {index} to center immediately. Computing new mutations...")

    def closeEvent(self, event) -> None:
        """
        Safely stops active background thread when application is closed.
        """
        if self.grading_worker is not None:
            if self.grading_worker.isRunning():
                self.grading_worker.cancel()
                self.grading_worker.wait()
        super().closeEvent(event)
