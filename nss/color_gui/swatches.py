from typing import Optional
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QDialog,
    QLineEdit,
    QDialogButtonBox,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QMenu,
)
from PyQt6.QtGui import (
    QMouseEvent,
    QDrag,
    QPixmap,
    QColor,
    QAction,
    QIntValidator,
)

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
