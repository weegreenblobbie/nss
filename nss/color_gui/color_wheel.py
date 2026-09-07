"""The colour wheel widget (painted like Lightroom's, hue = screen angle) and the double-clickable reset label."""
import numpy as np
from typing import Optional
from nss.color_gui.wheel_lut import wheel_image, hue_sat_at, offset_at
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtGui import (
    QImage,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QBrush,
)

class ResetLabel(QLabel):
    """
    A custom QLabel that detects double clicks and emits a signal to reset controls.
    """
    doubleClicked = pyqtSignal()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Emits doubleClicked."""
        self.doubleClicked.emit()


class ColorWheel(QWidget):
    """
    A custom circular color wheel widget, painted like Lightroom's (nss.color_gui.wheel_lut).
    The hue is the screen angle (0 at 3 o'clock, counter-clockwise); Sat 100 sits at wheel_lut.SAT_FULL_RADIUS of the painted radius.
    """
    colorChanged = pyqtSignal(float, float)  # Emits (hue, saturation)
    interactionFinished = pyqtSignal(float, float)  # Emits (hue, saturation) when mouse is released

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Starts at hue 0, Sat 0, no rotation; the painted wheel image is built on the first paint."""
        super().__init__(parent)
        self.hue: float = 0.0  # Range: [0.0, 360.0]
        self.sat: float = 0.0  # Range: [0.0, 1.0]
        self.rotation_offset: float = 0.0  # Real-time rotation offset
        self.show_all_tones: bool = False
        self.all_tones_data: dict = {}
        self._wheel_image: Optional[QImage] = None      # painted wheel, rebuilt only when the widget size changes
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

    def wheel_geometry(self) -> tuple[float, float, float]:
        """(cx, cy, radius) in widget pixels of the painted wheel: centred, 5 px margin."""
        rect = self.rect()
        return rect.width() / 2.0, rect.height() / 2.0, (min(rect.width(), rect.height()) - 10) / 2.0

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draws the painted wheel, then the handle (or the S / M / H markers) at their hues plus the rotation offset."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy, radius = self.wheel_geometry()
        size = int(round(2.0 * radius))         # painted disc diameter in pixels

        if radius <= 0:
            return

        # 1. Paint Lightroom's measured wheel (warped hues blending to the centre grey), cached per size
        if self._wheel_image is None or self._wheel_image.width() != size:
            self._wheel_image = wheel_image(size)
        painter.drawImage(QPointF(cx - size / 2.0, cy - size / 2.0), self._wheel_image)

        # 2. Draw indicator/handle using effective hue (including rotation offset); the hue number is the screen angle
        if self.show_all_tones and self.all_tones_data:
            for label, (base_hue, sat) in self.all_tones_data.items():
                dx, dy = offset_at(base_hue + self.rotation_offset, sat, radius)
                px, py = cx + dx, cy + dy

                # Draw text with no background color (transparent) and text color black
                painter.setBrush(Qt.BrushStyle.NoBrush)
                
                font = painter.font()
                font.setBold(True)
                font.setPointSize(9)
                painter.setFont(font)
                
                painter.setPen(QPen(Qt.GlobalColor.black))
                painter.drawText(QRectF(px - 8, py - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, label)
        else:
            dx, dy = offset_at(self.hue + self.rotation_offset, self.sat, radius)
            px, py = cx + dx, cy + dy

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            painter.drawEllipse(QPointF(px, py), 5.0, 5.0)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawEllipse(QPointF(px, py), 4.0, 4.0)

    def _update_color_from_mouse(self, pos: QPointF) -> None:
        """Sets hue (the screen angle) and saturation from a mouse position and emits colorChanged."""
        cx, cy, radius = self.wheel_geometry()

        if radius <= 0:
            return

        dx = pos.x() - cx
        dy = pos.y() - cy
        hue, sat = hue_sat_at(dx, dy, radius)
        if dx == 0 and dy == 0:     # the centre has no angle: keep the hue
            hue = self.hue

        self.hue = hue
        self.sat = sat
        self.update()
        self.colorChanged.emit(self.hue, self.sat)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """A left click picks the colour under the mouse."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._update_color_from_mouse(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Dragging with the left button keeps picking."""
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._update_color_from_mouse(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Releasing the left button picks once more and emits interactionFinished."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._update_color_from_mouse(event.position())
            self.interactionFinished.emit(self.hue, self.sat)