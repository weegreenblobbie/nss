import math
import numpy as np
from typing import Optional
from nss.color_math import WHEEL_ANGLES, HSL_ANGLES, wheel_to_hsl, hsl_to_wheel
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtGui import (
    QMouseEvent,
    QPainter,
    QConicalGradient,
    QRadialGradient,
    QPen,
    QColor,
    QBrush,
)

class ResetLabel(QLabel):
    """
    A custom QLabel that detects double clicks and emits a signal to reset controls.
    """
    doubleClicked = pyqtSignal()

    def mouseDoubleClickEvent(self, event) -> None:
        self.doubleClicked.emit()


class ColorWheel(QWidget):
    """
    A custom circular color wheel widget.
    Paints Hue using a conical gradient and Saturation using a radial gradient.
    """
    colorChanged = pyqtSignal(float, float)  # Emits (hue, saturation)
    interactionFinished = pyqtSignal(float, float)  # Emits (hue, saturation) when mouse is released

    # Perceptual Mapping: Calibrated against Adobe Lightroom color wheel (from nss.color_math)
    WHEEL_ANGLES = WHEEL_ANGLES
    HSL_ANGLES   = HSL_ANGLES

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.hue: float = 0.0  # Range: [0.0, 360.0]
        self.sat: float = 0.0  # Range: [0.0, 1.0]
        self.rotation_offset: float = 0.0  # Real-time rotation offset
        self.show_all_tones: bool = False
        self.all_tones_data: dict = {}
        self.setMinimumSize(160, 160)
        self.setMaximumSize(240, 240)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _wheel_to_hsl(self, angle: float) -> float:
        return float(wheel_to_hsl(angle))

    def _hsl_to_wheel(self, angle: float) -> float:
        return float(hsl_to_wheel(angle))

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

        # 1. Paint conical gradient (Hue spectrum) warped for human perception
        conical = QConicalGradient(cx, cy, 0.0)
        # Using 360 stops forces Qt to interpolate via our HSL path, preventing muddy RGB blending
        for i in range(361):
            hsl_h = self._wheel_to_hsl(i)
            conical.setColorAt(i / 360.0, QColor.fromHslF(hsl_h / 360.0, 1.0, 0.5))

        painter.setBrush(conical)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))

        # 2. Paint radial gradient (Saturation overlay)
        radial = QRadialGradient(cx, cy, radius)
        radial.setColorAt(0.0, QColor(255, 255, 255, 255))
        radial.setColorAt(1.0, QColor(255, 255, 255, 0))

        painter.setBrush(radial)
        painter.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))

        # 3. Draw indicator/handle using effective hue (including rotation offset)
        if self.show_all_tones and self.all_tones_data:
            for label, (base_hue, sat) in self.all_tones_data.items():
                visual_base_hue = self._hsl_to_wheel(base_hue)
                eff_hue = (visual_base_hue + self.rotation_offset) % 360.0
                angle_rad = math.radians(eff_hue)
                d = sat * radius
                px = cx + d * math.cos(angle_rad)
                py = cy - d * math.sin(angle_rad)
                
                # Draw text with no background color (transparent) and text color black
                painter.setBrush(Qt.BrushStyle.NoBrush)
                
                font = painter.font()
                font.setBold(True)
                font.setPointSize(9)
                painter.setFont(font)
                
                painter.setPen(QPen(Qt.GlobalColor.black))
                painter.drawText(QRectF(px - 8, py - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, label)
        else:
            visual_hue = self._hsl_to_wheel(self.hue)
            effective_hue = (visual_hue + self.rotation_offset) % 360.0
            angle_rad = math.radians(effective_hue)
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
            
            # Map the visual mouse angle back to mathematical HSL hue
            hue = self._wheel_to_hsl(angle_deg)

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