"""The single preview with Color Harmony samples: eyedropper picking and numbered sample circles anchored to image pixels."""
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QContextMenuEvent, QCursor, QMouseEvent, QPainter, QPaintEvent, QPen, QPixmap
from PyQt6.QtWidgets import QStyle, QWidget

from nss.color_gui.explore_widget import ImageContainer

MARKER_RADIUS = 9           # sample circle radius on the preview, screen pixels
EYEDROPPER_SIZE = 24        # eyedropper icon / cursor side, pixels


def eyedropper_pixmap(size: int = EYEDROPPER_SIZE) -> QPixmap:
    """An eyedropper glyph (tip at the bottom-left corner) on a transparent size x size pixmap."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 24.0
    for color, width in (("white", 5.0), ("black", 2.5)):           # white halo, visible on dark images
        pen = QPen(QColor(color), width * s)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(2.5 * s, 21.5 * s), QPointF(15.0 * s, 9.0 * s))
    painter.setPen(QPen(QColor("white"), 1.0 * s))
    painter.setBrush(QBrush(QColor("black")))
    painter.drawEllipse(QPointF(17.5 * s, 6.5 * s), 4.5 * s, 4.5 * s)                     # bulb
    painter.end()
    return pixmap


class SampleView(ImageContainer):
    """The single preview.  In sampling mode a left click emits sample_picked with the clicked position as fractions of the image width and
    height; the samples are drawn as numbered circles at their image positions (they follow the scaling of the preview)."""
    sample_picked = pyqtSignal(float, float)            # (u, v): 0..1 fractions of the image width and height
    sample_clicked = pyqtSignal(int, QPoint)            # slot, global position: left click on a sample circle
    sample_menu_requested = pyqtSignal(int, QPoint)     # slot, global position: right click on a sample circle

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """An empty preview, not sampling, with no samples."""
        super().__init__(-1, parent)
        self.image_array: Optional[np.ndarray] = None   # the array on screen, as passed to set_image
        self.markers: List[Tuple[int, float, float]] = []
        self.sampling = False

    def set_image(self, arr: Optional[np.ndarray]) -> None:
        """Shows arr and keeps it as the displayed image."""
        self.image_array = arr
        super().set_image(arr)

    def set_markers(self, markers: List[Tuple[int, float, float]]) -> None:
        """Sets the sample circles: (slot, u, v) with u, v the 0..1 fractions of the image width and height."""
        self.markers = list(markers)
        self.update()

    def set_sampling(self, on: bool) -> None:
        """Turns sampling mode on (eyedropper cursor) or off (default cursor)."""
        self.sampling = on
        if on:
            self.setCursor(QCursor(eyedropper_pixmap(), 2, EYEDROPPER_SIZE - 3))
        else:
            self.unsetCursor()

    def image_rect(self) -> Optional[QRectF]:
        """Widget rectangle of the shown (scaled) image, None without an image."""
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        size = pixmap.deviceIndependentSize().toSize()
        return QRectF(QStyle.alignedRect(self.layoutDirection(), self.alignment(), size, self.contentsRect()))

    def marker_center(self, rect: QRectF, u: float, v: float) -> QPointF:
        """Widget position of the image fractions (u, v) inside the image rectangle rect."""
        return QPointF(rect.left() + u * rect.width(), rect.top() + v * rect.height())

    def marker_at(self, pos: QPointF) -> Optional[int]:
        """Slot of the sample circle under pos, None if there is none."""
        rect = self.image_rect()
        if rect is None:
            return None
        for slot, u, v in reversed(self.markers):                  # the last drawn is on top
            d = pos - self.marker_center(rect, u, v)
            if d.x() * d.x() + d.y() * d.y() <= MARKER_RADIUS * MARKER_RADIUS:
                return slot
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Left click: on a sample circle emits sample_clicked, else in sampling mode on the image emits sample_picked."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        slot = self.marker_at(pos)
        if slot is not None:
            self.sample_clicked.emit(slot, event.globalPosition().toPoint())
            return
        rect = self.image_rect()
        if self.sampling and rect is not None and rect.contains(pos):
            self.sample_picked.emit((pos.x() - rect.left()) / rect.width(), (pos.y() - rect.top()) / rect.height())

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        """Right click on a sample circle emits sample_menu_requested; elsewhere the preview's own menu."""
        slot = self.marker_at(QPointF(event.pos()))
        if slot is None:
            super().contextMenuEvent(event)
        else:
            self.sample_menu_requested.emit(slot, event.globalPos())

    def paintEvent(self, event: QPaintEvent) -> None:
        """The image, then the numbered sample circles."""
        super().paintEvent(event)
        rect = self.image_rect()
        if rect is None or not self.markers:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = painter.font()
        font.setBold(True)
        font.setPointSize(8)
        painter.setFont(font)
        for slot, u, v in self.markers:
            c = self.marker_center(rect, u, v)
            painter.setPen(QPen(QColor("black"), 1.5))
            painter.setBrush(QBrush(QColor(255, 255, 255, 210)))
            painter.drawEllipse(c, MARKER_RADIUS, MARKER_RADIUS)
            painter.drawText(QRectF(c.x() - MARKER_RADIUS, c.y() - MARKER_RADIUS, 2 * MARKER_RADIUS, 2 * MARKER_RADIUS), Qt.AlignmentFlag.AlignCenter, str(slot))
        painter.end()
