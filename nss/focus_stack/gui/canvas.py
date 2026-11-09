"""The picture view of the focus stack window: zoom and pan, the brush, the active patch's outline, and a pin on every patch."""
import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from nss.focus_stack.edit import EditState
from nss.image_utils import to_uint8_display

PIN_RADIUS = 7.0  # widget pixels
PIN_HIT = 10.0  # widget pixels from a pin's middle that still count as clicking it
BRUSH_STEP = 1.25  # factor of one [ or ] press
BRUSH_RANGE = (1.0, 500.0)  # picture pixels
ZOOM_RANGE = (0.02, 64.0)  # widget pixels per picture pixel
ZOOM_STEPS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)  # the levels zoom_in and zoom_out move between
PAN_PER_NOTCH = 60.0  # widget pixels a mouse wheel notch pans
TINT_OPACITY = 0.35
TINT_COLORS = np.array(  # RGBA: the target is clear, then one colour per frame, repeating
    [(0, 0, 0, 0), (255, 0, 0, 255), (0, 200, 0, 255), (0, 90, 255, 255), (255, 200, 0, 255), (255, 0, 255, 255), (0, 220, 220, 255)],
    dtype=np.uint8,
)


class CanvasView(QtWidgets.QWidget):
    """Shows the merged picture of an `EditState` and turns mouse and key events into edits of it.

    Left-drag paints (starting a patch, or adding to the active one); Shift- or Alt-drag erases from the active patch; a click
    on a pin selects that patch; Ctrl+wheel or a pinch zooms about the cursor, a two-finger scroll or wheel pans, and so do a
    middle-button drag and a left-drag in pan mode (Space toggles it, Esc leaves it). Keys: [ ] brush size, Tab and
    Shift+Tab cycle the active patch's layer, Esc deselects, Delete removes the patch, hold T to see the target, L
    for the layer tint. Only the changed region of the picture is recomputed after an edit.
    """

    edited = QtCore.pyqtSignal()  # the patches or the selection changed
    status_changed = QtCore.pyqtSignal(str)  # the one-line description changed
    pan_mode_changed = QtCore.pyqtSignal(bool)  # pan mode was switched on or off
    view_changed = QtCore.pyqtSignal(object)  # the layer being viewed changed: its number, or None for the merged picture

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """An empty canvas; give it a picture with set_state."""
        super().__init__(parent)
        self.radius = 20.0  # brush radius in picture pixels
        self.scale = 1.0  # widget pixels per picture pixel
        self.show_tint = False
        self.pan_mode = False  # while on, a left-drag pans instead of painting
        self.pixels = np.zeros((1, 1, 3), dtype=np.uint8)
        self._offset = QtCore.QPointF(0.0, 0.0)
        self._state: EditState | None = None
        self._names: list[str] = []
        self._full_shape: tuple[int, int] | None = None
        self._target_pixels = self.pixels
        self._image = QtGui.QImage()
        self._target_image = QtGui.QImage()
        self._tint_image: QtGui.QImage | None = None
        self._outline: list[np.ndarray] | None = None
        self._show_target = False
        self.viewed_layer: int | None = None  # while set, the whole picture of that layer is shown and painting is off
        self._layer_pixels: dict[int, np.ndarray] = {}
        self._layer_images: dict[int, QtGui.QImage] = {}
        self._fitted = True
        self._mouse: QtCore.QPointF | None = None
        self._stroke: str | None = None  # "paint" or "erase" while a stroke is held
        self._last: tuple[float, float] | None = None
        self._pan_from: QtCore.QPointF | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._update_cursor()

    @property
    def state(self) -> EditState | None:
        """The edit state on show, or None before set_state."""
        return self._state

    def set_state(self, state: EditState, names: list[str], *, full_shape: tuple[int, int] | None = None) -> None:
        """Shows `state`; `names[k]` is what to call layer k (0 the target) in the status line.

        `full_shape` is the (rows, cols) of the full-size picture, when known, so the status can say what share of it the zoom shows.
        """
        self._state = state
        self._names = names
        self._full_shape = full_shape
        self.pixels = np.ascontiguousarray(to_uint8_display(state.result()))
        self._target_pixels = np.ascontiguousarray(to_uint8_display(state.target))
        self._image = self._make_image(self.pixels)
        self._target_image = self._make_image(self._target_pixels)
        self.viewed_layer, self._layer_pixels, self._layer_images = None, {}, {}
        self._tint_image = None
        self._outline = None
        self._fitted = True
        self.zoom_fit()
        self._announce()

    def zoom_fit(self) -> None:
        """Scales and centres the picture to fill the widget, and keeps it filling the widget as the window is resized."""
        rows, cols = self.pixels.shape[:2]
        self.scale = min(self.width() / cols, self.height() / rows)
        self._offset = QtCore.QPointF((self.width() - cols * self.scale) / 2.0, (self.height() - rows * self.scale) / 2.0)
        self._fitted = True
        self.update()
        self._announce()

    def set_zoom(self, scale: float) -> None:
        """Sets the widget pixels per picture pixel, within the zoom range, keeping the picture point at the middle of the widget still."""
        scale = min(max(scale, ZOOM_RANGE[0]), ZOOM_RANGE[1])
        middle_x, middle_y = self.width() / 2.0, self.height() / 2.0
        x, y = (middle_x - self._offset.x()) / self.scale, (middle_y - self._offset.y()) / self.scale
        self.scale = scale
        self._offset = QtCore.QPointF(middle_x - x * scale, middle_y - y * scale)
        self._fitted = False
        self.update()
        self._announce()

    def zoom_in(self) -> None:
        """Zooms to the next level above the current one, if there is one."""
        larger = [step for step in ZOOM_STEPS if step > self.scale * 1.001]
        if larger:
            self.set_zoom(larger[0])

    def zoom_out(self) -> None:
        """Zooms to the next level below the current one, if there is one."""
        smaller = [step for step in ZOOM_STEPS if step < self.scale / 1.001]
        if smaller:
            self.set_zoom(smaller[-1])

    def to_widget(self, x: float, y: float) -> QtCore.QPointF:
        """The widget position of the middle of picture pixel (column `x`, row `y`)."""
        return QtCore.QPointF((x + 0.5) * self.scale + self._offset.x(), (y + 0.5) * self.scale + self._offset.y())

    def to_image(self, position: QtCore.QPointF) -> tuple[float, float]:
        """The picture position (column, row) under a widget position."""
        return (position.x() - self._offset.x()) / self.scale - 0.5, (position.y() - self._offset.y()) / self.scale - 0.5

    def shown_pixels(self) -> np.ndarray:
        """The 8-bit picture on show: the viewed layer, or the target while T is held, otherwise the merged result."""
        if self.viewed_layer is not None:
            return self._viewed(self.viewed_layer)[0]
        return self._target_pixels if self._show_target else self.pixels

    def view_layer(self, layer: int | None) -> None:
        """Shows the whole picture of `layer` (0 the target) instead of the merged result, or goes back to the result with None.

        While a layer is viewed nothing can be painted or selected.
        """
        if self._state is None or layer == self.viewed_layer:
            return
        self.viewed_layer = layer
        self.update()
        self.view_changed.emit(layer)
        self._announce()

    def outline(self) -> list[np.ndarray]:
        """The contours of the active patch as (N, 2) arrays of (column, row), empty when nothing is active."""
        if self._outline is None:
            mask = None if self._state is None else self._state.active_mask()
            if mask is None:
                self._outline = []
            else:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                self._outline = [contour[:, 0, :] for contour in contours]
        return self._outline

    def describe(self) -> str:
        """One line for the status bar: brush size, zoom and, with a patch active, its number and the layer it shows."""
        text = f"brush {self.radius:.0f} px | zoom {self.scale * 100.0:.0f}%"
        if self._full_shape is not None:
            text += f" ({self.scale * 100.0 * self.pixels.shape[1] / self._full_shape[1]:.0f}% of full size)"
        if self.viewed_layer is not None:
            text += f" | viewing {self._names[self.viewed_layer]} (Esc to go back)"
        elif self._state is not None and self._state.active is not None:
            layer = self._names[self._state.patch_label(self._state.active)]
            text += f" | patch {self._state.active + 1} of {self._state.patch_count}: {layer}"
        if self.pan_mode:
            text += " | pan mode (Space or Esc to leave)"
        return text

    def undo(self) -> None:
        """Reverts the latest change."""
        if self._state is not None:
            self._state.undo()
            self._after_edit()

    def redo(self) -> None:
        """Applies again the change that undo reverted."""
        if self._state is not None:
            self._state.redo()
            self._after_edit()

    def cycle(self, step: int) -> None:
        """Gives the active patch the layer `step` places on."""
        if self._state is not None:
            self._state.cycle(step)
            self._after_edit()

    def set_layer(self, layer: int) -> None:
        """Gives the active patch layer `layer` (0 the target); ignored when there is no such frame."""
        if self._state is not None and layer <= self._state.frame_count:
            self._state.set_label(layer)
            self._after_edit()

    def deselect(self) -> None:
        """Makes no patch active."""
        if self._state is not None:
            self._state.deselect()
            self._after_edit()

    def delete_patch(self) -> None:
        """Removes the active patch."""
        if self._state is not None:
            self._state.delete_active()
            self._after_edit()

    def toggle_tint(self) -> None:
        """Shows or hides the colour that marks which frame each region uses."""
        self.show_tint = not self.show_tint
        self.update()

    def set_pan_mode(self, on: bool) -> None:
        """Switches pan mode on or off. In pan mode a left-drag moves the picture, and nothing is painted or selected."""
        if on == self.pan_mode:
            return
        self.pan_mode = on
        self._update_cursor()
        self.update()
        self.pan_mode_changed.emit(on)
        self._announce()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Brush size, layer, selection and view keys."""
        key = event.key()
        if key == QtCore.Qt.Key.Key_BracketLeft:
            self.radius = max(BRUSH_RANGE[0], self.radius / BRUSH_STEP)
            self._announce()
        elif key == QtCore.Qt.Key.Key_BracketRight:
            self.radius = min(BRUSH_RANGE[1], self.radius * BRUSH_STEP)
            self._announce()
        elif key == QtCore.Qt.Key.Key_Tab:
            self.cycle(1)
        elif key == QtCore.Qt.Key.Key_Backtab:
            self.cycle(-1)
        elif key == QtCore.Qt.Key.Key_Space and not event.isAutoRepeat():
            self.set_pan_mode(not self.pan_mode)
        elif key == QtCore.Qt.Key.Key_Escape:
            if self.viewed_layer is not None:
                self.view_layer(None)
            elif self.pan_mode:
                self.set_pan_mode(False)
            else:
                self.deselect()
        elif key == QtCore.Qt.Key.Key_Delete:
            self.delete_patch()
        elif key == QtCore.Qt.Key.Key_L:
            self.toggle_tint()
        elif key == QtCore.Qt.Key.Key_T and not event.isAutoRepeat():
            self._show_target = True
            self.update()
        else:
            super().keyPressEvent(event)
        self.update()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        """Releasing T shows the merged result again."""
        if event.key() == QtCore.Qt.Key.Key_T and not event.isAutoRepeat():
            self._show_target = False
            self.update()
        else:
            super().keyReleaseEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        """Losing focus while T is held would never see its release, so show the result again."""
        self._show_target = False
        self.update()
        super().focusOutEvent(event)

    def focusNextPrevChild(self, next_child: bool) -> bool:
        """Keeps Tab and Shift+Tab for cycling layers instead of moving the focus."""
        return False

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Left starts a stroke or selects the pin under the cursor; middle starts a pan."""
        if self._state is None:
            return
        self.setFocus()
        if event.button() == QtCore.Qt.MouseButton.MiddleButton or (event.button() == QtCore.Qt.MouseButton.LeftButton and self.pan_mode):
            self._pan_from = event.position()
            self._update_cursor()
        elif event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.viewed_layer is not None:
                return
            pin = self._pin_at(event.position())
            if pin is not None:
                self._state.select(pin)
                self._after_edit()
                return
            erase = bool(event.modifiers() & (QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.AltModifier))
            self._stroke = "erase" if erase else "paint"
            self._state.begin_stroke()
            self._last = self.to_image(event.position())
            self._apply(*self._last)
            self._after_edit()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Follows the cursor with the brush circle, and extends a stroke or a pan."""
        self._mouse = event.position()
        if self._pan_from is not None:
            self._offset += event.position() - self._pan_from
            self._pan_from = event.position()
            self._fitted = False
        elif self._stroke is not None and self._last is not None:
            x, y = self.to_image(event.position())
            distance = float(np.hypot(x - self._last[0], y - self._last[1]))
            steps = max(1, int(np.ceil(distance / max(1.0, self.radius / 3.0))))
            for i in range(1, steps + 1):
                t = i / steps
                self._apply(self._last[0] + (x - self._last[0]) * t, self._last[1] + (y - self._last[1]) * t)
            self._last = (x, y)
            self._after_edit()
        self._update_cursor()
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Ends the stroke or the pan."""
        if self._pan_from is not None and event.button() in (QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.LeftButton):
            self._pan_from = None
            self._update_cursor()
        elif event.button() == QtCore.Qt.MouseButton.LeftButton and self._stroke is not None:
            self._state.end_stroke()
            self._stroke = None
            self._last = None
            self._after_edit()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Ctrl+wheel zooms about the cursor; a plain wheel or two-finger scroll pans, by the touchpad's pixel delta or 60 pixels a notch."""
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self._zoom_about(event.position(), 1.15 ** (event.angleDelta().y() / 120.0))
            return
        pixels = event.pixelDelta()
        if pixels.isNull():
            angle = event.angleDelta()
            delta = QtCore.QPointF(angle.x(), angle.y()) * (PAN_PER_NOTCH / 120.0)
        else:
            delta = QtCore.QPointF(pixels)
        self._offset += delta
        self._fitted = False
        self.update()

    def event(self, event: QtCore.QEvent) -> bool:
        """Handles a touchpad pinch as a zoom about the fingers; everything else goes to the usual handlers."""
        if event.type() == QtCore.QEvent.Type.NativeGesture and event.gestureType() == QtCore.Qt.NativeGestureType.ZoomNativeGesture:
            self._zoom_about(event.position(), 1.0 + event.value())
            return True
        return super().event(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Hides the brush circle when the cursor leaves."""
        self._mouse = None
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Keeps the picture fitted until the user has zoomed or panned."""
        if self._fitted and self._state is not None:
            self.zoom_fit()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Draws the picture, the optional layer tint, the active outline, the pins and the brush circle."""
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(40, 40, 40))
        if self._state is None:
            return
        rows, cols = self.pixels.shape[:2]
        target = QtCore.QRectF(self._offset.x(), self._offset.y(), cols * self.scale, rows * self.scale)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, self.scale < 1.0)
        if self.viewed_layer is not None:
            painter.drawImage(target, self._viewed(self.viewed_layer)[1])
            return
        painter.drawImage(target, self._target_image if self._show_target else self._image)
        if self.show_tint:
            painter.setOpacity(TINT_OPACITY)
            painter.drawImage(target, self._tint())
            painter.setOpacity(1.0)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self._paint_outline(painter)
        self._paint_pins(painter)
        self._paint_brush(painter)

    def _zoom_about(self, position: QtCore.QPointF, factor: float) -> None:
        """Multiplies the zoom by `factor`, within the zoom range, keeping the picture point under a widget position still."""
        x, y = (position.x() - self._offset.x()) / self.scale, (position.y() - self._offset.y()) / self.scale
        self.scale = min(max(self.scale * factor, ZOOM_RANGE[0]), ZOOM_RANGE[1])
        self._offset = QtCore.QPointF(position.x() - x * self.scale, position.y() - y * self.scale)
        self._fitted = False
        self.update()
        self._announce()

    def _viewed(self, layer: int) -> tuple[np.ndarray, QtGui.QImage]:
        """The 8-bit picture of a layer and a QImage sharing its memory, made on first use."""
        if layer not in self._layer_pixels:
            pixels = np.ascontiguousarray(to_uint8_display(self._state.layer_image(layer)))
            self._layer_pixels[layer] = pixels
            self._layer_images[layer] = self._make_image(pixels)
        return self._layer_pixels[layer], self._layer_images[layer]

    def _update_cursor(self) -> None:
        """A closed hand while the picture is being dragged, an open hand in pan mode, a pointing hand over a pin, otherwise none so the brush circle shows."""
        if self._pan_from is not None:
            shape = QtCore.Qt.CursorShape.ClosedHandCursor
        elif self.pan_mode:
            shape = QtCore.Qt.CursorShape.OpenHandCursor
        elif self.hovered_pin() is not None:
            shape = QtCore.Qt.CursorShape.PointingHandCursor
        else:
            shape = QtCore.Qt.CursorShape.BlankCursor
        self.setCursor(shape)

    def _apply(self, x: float, y: float) -> None:
        """One dab or erase of the current stroke."""
        if self._stroke == "erase":
            self._state.erase(x, y, self.radius)
        else:
            self._state.dab(x, y, self.radius)

    def _after_edit(self) -> None:
        """Re-renders the changed region of the picture, then redraws and tells the window."""
        box = self._state.take_dirty()
        if box is not None:
            self.pixels[box] = to_uint8_display(self._state.result_region(*box))
            self._tint_image = None
        self._outline = None
        self.update()
        self.edited.emit()
        self._announce()

    def _announce(self) -> None:
        """Tells the window the status line may have changed."""
        self.status_changed.emit(self.describe())

    def hovered_pin(self) -> int | None:
        """The index of the pin under the cursor, or None when the cursor is elsewhere or the view is not ready to select a patch."""
        if self._mouse is None or self._state is None or self.pan_mode or self._pan_from is not None or self._stroke is not None or self.viewed_layer is not None:
            return None
        return self._pin_at(self._mouse)

    def _pin_at(self, position: QtCore.QPointF) -> int | None:
        """The index of the topmost pin within reach of a widget position, or None."""
        for index in reversed(range(self._state.patch_count)):
            centre = self.to_widget(*self._state.centers()[index])
            if np.hypot(centre.x() - position.x(), centre.y() - position.y()) <= PIN_HIT:
                return index
        return None

    @staticmethod
    def _make_image(pixels: np.ndarray) -> QtGui.QImage:
        """A QImage that shares the memory of an (rows, cols, 3) uint8 array, which must stay alive."""
        rows, cols = pixels.shape[:2]
        return QtGui.QImage(pixels.data, cols, rows, cols * 3, QtGui.QImage.Format.Format_RGB888)

    def _tint(self) -> QtGui.QImage:
        """The RGBA tint of the current labels, rebuilt after an edit."""
        if self._tint_image is None:
            self._tint_array = np.ascontiguousarray(TINT_COLORS[self._state.labels() % len(TINT_COLORS)])
            rows, cols = self._tint_array.shape[:2]
            self._tint_image = QtGui.QImage(self._tint_array.data, cols, rows, cols * 4, QtGui.QImage.Format.Format_RGBA8888)
        return self._tint_image

    def _paint_outline(self, painter: QtGui.QPainter) -> None:
        """A white outline, with a dark one beneath so it shows on light pictures, around the active patch."""
        contours = self.outline()
        if not contours:
            return
        painter.save()
        painter.translate(self._offset.x() + 0.5 * self.scale, self._offset.y() + 0.5 * self.scale)
        painter.scale(self.scale, self.scale)
        for color, width in ((QtGui.QColor(0, 0, 0, 160), 4.0), (QtGui.QColor(255, 255, 255), 2.0)):
            pen = QtGui.QPen(color, width)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            for contour in contours:
                painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in contour]))
        painter.restore()

    def _paint_pins(self, painter: QtGui.QPainter) -> None:
        """A small ring at the middle of every patch, filled for the active one and larger under the cursor."""
        hovered = self.hovered_pin()
        for index, (x, y) in enumerate(self._state.centers()):
            centre = self.to_widget(x, y)
            radius = PIN_RADIUS * 1.5 if index == hovered else PIN_RADIUS
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 3.0))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawEllipse(centre, radius, radius)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1.5))
            painter.setBrush(QtGui.QColor(255, 214, 0) if index == self._state.active else QtCore.Qt.BrushStyle.NoBrush)
            painter.drawEllipse(centre, radius, radius)

    def _paint_brush(self, painter: QtGui.QPainter) -> None:
        """The brush circle at the cursor, red while Shift or Alt is held to erase."""
        if self._mouse is None or self._pan_from is not None or self.pan_mode or self.hovered_pin() is not None:
            return
        erasing = bool(
            QtWidgets.QApplication.keyboardModifiers() & (QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.AltModifier)
        )
        radius = self.radius * self.scale
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 200), 3.0))
        painter.drawEllipse(self._mouse, radius, radius)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 80, 80) if erasing else QtGui.QColor(255, 255, 255), 1.0))
        painter.drawEllipse(self._mouse, radius, radius)
