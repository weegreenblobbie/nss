"""The Layers panel: the target and every extra frame, so you can see what a patch can show, pick it, look at it and make one the target."""
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

THUMBNAIL_SIZE = QtCore.QSize(120, 80)
HINT = (
    "Paint a patch, then click a layer to show it there, or press Tab. "
    "Esc finishes the patch, so the next stroke starts a new one; click a pin to edit an earlier patch. "
    "Ctrl+click a layer to look at it (Esc goes back); right-click a frame to make it the target."
)


class LayerList(QtWidgets.QListWidget):
    """A list that remembers which keyboard modifiers were held when the mouse was last pressed on it."""

    def __init__(self) -> None:
        """An empty list."""
        super().__init__()
        self.press_modifiers = QtCore.Qt.KeyboardModifier.NoModifier

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Notes the modifiers, then lets the list handle the press."""
        self.press_modifiers = event.modifiers()
        super().mousePressEvent(event)


class LayersPanel(QtWidgets.QWidget):
    """A list with the target first and then one row per extra frame (and the median, in clean mode), each with a thumbnail and its name.

    Clicking a row announces that layer; Ctrl+click announces that it should be looked at; the right-click menu of a frame offers
    to make it the target. `show_active` highlights the layer the active patch shows, `show_viewed` marks the one being looked at.
    """

    layer_chosen = QtCore.pyqtSignal(int)  # the user clicked layer k (0 the target)
    layer_viewed = QtCore.pyqtSignal(int)  # the user Ctrl+clicked layer k
    target_requested = QtCore.pyqtSignal(int)  # the user asked for frame k to be the target

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """An empty panel; fill it with set_layers."""
        super().__init__(parent)
        self._frame_count = 0
        self.list = LayerList()
        self.list.setIconSize(THUMBNAIL_SIZE)
        self.list.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        palette = self.list.palette()
        for role in (QtGui.QPalette.ColorRole.Highlight, QtGui.QPalette.ColorRole.HighlightedText):
            palette.setColor(QtGui.QPalette.ColorGroup.Inactive, role, palette.color(QtGui.QPalette.ColorGroup.Active, role))
        self.list.setPalette(palette)
        self.list.itemClicked.connect(self._clicked)
        self.list.customContextMenuRequested.connect(self._show_menu)
        self.hint = QtWidgets.QLabel(HINT)
        self.hint.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list)
        layout.addWidget(self.hint)

    def set_layers(self, frame_names: list[str], thumbnails: list[np.ndarray], frame_count: int) -> None:
        """Lists the target and then each of `frame_names`; `thumbnails` are (rows, cols, 3) uint8 pictures, the target's first.

        The first `frame_count` of the names are frames that can be made the target; any name after them (the median) cannot.
        """
        self._frame_count = frame_count
        self.list.clear()
        for layer, thumbnail in enumerate(thumbnails):
            name = "Target" if layer == 0 else frame_names[layer - 1]
            item = QtWidgets.QListWidgetItem(name)
            pixmap = self._pixmap(thumbnail)
            icon = QtGui.QIcon()
            icon.addPixmap(pixmap, QtGui.QIcon.Mode.Normal)
            icon.addPixmap(pixmap, QtGui.QIcon.Mode.Selected)  # without this Qt tints the thumbnail of the selected row
            item.setIcon(icon)
            self.list.addItem(item)

    def show_active(self, layer: int | None) -> None:
        """Highlights row `layer`, or clears the highlight when no patch is active. Does not emit layer_chosen."""
        self.list.setCurrentRow(-1 if layer is None else layer)
        if layer is None:
            self.list.clearSelection()

    def show_viewed(self, layer: int | None) -> None:
        """Puts row `layer` in bold, the layer being looked at, and every other row in the normal font."""
        for row in range(self.list.count()):
            font = self.list.item(row).font()
            font.setBold(row == layer)
            self.list.item(row).setFont(font)

    def target_menu(self, row: int) -> QtWidgets.QMenu | None:
        """The right-click menu of a row: one action to make a frame the target, or None for the target and the median."""
        if not 1 <= row <= self._frame_count:
            return None
        menu = QtWidgets.QMenu(self.list)
        action = menu.addAction(f"Set {self.list.item(row).text()} as the target")
        action.triggered.connect(lambda checked=False: self.target_requested.emit(row))
        return menu

    def _clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """A click chooses the layer; with Ctrl held it asks to look at it instead."""
        row = self.list.row(item)
        if self.list.press_modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.layer_viewed.emit(row)
        else:
            self.layer_chosen.emit(row)

    def _show_menu(self, position: QtCore.QPoint) -> None:
        """Pops up the right-click menu of the row under `position`."""
        item = self.list.itemAt(position)
        menu = None if item is None else self.target_menu(self.list.row(item))
        if menu is not None:
            menu.exec(self.list.viewport().mapToGlobal(position))

    @staticmethod
    def _pixmap(thumbnail: np.ndarray) -> QtGui.QPixmap:
        """A pixmap copy of an (rows, cols, 3) uint8 picture."""
        pixels = np.ascontiguousarray(thumbnail)
        rows, cols = pixels.shape[:2]
        return QtGui.QPixmap.fromImage(QtGui.QImage(pixels.data, cols, rows, cols * 3, QtGui.QImage.Format.Format_RGB888))
